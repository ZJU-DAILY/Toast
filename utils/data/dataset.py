import os
import random
import tqdm
import math
from chinese_calendar import is_holiday
import numpy as np
from typing import List
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops

from utils.data.point import SPoint, LAT_PER_METER, LNG_PER_METER
from utils.data.trajectory import Trajectory
from utils.parser.parser import ParseMMTraj


class RecoveryDataset(Dataset):
    def __init__(self, traj_dir, road_net, region, mode, time_interval, win_size, grid_size, ds_type, keep_ratio, neighbor_dist, search_dist, beta, gamma):
        self.road_net = road_net
        self.mode = mode
        self.region = region
        self.time_interval = time_interval
        self.neighbor_dist = neighbor_dist
        self.search_dist = search_dist
        self.beta = beta
        self.gamma = gamma
        self.grid_size = grid_size
        self.src_seq, self.tgt_seq = self.process(traj_dir, win_size, ds_type, keep_ratio)

    def __len__(self):
        return len(self.src_seq[0])

    def __getitem__(self, index):
        src_grid_seq = self.src_seq[0][index]
        src_gps_seq = self.src_seq[1][index]
        tgt_gps_seq = self.tgt_seq[0][index]
        tgt_rid = self.tgt_seq[1][index]
        tgt_rate = self.tgt_seq[2][index]

        src_grid_seq = self.add_start_token(src_grid_seq)
        src_gps_seq = self.add_start_token(src_gps_seq)
        tgt_gps_seq = self.add_start_token(tgt_gps_seq)
        tgt_rid = self.add_start_token(tgt_rid)
        tgt_rate = self.add_start_token(tgt_rate)
        src_feat = torch.tensor(self.src_seq[2][index], dtype=torch.float)
        src_len = torch.tensor([len(src_grid_seq)])
        tgt_len = torch.tensor([len(tgt_gps_seq)])

        src_influence, tgt_influence = self.get_influence_matrix(src_grid_seq, src_gps_seq, src_len, tgt_len)
        src_subgraphs, node_index = self.build_subgraph(src_influence, src_grid_seq, tgt_rid)
        return src_grid_seq, src_gps_seq, src_feat, tgt_gps_seq, tgt_rid, tgt_rate, tgt_influence, src_subgraphs, node_index

    def process(self, traj_dir, win_size, ds_type, keep_ratio):
        parser = ParseMMTraj(self.road_net)
        src_file = os.path.join(traj_dir, "{}/{}_input.txt".format(self.mode, self.mode))
        tgt_file = os.path.join(traj_dir, "{}/{}_output.txt".format(self.mode, self.mode))

        src_trajs = parser.parse(src_file, is_target=False, is_save=True)
        tgt_trajs = parser.parse(tgt_file, is_target=True, is_save=True)
        assert len(src_trajs) == len(tgt_trajs)

        src_grid_seq, src_gps_seq, src_feat_seq = [], [], []
        tgt_gps_seq, tgt_idx_seq, tgt_rate_seq = [], [], []
        keep_ratio = 1. if self.mode == "test" else keep_ratio
        for (src_traj, tgt_traj) in tqdm.tqdm(zip(src_trajs, tgt_trajs), total=len(src_trajs)):
            grid_seq, gps_seq, feat_seq = self.process_src_traj(src_traj, win_size, ds_type, keep_ratio)
            mm_gps_seq, mm_rid_seq, mm_rate_seq = self.process_tgt_traj(tgt_traj, win_size)

            src_grid_seq.extend(grid_seq)
            src_gps_seq.extend(gps_seq)
            src_feat_seq.extend(feat_seq)
            tgt_gps_seq.extend(mm_gps_seq)
            tgt_idx_seq.extend(mm_rid_seq)
            tgt_rate_seq.extend(mm_rate_seq)

        assert len(src_grid_seq) == len(src_gps_seq) == len(src_feat_seq) == \
               len(tgt_gps_seq) == len(tgt_idx_seq) == len(tgt_rate_seq),\
            "The number of source and target sequence must be equal."
        return (src_grid_seq, src_gps_seq, src_feat_seq), (tgt_gps_seq, tgt_idx_seq, tgt_rate_seq)

    def process_src_traj(self, traj, win_size, ds_type, keep_ratio):
        traj_splits = self.split_traj(traj, win_size)
        grid_seq, gps_seq, feat_seq = [], [], []
        for sub_traj in traj_splits:
            pt_list = sub_traj.pt_list
            if keep_ratio != 1:
                ds_pt_list = self.downsample_traj(pt_list, ds_type, keep_ratio)
            else:
                ds_pt_list = pt_list
            hours = []
            ls_grid_seq = []
            ls_gps_seq = []
            first_pt = ds_pt_list[0]
            for ds_pt in ds_pt_list:
                hours.append(ds_pt.time.hour)
                t = self.normalize_time(first_pt, ds_pt, self.time_interval)
                ls_gps_seq.append([ds_pt.lat, ds_pt.lng])
                grid_x, grid_y = self.road_net.point2grid(ds_pt, self.grid_size)
                ls_grid_seq.append([grid_x, grid_y, t])
            feat = self.get_pro_feats(ds_pt_list, hours)
            grid_seq.append(ls_grid_seq)
            gps_seq.append(ls_gps_seq)
            feat_seq.append(feat)
        return grid_seq, gps_seq, feat_seq

    def process_tgt_traj(self, traj, win_size):
        traj_splits = self.split_traj(traj, win_size)
        mm_gps_seq, mm_eid_seq, mm_rate_seq = [], [], []
        for sub_traj in traj_splits:
            pt_list = sub_traj.pt_list
            sub_gps_seq, sub_eid_seq, sub_rate_seq = [], [], []
            for pt in pt_list:
                sub_gps_seq.append([pt.lat, pt.lng])
                sub_eid_seq.append([self.road_net.reindex_nodes[pt.segment_id] + 1])
                sub_rate_seq.append([pt.rate])
            mm_gps_seq.append(sub_gps_seq)
            mm_eid_seq.append(sub_eid_seq)
            mm_rate_seq.append(sub_rate_seq)
        return mm_gps_seq, mm_eid_seq, mm_rate_seq

    @staticmethod
    def add_start_token(seq):
        dimension = len(seq[0])
        start_seq = [0] * dimension
        return torch.tensor([start_seq] + seq)

    @staticmethod
    def split_traj(traj, win_size):
        pt_list = traj.pt_list
        traj_len = len(pt_list)
        if traj_len < win_size:
            return [traj]

        num_win = traj_len // win_size
        last_win_size = traj_len % win_size + 1
        split_traj = []
        for w in range(num_win):
            if w == num_win - 1 and last_win_size <= 15:
                idx = win_size * w - 1 if win_size * w - 1 > 0 else 0
                sub_traj = pt_list[idx:]
            else:
                sub_traj = pt_list[max(0, win_size * w - 1):win_size * (w + 1)]
            sub_traj = Trajectory(sub_traj)
            split_traj.append(sub_traj)
        return split_traj

    @staticmethod
    def downsample_traj(pt_list, ds_type, keep_ratio):
        assert ds_type in ["uniform", "random"], "only `uniform` or `random` is supported"

        old_pt_list = pt_list.copy()
        start_pt = old_pt_list[0]
        end_pt = old_pt_list[-1]
        if ds_type == "uniform":
            if (len(old_pt_list) - 1) % int(1 / keep_ratio) == 0:
                new_pt_list = old_pt_list[::int(1 / keep_ratio)]
            else:
                new_pt_list = old_pt_list[::int(1 / keep_ratio)] + [end_pt]
        else:
            sampled_inds = sorted(
                random.sample(range(1, len(old_pt_list) - 1), int((len(old_pt_list) - 2) * keep_ratio)))
            new_pt_list = [start_pt] + list(np.array(old_pt_list)[sampled_inds]) + [end_pt]
        return new_pt_list

    @staticmethod
    def normalize_time(start_pt, end_pt, time_interval):
        norm_time = int(1 + ((end_pt.time - start_pt.time).seconds / time_interval))
        return norm_time

    @staticmethod
    def get_pro_feats(pt_list, hours):
        holiday = is_holiday(pt_list[0].time) * 1
        freq_hour = np.bincount(hours).argmax()
        encoded_hour = [0] * 24
        encoded_hour[freq_hour] = 1
        feat = encoded_hour + [holiday]
        return feat

    def get_influence_matrix(self, grid_seq, gps_seq, src_seq_len, tgt_seq_len):
        num_nodes = len(self.road_net.node_lst) + 1
        src_influence_score = torch.zeros((src_seq_len, num_nodes), dtype=torch.float)
        tgt_influence_score = torch.zeros((tgt_seq_len, num_nodes), dtype=torch.float) + 1e-6
        pre_t = 1
        pre_pt = SPoint(*gps_seq[pre_t].tolist())
        src_influence_score[pre_t] = self.cal_influence_score(pre_pt, self.neighbor_dist, self.gamma)
        tgt_influence_score[pre_t] = self.cal_influence_score(pre_pt, self.search_dist, self.beta)
        for idx in range(2, src_seq_len):
            cur_t = grid_seq[idx, 2].tolist()
            cur_pt = SPoint(*gps_seq[idx].tolist())
            for t in range(pre_t + 1, cur_t):
                tgt_influence_score[t] = 1.
            src_influence_score[idx] = self.cal_influence_score(cur_pt, self.neighbor_dist, self.gamma)
            tgt_influence_score[cur_t] = self.cal_influence_score(cur_pt, self.search_dist, self.beta)
            pre_t = cur_t
        tgt_influence_score = torch.clip(tgt_influence_score, 1e-6, 1)
        return src_influence_score, tgt_influence_score

    def cal_influence_score(self, pt, search_dist, gamma):
        influence_score = torch.zeros(len(self.road_net.node_lst) + 1, dtype=torch.float)
        region = (pt.lng - search_dist * LNG_PER_METER, pt.lat - search_dist * LAT_PER_METER,
                  pt.lng + search_dist * LNG_PER_METER, pt.lat + search_dist * LAT_PER_METER)
        query_results = self.road_net.range_query(pt, *region)
        if query_results is not None:
            for pt in query_results:
                nid = self.road_net.reindex_nodes[pt.segment_id] + 1
                influence_score[nid] = math.exp(-pow(pt.error, 2) / pow(gamma, 2))
        else:
            influence_score = torch.ones(len(self.road_net.node_lst) + 1, dtype=torch.float)
        return influence_score

    def build_subgraph(self, influence_mat, grid_seq, nid_seq):
        neighbors = self.road_net.get_neighbors()
        src_len = influence_mat.size(0)
        subgraphs, node_index = [], []
        for idx in range(src_len):
            if idx == 0:
                subg = Data(edge_index=torch.tensor([]), weight=torch.tensor([1.]), gt=torch.tensor([1.]))
                subg.num_nodes = 1
                subg.edge_index, _ = add_self_loops(subg.edge_index, num_nodes=subg.num_nodes)
                node_index.append(0)
            else:
                nodes = torch.where(influence_mat[idx] > 0)[0].tolist()
                if nid_seq[grid_seq[idx][-1]] not in nodes:
                    nodes.append(nid_seq[grid_seq[idx][-1]].item())
                neighbor_nodes = []
                for n in nodes:
                    neighbor_nodes.extend(neighbors[n - 1])
                neighbor_nodes = [nid + 1 for nid in neighbor_nodes]
                nodes = list(set.union(set(nodes), set(neighbor_nodes)))
                node_index.extend(nodes)
                reindex_map = {nid: rid for rid, nid in enumerate(nodes)}
                edge_index, weight = [], []
                for n in nodes:
                    weight.append(influence_mat[idx][n] / influence_mat[idx].sum())
                    for nb in neighbors[n - 1]:
                        if ((nb + 1) in reindex_map) and ((nb + 1) != n):
                            edge_index.append([reindex_map[n], reindex_map[nb + 1]])
                if len(edge_index) > 0:
                    edge_index = torch.tensor(edge_index, dtype=torch.long).T
                    edge_index, _ = add_self_loops(edge_index, num_nodes=len(nodes))
                else:
                    edge_index, _ = add_self_loops(torch.tensor([], dtype=torch.long), num_nodes=len(nodes))
                weight = torch.tensor(weight)
                gt = torch.zeros_like(weight)
                gt[reindex_map[nid_seq[grid_seq[idx][-1]].item()]] = 1.
                subg = Data(edge_index=edge_index, weight=weight, gt=gt)
                subg.num_nodes = len(nodes)
            subgraphs.append(subg)
        node_index = torch.tensor(node_index)
        return Batch.from_data_list(subgraphs), node_index

    @staticmethod
    def collate_fn(batch):
        def collate_batch(idx):
            return [sample[idx] for sample in batch]

        def pad_seq(seq_list, pad_val=.0):
            seq_len = [len(seq) for seq in seq_list]
            dimension = seq_list[0].size(1)
            pad_seqs = torch.full((len(seq_list), max(seq_len), dimension), pad_val)
            for idx, seq in enumerate(seq_list):
                length = seq_len[idx]
                pad_seqs[idx, :length] = seq
            return pad_seqs, torch.tensor(seq_len)

        def pad_graph(graph_list: List[Batch], node_idx_list: List[torch.Tensor], max_seq_len):
            pad_graph_list, pad_node_list = [], []
            for graph, node_list in zip(graph_list, node_idx_list):
                pad_len = max_seq_len - graph.num_graphs
                if pad_len > 0:
                    pad_g = Data(edge_index=torch.tensor([], dtype=torch.long),
                                 weight=torch.tensor([1.]), gt=torch.tensor([1.]))
                    pad_g.num_nodes = 1
                    pad_g.edge_index, _ = add_self_loops(pad_g.edge_index, num_nodes=pad_g.num_nodes)
                    pad_batch = Batch.from_data_list([pad_g] * pad_len)
                    pad_node_list.append(torch.tensor(node_list.tolist() + [0] * pad_len))
                    pad_graph_list.append(Batch.from_data_list([graph, pad_batch]))
                else:
                    if hasattr(graph, "ptr"):
                        del graph.ptr
                    pad_node_list.append(node_list)
                    pad_graph_list.append(graph)
            batched_graph = Batch.from_data_list(pad_graph_list)
            batched_node_index = torch.cat(pad_node_list, dim=-1)
            return batched_graph, batched_node_index

        batch.sort(key=lambda x: len(x[0]), reverse=True)
        src_grid_seq, src_seq_len = pad_seq(collate_batch(0))
        src_gps_seq, _ = pad_seq(collate_batch(1))
        feat_seq = torch.stack(collate_batch(2), dim=0)
        tgt_gps_seq, tgt_seq_len = pad_seq(collate_batch(3))
        tgt_node_idx, _ = pad_seq(collate_batch(4))
        tgt_rate_seq, _ = pad_seq(collate_batch(5))
        tgt_inf_scores, _ = pad_seq(collate_batch(6))
        graph_batch, node_index_batch = pad_graph(collate_batch(7), collate_batch(8), src_seq_len.max())
        return src_grid_seq, src_gps_seq, feat_seq, src_seq_len, tgt_gps_seq, tgt_node_idx, tgt_rate_seq, tgt_seq_len, tgt_inf_scores, graph_batch, node_index_batch

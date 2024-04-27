import os
import random
import tqdm
from chinese_calendar import is_holiday
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

from utils.data import SPoint, LAT_PER_METER, LNG_PER_METER
from utils.data import Trajectory
from utils.parser import ParseMMTraj


class RecoveryDataset(Dataset):
    def __init__(self, traj_dir, road_net, region, mode, time_interval, win_size, grid_size, ds_type, keep_ratio, search_dist, gamma):
        self.road_net = road_net
        self.mode = mode
        self.region = region
        self.time_interval = time_interval
        self.search_dist = search_dist
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
        src_feat = torch.tensor(self.src_seq[2][index])
        src_len = torch.tensor([len(src_grid_seq)])
        tgt_len = torch.tensor([len(tgt_gps_seq)])

        src_influence, tgt_influence = self.get_influence_matrix(src_grid_seq, src_gps_seq, src_len, tgt_len)
        src_subgraphs = self.build_subgraph(src_influence, src_grid_seq, tgt_rid)
        return src_grid_seq, src_gps_seq, src_feat, tgt_gps_seq, tgt_rid, tgt_rate, tgt_influence, src_subgraphs

    def process(self, traj_dir, win_size, ds_type, keep_ratio):
        parser = ParseMMTraj(self.road_net)
        src_file = os.path.join(traj_dir, "{}/{}_input.txt".format(self.mode, self.mode))
        tgt_file = os.path.join(traj_dir, "{}/{}_output.txt".format(self.mode, self.mode))

        src_trajs = parser.parse(src_file, is_target=False)
        tgt_trajs = parser.parse(tgt_file, is_target=True)
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
                locgrid_xid, locgrid_yid = self.gps2grid(ds_pt, self.grid_size)
                ls_grid_seq.append([locgrid_xid, locgrid_yid, t])
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
        return [start_seq] + seq

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

    def gps2grid(self, pt, grid_size):
        lat_unit = LAT_PER_METER * grid_size
        lng_unit = LNG_PER_METER * grid_size

        lat = pt.lat
        lng = pt.lng
        min_lat, min_lng = self.region[0], self.region[1]
        grid_x = int((lat - min_lat) / lat_unit) + 1
        grid_y = int((lng - min_lng) / lng_unit) + 1
        return grid_x, grid_y

    def get_influence_matrix(self, grid_seq, gps_seq, src_seq_len, tgt_seq_len):
        num_nodes = len(self.road_net.node_set) + 1
        src_influence_score = torch.zeros((src_seq_len, num_nodes), dtype=torch.float)
        tgt_influence_score = torch.zeros((tgt_seq_len, num_nodes), dtype=torch.float) + 1e-6
        pre_t = 1
        pre_pt = SPoint(*gps_seq[pre_t])
        src_influence_score[pre_t] = self.cal_influence_score(pre_pt, self.search_dist, self.gamma)
        tgt_influence_score[pre_t] = self.cal_influence_score(pre_pt, self.search_dist, self.gamma)
        for idx in range(2, src_seq_len):
            cur_t = grid_seq[idx, 2].tolist()
            cur_pt = SPoint(*gps_seq[idx])
            for t in range(pre_t + 1, cur_t):
                tgt_influence_score[t] = 1.
            src_influence_score[idx] = self.cal_influence_score(cur_pt, self.search_dist, self.gamma)
            tgt_influence_score[cur_t] = self.cal_influence_score(cur_pt, self.search_dist, self.gamma)
            pre_t = cur_t
        tgt_influence_score = torch.clip(tgt_influence_score, 1e-6, 1)
        return src_influence_score, tgt_influence_score

    def cal_influence_score(self, pt, search_dist, gamma):
        influence_score = torch.zeros(len(self.road_net.node_set) + 1, dtype=torch.float)
        region = (pt.lng - search_dist * LNG_PER_METER, pt.lat - search_dist * LAT_PER_METER,
                  pt.lng + search_dist * LNG_PER_METER, pt.lat + search_dist * LAT_PER_METER)
        query_results = self.road_net.range_query(pt, *region)
        if len(query_results) == 0:
            influence_score = torch.ones(len(self.road_net.node_set) + 1, dtype=torch.float)
        else:
            for pt in query_results:
                nid = self.road_net.reindex_nodes[pt.segment_id]
                influence_score[nid] = torch.exp(-pt.error ** 2 / gamma ** 2)
        return influence_score

    def build_subgraph(self, influence_mat, grid_seq, nid_seq):
        # todo
        neighbors = self.road_net.get_neighbors()
        src_len = influence_mat.size(0)
        subgraphs = []
        for idx in range(src_len):
            if idx == 0:
                subg = Data(edge_index=None, node_index=torch.tensor(0),
                            node_weight=torch.tensor([1.]), y=torch.tensor([1.]))
                subgraphs.append(subg)
            else:
                node_index = torch.where(influence_mat[idx] > 0)[0].tolist()
                if nid_seq[grid_seq[idx][-1]] not in node_index:
                    node_index.append(nid_seq[grid_seq[idx][-1]])
        return subgraphs

    @staticmethod
    def collate_fn(batch):
        def recollect_batch(idx):
            return [sample[idx] for sample in batch]

        def pad_seq(seq_batch, pad_val=.0):
            seq_len = [len(seq) for seq in seq_batch]
            dimension = seq_batch[0].size(1)
            pad_seqs = torch.full((len(seq_batch), max(seq_len), dimension), pad_val)
            for idx, seq in enumerate(seq_batch):
                length = seq_len[idx]
                pad_seqs[idx, :length] = seq
            return pad_seqs, torch.tensor(seq_len)

        batch.sort(key=lambda x: len(x[0]), reverse=True)
        src_grid_seq, src_seq_len = pad_seq(recollect_batch(0))
        src_gps_seq, _ = pad_seq(recollect_batch(1))
        feat_seq = torch.tensor(recollect_batch(2))
        tgt_gps_seq, tgt_seq_len = pad_seq(recollect_batch(3))
        tgt_node_idx, _ = pad_seq(recollect_batch(4))
        tgt_rate_seq, _ = pad_seq(recollect_batch(5))
        tgt_inf_scores = pad_seq(recollect_batch(6))
        return src_grid_seq, src_gps_seq, feat_seq, src_seq_len, tgt_gps_seq, tgt_node_idx, tgt_rate_seq, tgt_seq_len, tgt_inf_scores

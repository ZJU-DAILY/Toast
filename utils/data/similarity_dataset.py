import os
import pickle as pkl
import pandas as pd
import datetime
import numpy as np
import torch
from torch.utils.data import Dataset


class SimilarityDataset(Dataset):
    model_type = None

    def __init__(
            self,
            traj_dir,
            groundtruth_dir,
            mode,
            model_type,
            train_size=10000,
            valid_size=4000,
            test_size=16000
    ):
        self.mode = mode
        SimilarityDataset.model_type = model_type
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.node_data, self.time_data, self.seq_len, self.dist_data = self.process(traj_dir, groundtruth_dir)

    def __len__(self):
        assert len(self.node_data) == len(self.time_data) == len(self.seq_len) == len(self.dist_data), \
            f"{len(self.node_data)}\t{len(self.time_data)}\t{len(self.seq_len)}\t{len(self.dist_data)}\n"
        return len(self.node_data)

    def __getitem__(self, item):
        if self.mode == "train":
            return (
                torch.tensor(self.node_data[item][0], dtype=torch.long).unsqueeze(dim=1),
                torch.tensor(self.time_data[item][0], dtype=torch.float),
                torch.tensor(self.seq_len[item][0], dtype=torch.long),
                torch.tensor(self.node_data[item][1], dtype=torch.long).unsqueeze(dim=1),
                torch.tensor(self.time_data[item][1], dtype=torch.float),
                torch.tensor(self.seq_len[item][1], dtype=torch.long),
                torch.tensor(self.node_data[item][2], dtype=torch.long).unsqueeze(dim=1),
                torch.tensor(self.time_data[item][2], dtype=torch.float),
                torch.tensor(self.seq_len[item][2], dtype=torch.long),
                torch.tensor(self.dist_data[item][0], dtype=torch.float),
                torch.tensor(self.dist_data[item][1], dtype=torch.float)
            )
        else:
            return (
                torch.tensor(self.node_data[item], dtype=torch.long).unsqueeze(dim=1),
                torch.tensor(self.time_data[item], dtype=torch.float),
                torch.tensor(self.seq_len[item], dtype=torch.long),
                torch.tensor(self.dist_data[item], dtype=torch.float)
            )

    def process(self, traj_dir, gt_dir=None):
        if self.mode == "train":
            node_triple_path = os.path.join(traj_dir, "triplet/TP/node_triplets_2w_STBall")
            with open(node_triple_path, "rb") as n_file:
                node_triples = pkl.load(n_file)
            dist_triple_path = os.path.join(gt_dir, "TP/train_triplet_2w_STBall.npy")
            dist_data = np.load(dist_triple_path)
            time_triple_path = os.path.join(traj_dir, "triplet/TP/d2vec_triplets_2w_STBall")
            with open(time_triple_path, "rb") as t_file:
                transform_time_triples = pkl.load(t_file)
            seq_len_triples = []
            for a_seq, p_seq, n_seq in transform_time_triples:
                seq_len = [len(a_seq), len(p_seq), len(n_seq)]
                seq_len_triples.append(seq_len)
            return node_triples, transform_time_triples, seq_len_triples, dist_data
        else:
            node_path = os.path.join(traj_dir, "st_traj/shuffle_node_list.npy")
            node_list = np.load(node_path, allow_pickle=True)
            time_path = os.path.join(traj_dir, "st_traj/shuffle_d2vec_list.npy")
            time_data = np.load(time_path, allow_pickle=True)
            lengths = []
            for n_seq, t_seq in zip(node_list, time_data):
                assert len(n_seq) == len(t_seq), "length of node list and time list should be equal"
                lengths.append(len(n_seq))
            if self.mode == "valid":
                dist_path = os.path.join(gt_dir, "TP/vali_st_distance.npy")
                dist = np.load(dist_path)
                num_tests, dimension = dist.shape
                dist_data = np.full((self.valid_size, dimension), -1, dtype=float)
                dist_data[:num_tests] = dist
                valid_end = self.train_size + self.valid_size
                return (
                    node_list[self.train_size:valid_end],
                    time_data[self.train_size:valid_end],
                    lengths[self.train_size:valid_end],
                    dist_data
                )
            else:
                dist_path = os.path.join(gt_dir, "TP/test_st_distance.npy")
                dist = np.load(dist_path)
                num_tests, dimension = dist.shape
                dist_data = np.full((self.test_size, dimension), -1, dtype=float)
                dist_data[:num_tests] = dist
                valid_end = self.train_size + self.valid_size
                test_end = valid_end + self.test_size
                node_data = node_list[valid_end:test_end]
                time_data = time_data[valid_end:test_end]
                lengths = lengths[valid_end:test_end]
                dist_data = dist_data[:min(node_data.shape[0], self.test_size)]
                return (
                    node_data,
                    time_data,
                    lengths,
                    dist_data
                )

    @staticmethod
    def collate_fn(batch):
        def collate_batch(idx):
            return [sample[idx] for sample in batch]

        def padding(seq_list, seq_len, pad_val=.0):
            dimension = seq_list[0].shape[-1]
            pad_seqs = torch.full((len(seq_list), max(seq_len), dimension), pad_val)
            for idx, seq in enumerate(seq_list):
                length = seq_len[idx]
                pad_seqs[idx, :length] = seq
            return pad_seqs, torch.tensor(seq_len)

        seq_len = collate_batch(2)
        batch_anchor_nodes, anchor_length_seq = padding(collate_batch(0), seq_len, 0)
        batch_anchor_times, _ = padding(collate_batch(1), seq_len, 0.)
        if len(batch[0]) > 4:
            seq_len = collate_batch(5)
            batch_pos_nodes, pos_length_seq = padding(collate_batch(3), seq_len, 0)
            batch_pos_times, _ = padding(collate_batch(4), seq_len, 0.)
            seq_len = collate_batch(8)
            batch_neg_nodes, neg_length_seq = padding(collate_batch(6), seq_len, 0)
            batch_neg_times, _ = padding(collate_batch(7), seq_len, 0.)
            pos_dist = torch.exp(-torch.tensor(collate_batch(9)))
            neg_dist = torch.exp(-torch.tensor(collate_batch(10)))
            return (
                batch_anchor_nodes, batch_anchor_times, anchor_length_seq,
                batch_pos_nodes, batch_pos_times, pos_length_seq,
                batch_neg_nodes, batch_neg_times, neg_length_seq,
                pos_dist, neg_dist
            )
        else:
            dist = torch.stack(collate_batch(3), dim=0)
            return batch_anchor_nodes, batch_anchor_times, anchor_length_seq, dist

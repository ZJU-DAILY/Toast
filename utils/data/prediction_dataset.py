import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class Scaler:
    def __init__(self, data):
        self.mean = np.mean(data)
        self.std = np.std(data)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


class PredictDataset(Dataset):
    scaler = None

    def __init__(
            self,
            data_dir,
            feat_dim,
            features,
            mode,
            input_len,
            output_len,
            train_ratio=0.6,
            valid_ratio=0.2,
    ):
        self.data_dir = data_dir
        self.feat_dim = feat_dim
        if feat_dim < features.shape[-1]:
            features[feat_dim:] = 0.
        self.features = features
        self.mode = mode
        self.input_len = input_len
        self.output_len = output_len
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.flow_data, self.feat_data, self.labels = self.process(data_dir, "data")

    def __len__(self):
        return self.flow_data.shape[0]

    def __getitem__(self, item):
        return self.flow_data[item], self.feat_data[item], self.labels[item]

    def construct_dataset(self, data, features):
        mask = np.sum(data, axis=(1, 2)) > 5000
        data = self.scaler.transform(data)
        num_time, num_nodes, _ = data.shape
        timestamps = (np.arange(num_time) % 24) / 24
        timestamps = np.tile(timestamps, (1, num_nodes, 1)).T
        transform_data = np.concatenate((data, timestamps), axis=2)
        split_data, split_feat, split_label = [], [], []
        for i in range(num_time - self.input_len - self.output_len + 1):
            if mask[i + self.input_len:i + self.input_len + self.output_len].sum() != self.output_len:
                continue
            split_data.append(transform_data[i:i + self.input_len])
            split_feat.append(features)
            split_label.append(transform_data[i + self.input_len:i + self.input_len + self.output_len])
        split_data, split_feat, split_label = (
            np.stack(split_data, axis=0),
            np.stack(split_feat, axis=0),
            np.stack(split_label, axis=0)
        )
        return (
            torch.tensor(split_data, dtype=torch.float32),
            torch.tensor(split_feat, dtype=torch.float32),
            torch.tensor(split_label, dtype=torch.float32)
        )

    def process(self, data_dir, key):
        print("[read traffic flow data...]")
        flow_path = os.path.join(data_dir, "BJ_FLOW.h5")
        with h5py.File(flow_path, 'r') as flow_f:
            flow_data = np.array(flow_f[key])
        days, hours, rows, cols, _ = flow_data.shape
        flow_data = flow_data.reshape((days * hours, rows * cols, -1))

        data_size = flow_data.shape[0]
        train_size = int(data_size * self.train_ratio)
        valid_size = int(data_size * self.valid_ratio)
        test_size = data_size - train_size - valid_size
        if self.mode == "train":
            PredictDataset.scaler = Scaler(flow_data[:train_size])
            return self.construct_dataset(flow_data[:train_size], self.features)
        elif self.mode == "valid":
            return self.construct_dataset(flow_data[train_size:train_size + valid_size], self.features)
        else:
            return self.construct_dataset(flow_data[-test_size:], self.features)

    @staticmethod
    def collate_fn(batch):
        def collate_batch(idx):
            return [sample[idx] for sample in batch]
        batch_data = torch.stack(collate_batch(0), dim=0)
        batch_feat = torch.stack(collate_batch(1), dim=0)
        batch_label = torch.stack(collate_batch(2), dim=0)
        return batch_data, batch_feat, batch_label

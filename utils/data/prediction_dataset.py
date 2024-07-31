import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class Scaler:
    def __init__(self, data):
        self.mean = np.mean(data)
        self.std = np.std(data)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


class PredictDataset(Dataset):
    model_name = None
    scaler = None

    def __init__(
            self,
            data_dir,
            features,
            mode,
            model_name,
            train_ratio=0.8,
            valid_ratio=0.1,
            **kwargs
    ):
        self.data_dir = data_dir
        self.features = torch.tensor(features)
        self.mode = mode
        PredictDataset.model_name = model_name
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        if model_name == "STMetaNet":
            self.input_len = kwargs["input_len"]
            self.output_len = kwargs["output_len"]
            self.flow_data, self.labels = self.process(data_dir, "data")
        else:
            self.close_len = kwargs["closeness_length"]
            self.period_len = kwargs["period_length"]
            self.trend_len = kwargs["trend_length"]
            self.period_interval = kwargs["period_interval"]
            self.trend_interval = kwargs["trend_interval"]
            self.close_data, self.period_data, self.trend_data, self.labels = self.process(data_dir, "data")

    def __len__(self):
        if PredictDataset.model_name == "STMetaNet":
            return self.flow_data.shape[0]
        else:
            return self.close_data.shape[0]

    def __getitem__(self, item):
        if PredictDataset.model_name == "STMetaNet":
            return self.flow_data[item], self.features, self.labels[item]
        else:
            return (
                self.close_data[item],
                self.period_data[item],
                self.trend_data[item],
                self.features,
                self.labels[item]
            )

    @classmethod
    def get_model_name(cls):
        return cls.model_name

    def construct_dataset(self, data, **kwargs):
        num_time, num_nodes, _ = data.shape
        if PredictDataset.model_name == "STMetaNet":
            mask = np.sum(data, axis=(1, 2)) > 5000
            data = self.scaler.transform(data)
            timestamps = (np.arange(num_time) % 24) / 24
            timestamps = np.tile(timestamps, (1, num_nodes, 1)).T
            transform_data = np.concatenate((data, timestamps), axis=2)
            split_data, split_label = [], []
            for i in range(num_time - self.input_len - self.output_len + 1):
                if mask[i + self.input_len:i + self.input_len + self.output_len].sum() != self.output_len:
                    continue
                split_data.append(transform_data[i:i + self.input_len])
                split_label.append(transform_data[i + self.input_len:i + self.input_len + self.output_len])
            split_data, split_label = (
                np.stack(split_data, axis=0),
                np.stack(split_label, axis=0)
            )
            return (
                torch.tensor(split_data, dtype=torch.float32),
                torch.tensor(split_label, dtype=torch.float32)
            )
        else:
            data = self.scaler.transform(data)
            num_rows, num_cols = kwargs["rows"], kwargs["cols"]
            self.features = self.features.reshape(num_rows, num_cols, -1)
            start_idx = max([
                self.close_len,
                self.period_len * self.period_interval * 24,
                self.trend_len * self.trend_interval * 24
            ])
            data = data.reshape(num_time, num_rows, num_cols, -1)
            data = data.transpose((0, 3, 1, 2))
            close_data, period_data, trend_data, labels = [], [], [], []
            for i in range(start_idx, num_time):
                close_data.append(data[i - self.close_len:i].reshape(-1, num_rows, num_cols))
                period_data.append(
                    data[i - self.period_len * self.period_interval * 24:i:self.period_interval * 24]
                    .reshape(-1, num_rows, num_cols)
                )
                trend_data.append(
                    data[i - self.trend_len * self.trend_interval * 24:i:self.trend_interval * 24]
                    .reshape(-1, num_rows, num_cols)
                )
                labels.append(data[i])
            close_data, period_data, trend_data, labels = (
                np.stack(close_data, axis=0),
                np.stack(period_data, axis=0),
                np.stack(trend_data, axis=0),
                np.stack(labels, axis=0)
            )
            return (
                torch.tensor(close_data, dtype=torch.float32),
                torch.tensor(period_data, dtype=torch.float32),
                torch.tensor(trend_data, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.float32)
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
            flow_data[:train_size] += 2
            PredictDataset.scaler = Scaler(flow_data[:train_size])
            return self.construct_dataset(flow_data[:train_size], rows=rows, cols=cols)
        elif self.mode == "valid":
            return self.construct_dataset(flow_data[train_size:train_size + valid_size], rows=rows, cols=cols)
        else:
            return self.construct_dataset(flow_data[-test_size:], rows=rows, cols=cols)

    @staticmethod
    def collate_fn(batch):
        def collate_batch(idx):
            return [sample[idx] for sample in batch]
        if PredictDataset.model_name == "STMetaNet":
            batch_data = torch.stack(collate_batch(0), dim=0)
            batch_feat = torch.stack(collate_batch(1), dim=0)
            batch_label = torch.stack(collate_batch(2), dim=0)
            return batch_data, batch_feat, batch_label
        else:
            batch_close = torch.stack(collate_batch(0), dim=0)
            batch_period = torch.stack(collate_batch(1), dim=0)
            batch_trend = torch.stack(collate_batch(2), dim=0)
            batch_feat = torch.stack(collate_batch(3), dim=0)
            batch_label = torch.stack(collate_batch(4), dim=0)
            return batch_close, batch_period, batch_trend, batch_feat, batch_label

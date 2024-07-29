import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

Mode_Index = {"walk": 0, "run": 9, "bike": 1, "bus": 2, "car": 3, "taxi": 3, "subway": 4, "railway": 4,
              "train": 4, "motocycle": 8, "boat": 9, "airplane": 9, "other": 9}
Ground_Mode = ["walk", "bike", "bus", "car", "taxi", "subway", "railway", "train"]


# each line contains `Latitude`, `Longitude`, `Flag`, `Altitude`, `Number of days`, `Datetime`,
# `Transportation Mode`, `Distance`, `Bearing Rate`, `Time Diff`, `Speed`, `Acceleration`, `Jerk`
class ModeIdentifyDataset(Dataset):
    def __init__(
            self,
            traj_dir,
            input_dim,
            mode,
            minimal_num_points=20,
            maximal_num_points=248,
            minimal_distance=150,
            minimal_triptime=60
    ):
        self.traj_dir = traj_dir
        self.input_dim = input_dim
        self.mode = mode
        self.max_length = maximal_num_points
        trip_data, labels = self.process(maximal_triptime=20 * 60)
        self.trip_data, self.labels = self.filter(
            trip_data, labels,
            minimal_num_points,
            minimal_distance,
            minimal_triptime
        )

    def __len__(self):
        return len(self.trip_data)

    def __getitem__(self, item):
        pad_size = self.max_length - self.trip_data[item].shape[0]
        data = self.trip_data[item][:, 3:3 + self.input_dim]
        data = F.pad(data, (0, 0, 0, pad_size), "constant", 0.0)
        return data.unsqueeze(0), self.labels[item]

    def process(self, maximal_triptime):
        traj_path = os.path.join(self.traj_dir, f"{self.mode}.txt")
        trajectories_with_label, single_trajectory = [], []
        with open(traj_path, 'r') as r_file:
            for lid, line in enumerate(r_file.readlines()):
                if line[0] == '-':
                    trajectories_with_label.append(single_trajectory)
                    single_trajectory = []
                else:
                    row = line.strip().split(',')
                    # each data contains `latitude, longitude, timestamp,
                    # relative distance, bearing rate, time interval,
                    # speed, acceleration, jerk`
                    if row[7] == '' or row[8] == '' or row[9] == '' or row[10] == '' or row[11] == '' or row[12] == '':
                        continue
                    (lat, lng, time, rd, br, tdiff, speed, accl, jerk, label) = (
                        float(row[0]), float(row[1]), float(row[4]),
                        float(row[7]), float(row[8]), float(row[9]),
                        float(row[10]), float(row[11]), float(row[12]), row[6]
                    )
                    if label not in Ground_Mode:
                        continue
                    label = Mode_Index[label]
                    single_trajectory.append([lat, lng, time, rd, speed, accl, jerk, br, tdiff, label])
        trip_data, trip, labels = [], [], []
        for trajectory in trajectories_with_label:
            i = 0
            while i < len(trajectory) - 1:
                delta_time = (trajectory[i + 1][2] - trajectory[i][2]) * 24 * 3600
                mode_not_change = (trajectory[i + 1][-1] == trajectory[i][-1])
                if 0 < delta_time <= maximal_triptime and mode_not_change and len(trip) + 1 < self.max_length:
                    trip.append(trajectory[i][:-1])
                    i += 1
                elif delta_time > maximal_triptime or not mode_not_change or len(trip) + 1 == self.max_length:
                    trip.append(trajectory[i][:-1])
                    trip_data.append(torch.tensor(trip))
                    labels.append(trajectory[i][-1])
                    trip = []
                    i += 1
                elif delta_time <= 0:
                    trajectory.remove(trajectory[i + 1])
        return trip_data, labels

    def filter(self, trip_data, labels, minimal_num_point, minimal_distance, minimal_triptime):
        filter_data, filter_labels = [], []
        for trip, label in zip(trip_data, labels):
            if len(trip) < minimal_num_point:
                continue
            if (trip[:, 3]).sum() < minimal_distance:
                continue
            if (trip[:, 8]).sum() < minimal_triptime:
                continue
            filter_data.append(trip)
            filter_labels.append(label)
        return filter_data, filter_labels

    @staticmethod
    def collate_fn(batch):
        func = lambda x: [sample[x] for sample in batch]
        batch_data = torch.stack(func(0), dim=0)
        batch_labels = torch.tensor(func(1), dtype=torch.long)
        return batch_data, batch_labels

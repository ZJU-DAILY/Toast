import os
import math
import numpy as np
import torch
from queue import Queue
from rtree import Rtree
from collections import defaultdict
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops

from utils.data.point import STPoint, project_pt_to_road, rate2gps, LAT_PER_METER, LNG_PER_METER


def calculate_spatial_distance(x1, y1, x2, y2):
    lat1 = (math.pi / 180.0) * x1
    lat2 = (math.pi / 180.0) * x2
    lng1 = (math.pi / 180.0) * y1
    lng2 = (math.pi / 180.0) * y2
    R = 6378.137
    t = math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(lng2 - lng1)
    if t > 1.0:
        t = 1.0
    d = math.acos(t) * R * 1000
    return d


class SegmentCentricRoadNetwork:
    def __init__(self, road_dir, file_name, zone_range):
        self.edge_path = os.path.join(road_dir, file_name)
        self.zone_range = zone_range
        self.node_lst = None
        self.edge_lst = None
        self.reindex_nodes = None
        self.cord_on_segment = None
        self.cord_offset = None
        self.segment_distance = None
        self.road_type = None
        self.spatial_index = Rtree()

    def is_within_range(self, lat, lng, region=None):
        if region is None:
            return self.zone_range[0] <= lat <= self.zone_range[2] and \
                   self.zone_range[1] <= lng <= self.zone_range[3]
        else:
            return region[0] <= lng <= region[2] and \
                   region[1] <= lat <= region[3]

    def read_roadnet(self):
        node_set = {}
        reindex_nodes = {}
        node_idx = 0
        cord_on_segment = []
        cord_offset = []
        segment_distance = []
        start2segment = defaultdict(list)
        with open(self.edge_path, 'r') as edge_file:
            for line in edge_file.readlines():
                items = line.strip().split()
                segment_id = int(items[0])
                start_id, end_id = int(items[1]), int(items[2])
                n_cords = int(items[3])
                in_range = True
                min_lng, min_lat, max_lng, max_lat = 1e18, 1e18, -1e18, -1e18
                for cord_id in range(n_cords):
                    lat = float(items[4 + cord_id * 2])
                    lng = float(items[5 + cord_id * 2])
                    min_lng, min_lat = min(min_lng, lng), min(min_lat, lat)
                    max_lng, max_lat = max(max_lng, lng), max(max_lat, lat)
                    in_range = in_range and self.is_within_range(lat, lng)
                if in_range:
                    node_set[segment_id] = (start_id, end_id)
                    reindex_nodes[segment_id] = node_idx
                    start2segment[start_id].append(segment_id)
                    self.spatial_index.insert(segment_id, (min_lng, min_lat, max_lng, max_lat))
                    node_idx += 1
                cord_on_segment.append(list(map(float, items[4:])))
                total_dist = .0
                offsets = []
                for idx in range(n_cords - 1):
                    dist = calculate_spatial_distance(float(items[4 + idx * 2]), float(items[5 + idx * 2]),
                                                      float(items[6 + idx * 2]), float(items[7 + idx * 2]))
                    total_dist += dist
                    offsets.append(dist)
                segment_distance.append(total_dist)
                for i in range(len(offsets) - 1, 0, -1):
                    offsets[i - 1] = offsets[i - 1] + offsets[i]
                offsets.append(0)
                cord_offset.append(offsets)

        edge_lst = []
        for node_1, (sid, eid) in node_set.items():
            for node_2 in start2segment[eid]:
                edge_lst.append([node_1, node_2])

        node_set = list(map(lambda x: reindex_nodes[x], node_set.keys()))
        edge_lst = list(map(lambda x: [reindex_nodes[x[0]], reindex_nodes[x[1]]], edge_lst))
        self.reindex_nodes = reindex_nodes
        self.node_lst = node_set
        self.edge_lst = np.array(edge_lst)
        self.cord_on_segment = cord_on_segment
        self.cord_offset = cord_offset
        self.segment_distance = segment_distance

    def read_road_type(self, type_path):
        self.road_type = {}
        with open(type_path, 'r') as type_file:
            for line in type_file.readlines():
                items = line.strip().split()
                segment_id, type_id = int(items[0]), int(items[2])
                self.road_type[segment_id] = type_id

    def range_query(self, query_point, left, bottom, right, top):
        segment_ids = self.spatial_index.intersection((left, bottom, right, top))
        query_result = []
        for sid in segment_ids:
            projection, rate, dist = project_pt_to_road(self, query_point, sid)
            if self.is_within_range(projection.lat, projection.lng, region=(left, bottom, right, top)):
                cand_point = STPoint(projection.lat, projection.lng, None, segment_id=sid, error=dist)
                query_result.append(cand_point)
        return query_result

    def get_neighbors(self):
        neighbors = defaultdict(list)
        for n1, n2 in self.edge_lst:
            neighbors[n1].append(n2)
        return neighbors

    def to_full_graph(self):
        node_index = torch.arange(0, len(self.node_lst) + 1)
        edge_index = (self.edge_lst + 1).T
        road_net = Data(edge_index=edge_index, node_index=node_index)
        road_net.num_nodes = len(self.node_lst) + 1
        return road_net

    def to_subgraphs(self, max_depth=3):
        num_nodes = len(self.node_lst) + 1
        neighbors = self.get_neighbors()

        subgraphs = []
        node_index = []
        for nid in range(num_nodes):
            edge_index = []
            if nid == 0:
                subg = Data(edge_index=torch.tensor(edge_index))
                subg.num_nodes = 1
                subg.edge_index, _ = add_self_loops(subg.edge_index, num_nodes=subg.num_nodes)
                node_index.append(0)
                subgraphs.append(subg)
            else:
                rid = nid - 1
                rset, cset = set(), set()
                q = Queue()
                q.put((rid, 0))
                rset.add(rid)
                while not q.empty():
                    idx, depth = q.get()
                    if depth == max_depth:
                        continue
                    if idx in cset:
                        continue
                    cset.add(idx)
                    for nb in neighbors[idx]:
                        edge_index.append([idx, nb])
                        rset.add(nb)
                        q.put((nb, depth + 1))
                rset = list(rset)
                node_reindex = {node: nid for nid, node in enumerate(rset)}
                edge_index = [[node_reindex[edge[0]], node_reindex[edge[1]]] for edge in edge_index]
                if len(edge_index) > 0:
                    edge_index, _ = add_self_loops(torch.tensor(edge_index, dtype=torch.long).T, num_nodes=len(rset))
                else:
                    edge_index, _ = add_self_loops(torch.tensor([], dtype=torch.long), num_nodes=len(rset))
                rset = [nid + 1 for nid in rset]
                node_index.extend(rset)
                subg = Data(edge_index=edge_index)
                subg.num_nodes = len(rset)
                subgraphs.append(subg)
        subgraph = Batch.from_data_list(subgraphs)
        node_index = torch.tensor(node_index)
        return subgraph, node_index

    def get_road_node_feat(self, type_path):
        self.read_road_type(type_path)
        neighbor = self.get_neighbors()
        features = torch.zeros(len(self.node_lst) + 1, 11, dtype=torch.float)
        max_segment_len = max(self.segment_distance)
        for seg_id, nid in self.reindex_nodes.items():
            features[nid + 1][0] = math.log10(self.segment_distance[seg_id] + 1e-6) / math.log10(max_segment_len)
            features[nid + 1][self.road_type[seg_id] + 1] = 1.
            neighbor_nodes = neighbor[nid]
            features[nid + 1][9] += len(neighbor_nodes)
            for nb in neighbor_nodes:
                features[nb + 1][10] += 1.
        return features

    def point2grid(self, pt, grid_size):
        lat_unit = LAT_PER_METER * grid_size
        lng_unit = LNG_PER_METER * grid_size
        lat = pt.lat
        lng = pt.lng
        min_lat, min_lng = self.zone_range[0], self.zone_range[1]
        grid_x = int((lat - min_lat) / lat_unit) + 1
        grid_y = int((lng - min_lng) / lng_unit) + 1
        return grid_x, grid_y

    def roadnet2seq(self, grid_size):
        def pad_seq(seq_list, pad_val=0):
            seq_len = [seq.shape[0] for seq in seq_list]
            pad_seqs = torch.full((len(seq_list), max(seq_len), 2), pad_val, dtype=torch.long)
            for idx, seq in enumerate(seq_list):
                length = seq_len[idx]
                pad_seqs[idx, :length] = seq
            return pad_seqs, torch.tensor(seq_len)

        grid_seq = [torch.zeros(1, 2)]
        for seg_id, _ in self.reindex_nodes.items():
            grid = []
            for rate in range(1000):
                r = rate / 1000
                pt = rate2gps(self, seg_id, r)
                grid_x, grid_y = self.point2grid(pt, grid_size)
                if len(grid) == 0 or [grid_x, grid_y] != grid[-1]:
                    grid.append([grid_x, grid_y])
            grid_seq.append(torch.tensor(grid))
        grid_seq, grid_len = pad_seq(grid_seq)
        return grid_seq, grid_len

import os
import math
import numpy as np
import torch
from queue import Queue
from rtree import Rtree
from collections import defaultdict
from torch_geometric.data import Data

from utils.data.point import STPoint, project_pt_to_road


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

        node2batch = {}
        subg_nid = 0
        node_index_map = {}  # node id (in subgraph) -> node id (in road net)
        node_index = []
        edge_index = []
        subgraph_idx = 0
        for nid in range(num_nodes):
            if nid == 0:
                node_index.append(subg_nid)
                node_index_map[subg_nid] = nid
                node2batch[subg_nid] = subgraph_idx
            else:
                rset, cset = set(), set()
                q = Queue()
                subg_nid += 1
                node_index.append(subg_nid)
                node_index_map[subg_nid] = nid
                node2batch[subg_nid] = subgraph_idx
                q.put((subg_nid, 0))
                rset.add(nid)
                while not q.empty():
                    idx, depth = q.get()
                    rid = node_index_map[idx]
                    if depth == max_depth:
                        continue
                    if rid in cset:
                        continue
                    cset.add(rid)
                    for nrid in neighbors[rid - 1]:
                        subg_nid += 1
                        node_index.append(subg_nid)
                        node_index_map[subg_nid] = nrid + 1
                        node2batch[subg_nid] = subgraph_idx
                        edge_index.append([idx, subg_nid])
                        rset.add(nrid + 1)
                        q.put((subg_nid, depth + 1))
            subgraph_idx += 1
        edge_index = torch.tensor(edge_index).T
        node2batch = torch.tensor(list(node2batch.keys()))
        node_index = list(map(lambda x: node_index_map[x], node_index))
        node_index = torch.tensor(node_index)
        subgraph = Data(edge_index=edge_index, node_index=node_index, batch=node2batch)
        subgraph.num_nodes = subg_nid + 1
        return subgraph

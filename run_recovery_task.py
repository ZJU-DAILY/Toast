import argparse
import os
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.roadmap import SegmentCentricRoadNetwork
from utils.data import SPoint, RecoveryDataset
from models.recovery_model import RNTrajRec


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="trajectory recovery")
    parser.add_argument("--city", type=str, default="Shanghai")
    parser.add_argument("--num_epoch", type=int, default=30, help="epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--load_pretrained_flag", action='store_true', help="flag of load pretrained model")
    parser.add_argument("--gpu", type=int, default=0, help="gpu device")
    parser.add_argument("--seed", type=int, default=2024, help="random seed")
    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def initialize_model(model):
    ih = (param.data for name, param in model.named_parameters() if "weight_ih" in name)
    hh = (param.data for name, param in model.named_parameters() if "weight_hh" in name)
    b = (param.data for name, param in model.named_parameters() if "bias" in name)

    for t in ih:
        torch.nn.init.xavier_uniform_(t)
    for t in hh:
        torch.nn.init.orthogonal_(t)
    for t in b:
        torch.nn.init.constant_(t, 0)


def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    traj_dir = os.path.join("./data", args.city)
    road_dir = "./data/roadnet"
    file_name = "edgeOSM.txt"
    type_path = "wayTypeOSM.txt"
    zone_range = [41.111975, -8.667057, 41.177462, -8.585305]
    with open("configs/recovery_config.json", 'r') as config_file:
        config = json.load(config_file)

    road_net = SegmentCentricRoadNetwork(road_dir, file_name, zone_range)
    road_net.read_roadnet()
    subgraph = road_net.to_subgraphs(max_depth=config["max_depth"])
    grid_shape = road_net.point2grid(SPoint(zone_range[2], zone_range[3]), config["grid_size"])
    grid_shape = (grid_shape[0] + 1, grid_shape[1] + 1)
    road_grid, road_len = road_net.roadnet2seq(args.grid_size)
    road_feat = road_net.get_road_node_feat(os.path.join(road_dir, type_path))

    dataset_params = {"time_interval": config["time_interval"],
                      "win_size": config["win_size"],
                      "grid_size": config["grid_size"],
                      "ds_type": config["ds_type"],
                      "keep_ratio": config["keep_ratio"],
                      "search_dist": config["search_dist"],
                      "beta": config["beta"],
                      "gamma": config["gamma"]}
    train_dataset = RecoveryDataset(traj_dir, road_net, zone_range, "train", **dataset_params)
    valid_dataset = RecoveryDataset(traj_dir, road_net, zone_range, "valid", **dataset_params)
    test_dataset = RecoveryDataset(traj_dir, road_net, zone_range, "test", **dataset_params)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=config["shuffle"], collate_fn=RecoveryDataset.collate_fn,
                              num_workers=8, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                              shuffle=config["shuffle"], collate_fn=RecoveryDataset.collate_fn,
                              num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, collate_fn=RecoveryDataset.collate_fn,
                             num_workers=8, pin_memory=True)

    model_params = {"hidden_dim": config["hidden_dim"],
                    "num_gnn_layers": config["num_gnn_layers"],
                    "num_encoder_layers": config["num_encoder_layers"],
                    "extra_input_dim": config["extra_input_dim"],
                    "extra_output_dim": config["extra_output_dim"],
                    "road_feat_dim": config["road_feat_dim"],
                    "grid_shape": grid_shape,
                    "grid_embed_size": config["grid_embed_size"],
                    "batch_size": args.batch_size,
                    "device": device,
                    "dropout": config["dropout"],
                    "tf_ratio": config["tf_ratio"]}
    model = RNTrajRec(num_rn_node=len(road_net.node_lst) + 1, **model_params)
    model.apply(initialize_model)

import argparse
import torch

from utils.roadmap import SegmentCentricRoadNetwork
from utils.data import RecoveryDataset


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="trajectory recovery")
    parser.add_argument("--city", type=str, default="Shanghai")
    parser.add_argument("--keep_ratio", type=float, default=0.125, help="keep ratio in float")
    parser.add_argument("--lambda1", type=int, default=10, help="weight for multi task rate")
    parser.add_argument("--lambda2", type=int, default=0.1, help="weight for multi task rate")
    parser.add_argument("--hid_dim", type=int, default=512, help="hidden dimension")
    parser.add_argument("--epochs", type=int, default=30, help="epochs")
    parser.add_argument("--grid_size", type=int, default=50, help="grid size in int")
    parser.add_argument("--pro_features_flag", action='store_true', help="flag of using profile features")
    parser.add_argument("--tandem_fea_flag", action='store_true', help="flag of using tandem rid features")
    parser.add_argument("--no_attn_flag", action='store_false', help="flag of using attention")
    parser.add_argument("--load_pretrained_flag", action='store_true', help="flag of load pretrained model")
    parser.add_argument("--model_old_path", type=str, default='', help="old model path")
    parser.add_argument("--decay_flag", action="store_true")
    parser.add_argument("--grid_flag", action="store_true")
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0, help="gpu device")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    road_dir = "./data/roadnet"
    file_name = "edgeOSM.txt"
    zone_range = [41.111975, -8.667057, 41.177462, -8.585305]
    time_interal = 15

    road_net = SegmentCentricRoadNetwork(road_dir, file_name, zone_range)
    road_net.read_roadnet()
    subgraph = road_net.to_subgraphs()

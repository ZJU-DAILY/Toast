import argparse
import os
import json
import random
import numpy as np
import torch
from torch.optim import AdamW

from utils.roadmap import SegmentCentricRoadNetwork
from utils.data import SPoint, RecoveryDataset
from utils.train import RecoveryTrainer
from models.recovery_model import RNTrajRec, MTrajRec
from augment.augment_config import TaskType, PointUnionConfig
from augment.point_union import PointUnion


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="trajectory recovery")
    parser.add_argument("--dataset", type=str, default="Shanghai")
    parser.add_argument("--model_type", type=str, default="RNTrajRec")
    parser.add_argument("--num_epochs", type=int, default=30, help="epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--phase", type=str, default=None, help="select from `train`, `test`, `augment`")
    parser.add_argument("--saved_path", type=str, default=None, help="model checkpoint")
    parser.add_argument("--gpu", type=int, default=0, help="gpu device")
    parser.add_argument("--seed", type=int, default=2024, help="random seed")
    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    traj_dir = os.path.join("./data", args.dataset)
    phase = args.phase
    road_dir = f"./data/{args.dataset}/roadnet"
    file_name = "edgeOSM.txt"
    type_path = "wayTypeOSM.txt"
    zone_range = [41.111975, -8.667057, 41.177462, -8.585305]
    model_type = args.model_type
    ckpt_dir = f"./ckpt/{model_type}-" + args.dataset if args.saved_path is None else args.saved_path
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    with open("configs/recovery_config.json", 'r') as config_file:
        configs = json.load(config_file)
        config = configs[model_type]

    road_net = SegmentCentricRoadNetwork(road_dir, file_name, zone_range)
    road_net.read_roadnet()
    subgraph, road_node_index = road_net.to_subgraphs(max_depth=config["max_depth"])
    grid_shape = road_net.point2grid(SPoint(zone_range[2], zone_range[3]), config["grid_size"])
    grid_shape = [grid_shape[0] + 1, grid_shape[1] + 1]
    road_grid, road_len = road_net.roadnet2seq(config["grid_size"])
    road_feat = road_net.get_road_node_feat(os.path.join(road_dir, type_path))

    dataset_params = {"time_interval": config["time_interval"],
                      "win_size": config["win_size"],
                      "grid_size": config["grid_size"],
                      "ds_type": config["ds_type"],
                      "keep_ratio": config["keep_ratio"],
                      "neighbor_dist": config["neighbor_dist"],
                      "search_dist": config["search_dist"],
                      "beta": config["beta"],
                      "gamma": config["gamma"]}
    train_dataset = RecoveryDataset(traj_dir, road_net, zone_range, "train", model_type, **dataset_params)
    valid_dataset = RecoveryDataset(traj_dir, road_net, zone_range, "valid", model_type, **dataset_params)
    test_dataset = RecoveryDataset(traj_dir, road_net, zone_range, "test", model_type, **dataset_params)

    if model_type == "RNTrajRec":
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
                        "dropout": config["dropout"]}
        model = RNTrajRec(num_rn_node=len(road_net.node_lst) + 1, **model_params).to(device)
        model.apply(initialize_model)
    else:
        model_params = {"hidden_dim": config["hidden_dim"],
                        "extra_input_dim": config["extra_input_dim"],
                        "extra_output_dim": config["extra_output_dim"],
                        "road_feat_dim": config["road_feat_dim"],
                        "grid_embed_size": config["grid_embed_size"],
                        "dropout": config["dropout"]}
        model = MTrajRec(node_size=len(road_net.node_lst) + 1, **model_params).to(device)
        model.apply(initialize_model)

    optim = AdamW(model.parameters(), lr=config["learning_rate"])
    trainer = RecoveryTrainer(model, train_dataset, valid_dataset, optim,
                              num_epochs=args.num_epochs,
                              data_collator=RecoveryDataset.collate_fn,
                              saved_dir=ckpt_dir,
                              batch_size=args.batch_size,
                              shuffle=config["shuffle"],
                              num_workers=8,
                              pin_memory=True,
                              device=device)
    if (phase is None) or (phase == "train"):
        if model_type == "RNTrajRec":
            trainer.train(road_grid, road_len, road_node_index, subgraph.edge_index, subgraph.batch,
                          road_feat, road_net, [config["lambda1"], config["lambda2"]], config["decay_ratio"], config["tf_ratio"])
            metric_result = trainer.evaluate(test_dataset, road_grid, road_len, road_node_index,
                                             subgraph.edge_index, subgraph.batch, road_feat, .0, road_net,
                                             [config["lambda1"], config["lambda2"]])
        else:
            trainer.train_mtrajrec(
                road_net, road_feat,
                [config["lambda1"], config["lambda2"]],
                config["decay_ratio"], config["tf_ratio"]
            )
            metric_result = trainer.evaluate_mtrajrec(
                test_dataset, road_feat, .0, road_net,
                [config["lambda1"], config["lambda2"]]
            )

        print("ACC: {:.4f}\tRecall: {:.4f}\tPrec: {:.4f}\tF1: {:.4f}\tMAE: {:.4f}\tRMSE: {:.4f}\n"
              .format(metric_result[0],
                      metric_result[1],
                      metric_result[2],
                      metric_result[3],
                      metric_result[4],
                      metric_result[5]))
    if (phase is None) or (phase == "test"):
        trainer.load_model()
        if model_type == "RNTrajRec":
            metric_result = trainer.evaluate(test_dataset, road_grid, road_len, road_node_index,
                                             subgraph.edge_index, subgraph.batch, road_feat, .0, road_net,
                                             [config["lambda1"], config["lambda2"]])
        else:
            metric_result = trainer.evaluate_mtrajrec(
                test_dataset, road_feat, .0, road_net,
                [config["lambda1"], config["lambda2"]]
            )

        print("ACC: {:.4f}\tRecall: {:.4f}\tPrec: {:.4f}\tF1: {:.4f}\tMAE: {:.4f}\tRMSE: {:.4f}\n"
              .format(metric_result[0],
                      metric_result[1],
                      metric_result[2],
                      metric_result[3],
                      metric_result[4],
                      metric_result[5]))
    if (phase is None) or (phase == "augment"):
        trainer.load_model()
        augment_config = PointUnionConfig(
            TaskType["TRAJ_REC"],
            num_virtual_tokens=20,
            num_epochs=args.num_epochs
        )
        augmentor = PointUnion(augment_config, trainer, None, device)
        augment_result = augmentor.augment_points(
            test_dataset,
            model_type,
            road_net,
            road_grid,
            road_len,
            road_node_index,
            subgraph.edge_index,
            subgraph.batch,
            road_feat,
            [config["lambda1"], config["lambda2"]]
        )
        print("ACC: {:.4f}\tRecall: {:.4f}\tPrec: {:.4f}\tF1: {:.4f}\tMAE: {:.4f}\tRMSE: {:.4f}\n"
              .format(augment_result[0],
                      augment_result[1],
                      augment_result[2],
                      augment_result[3],
                      augment_result[4],
                      augment_result[5]))


if __name__ == "__main__":
    main()

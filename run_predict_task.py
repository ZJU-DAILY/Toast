import argparse
import os
import json
import h5py
import random
import numpy as np
import torch
from torch.optim import AdamW
from torch_geometric.data import Data

from utils.data import PredictDataset
from utils.train import PredictTrainer
from models.prediction_model import STMetaNet, STResNet


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="traffic flow prediction")
    parser.add_argument("--dataset", type=str, default="Beijing")
    parser.add_argument("--model_name", type=str, default="STMetaNet")
    parser.add_argument("--num_epochs", type=int, default=20, help="epochs")
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


def read_feature(data_dir, feat_dim):
    print(f"[read data features with dimension {feat_dim}...]")
    feat_path = os.path.join(data_dir, "BJ_FEATURE.h5")
    with h5py.File(feat_path, 'r') as feat_f:
        feat_data = np.array(feat_f["embeddings"])
    row_size, col_size, dimension = feat_data.shape
    feat_data = feat_data.reshape(row_size * col_size, -1)
    if feat_dim < dimension:
        feat_data[:, feat_dim:] = 0.
        # feat_data = feat_data[:, :feat_dim]
    feat_data = (feat_data - feat_data.mean(axis=0)) / (np.std(feat_data, axis=0) + 1e-8)
    return feat_data


def read_graph(data_dir):
    print("[read graph...]")
    graph_path = os.path.join(data_dir, "BJ_GRAPH.h5")
    with h5py.File(graph_path, 'r') as graph_f:
        adj = np.array(graph_f["data"])
    src, dst = np.where(np.sum(adj, axis=2) > 0)
    values = adj[src, dst]
    adj = (adj - np.mean(values, axis=0)) / (np.std(values, axis=0) + 1e-8)
    edge_index = torch.tensor(np.stack((src, dst), axis=0))
    return Data(edge_index=edge_index, adj=adj)


def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    data_dir = os.path.join("./data", args.dataset)
    model_name = args.model_name
    phase = args.phase
    with open("configs/predict_config.json", 'r') as config_file:
        configs = json.load(config_file)
        config = configs[model_name]
    feat_dim = config["feat_dim"]
    ckpt_dir = f"./ckpt/{model_name}-{args.dataset}-{feat_dim}-tu" if args.saved_path is None else args.saved_path
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    features = read_feature(data_dir, config["feat_dim"])
    graph = read_graph(data_dir)

    if model_name == "STMetaNet":
        dataset_config = {
            "input_len": config["input_len"],
            "output_len": config["output_len"],
        }
    else:
        dataset_config = {
            "closeness_length": config["closeness_length"],
            "period_length": config["period_length"],
            "trend_length": config["trend_length"],
            "period_interval": config["period_interval"],
            "trend_interval": config["trend_interval"]
        }
    train_dataset = PredictDataset(data_dir, features=features, mode="train", model_name=model_name, **dataset_config)
    valid_dataset = PredictDataset(data_dir, features=features, mode="valid", model_name=model_name, **dataset_config)
    test_dataset = PredictDataset(data_dir, features=features, mode="test", model_name=model_name, **dataset_config)

    if model_name == "STMetaNet":
        model_params = {
            "input_dim": config["input_dim"],
            "output_dim": config["output_dim"],
            "feat_dim": config["feat_dim"],
            "hidden_dim": config["hidden_dim"]
        }
        model = STMetaNet(**model_params).to(device)
    else:
        model_params = {
            "closeness_len": config["closeness_length"],
            "period_len": config["period_length"],
            "trend_len": config["trend_length"],
            "feat_dim": 989,
            "num_layers": config["num_layers"]
        }
        model = STResNet(2, 32, 32, **model_params).to(device)

    optim = AdamW(model.parameters(), lr=config["learning_rate"])
    trainer = PredictTrainer(model, train_dataset, valid_dataset, optim,
                             num_epochs=args.num_epochs,
                             data_collator=PredictDataset.collate_fn,
                             saved_dir=ckpt_dir,
                             batch_size=args.batch_size,
                             shuffle=config["shuffle"],
                             num_workers=8,
                             pin_memory=True,
                             device=device)
    if (phase is None) or (phase == "train"):
        trainer.train(edge_index=graph.edge_index, scaler=PredictDataset.scaler)
        metric_result = trainer.evaluate(test_dataset, graph.edge_index, PredictDataset.scaler)
        print("rsme: {:.4f}\tmae: {:.4f}\n"
              .format(metric_result[0],
                      metric_result[1]))
    if (phase is None) or (phase == "test"):
        trainer.load_model()
        metric_result = trainer.evaluate(test_dataset, graph.edge_index, PredictDataset.scaler)
        print("rsme: {:.4f}\tmae: {:.4f}\n"
              .format(metric_result[0],
                      metric_result[1]))


if __name__ == "__main__":
    main()

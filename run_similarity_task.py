import argparse
import os
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch_geometric.nn import Node2Vec

from utils.data import SimilarityDataset
from utils.train import SimilarityTrainer
from models.similarity_model import ST2Vec, GTS
from augment.augment_config import PointUnionConfig, TaskType
from augment.point_union import PointUnion


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="traffic flow prediction")
    parser.add_argument("--dataset", type=str, default="tdrive")
    parser.add_argument("--model_name", type=str, default="ST2Vec")
    parser.add_argument("--gt_file", type=str, default="ground_truth")
    parser.add_argument("--num_epochs", type=int, default=150, help="epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--phase", type=str, default=None, help="select from `train`, `test`, `augment`")
    parser.add_argument("--saved_path", type=str, default=None, help="model checkpoint")
    parser.add_argument("--gpu", type=int, default=0, help="gpu device")
    parser.add_argument("--seed", type=int, default=2024, help="random seed")

    parser.add_argument("--num_virtual_tokens", type=int, default=20)
    parser.add_argument("--num_augment_epochs", type=int, default=20)
    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_graph_features(graph_dir, feat_dim, device):
    print("[construct graph features...]")
    edge_path = os.path.join(graph_dir, "road/edge_weight.csv")
    node_path = os.path.join(graph_dir, "road/node.csv")
    df_dege = pd.read_csv(edge_path, sep=',')
    df_node = pd.read_csv(node_path, sep=',')
    edge_index = df_dege[["s_node", "e_node"]].to_numpy()
    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).T
    edge_attr = torch.tensor(df_dege["length"].to_numpy(), dtype=torch.float, device=device)
    num_nodes = df_node["node"].size

    if os.path.exists(graph_dir + "/node_features.npy"):
        node_feats = np.load(graph_dir + "/node_features.npy")
        node_feats = torch.tensor(node_feats, dtype=torch.float, device=device)
    else:
        model = Node2Vec(
            edge_index,
            embedding_dim=feat_dim,
            walk_length=20,
            context_size=10,
            walks_per_node=10,
            num_negative_samples=1,
            p=1,
            q=1,
            sparse=True,
            num_nodes=num_nodes
        ).to(device)
        loader = model.loader(batch_size=128, shuffle=True)
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)

        last_loss = 1.
        print("[train node embedding with node2vec...]")
        for i in range(100):
            model.train()
            total_loss = 0.
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss
            print(f"[Epoch: {i}\tLoss: {total_loss / len(loader):.4f}]")
            if abs(last_loss - total_loss / len(loader)) < 1e-5:
                break
            else:
                last_loss = total_loss / len(loader)
        model.eval()
        with torch.no_grad():
            node_feats = model(torch.arange(num_nodes, device=device))
    return edge_index, node_feats, edge_attr


def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    data_dir = os.path.join("./data", args.dataset)
    gt_dir = os.path.join(data_dir, args.gt_file)
    model_name = args.model_name
    phase = args.phase
    with open("configs/similarity_config.json", 'r') as config_file:
        configs = json.load(config_file)
        config = configs[model_name]
    ckpt_dir = f"./ckpt/{model_name}-{args.dataset}" if args.saved_path is None else args.saved_path
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    edge_index, node_feats, edge_attr = get_graph_features(data_dir, config["spatial_dim"], device)

    train_dataset = SimilarityDataset(data_dir, gt_dir, mode="train", model_name=model_name)
    valid_dataset = SimilarityDataset(data_dir, gt_dir, mode="valid", model_name=model_name)
    test_dataset = SimilarityDataset(data_dir, gt_dir, mode="test", model_name=model_name)

    if model_name == "ST2Vec":
        model_params = {
            "spatial_dim": config["spatial_dim"],
            "temporal_dim": config["temporal_dim"],
            "feature_dim": config["feature_dim"],
            "hidden_dim": config["hidden_dim"],
            "num_layers": config["num_layers"],
            "dropout_rate": config["dropout_rate"],
            "pretrain": False,
            "pretrain_model": data_dir + "/d2v_98291_17.169918439404636.pth"
        }
        model = ST2Vec(**model_params).to(device)
    else:
        model_params = {
            "spatial_dim": config["spatial_dim"],
            "feature_dim": config["feature_dim"],
            "hidden_dim": config["hidden_dim"],
            "num_layers": config["num_layers"],
            "dropout_rate": config["dropout_rate"],
        }
        model = GTS(**model_params).to(device)

    optim = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.0001)
    trainer = SimilarityTrainer(model, train_dataset, valid_dataset, optim,
                                num_epochs=args.num_epochs,
                                data_collator=SimilarityDataset.collate_fn,
                                saved_dir=ckpt_dir,
                                batch_size=args.batch_size,
                                shuffle=config["shuffle"],
                                num_workers=8,
                                pin_memory=True,
                                device=device)
    if (phase is None) or (phase == "train"):
        trainer.train(node_feats, edge_index, edge_attr)
        metric_result = trainer.evaluate(test_dataset, node_feats, edge_index, edge_attr)
        print("hr10: {:.4f}\thr50: {:.4f}\thr10_50: {:.4f}\n"
              .format(metric_result[0],
                      metric_result[1],
                      metric_result[2]))
    if (phase is None) or (phase == "test"):
        trainer.load_model()
        metric_result = trainer.evaluate(test_dataset, node_feats, edge_index)
        print("hr10: {:.4f}\thr50: {:.4f}\thr10_50: {:.4f}\n"
              .format(metric_result[0],
                      metric_result[1],
                      metric_result[2]))
    if (phase is None) or (phase == "augment"):
        trainer.load_model()
        augment_config = PointUnionConfig(
            TaskType.TRAJ_SIMILAR,
            model_name=model_name,
            virtual_dim=config["feature_dim"],
            num_virtual_tokens=args.num_virtual_tokens,
            num_epochs=args.num_augment_epochs
        )
        augmentor = PointUnion(augment_config, trainer, None, device)
        augment_result = augmentor.augment_points(
            test_dataset,
            node_feat=node_feats,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        print("hr10: {:.4f}\thr50: {:.4f}\thr10_50: {:.4f}\n"
              .format(augment_result[0],
                      augment_result[1],
                      augment_result[2]))


if __name__ == "__main__":
    main()

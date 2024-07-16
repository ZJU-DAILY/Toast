import tqdm
from typing import Union, Callable, Optional, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer

from models.base_model import BaseModel
from utils.roadmap.road_network import SegmentCentricRoadNetwork
from utils.train.trainer import Trainer
from utils.metrics.recovery_metrics import evaluate_point_prediction, evaluate_distance_regression


class RecoveryTrainer(Trainer):
    def __init__(
            self,
            model: BaseModel,
            train_dataset: Dataset,
            eval_dataset: Dataset,
            optimizer: Optimizer,
            num_epochs: int,
            data_collator: Callable,
            saved_dir: str,
            batch_size: int,
            shuffle: bool,
            num_workers: int,
            pin_memory: bool,
            device: Union[torch.device, str] = "cpu",
            **kwargs
    ):
        super(RecoveryTrainer, self).__init__(model,
                                              train_dataset,
                                              eval_dataset,
                                              optimizer,
                                              num_epochs,
                                              data_collator,
                                              saved_dir,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              pin_memory=pin_memory,
                                              device=device)
        for key, val in kwargs.items():
            setattr(self, key, val)

    def compute_loss(self, pred_node, pred_rate, pred_logit, true_node, true_rate, true_logit,
                     src_len, tgt_len, weights):
        regression_criterion = nn.MSELoss(reduction="sum")
        classification_criterion = nn.NLLLoss(reduction="sum")

        node_loss = classification_criterion(pred_node, true_node) / tgt_len.sum()
        rate_loss = regression_criterion(pred_rate, true_rate) / tgt_len.sum()
        if pred_logit is not None:
            enc_loss = -1 * (pred_logit.squeeze(-1) * true_logit).sum() / src_len.sum()
            return node_loss + rate_loss * weights[0] + enc_loss * weights[1]
        else:
            return node_loss + rate_loss * weights[0]

    def train_mtrajrec(self, road_net, road_feat, weights, decay_param, tf_ratio):
        road_feat = road_feat.to(self.device)
        best_loss = float("inf")
        train_loader = self.get_train_dataloader()
        for epoch in range(self.num_epochs):
            print("[start {}-th training]".format(epoch + 1))
            self.model.train()
            for batch in tqdm.tqdm(train_loader, total=len(train_loader), desc="train"):
                src_grid_seq, src_gps_seq, feat_seq, src_seq_len, \
                tgt_gps_seq, tgt_node_seq, tgt_rate_seq, tgt_seq_len, tgt_inf_scores, \
                batched_graph, node_index = batch
                src_grid_seq, src_gps_seq, feat_seq = src_grid_seq.to(self.device), \
                                                      src_gps_seq.to(self.device), \
                                                      feat_seq.to(self.device)
                tgt_gps_seq, tgt_node_seq, tgt_rate_seq, tgt_inf_scores = tgt_gps_seq.to(self.device), \
                                                                          tgt_node_seq.to(self.device), \
                                                                          tgt_rate_seq.to(self.device), \
                                                                          tgt_inf_scores.to(self.device)
                self.optimizer.zero_grad()
                output_node, output_rate, logits = self.model(road_feat, src_grid_seq, src_seq_len, feat_seq,
                                                              tgt_node_seq, tgt_rate_seq, tgt_seq_len, tgt_inf_scores,
                                                              tf_ratio)
                output_node_dim = output_node.size(2)
                output_node = output_node.permute(1, 0, 2)[1:]
                output_node = output_node.reshape(-1, output_node_dim)  # ((len - 1) * batch_size, node_size)
                output_rate = output_rate.permute(1, 0, 2).squeeze(-1)[1:]  # ((len - 1), batch_size)
                logits = logits.squeeze(-1) if logits is not None else logits  # (num_nodes)

                tgt_node_seq = tgt_node_seq.permute(1, 0, 2)[1:].squeeze(-1)
                tgt_node_seq = tgt_node_seq.reshape(-1)  # ((len - 1) * batch_size)
                tgt_rate_seq = tgt_rate_seq.permute(1, 0, 2)[1:].squeeze(-1)  # ((len - 1), batch_size)
                loss = self.compute_loss(output_node, output_rate, logits,
                                         tgt_node_seq, tgt_rate_seq, None,
                                         src_seq_len, tgt_seq_len, weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()
            eval_loss = self.evaluate_mtrajrec(
                None, road_feat=road_feat, tf_ratio=.0, road_net=road_net, weights=weights
            )
            if self.saved_dir is not None:
                if eval_loss < best_loss:
                    print("saving best model.")
                    best_loss = eval_loss
                    self.save_model(save_optim=True)
            if (epoch + 1) % 5 == 0:
                print("valid loss: {:.4f}".format(eval_loss))
            tf_ratio = tf_ratio * decay_param

    def train(self, road_grid, road_len, road_nodes, road_edge, road_batch, road_feat, road_net, weights, decay_param, tf_ratio):
        road_grid, road_nodes = road_grid.to(self.device), road_nodes.to(self.device)
        road_edge, road_batch = road_edge.long().to(self.device), road_batch.to(self.device)
        road_feat = road_feat.to(self.device)
        best_loss = float("inf")
        train_loader = self.get_train_dataloader()
        for epoch in range(self.num_epochs):
            print("[start {}-th training]".format(epoch + 1))
            self.model.train()
            for batch in tqdm.tqdm(train_loader, total=len(train_loader), desc="train"):
                src_grid_seq, src_gps_seq, feat_seq, src_seq_len, \
                tgt_gps_seq, tgt_node_seq, tgt_rate_seq, tgt_seq_len, tgt_inf_scores, \
                batched_graph, node_index = batch
                src_grid_seq, src_gps_seq, feat_seq = src_grid_seq.to(self.device), \
                                                      src_gps_seq.to(self.device), \
                                                      feat_seq.to(self.device)
                tgt_gps_seq, tgt_node_seq, tgt_rate_seq, tgt_inf_scores = tgt_gps_seq.to(self.device), \
                                                                          tgt_node_seq.to(self.device), \
                                                                          tgt_rate_seq.to(self.device), \
                                                                          tgt_inf_scores.to(self.device)
                batched_graph, node_index = batched_graph.to(self.device), node_index.to(self.device)
                traj_edge, tgt_node_seq = batched_graph.edge_index.long(), tgt_node_seq.long()

                self.optimizer.zero_grad()
                output_node, output_rate, logits = self.model(road_grid, road_len, road_nodes, road_edge, road_batch,
                                                              road_feat, src_grid_seq, src_seq_len, feat_seq, node_index,
                                                              traj_edge, batched_graph.batch, batched_graph.weight,
                                                              tgt_node_seq, tgt_rate_seq, tgt_seq_len, tgt_inf_scores, tf_ratio)
                output_node_dim = output_node.size(2)
                output_node = output_node.permute(1, 0, 2)[1:]
                output_node = output_node.reshape(-1, output_node_dim)  # ((len - 1) * batch_size, node_size)
                output_rate = output_rate.permute(1, 0, 2).squeeze(-1)[1:]  # ((len - 1), batch_size)
                logits = logits.squeeze(-1)  # (num_nodes)

                tgt_node_seq = tgt_node_seq.permute(1, 0, 2)[1:].squeeze(-1)
                tgt_node_seq = tgt_node_seq.reshape(-1)  # ((len - 1) * batch_size)
                tgt_rate_seq = tgt_rate_seq.permute(1, 0, 2)[1:].squeeze(-1)  # ((len - 1), batch_size)
                loss = self.compute_loss(output_node, output_rate, logits,
                                         tgt_node_seq, tgt_rate_seq, batched_graph.gt,
                                         src_seq_len, tgt_seq_len, weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()
            eval_loss = self.evaluate(
                None, road_grid, road_len, road_nodes, road_edge, road_batch, road_feat, .0, road_net, weights
            )
            if self.saved_dir is not None:
                if eval_loss < best_loss:
                    print("saving best model.")
                    best_loss = eval_loss
                    self.save_model(save_optim=True)
            if (epoch + 1) % 5 == 0:
                print("valid loss: {:.4f}".format(eval_loss))
            tf_ratio = tf_ratio * decay_param

    def evaluate(
            self,
            test_dataset: Dataset = None,
            road_grid: torch.Tensor = None,
            road_len: torch.Tensor = None,
            road_nodes: torch.Tensor = None,
            road_edge: torch.Tensor = None,
            road_batch: torch.Tensor = None,
            road_feat: torch.Tensor = None,
            tf_ratio: float = .0,
            road_net: SegmentCentricRoadNetwork = None,
            weights: Optional[List[float]] = None
    ):
        if test_dataset is None:
            eval_loader = self.get_eval_dataloader()
            mode = "eval"
        else:
            eval_loader = self.get_test_dataloader(test_dataset)
            self.load_model()
            mode = "test"
        self.model.eval()
        road_grid, road_nodes = road_grid.to(self.device), road_nodes.to(self.device)
        road_edge, road_batch = road_edge.long().to(self.device), road_batch.to(self.device)
        road_feat = road_feat.to(self.device)

        total_acc, total_prec, total_recall, total_f1, total_mae, total_rmse, total_loss = [], [], [], [], [], [], .0
        with torch.no_grad():
            for batch in tqdm.tqdm(eval_loader, total=len(eval_loader), desc=mode):
                src_grid_seq, src_gps_seq, feat_seq, src_seq_len, \
                tgt_gps_seq, tgt_node_seq, tgt_rate_seq, tgt_seq_len, tgt_inf_scores, \
                batched_graph, node_index = batch
                src_grid_seq, src_gps_seq, feat_seq = src_grid_seq.to(self.device), \
                                                      src_gps_seq.to(self.device), \
                                                      feat_seq.to(self.device)
                tgt_gps_seq, tgt_node_seq, tgt_rate_seq, tgt_inf_scores = tgt_gps_seq.to(self.device), \
                                                                          tgt_node_seq.to(self.device), \
                                                                          tgt_rate_seq.to(self.device), \
                                                                          tgt_inf_scores.to(self.device)
                batched_graph, node_index = batched_graph.to(self.device), node_index.to(self.device)
                traj_edge, tgt_node_seq = batched_graph.edge_index.long(), tgt_node_seq.long()

                output_node, output_rate, logits = self.model(road_grid, road_len, road_nodes, road_edge, road_batch,
                                                              road_feat, src_grid_seq, src_seq_len, feat_seq, node_index,
                                                              traj_edge, batched_graph.batch, batched_graph.weight,
                                                              tgt_node_seq, tgt_rate_seq, tgt_seq_len, tgt_inf_scores, tf_ratio)
                output_rate = output_rate.squeeze(-1)
                tgt_node_seq, tgt_rate_seq = tgt_node_seq.squeeze(-1), tgt_rate_seq.squeeze(-1)
                output_node, output_rate = output_node[:, 1:, :], output_rate[:, 1:]
                tgt_node_seq, tgt_rate_seq = tgt_node_seq[:, 1:], tgt_rate_seq[:, 1:]
                if mode == "test":
                    acc, prec, recall, f1 = evaluate_point_prediction(output_node,
                                                                      tgt_node_seq,
                                                                      tgt_seq_len - 1)
                    mae, rmse = evaluate_distance_regression(output_node,
                                                             output_rate,
                                                             tgt_gps_seq[:, 1:, :],
                                                             tgt_seq_len - 1,
                                                             road_net)
                    total_acc += acc
                    total_prec += prec
                    total_recall += recall
                    total_f1 += f1
                    total_mae += mae
                    total_rmse += rmse
                else:
                    output_node_dim = output_node.size(2)
                    output_node = output_node.permute(1, 0, 2)
                    output_node = output_node.reshape(-1, output_node_dim)  # ((len - 1) * batch_size, node_size)
                    output_rate = output_rate.permute(1, 0)  # ((len - 1), batch_size)

                    tgt_node_seq = tgt_node_seq.permute(1, 0).reshape(-1)  # ((len - 1) * batch_size)
                    tgt_rate_seq = tgt_rate_seq.permute(1, 0)  # ((len - 1), batch_size)
                    loss = self.compute_loss(output_node, output_rate, None,
                                             tgt_node_seq, tgt_rate_seq, None,
                                             src_seq_len, tgt_seq_len, weights)
                    total_loss += loss
        if mode == "test":
            return np.mean(total_acc), np.mean(total_prec), np.mean(total_recall), \
                   np.mean(total_f1), np.mean(total_mae), np.mean(total_rmse)
        else:
            return total_loss / len(eval_loader)

    def evaluate_mtrajrec(
            self,
            test_dataset: Dataset = None,
            road_feat: torch.Tensor = None,
            tf_ratio: float = .0,
            road_net: SegmentCentricRoadNetwork = None,
            weights: Optional[List[float]] = None
    ):
        if test_dataset is None:
            eval_loader = self.get_eval_dataloader()
            mode = "eval"
        else:
            eval_loader = self.get_test_dataloader(test_dataset)
            self.load_model()
            mode = "test"
        self.model.eval()
        road_feat = road_feat.to(self.device)

        total_acc, total_prec, total_recall, total_f1, total_mae, total_rmse, total_loss = [], [], [], [], [], [], .0
        with torch.no_grad():
            for batch in tqdm.tqdm(eval_loader, total=len(eval_loader), desc=mode):
                src_grid_seq, src_gps_seq, feat_seq, src_seq_len, \
                tgt_gps_seq, tgt_node_seq, tgt_rate_seq, tgt_seq_len, tgt_inf_scores, \
                batched_graph, node_index = batch
                src_grid_seq, src_gps_seq, feat_seq = src_grid_seq.to(self.device), \
                                                      src_gps_seq.to(self.device), \
                                                      feat_seq.to(self.device)
                tgt_gps_seq, tgt_node_seq, tgt_rate_seq, tgt_inf_scores = tgt_gps_seq.to(self.device), \
                                                                          tgt_node_seq.to(self.device), \
                                                                          tgt_rate_seq.to(self.device), \
                                                                          tgt_inf_scores.to(self.device)

                output_node, output_rate, _ = self.model(road_feat, src_grid_seq, src_seq_len, feat_seq,
                                                         tgt_node_seq, tgt_rate_seq, tgt_seq_len, tgt_inf_scores,
                                                         tf_ratio)
                output_rate = output_rate.squeeze(-1)
                tgt_node_seq, tgt_rate_seq = tgt_node_seq.squeeze(-1), tgt_rate_seq.squeeze(-1)
                output_node, output_rate = output_node[:, 1:, :], output_rate[:, 1:]
                tgt_node_seq, tgt_rate_seq = tgt_node_seq[:, 1:], tgt_rate_seq[:, 1:]
                if mode == "test":
                    acc, prec, recall, f1 = evaluate_point_prediction(output_node,
                                                                      tgt_node_seq,
                                                                      tgt_seq_len - 1)
                    mae, rmse = evaluate_distance_regression(output_node,
                                                             output_rate,
                                                             tgt_gps_seq[:, 1:, :],
                                                             tgt_seq_len - 1,
                                                             road_net)
                    total_acc += acc
                    total_prec += prec
                    total_recall += recall
                    total_f1 += f1
                    total_mae += mae
                    total_rmse += rmse
                else:
                    output_node_dim = output_node.size(2)
                    output_node = output_node.permute(1, 0, 2)
                    output_node = output_node.reshape(-1, output_node_dim)  # ((len - 1) * batch_size, node_size)
                    output_rate = output_rate.permute(1, 0)  # ((len - 1), batch_size)

                    tgt_node_seq = tgt_node_seq.permute(1, 0).reshape(-1)  # ((len - 1) * batch_size)
                    tgt_rate_seq = tgt_rate_seq.permute(1, 0)  # ((len - 1), batch_size)
                    loss = self.compute_loss(output_node, output_rate, None,
                                             tgt_node_seq, tgt_rate_seq, None,
                                             src_seq_len, tgt_seq_len, weights)
                    total_loss += loss
        if mode == "test":
            return np.mean(total_acc), np.mean(total_prec), np.mean(total_recall), \
                   np.mean(total_f1), np.mean(total_mae), np.mean(total_rmse)
        else:
            return total_loss / len(eval_loader)

import tqdm
from typing import Union, Callable, Optional, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim import Optimizer
from sklearn.metrics import accuracy_score

from models.base_model import BaseModel
from utils.roadmap.road_network import SegmentCentricRoadNetwork
from utils.train.trainer import Trainer
from utils.data.point import SPoint, haversine_distance, rate2gps


def remove_repeats(seq):
    new_seq = [seq[0]]
    prev = seq[0]
    for curr in seq[1:]:
        if curr == prev:
            continue
        else:
            new_seq.append(curr)
        prev = curr
    return new_seq


def memoize(fn):
    cache = dict()

    def wrapped(*v):
        key = tuple(v)  # tuples are hashable, and can be used as dict keys
        if key not in cache:
            cache[key] = fn(*v)
        return cache[key]
    return wrapped


def lcs(xs, ys):
    @memoize
    def lcs_(i, j):
        if i and j:
            xe, ye = xs[i - 1], ys[j - 1]
            if xe == ye:
                return lcs_(i - 1, j - 1) + [xe]
            else:
                return max(lcs_(i, j - 1), lcs_(i - 1, j), key=len)
        else:
            return []

    return lcs_(len(xs), len(ys))


def evaluate_point_prediction(pred, label, seq_len):
    batch_size = pred.size(0)
    pred, label = pred.argmax(-1).detach().cpu().numpy(), label.detach().cpu().numpy()
    acc_batch, prec_batch, recall_batch, f1_batch = [], [], [], []
    for bs in range(batch_size):
        length = seq_len[bs]
        acc_batch.append(accuracy_score(label[bs, :length], pred[bs, :length]))
        remove_repeat_pred = remove_repeats(pred[bs, :length])
        remove_repeat_label = remove_repeats(label[bs, :length])
        true_positive = len(lcs(remove_repeat_pred, remove_repeat_label))
        pred_positive = len(remove_repeat_pred)
        real_positive = len(remove_repeat_label)
        precision = true_positive / pred_positive
        recall = true_positive / real_positive
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 1e-6 else 0
        prec_batch.append(precision)
        recall_batch.append(recall)
        f1_batch.append(f1)
    return acc_batch, prec_batch, recall_batch, f1_batch


def node2loc(road_net, node_seq, rate_seq):
    reverse_map = {val: key for key, val in road_net.reindex_nodes.items()}
    batch_size, seq_len = node_seq.size(0), node_seq.size(1)
    node_seq, rate_seq = node_seq.detach().cpu().numpy(), rate_seq.detach().cpu().numpy()
    loc_seq = torch.zeros(batch_size, seq_len, 2)
    for bs in range(batch_size):
        for idx in range(seq_len):
            if node_seq[bs][idx].argmax() != 0:
                node_idx = reverse_map[node_seq[bs][idx].argmax() - 1]
                rate = rate_seq[bs][idx]
                pt = rate2gps(road_net, node_idx, rate)
                loc_seq[bs, idx, 0], loc_seq[bs, idx, 1] = pt.lat, pt.lng
            else:
                loc_seq[bs, idx, 0] = (road_net.zone_range[0] + road_net.zone_range[2]) / 2
                loc_seq[bs, idx, 1] = (road_net.zone_range[1] + road_net.zone_range[3]) / 2
    return loc_seq


def evaluate_distance_regression(pred_node, pred_rate, label_loc, seq_len, road_net):
    batch_size = pred_node.size(0)
    pred_loc = node2loc(road_net, pred_node, pred_rate)
    label_loc = label_loc.detach().cpu().numpy()
    mae_batch, rmse_batch = [], []
    for bs in range(batch_size):
        length = seq_len[bs]
        dist = []
        for idx in range(length - 1):
            dist.append(haversine_distance(SPoint(pred_loc[bs, idx, 0].item(), pred_loc[bs, idx, 1].item()),
                                           SPoint(label_loc[bs, idx, 0].item(), label_loc[bs, idx, 1].item())))
        dist = np.array(dist)
        mae_batch.append(dist.mean())
        rmse_batch.append(np.sqrt((dist ** 2).mean()))
    return mae_batch, rmse_batch


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

    def compute_metrics(self, pred_node, pred_rate, true_node, true_position, seq_len, road_net):
        acc, prec, recall, f1 = evaluate_point_prediction(pred_node, true_node, seq_len)
        mae, rmse = evaluate_distance_regression(pred_node, pred_rate, true_position, seq_len, road_net)
        return acc, prec, recall, f1, mae, rmse

    def train(
            self,
            road_grid: Optional[torch.Tensor] = None,
            road_len: Optional[torch.Tensor] = None,
            road_nodes: Optional[torch.Tensor] = None,
            road_edge: Optional[torch.Tensor] = None,
            road_batch: Optional[torch.Tensor] = None,
            road_feat: Optional[torch.Tensor] = None,
            road_net: Optional[SegmentCentricRoadNetwork] = None,
            weights: Optional[List[float]] = None,
            decay_param: Optional[float] = None,
            tf_ratio: float = 0.
    ):
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
                if self.train_dataset.model_name == "RNTrajRec":
                    batched_graph, node_index = batched_graph.to(self.device), node_index.to(self.device)
                    traj_edge, tgt_node_seq = batched_graph.edge_index.long(), tgt_node_seq.long()

                    self.optimizer.zero_grad()
                    output_node, output_rate, logits = self.model(road_grid, road_len, road_nodes, road_edge, road_batch,
                                                                  road_feat, src_grid_seq, src_seq_len, feat_seq, node_index,
                                                                  traj_edge, batched_graph.batch, batched_graph.weight,
                                                                  tgt_node_seq, tgt_rate_seq, tgt_seq_len, tgt_inf_scores, tf_ratio)
                else:
                    self.optimizer.zero_grad()
                    output_node, output_rate, logits = self.model(road_feat, src_grid_seq, src_seq_len, feat_seq,
                                                                  tgt_node_seq, tgt_rate_seq, tgt_seq_len,
                                                                  tgt_inf_scores,
                                                                  tf_ratio)
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
                test_dataset=None,
                road_grid=road_grid,
                road_len=road_len,
                road_nodes=road_nodes,
                road_edge=road_edge,
                road_batch=road_batch,
                road_feat=road_feat,
                tf_ratio=.0,
                road_net=road_net,
                weights=weights
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
                self.optimizer.zero_grad()
                if self.train_dataset.model_name == "RNTrajRec":
                    batched_graph, node_index = batched_graph.to(self.device), node_index.to(self.device)

                    output_node, output_rate, logits = self.model(road_grid, road_len, road_nodes, road_edge,
                                                                  road_batch,
                                                                  road_feat, src_grid_seq, src_seq_len, feat_seq,
                                                                  node_index,
                                                                  traj_edge, batched_graph.batch, batched_graph.weight,
                                                                  tgt_node_seq, tgt_rate_seq, tgt_seq_len,
                                                                  tgt_inf_scores, tf_ratio)
                else:
                    output_node, output_rate, logits = self.model(road_feat, src_grid_seq, src_seq_len, feat_seq,
                                                                  tgt_node_seq, tgt_rate_seq, tgt_seq_len,
                                                                  tgt_inf_scores,
                                                                  tf_ratio)
                output_rate = output_rate.squeeze(-1)
                tgt_node_seq, tgt_rate_seq = tgt_node_seq.squeeze(-1), tgt_rate_seq.squeeze(-1)
                output_node, output_rate = output_node[:, 1:, :], output_rate[:, 1:]
                tgt_node_seq, tgt_rate_seq = tgt_node_seq[:, 1:], tgt_rate_seq[:, 1:]
                if mode == "test":
                    acc, prec, recall, f1, mae, rmse = self.compute_metrics(
                        output_node,
                        output_rate,
                        tgt_node_seq,
                        tgt_gps_seq[:, 1:],
                        tgt_seq_len - 1,
                        road_net
                    )
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

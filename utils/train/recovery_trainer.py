import tqdm
from typing import Union, Callable
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer

from models.base_model import BaseModel
from utils.train.trainer import Trainer


class RecoveryTrainer(Trainer):
    def __init__(
            self,
            model: BaseModel,
            train_dataset: Dataset,
            eval_dataset: Dataset,
            optimizer: Optimizer,
            num_epochs: int,
            data_collator: Callable,
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
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              pin_memory=pin_memory)
        self.device = device
        for key, val in kwargs.items():
            setattr(self, key, val)

    def compute_loss(self, pred_node, pred_rate, pred_logit, true_node, true_rate, true_logit,
                     src_len, tgt_len, weights):
        regression_criterion = nn.MSELoss(reduction="sum")
        classfication_criterion = nn.NLLLoss(reduction="sum")

        node_loss = classfication_criterion(pred_node, true_node) / tgt_len.sum()
        rate_loss = regression_criterion(pred_rate, true_rate) / tgt_len.sum()
        enc_loss = -1 * (pred_logit * true_logit).sum() / src_len.sum()
        return node_loss + rate_loss * weights[0] + enc_loss * weights[1]

    def train(self, road_grid, road_len, road_nodes, road_edge, road_batch, road_feat, weights):
        road_grid, road_nodes = road_grid.to(self.device), road_nodes.to(self.device)
        road_edge, road_batch = road_edge.to(self.device), road_batch.to(self.device)
        road_feat = road_feat.to(self.device)
        self.model.train()
        train_loader = DataLoader(self.train_dataset, **self.dataloader_params)
        for epoch in tqdm.tqdm(range(self.num_epochs), total=self.num_epochs):
            for batch in train_loader:
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

                self.optimizer.zero_grad()
                output_node, output_rate, logits = self.model(road_grid, road_len, road_nodes, road_edge, road_batch,
                                                              road_feat, src_grid_seq, src_seq_len, feat_seq, node_index,
                                                              batched_graph.edge_index, batched_graph.batch, batched_graph.weight,
                                                              tgt_node_seq, tgt_rate_seq, tgt_seq_len, tgt_inf_scores)
                output_node_dim = output_node.size(2)
                output_node = output_node.permute(1, 0, 2)[1:]
                output_node = output_node.reshape(-1, output_node_dim)  # ((len - 1) * batch_size, node_size)
                output_rate = output_rate.permute(1, 0, 2).squeeze(-1)[1:]  # ((len - 1) * batch_size)
                logits = logits.squeeze(-1)  # (num_nodes)

                tgt_node_seq = tgt_node_seq.permute(1, 0, 2)[1:]
                tgt_node_seq = tgt_node_seq.squeeze(-1)  # ((len - 1) * batch_size)
                tgt_rate_seq = tgt_rate_seq.permute(1, 0, 2)[1:]
                tgt_rate_seq = tgt_rate_seq.squeeze(-1)  # ((len - 1) * batch_size)
                loss = self.compute_loss(output_node, output_rate, logits,
                                         tgt_node_seq, tgt_rate_seq, batched_graph.gt,
                                         src_seq_len, tgt_seq_len, weights)
                loss.backward()
                self.optimizer.step()
        pass

    def evaluate(self, test_dataset: Dataset=None):
        if test_dataset is None:
            eval_loader = DataLoader(self.eval_dataset, **self.dataloader_params)
        else:
            eval_loader = DataLoader(test_dataset,
                                     batch_size=self.dataloader_params["batch_size"],
                                     shuffle=False,
                                     collate_fn=self.dataloader_params["collate_fn"],
                                     num_workers=self.dataloader_params["num_workers"],
                                     pin_memory=self.dataloader_params["pin_memory"])
        pass

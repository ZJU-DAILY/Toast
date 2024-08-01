import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from typing import Union, Optional, List

from augment.augment_config import PointUnionConfig, TaskType
from utils.train.trainer import Trainer


class PointUnion(nn.Module):
    def __init__(
            self,
            config: PointUnionConfig,
            trainer: Trainer,
            augment_dataset: Optional[Dataset],
            device: Union[torch.device, str] = "cpu",
            **kwargs
    ):
        super(PointUnion, self).__init__()
        self.config = config
        self.augment_dataset = augment_dataset
        self.trainer = trainer
        self.device = torch.device(device) if isinstance(device, str) else device
        # self.point_index = faiss.IndexHNSWFlat(config.virtual_dim, 32)
        if config.projection:
            self.virtual_embeds = nn.Embedding(config.num_virtual_tokens, config.projection_hidden_dim).to(self.device)
            self.transform = nn.Sequential(
                nn.Linear(config.projection_hidden_dim, config.projection_hidden_dim),
                nn.Tanh(),
                nn.Linear(config.projection_hidden_dim, config.virtual_dim)
            ).to(self.device)
        else:
            self.virtual_embeds = nn.Embedding(config.num_virtual_tokens, config.virtual_dim).to(self.device)

        for key, val in kwargs.items():
            setattr(self, key, val)

    def get_optimizer(self, parameters):
        return AdamW(parameters, lr=self.config.lr)

    def append_virtual_embedding(self, inputs, seq_len):
        batch_size = inputs.size(0)
        indices = torch.arange(self.config.num_virtual_tokens, dtype=torch.long, device=self.device)
        indices = torch.stack([indices] * batch_size, dim=0)
        if self.config.projection:
            virtual_tokens = self.virtual_embeds(indices)
            virtual_inputs = self.transform(virtual_tokens)
        else:
            virtual_inputs = self.virtual_embeds(indices)
        if self.config.task_type == TaskType.TYPE_IDENTIFY:
            output_shape = inputs.shape
            augment_length = seq_len
        else:
            output_shape = (
                inputs.size(0),
                inputs.size(1) + self.config.num_virtual_tokens,
                inputs.size(2)
            )
            augment_length = seq_len + self.config.num_virtual_tokens
        outputs = torch.zeros(output_shape, dtype=torch.float, device=self.device)
        for bs in range(batch_size):
            length = seq_len[bs]
            outputs[bs, :length] = inputs[bs, :length]
            outputs[bs, length:length + self.config.num_virtual_tokens] = virtual_inputs[bs]
        return outputs, augment_length

    def train_virtual_embedding(
            self,
            dataloader: Optional[DataLoader],
            road_embeds: Optional[torch.Tensor] = None,
            road_feats: Optional[torch.Tensor] = None,
            tf_ratio: float = 0.0,
            weights: Optional[List[float]] = None
    ):
        optimizer = self.get_optimizer(self.parameters())
        for epoch in range(self.config.num_epochs):
            self.trainer.model.train()
            for batch in tqdm.tqdm(dataloader, total=len(dataloader),
                                   desc="train virtual embeddings @ {}".format(epoch + 1)):
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

                optimizer.zero_grad()
                if self.config.model_name == "RNTrajRec":
                    batched_graph, node_index = batched_graph.to(self.device), node_index.to(self.device)
                    traj_edge, tgt_node_seq = batched_graph.edge_index.long(), tgt_node_seq.long()
                    encoder_embed, hidden_embed, _ = self.trainer.model.encoding(
                        road_embeds, src_grid_seq, src_seq_len,
                        feat_seq, node_index, traj_edge,
                        batched_graph.batch, batched_graph.weight
                    )
                    decode_inputs, src_seq_len = self.append_virtual_embedding(
                        encoder_embed,
                        src_seq_len
                    )
                    pred_node, pred_rate = self.trainer.model.decoding(
                        decode_inputs, hidden_embed, road_embeds, road_feats,
                        src_seq_len, tgt_node_seq, tgt_rate_seq, tgt_seq_len,
                        tgt_inf_scores, tf_ratio
                    )
                else:
                    encoder_embed, hidden_embed = self.trainer.model.encoding(src_grid_seq, src_seq_len, feat_seq)
                    decode_inputs, src_seq_len = self.append_virtual_embedding(
                        encoder_embed,
                        src_seq_len
                    )
                    pred_node, pred_rate = self.trainer.model.decoding(
                        decode_inputs, hidden_embed, road_feats,
                        src_seq_len, tgt_node_seq, tgt_rate_seq, tgt_seq_len, tgt_inf_scores, tf_ratio
                    )
                num_pred_node = pred_node.size(2)
                pred_rate = pred_rate.squeeze(-1)[:, 1:]
                pred_node = pred_node.permute(1, 0, 2)[1:]
                pred_node = pred_node.reshape(-1, num_pred_node)  # ((len - 1) * batch_size, node_size)
                pred_rate = pred_rate.permute(1, 0)  # ((len - 1), batch_size)

                true_node, true_rate = tgt_node_seq.squeeze(-1)[:, 1:], tgt_rate_seq.squeeze(-1)[:, 1:]
                true_node = true_node.permute(1, 0).reshape(-1)
                true_rate = true_rate.permute(1, 0)
                loss = self.trainer.compute_loss(pred_node, pred_rate, None,
                                                 true_node, true_rate, None,
                                                 None, tgt_seq_len, weights)
                loss.backward()
                optimizer.step()
            # eval_loss = self.evaluate_virtual_embedding(dataloader, road_embeds, road_feats, 0.0, weights)
            # print("eval loss: {:.4f}".format(eval_loss.item()))

    @torch.no_grad()
    def evaluate_virtual_embedding(
            self,
            dataloader: Optional[DataLoader] = None,
            road_embeds: Optional[torch.Tensor] = None,
            road_feats: Optional[torch.Tensor] = None,
            tf_ratio: float = 0.0,
            weights: Optional[List[float]] = None
    ):
        self.trainer.model.eval()
        total_loss = .0
        for batch in tqdm.tqdm(dataloader, total=len(dataloader),
                               desc="evaluate virtual embeddings"):
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

            if self.config.model_name == "RNTrajRec":
                batched_graph, node_index = batched_graph.to(self.device), node_index.to(self.device)
                traj_edge, tgt_node_seq = batched_graph.edge_index.long(), tgt_node_seq.long()
                encoder_embed, hidden_embed, _ = self.trainer.model.encoding(
                    road_embeds, src_grid_seq, src_seq_len,
                    feat_seq, node_index, traj_edge,
                    batched_graph.batch, batched_graph.weight
                )
                decode_inputs, src_seq_len = self.append_virtual_embedding(
                    encoder_embed,
                    src_seq_len
                )
                pred_node, pred_rate = self.trainer.model.decoding(
                    decode_inputs, hidden_embed, road_embeds, road_feats,
                    src_seq_len, tgt_node_seq, tgt_rate_seq, tgt_seq_len,
                    tgt_inf_scores, tf_ratio
                )
            else:
                encoder_embed, hidden_embed = self.trainer.model.encoding(src_grid_seq, src_seq_len, feat_seq)
                decode_inputs, src_seq_len = self.append_virtual_embedding(
                    encoder_embed,
                    src_seq_len
                )
                pred_node, pred_rate = self.trainer.model.decoding(
                    decode_inputs, hidden_embed, road_feats,
                    src_seq_len, tgt_node_seq, tgt_rate_seq, tgt_seq_len, tgt_inf_scores, tf_ratio
                )

            num_pred_node = pred_node.size(2)
            pred_rate = pred_rate.squeeze(-1)[:, 1:]
            pred_node = pred_node.permute(1, 0, 2)[1:]
            pred_node = pred_node.reshape(-1, num_pred_node)  # ((len - 1) * batch_size, node_size)
            pred_rate = pred_rate.permute(1, 0)  # ((len - 1), batch_size)

            true_node, true_rate = tgt_node_seq.squeeze(-1)[:, 1:], tgt_rate_seq.squeeze(-1)[:, 1:]
            true_node = true_node.permute(1, 0).reshape(-1)
            true_rate = true_rate.permute(1, 0)
            loss = self.trainer.compute_loss(pred_node, pred_rate, None,
                                             true_node, true_rate, None,
                                             None, tgt_seq_len, weights)
            total_loss += loss
        return total_loss / len(dataloader)

    @torch.no_grad()
    def evaluate_recovery_augment(
            self,
            test_set,
            road_net=None,
            road_grid=None,
            road_len=None,
            road_nodes=None,
            road_edge=None,
            road_batch=None,
            road_feats=None
    ):
        self.trainer.model.eval()
        test_loader = self.trainer.get_test_dataloader(test_set)
        road_feats = road_feats.to(self.device)

        total_acc, total_prec, total_recall, total_f1, total_mae, total_rmse, total_loss = [], [], [], [], [], [], .0
        for batch in tqdm.tqdm(test_loader, total=len(test_loader), desc="test augmentation"):
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

            if self.config.model_name == "RNTrajRec":
                road_grid, road_nodes = road_grid.to(self.device), road_nodes.to(self.device)
                road_edge, road_batch = road_edge.long().to(self.device), road_batch.to(self.device)
                road_embeds = self.trainer.model.extract_feat(road_grid, road_len, road_nodes, road_edge, road_batch)
                batched_graph, node_index = batched_graph.to(self.device), node_index.to(self.device)
                traj_edge, tgt_node_seq = batched_graph.edge_index.long(), tgt_node_seq.long()
                encoder_embed, hidden_embed, _ = self.trainer.model.encoding(
                    road_embeds, src_grid_seq, src_seq_len,
                    feat_seq, node_index, traj_edge,
                    batched_graph.batch, batched_graph.weight
                )
                decode_inputs, src_seq_len = self.append_virtual_embedding(
                    encoder_embed,
                    src_seq_len
                )
                pred_node, pred_rate = self.trainer.model.decoding(
                    decode_inputs, hidden_embed, road_embeds, road_feats,
                    src_seq_len, tgt_node_seq, tgt_rate_seq, tgt_seq_len,
                    tgt_inf_scores, 0.0
                )
            else:
                encoder_embed, hidden_embed = self.trainer.model.encoding(src_grid_seq, src_seq_len, feat_seq)
                decode_inputs, src_seq_len = self.append_virtual_embedding(
                    encoder_embed,
                    src_seq_len
                )
                pred_node, pred_rate = self.trainer.model.decoding(
                    decode_inputs, hidden_embed, road_feats,
                    src_seq_len, tgt_node_seq, tgt_rate_seq, tgt_seq_len, tgt_inf_scores, 0.0
                )

            pred_rate = pred_rate.squeeze(-1)
            tgt_node_seq, tgt_rate_seq = tgt_node_seq.squeeze(-1), tgt_rate_seq.squeeze(-1)
            pred_node, pred_rate = pred_node[:, 1:, :], pred_rate[:, 1:]
            tgt_node_seq, tgt_rate_seq = tgt_node_seq[:, 1:], tgt_rate_seq[:, 1:]
            acc, prec, recall, f1, mae, rmse = self.trainer.compute_metrics(
                pred_node,
                pred_rate,
                tgt_node_seq,
                tgt_gps_seq[:, 1:, :],
                tgt_seq_len - 1,
                road_net
            )
            total_acc += acc
            total_prec += prec
            total_recall += recall
            total_f1 += f1
            total_mae += mae
            total_rmse += rmse
        return np.mean(total_acc), np.mean(total_prec), np.mean(total_recall), \
               np.mean(total_f1), np.mean(total_mae), np.mean(total_rmse)

    def train_similarity_virtual_embeddings(
            self,
            dataloader: Optional[DataLoader],
            node_feat: Optional[torch.Tensor] = None,
            edge_index: Optional[torch.Tensor] = None,
            edge_attr: Optional[torch.Tensor] = None
    ):
        optimizer = self.get_optimizer(self.parameters())
        for epoch in range(self.config.num_epochs):
            self.trainer.model.train()
            for batch in tqdm.tqdm(dataloader, total=len(dataloader),
                                   desc="train virtual embeddings @ {}".format(epoch + 1)):
                optimizer.zero_grad()
                if self.config.model_name == "ST2Vec":
                    a_nodes, a_time, a_len, p_nodes, p_time, p_len, n_nodes, n_time, n_len, pos_dist, neg_dist = batch
                    a_nodes, a_time = a_nodes.to(self.device), a_time.to(self.device)
                    p_nodes, p_time = p_nodes.to(self.device), p_time.to(self.device)
                    n_nodes, n_time = n_nodes.to(self.device), n_time.to(self.device)
                    pos_dist, neg_dist = pos_dist.to(self.device), neg_dist.to(self.device)
                    spatial_embeds, time_embeds = self.trainer.model.extract_feat(
                        a_nodes, a_time, a_len, node_feat, edge_index, edge_attr
                    )
                    suffix_spatial_embeds, augment_lengths = self.append_virtual_embedding(
                        spatial_embeds, a_len
                    )
                    suffix_temporal_embeds, _ = self.append_virtual_embedding(
                        time_embeds, a_len
                    )
                    a_embeds = self.trainer.model.encoding(
                        suffix_spatial_embeds,
                        suffix_temporal_embeds,
                        augment_lengths
                    )
                    p_embeds = self.trainer.model(p_nodes, p_time, p_len, node_feat, edge_index, edge_attr)
                    n_embeds = self.trainer.model(n_nodes, n_time, n_len, node_feat, edge_index, edge_attr)
                else:
                    a_nodes, _, a_len, p_nodes, _, p_len, n_nodes, _, n_len, pos_dist, neg_dist = batch
                    a_nodes, p_nodes, n_nodes = a_nodes.to(self.device), p_nodes.to(self.device), n_nodes.to(self.device)
                    pos_dist, neg_dist = pos_dist.to(self.device), neg_dist.to(self.device)
                    spatial_embeds = self.trainer.model.extract_feat(
                        a_nodes, a_len, node_feat, edge_index, edge_attr
                    )
                    suffix_spatial_embeds, augment_lengths = self.append_virtual_embedding(
                        spatial_embeds, a_len
                    )
                    a_embeds = self.trainer.model.encoding(
                        suffix_spatial_embeds,
                        augment_lengths
                    )
                    p_embeds = self.trainer.model(p_nodes, p_len, node_feat, edge_index, edge_attr)
                    n_embeds = self.trainer.model(n_nodes, n_len, node_feat, edge_index, edge_attr)
                loss = self.trainer.compute_loss(a_embeds, p_embeds, n_embeds, pos_dist, neg_dist)
                loss.backward()
                optimizer.step()

    @torch.no_grad()
    def evaluate_similarity_virtual_embeddings(
            self,
            dataloader: DataLoader,
            node_feat=None,
            edge_index=None,
            edge_attr=None
    ):
        self.trainer.model.eval()
        embeddings, true_dist = [], []
        for batch in tqdm.tqdm(dataloader, total=len(dataloader),
                               desc="evaluate virtual embeddings"):
            if self.config.model_name == "ST2Vec":
                nodes, time, seq_len, dist = batch
                nodes, time = nodes.to(self.device), time.to(self.device)
                spatial_embeds, time_embeds = self.trainer.model.extract_feat(
                    nodes, time, seq_len, node_feat, edge_index, edge_attr
                )
                suffix_spatial_embeds, augment_lengths = self.append_virtual_embedding(
                    spatial_embeds, seq_len
                )
                suffix_temporal_embeds, _ = self.append_virtual_embedding(
                    time_embeds, seq_len
                )
                batch_embeds = self.trainer.model.encoding(
                    suffix_spatial_embeds,
                    suffix_temporal_embeds,
                    augment_lengths
                )
            else:
                nodes, _, seq_len, dist = batch
                nodes = nodes.to(self.device)
                spatial_embeds = self.trainer.model.extract_feat(
                    nodes, seq_len, node_feat, edge_index, edge_attr
                )
                suffix_spatial_embeds, augment_lengths = self.append_virtual_embedding(
                    spatial_embeds, seq_len
                )
                batch_embeds = self.trainer.model.encoding(
                    suffix_spatial_embeds, augment_lengths
                )
            embeddings.append(batch_embeds)
            true_dist.append(dist)
        embeddings = torch.cat(embeddings, dim=0)
        true_dist = torch.cat(true_dist, dim=0)
        hr10, hr50, hr10_50 = self.trainer.compute_metrics(embeddings.cpu().numpy(), true_dist[:5000].numpy())
        return hr10, hr50, hr10_50

    @torch.no_grad()
    def evaluate_similarity_augment(
            self,
            test_set,
            node_feat=None,
            edge_index=None,
            edge_attr=None
    ):
        self.trainer.model.eval()
        test_loader = self.trainer.get_test_dataloader(test_set)
        embeddings, true_dist = [], []
        for batch in tqdm.tqdm(test_loader, total=len(test_loader),
                               desc="evaluate virtual embeddings"):
            if self.config.model_name == "ST2Vec":
                nodes, time, seq_len, dist = batch
                nodes, time = nodes.to(self.device), time.to(self.device)
                spatial_embeds, time_embeds = self.trainer.model.extract_feat(
                    nodes, time, seq_len, node_feat, edge_index, edge_attr
                )
                suffix_spatial_embeds, augment_lengths = self.append_virtual_embedding(
                    spatial_embeds, seq_len
                )
                suffix_temporal_embeds, _ = self.append_virtual_embedding(
                    time_embeds, seq_len
                )
                batch_embeds = self.trainer.model.encoding(
                    suffix_spatial_embeds,
                    suffix_temporal_embeds,
                    augment_lengths
                )
            else:
                nodes, _, seq_len, dist = batch
                nodes = nodes.to(self.device)
                spatial_embeds = self.trainer.model.extract_feat(
                    nodes, seq_len, node_feat, edge_index, edge_attr
                )
                suffix_spatial_embeds, augment_lengths = self.append_virtual_embedding(
                    spatial_embeds, seq_len
                )
                batch_embeds = self.trainer.model.encoding(
                    suffix_spatial_embeds, augment_lengths
                )
            embeddings.append(batch_embeds)
            true_dist.append(dist)
        embeddings = torch.cat(embeddings, dim=0)
        true_dist = torch.cat(true_dist, dim=0)
        hr10, hr50, hr10_50 = self.trainer.compute_metrics(embeddings.cpu().numpy(), true_dist[:5000].numpy())
        return hr10, hr50, hr10_50

    @torch.no_grad()
    def evaluate_similarity_augment(
            self,
            test_set,
            node_feat=None,
            edge_index=None,
            edge_attr=None
    ):
        self.trainer.model.eval()
        test_loader = self.trainer.get_test_dataloader(test_set)
        embeddings, true_dist = [], []
        for batch in tqdm.tqdm(test_loader, total=len(test_loader),
                               desc="evaluate virtual embeddings"):
            if self.config.model_name == "ST2Vec":
                nodes, time, seq_len, dist = batch
                nodes, time = nodes.to(self.device), time.to(self.device)
                spatial_embeds, time_embeds = self.trainer.model.extract_feat(
                    nodes, time, seq_len, node_feat, edge_index, edge_attr
                )
                suffix_spatial_embeds, augment_lengths = self.append_virtual_embedding(
                    spatial_embeds, seq_len
                )
                suffix_temporal_embeds, _ = self.append_virtual_embedding(
                    time_embeds, seq_len
                )
                batch_embeds = self.trainer.model.encoding(
                    suffix_spatial_embeds,
                    suffix_temporal_embeds,
                    augment_lengths
                )
            else:
                nodes, _, seq_len, dist = batch
                nodes = nodes.to(self.device)
                spatial_embeds = self.trainer.model.extract_feat(
                    nodes, seq_len, node_feat, edge_index, edge_attr
                )
                suffix_spatial_embeds, augment_lengths = self.append_virtual_embedding(
                    spatial_embeds, seq_len
                )
                batch_embeds = self.trainer.model.encoding(
                    suffix_spatial_embeds, augment_lengths
                )
            embeddings.append(batch_embeds)
            true_dist.append(dist)
        embeddings = torch.cat(embeddings, dim=0)
        true_dist = torch.cat(true_dist, dim=0)
        hr10, hr50, hr10_50 = self.trainer.compute_metrics(embeddings.cpu().numpy(), true_dist[:5000].numpy())
        return hr10, hr50, hr10_50

    def train_identify_virtual_embeddings(
            self,
            dataloader: Optional[DataLoader],
            max_length: int
    ):
        optimizer = self.get_optimizer(self.parameters())
        for epoch in range(self.config.num_epochs):
            self.trainer.model.train()
            for batch in tqdm.tqdm(dataloader, total=len(dataloader),
                                   desc="train virtual embeddings @ {}".format(epoch + 1)):
                optimizer.zero_grad()
                batch_data, seq_len, batch_labels = batch
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                suffix_data, _ = self.append_virtual_embedding(
                    batch_data.squeeze(dim=1), seq_len
                )
                if self.config.model_name == "SECA":
                    hidden_unlabel, pool_indices, hidden_label = self.trainer.model.encoding(
                        suffix_data.unsqueeze(dim=1), suffix_data.unsqueeze(dim=1)
                    )
                    recover_data = self.trainer.model.decoding(hidden_unlabel, pool_indices[::-1])
                    pred_logits = self.trainer.model.cls_layer(hidden_label)
                    recover_data = recover_data.permute(0, 2, 3, 1)[:, :max_length]
                else:
                    hiddens, _ = self.trainer.model.encoding(suffix_data.unsqueeze(dim=1))
                    pred_logits = self.trainer.model.decoding(hiddens)
                    recover_data = None

                loss = self.trainer.compute_loss(recover_data, pred_logits, batch_data, batch_labels)
                loss.backward()
                optimizer.step()

    @torch.no_grad()
    def evaluate_identify_augment(
            self,
            test_set
    ):
        self.trainer.model.eval()
        test_loader = self.trainer.get_test_dataloader(test_set)
        pred_labels, true_labels = [], []
        for batch in tqdm.tqdm(test_loader, total=len(test_loader),
                               desc="evaluate virtual embeddings"):
            batch_data, seq_len, batch_labels = batch
            batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
            suffix_data, _ = self.append_virtual_embedding(
                batch_data.squeeze(dim=1), seq_len
            )
            if self.config.model_name == "SECA":
                hidden_unlabel, pool_indices, hidden_label = self.trainer.model.encoding(
                    suffix_data.unsqueeze(dim=1), suffix_data.unsqueeze(dim=1)
                )
                pred_logits = self.trainer.model.cls_layer(hidden_label)
            else:
                hiddens, _ = self.trainer.model.encoding(suffix_data.unsqueeze(dim=1))
                pred_logits = self.trainer.model.decoding(hiddens)
            pred_labels.append(pred_logits.argmax(dim=-1).cpu())
            true_labels.append(batch_labels.cpu())
        true_labels, pred_labels = torch.cat(true_labels, dim=0), torch.cat(pred_labels, dim=0)
        acc, prec, recall, macro_f1, weighted_f1 = self.trainer.compute_metrics(pred_labels, true_labels)
        return acc, prec, recall, macro_f1, weighted_f1

    def check_kwargs(self, kwargs):
        if self.config.task_type == TaskType.TRAJ_RECOVERY:
            if self.config.model_name == "RNTrajRec":
                return "road_grid" in kwargs
            else:
                return "road_feat" in kwargs
        elif self.config.task_type == TaskType.TRAJ_SIMILAR:
            return "edge_index" in kwargs
        elif self.config.task_type == TaskType.TYPE_IDENTIFY:
            return "max_length" in kwargs

    def augment_points(
            self,
            test_set,
            **kwargs
    ):
        self.trainer.freeze_model()
        if self.config.task_type == TaskType.TRAJ_RECOVERY:
            valid_dataloader = self.trainer.get_eval_dataloader()
            augment_dataloader = self.trainer.get_test_dataloader(self.augment_dataset)
            if self.config.model_name == "RNTrajRec" and self.check_kwargs(kwargs):
                road_net, road_grid, road_len, road_nodes, road_edge, road_batch, road_feat, weight = (
                    kwargs["road_net"], kwargs["road_grid"],
                    kwargs["road_len"], kwargs["road_nodes"],
                    kwargs["road_edge"], kwargs["road_batch"],
                    kwargs["road_feat"], kwargs["weights"]
                )
                road_grid, road_nodes = road_grid.to(self.device), road_nodes.to(self.device)
                road_edge, road_batch = road_edge.long().to(self.device), road_batch.to(self.device)
                road_feat = road_feat.to(self.device)
                road_embed = self.trainer.model.extract_feat(road_grid, road_len, road_nodes, road_edge, road_batch)
                self.train_virtual_embedding(
                    dataloader=valid_dataloader,
                    road_embeds=road_embed,
                    road_feats=road_feat,
                    tf_ratio=0.0,
                    weights=weight
                )
                results = self.evaluate_recovery_augment(
                    test_set,
                    road_net,
                    road_grid,
                    road_len,
                    road_nodes,
                    road_edge,
                    road_batch,
                    road_feat
                )
            elif self.check_kwargs(kwargs):
                road_net, road_feat, weight = kwargs["road_net"], kwargs["road_feat"], kwargs["weights"]
                road_feat = road_feat.to(self.device)
                self.train_virtual_embedding(
                    valid_dataloader,
                    road_feats=road_feat,
                    tf_ratio=0.0,
                    weights=weight
                )
                results = self.evaluate_recovery_augment(
                    test_set,
                    road_net=road_net,
                    road_feats=road_feat
                )
        elif self.config.task_type == TaskType.TRAJ_SIMILAR and self.check_kwargs(kwargs):
            train_dataloader = self.trainer.get_train_dataloader()
            augment_dataloader = self.trainer.get_test_dataloader(self.augment_dataset)
            node_feat, edge_index, edge_attr = kwargs["node_feat"], kwargs["edge_index"], kwargs["edge_attr"]
            node_feat, edge_index, edge_attr = (
                node_feat.to(self.device),
                edge_index.to(self.device),
                edge_attr.to(self.device)
            )
            self.train_similarity_virtual_embeddings(
                train_dataloader,
                node_feat=node_feat,
                edge_index=edge_index,
                edge_attr=edge_attr
            )
            results = self.evaluate_similarity_augment(
                test_set,
                node_feat=node_feat,
                edge_index=edge_index,
                edge_attr=edge_attr
            )
        elif self.config.task_type == TaskType.TYPE_IDENTIFY and self.check_kwargs(kwargs):
            train_dataloader = self.trainer.get_train_dataloader()
            augment_dataloader = self.trainer.get_test_dataloader(self.augment_dataset)
            self.train_identify_virtual_embeddings(
                train_dataloader,
                max_length=kwargs["max_length"]
            )
            results = self.evaluate_identify_augment(
                test_set
            )
        return results

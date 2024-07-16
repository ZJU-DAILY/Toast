import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import Dataset
from torch.optim import AdamW
from typing import Union, Optional
import matplotlib.pyplot as plt

from augment.augment_config import PointUnionConfig
from utils.train.trainer import Trainer
from utils.metrics.recovery_metrics import evaluate_point_prediction, evaluate_distance_regression


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

    @torch.no_grad()
    def visualize_virtual_embedding(self, axs, batch):
        if self.config.projection:
            virtual_embeds = self.transform(self.virtual_embeds.weight).detach()
        else:
            virtual_embeds = self.virtual_embeds.weight.detach()
        # rerank = [0, 3, 4, 8, 10, 11, 12, 13, 16, 18,
        #           1, 2, 5, 6, 7, 9, 14, 15, 17, 19]
        # rerank = torch.tensor(rerank, dtype=torch.long, device=self.device)
        # virtual_embeds = torch.index_select(virtual_embeds, dim=0, index=rerank)
        cosine_similarity = F.cosine_similarity(virtual_embeds.unsqueeze(1),
                                                virtual_embeds.unsqueeze(0),
                                                dim=-1).cpu().numpy()
        axs.flat[batch // 5].imshow(cosine_similarity, cmap="viridis")
        axs.flat[batch // 5].set_title("At epoch {}".format(batch + 1))

    def append_virtual_embedding(self, inputs, seq_len):
        batch_size = inputs.size(0)
        indices = torch.arange(self.config.num_virtual_tokens, dtype=torch.long, device=self.device)
        indices = torch.stack([indices] * batch_size, dim=0)
        if self.config.projection:
            virtual_tokens = self.virtual_embeds(indices)
            virtual_inputs = self.transform(virtual_tokens)
        else:
            virtual_inputs = self.virtual_embeds(indices)
        output_shape = (
            inputs.size(0),
            inputs.size(1) + self.config.num_virtual_tokens,
            inputs.size(2)
        )
        outputs = torch.zeros(output_shape, dtype=torch.float, device=self.device)
        for bs in range(batch_size):
            length = seq_len[bs]
            outputs[bs, :length] = inputs[bs, :length]
            outputs[bs, length:length + self.config.num_virtual_tokens] = virtual_inputs[bs]
        seq_len += self.config.num_virtual_tokens
        return outputs, seq_len

    def virtual_train_mtrajrec(self, dataloader, road_feats, tf_ratio, weights):
        optimizer = self.get_optimizer(self.parameters())
        for epoch in range(self.config.num_epochs):
            self.trainer.model.train()
            for batch in tqdm.tqdm(dataloader, total=len(dataloader), desc=f"train virtual embedding @ {epoch + 1}"):
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
            eval_loss = self.virtual_evaluate_mtrajrec(dataloader, road_feats, 0.0, weights)
            print("eval loss: {:.4f}".format(eval_loss.item()))

    def train_virtual_embedding(self, dataloader, road_embeds, road_feats, tf_ratio, weights):
        optimizer = self.get_optimizer(self.parameters())
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
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
                batched_graph, node_index = batched_graph.to(self.device), node_index.to(self.device)
                traj_edge, tgt_node_seq = batched_graph.edge_index.long(), tgt_node_seq.long()

                optimizer.zero_grad()
                encoder_embed, hidden_embed, _ = self.trainer.model.encoding(road_embeds, src_grid_seq, src_seq_len,
                                                                             feat_seq, node_index, traj_edge,
                                                                             batched_graph.batch, batched_graph.weight)
                decode_inputs, src_seq_len = self.append_virtual_embedding(
                    encoder_embed,
                    src_seq_len
                )
                pred_node, pred_rate = self.trainer.model.decoding(decode_inputs, hidden_embed, road_embeds, road_feats,
                                                                   src_seq_len, tgt_node_seq, tgt_rate_seq, tgt_seq_len,
                                                                   tgt_inf_scores, tf_ratio)
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
            eval_loss = self.evaluate_virtual_embedding(dataloader, road_embeds, road_feats, 0.0, weights)
            print("eval loss: {:.4f}".format(eval_loss.item()))
            if (epoch + 1) % 5 == 0:
                self.visualize_virtual_embedding(axs, epoch)
        plt.savefig("./ckpt/virtual_embedding.png", dpi=300)

    @torch.no_grad()
    def virtual_evaluate_mtrajrec(self, dataloader, road_feats, tf_ratio, weights):
        self.trainer.model.eval()
        total_loss = .0
        for batch in tqdm.tqdm(dataloader, total=len(dataloader), desc="evaluate virtual embeddings"):
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
    def evaluate_virtual_embedding(self, dataloader, road_embeds, road_feats, tf_ratio, weights):
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
            batched_graph, node_index = batched_graph.to(self.device), node_index.to(self.device)
            traj_edge, tgt_node_seq = batched_graph.edge_index.long(), tgt_node_seq.long()

            encoder_embed, hidden_embed, _ = self.trainer.model.encoding(road_embeds, src_grid_seq, src_seq_len,
                                                                         feat_seq, node_index, traj_edge,
                                                                         batched_graph.batch, batched_graph.weight)
            decode_inputs, src_seq_len = self.append_virtual_embedding(
                encoder_embed,
                src_seq_len
            )
            pred_node, pred_rate = self.trainer.model.decoding(decode_inputs, hidden_embed, road_embeds, road_feats,
                                                               src_seq_len, tgt_node_seq, tgt_rate_seq, tgt_seq_len,
                                                               tgt_inf_scores, tf_ratio)
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

    def augment_points(self, test_set, model_type, road_net, road_grid, road_len, road_nodes, road_edge, road_batch, road_feat, weight):
        self.trainer.freeze_model()
        valid_dataloader = self.trainer.get_eval_dataloader()
        augment_dataloader = self.trainer.get_test_dataloader(self.augment_dataset)
        if model_type == "RNTrajRec":
            road_grid, road_nodes = road_grid.to(self.device), road_nodes.to(self.device)
            road_edge, road_batch = road_edge.long().to(self.device), road_batch.to(self.device)
            road_feat = road_feat.to(self.device)
            road_embed = self.trainer.model.extract_feat(road_grid, road_len, road_nodes, road_edge, road_batch)

            self.train_virtual_embedding(valid_dataloader, road_embed, road_feat, 0.0, weight)
            results = self.evaluate_virtual_embedding(
                test_set, road_net, road_grid, road_len, road_nodes, road_edge, road_batch, road_feat
            )
        else:
            road_feat = road_feat.to(self.device)
            self.virtual_train_mtrajrec(valid_dataloader, road_feat, 0.0, weight)
            results = self.evaluate_augment_mtrajrec(test_set, road_net, road_feat)
        return results
        # with torch.no_grad():
        #     if self.config.projection:
        #         self.virtual_embeds = self.transform(self.virtual_embeds).detach()
        #
        #     for batch in augment_dataloader:
        #         src_grid_seq, src_gps_seq, feat_seq, src_seq_len, \
        #         tgt_gps_seq, tgt_node_seq, tgt_rate_seq, tgt_seq_len, tgt_inf_scores, \
        #         batched_graph, node_index = batch
        #         src_grid_seq, src_gps_seq, feat_seq = src_grid_seq.to(self.device), \
        #                                               src_gps_seq.to(self.device), \
        #                                               feat_seq.to(self.device)
        #         tgt_gps_seq, tgt_node_seq, tgt_rate_seq, tgt_inf_scores = tgt_gps_seq.to(self.device), \
        #                                                                   tgt_node_seq.to(self.device), \
        #                                                                   tgt_rate_seq.to(self.device), \
        #                                                                   tgt_inf_scores.to(self.device)
        #         batched_graph, node_index = batched_graph.to(self.device), node_index.to(self.device)
        #         traj_edge, tgt_node_seq = batched_graph.edge_index.long(), tgt_node_seq.long()
        #
        #         point_embed, _, _ = self.trainer.model.encoding(road_embed, src_grid_seq, src_seq_len, feat_seq,
        #                                                         node_index, traj_edge, batched_graph.batch,
        #                                                         batched_graph.weight)
        #         self.point_index.add(point_embed)
        # augment_embeds, _ = self.point_index.search(self.virtual_embeds, self.config.num_virtual_tokens)
        # return augment_embeds

    @torch.no_grad()
    def evaluate_augment(self, test_set, road_net, road_grid, road_len, road_nodes, road_edge, road_batch, road_feats):
        self.trainer.model.eval()
        test_loader = self.trainer.get_test_dataloader(test_set)
        road_grid, road_nodes = road_grid.to(self.device), road_nodes.to(self.device)
        road_edge, road_batch = road_edge.long().to(self.device), road_batch.to(self.device)
        road_feats = road_feats.to(self.device)
        road_embeds = self.trainer.model.extract_feat(road_grid, road_len, road_nodes, road_edge, road_batch)

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
            batched_graph, node_index = batched_graph.to(self.device), node_index.to(self.device)
            traj_edge, tgt_node_seq = batched_graph.edge_index.long(), tgt_node_seq.long()
            encoder_embed, hidden_embed, _ = self.trainer.model.encoding(road_embeds, src_grid_seq, src_seq_len,
                                                                         feat_seq, node_index, traj_edge,
                                                                         batched_graph.batch, batched_graph.weight)
            decode_inputs, src_seq_len = self.append_virtual_embedding(
                encoder_embed,
                src_seq_len
            )
            pred_node, pred_rate = self.trainer.model.decoding(decode_inputs, hidden_embed, road_embeds, road_feats,
                                                               src_seq_len, tgt_node_seq, tgt_rate_seq, tgt_seq_len,
                                                               tgt_inf_scores, .0)
            pred_rate = pred_rate.squeeze(-1)
            tgt_node_seq, tgt_rate_seq = tgt_node_seq.squeeze(-1), tgt_rate_seq.squeeze(-1)
            pred_node, pred_rate = pred_node[:, 1:, :], pred_rate[:, 1:]
            tgt_node_seq, tgt_rate_seq = tgt_node_seq[:, 1:], tgt_rate_seq[:, 1:]

            acc, prec, recall, f1 = evaluate_point_prediction(pred_node,
                                                              tgt_node_seq,
                                                              tgt_seq_len - 1)
            mae, rmse = evaluate_distance_regression(pred_node,
                                                     pred_rate,
                                                     tgt_gps_seq[:, 1:, :],
                                                     tgt_seq_len - 1,
                                                     road_net)
            total_acc += acc
            total_prec += prec
            total_recall += recall
            total_f1 += f1
            total_mae += mae
            total_rmse += rmse
        return np.mean(total_acc), np.mean(total_prec), np.mean(total_recall), \
               np.mean(total_f1), np.mean(total_mae), np.mean(total_rmse)

    @torch.no_grad()
    def evaluate_augment_mtrajrec(self, test_set, road_net, road_feats):
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

            encoder_embed, hidden_embed = self.trainer.model.encoding(src_grid_seq, src_seq_len, feat_seq)
            decode_inputs, src_seq_len = self.append_virtual_embedding(
                encoder_embed,
                src_seq_len
            )
            pred_node, pred_rate = self.trainer.model.decoding(
                decode_inputs, hidden_embed, road_feats,
                src_seq_len, tgt_node_seq, tgt_rate_seq, tgt_seq_len, tgt_inf_scores, 0
            )
            pred_rate = pred_rate.squeeze(-1)
            tgt_node_seq, tgt_rate_seq = tgt_node_seq.squeeze(-1), tgt_rate_seq.squeeze(-1)
            pred_node, pred_rate = pred_node[:, 1:, :], pred_rate[:, 1:]
            tgt_node_seq, tgt_rate_seq = tgt_node_seq[:, 1:], tgt_rate_seq[:, 1:]

            acc, prec, recall, f1 = evaluate_point_prediction(pred_node,
                                                              tgt_node_seq,
                                                              tgt_seq_len - 1)
            mae, rmse = evaluate_distance_regression(pred_node,
                                                     pred_rate,
                                                     tgt_gps_seq[:, 1:, :],
                                                     tgt_seq_len - 1,
                                                     road_net)
            total_acc += acc
            total_prec += prec
            total_recall += recall
            total_f1 += f1
            total_mae += mae
            total_rmse += rmse
        return np.mean(total_acc), np.mean(total_prec), np.mean(total_recall), \
               np.mean(total_f1), np.mean(total_mae), np.mean(total_rmse)

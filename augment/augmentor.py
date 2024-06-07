# import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from typing import Union, Optional, List
import matplotlib.pyplot as plt

from utils.train.trainer import Trainer


class AugmentConfig:
    def __init__(
            self,
            num_virtual_tokens: int = 5,
            augment_type: Optional[Union[str, List[str]]] = None,
            virtual_dim: int = 512,
            num_epochs: int = 10,
            learning_rate: float = 1e-3,
            projection: bool = True,
            project_hidden_dim: int = 256
    ):
        self.num_virtual_tokens = num_virtual_tokens
        self.augment_type = augment_type
        self.virtual_dim = virtual_dim
        self.num_epochs = num_epochs

        self.lr = learning_rate
        self.projection = projection
        self.projection_hidden_dim = project_hidden_dim


class Augmentor(nn.Module):
    def __init__(
            self,
            config: AugmentConfig,
            trainer: Trainer,
            augment_dataset: Optional[Dataset],
            device: Union[torch.device, str] = "cpu",
            **kwargs
    ):
        super(Augmentor, self).__init__()
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

    def train_virtual_embedding(self, dataloader, road_embeds, road_feats, tf_ratio, weights):
        optimizer = self.get_optimizer(self.parameters())
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
        for epoch in range(self.config.num_epochs):
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
                batch_size = encoder_embed.size(0)
                indices = torch.arange(self.config.num_virtual_tokens, dtype=torch.long, device=self.device)
                indices = torch.stack([indices] * batch_size, dim=0)
                if self.config.projection:
                    virtual_tokens = self.virtual_embeds(indices)
                    virtual_inputs = self.transform(virtual_tokens)
                else:
                    virtual_inputs = self.virtual_embeds(indices)
                decode_input_shape = (encoder_embed.size(0),
                                      encoder_embed.size(1) + self.config.num_virtual_tokens,
                                      encoder_embed.size(2))
                decode_inputs = torch.zeros(decode_input_shape, dtype=torch.float, device=self.device)
                for bs in range(batch_size):
                    length = src_seq_len[bs]
                    decode_inputs[bs, :length] = encoder_embed[bs, :length]
                    decode_inputs[bs, length:length + self.config.num_virtual_tokens] = virtual_inputs[bs]
                src_seq_len += self.config.num_virtual_tokens

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
            if (epoch + 1) % 5 == 0:
                self.visualize_virtual_embedding(axs, epoch)
        plt.savefig("./ckpt/virtual_embedding.png", dpi=300)

    def augment_points(self, road_grid, road_len, road_nodes, road_edge, road_batch, road_feat, weight):
        self.trainer.freeze_model()
        valid_dataloader = self.trainer.get_eval_dataloader()
        augment_dataloader = self.trainer.get_test_dataloader(self.augment_dataset)
        road_grid, road_nodes = road_grid.to(self.device), road_nodes.to(self.device)
        road_edge, road_batch = road_edge.long().to(self.device), road_batch.to(self.device)
        road_feat = road_feat.to(self.device)
        road_embed = self.trainer.model.extract_feat(road_grid, road_len, road_nodes, road_edge, road_batch)

        self.train_virtual_embedding(valid_dataloader, road_embed, road_feat, 0.0, weight)
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
    def evaluate_augment(self, test_set, augment_embeds, road_net, road_grid, road_len, road_nodes, road_edge, road_batch, road_feat, weight):
        test_loader = self.trainer.get_test_dataloader(test_set)
        road_embed = self.trainer.model.extract_feat(road_grid, road_len, road_nodes, road_edge, road_batch)
        for batch in tqdm.tqdm(test_loader, total=len(test_loader), desc="test augment"):
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
            encoder_embed, hidden_embed, _ = self.trainer.model.encoding(road_embed, src_grid_seq, src_seq_len,
                                                                         feat_seq, node_index, traj_edge,
                                                                         batched_graph.batch, batched_graph.weight)

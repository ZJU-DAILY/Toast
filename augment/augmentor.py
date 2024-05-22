import faiss
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim import AdamW
from typing import Union, Optional, List

from utils.train.trainer import Trainer


class AugmentConfig:
    def __init__(
            self,
            num_augments: int = 1,
            augment_type: Optional[str, List[str]] = None,
            virtual_dim: Optional[int] = 512,
            learning_rate: Optional[float] = 1e-3
    ):
        self.num_augments = num_augments
        self.augment_type = augment_type

        assert (virtual_dim is None) and ((augment_type is None) or ("AJ" in augment_type) or ("PU" in augment_type)), \
            "`virtual_dim` is necessary when `AJ`/`PU` contained in `augment_type`"
        self.virtual_dim = virtual_dim

        self.lr = learning_rate


class Augmentor:
    def __init__(
            self,
            config: AugmentConfig,
            trainer: Trainer,
            augment_dataset: Dataset,
            device: Union[torch.device, str] = "cpu",
            **kwargs
    ):
        self.config = config
        self.augment_dataset = augment_dataset
        self.trainer = trainer
        self.device = torch.device(device) if isinstance(device, str) else device
        self.point_index = faiss.IndexHNSWFlat(config.virtual_dim, 32)

        for key, val in kwargs.items():
            setattr(self, key, val)

    def get_optimizer(self, parameters):
        return AdamW(parameters, lr=self.config.lr)

    def get_optimal_embed(self, encoded_embeds, labels, *args):
        batch_size = encoded_embeds.size(0)
        indices = torch.arange(self.config.num_augments, dtype=torch.long, device=self.device)
        indices = torch.stack([indices] * batch_size, dim=0)
        virtual_embeds = nn.Embedding(self.config.num_augments, self.config.virtual_dim).to(self.device)
        optimizer = self.get_optimizer(virtual_embeds)
        input_embeds = virtual_embeds(indices)
        decoder_input = torch.cat((encoded_embeds, input_embeds), dim=1)

        for epoch in range(self.trainer.num_epochs):
            optimizer.zero_grad()
            decoder_output = self.trainer.model.decoding(decoder_input, *args)
            loss = self.trainer.compute_loss(*decoder_output, labels)
            loss.backward()
            optimizer.step()
        return virtual_embeds.detach()

    def augment_points(self, *args):
        self.trainer.freeze_model()
        augment_dataloader = self.trainer.get_test_dataloader(self.augment_dataset)

        with torch.no_grad():
            for batch in augment_dataloader:
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

                road_embed = self.trainer.model.extract_feat(*args)
                point_embed, _, _ = self.trainer.model.encoding(road_embed, src_grid_seq, src_seq_len, feat_seq,
                                                                node_index, traj_edge, batched_graph.batch,
                                                                batched_graph.weight)


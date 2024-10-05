import tqdm
from typing import Union, Callable, Optional, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim import Optimizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from models.base_model import BaseModel
from utils.train.trainer import Trainer


class SimilarityTrainer(Trainer):
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
            device: Union[str, torch.device] = "cpu",
            **kwargs
    ):
        super(SimilarityTrainer, self).__init__(model,
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

    def compute_loss(self, anchors, positives, negatives, pos_dist, neg_dist):
        pos_embed_dist = torch.exp(-torch.norm(anchors - positives, p=2, dim=-1))
        neg_embed_dist = torch.exp(-torch.norm(anchors - negatives, p=2, dim=-1))
        pos_loss = pos_dist * ((pos_dist - pos_embed_dist) ** 2)
        neg_loss = neg_dist * ((neg_dist - neg_embed_dist) ** 2)
        loss = (pos_loss + neg_loss).mean()
        return loss

    def forward_once(self, model_kwargs, batch, augment_fn=None):
        node_feat, edge_index, edge_attr = (
            model_kwargs["node_feat"], model_kwargs["edge_index"], model_kwargs["edge_attr"]
        )
        if self.train_dataset.model_name == "ST2Vec":
            a_nodes, a_time, a_len, p_nodes, p_time, p_len, n_nodes, n_time, n_len, pos_dist, neg_dist = batch
            a_nodes, a_time = a_nodes.to(self.device), a_time.to(self.device)
            p_nodes, p_time = p_nodes.to(self.device), p_time.to(self.device)
            n_nodes, n_time = n_nodes.to(self.device), n_time.to(self.device)
            pos_dist, neg_dist = pos_dist.to(self.device), neg_dist.to(self.device)
            spatial_embeds,time_embeds = self.model.extract_feat(
                a_nodes, a_time, a_len, node_feat, edge_index, edge_attr
            )
            if augment_fn:
                spatial_embeds, augment_len = augment_fn(
                    spatial_embeds, a_len
                )
                time_embeds, _ = augment_fn(
                    time_embeds, a_len
                )
            else:
                augment_len = a_len
            a_embeds = self.model.encoding(spatial_embeds, time_embeds, augment_len)
            p_embeds = self.model(p_nodes, p_time, p_len, node_feat, edge_index, edge_attr)
            n_embeds = self.model(n_nodes, n_time, n_len, node_feat, edge_index, edge_attr)
        else:
            a_nodes, _, a_len, p_nodes, _, p_len, n_nodes, _, n_len, pos_dist, neg_dist = batch
            a_nodes, p_nodes, n_nodes = a_nodes.to(self.device), p_nodes.to(self.device), n_nodes.to(self.device)
            pos_dist, neg_dist = pos_dist.to(self.device), neg_dist.to(self.device)
            spatial_embeds = self.model.extract_feat(
                a_nodes, a_len, node_feat, edge_index, edge_attr
            )
            if augment_fn:
                spatial_embeds, augment_len = augment_fn(
                    spatial_embeds, a_len
                )
            else:
                augment_len = a_len
            a_embeds = self.model.encoding(spatial_embeds, augment_len)
            p_embeds = self.model(p_nodes, p_len, node_feat, edge_index, edge_attr)
            n_embeds = self.model(n_nodes, n_len, node_feat, edge_index, edge_attr)
        loss = self.compute_loss(a_embeds, p_embeds, n_embeds, pos_dist, neg_dist)
        return loss

    def train(
            self,
            node_feat: Optional[torch.Tensor] = None,
            edge_index: Optional[torch.Tensor] = None,
            edge_attr: Optional[torch.Tensor] = None
    ):
        train_loader = self.get_train_dataloader()
        node_feat, edge_index, edge_attr = (
            node_feat.to(self.device),
            edge_index.to(self.device),
            edge_attr.to(self.device)
        )
        best_hr10 = float("-inf")
        for epoch in range(self.num_epochs):
            print("[start {}-th training]".format(epoch + 1))
            self.model.train()
            for batch in tqdm.tqdm(train_loader, total=len(train_loader), desc="train"):
                self.optimizer.zero_grad()
                if self.train_dataset.model_name == "ST2Vec":
                    a_nodes, a_time, a_len, p_nodes, p_time, p_len, n_nodes, n_time, n_len, pos_dist, neg_dist = batch
                    a_nodes, a_time = a_nodes.to(self.device), a_time.to(self.device)
                    p_nodes, p_time = p_nodes.to(self.device), p_time.to(self.device)
                    n_nodes, n_time = n_nodes.to(self.device), n_time.to(self.device)
                    pos_dist, neg_dist = pos_dist.to(self.device), neg_dist.to(self.device)
                    a_embeds = self.model(a_nodes, a_time, a_len, node_feat, edge_index, edge_attr)
                    p_embeds = self.model(p_nodes, p_time, p_len, node_feat, edge_index, edge_attr)
                    n_embeds = self.model(n_nodes, n_time, n_len, node_feat, edge_index, edge_attr)
                else:
                    a_nodes, _, a_len, p_nodes, _, p_len, n_nodes, _, n_len, pos_dist, neg_dist = batch
                    a_nodes, p_nodes, n_nodes = a_nodes.to(self.device), p_nodes.to(self.device), n_nodes.to(self.device)
                    pos_dist, neg_dist = pos_dist.to(self.device), neg_dist.to(self.device)
                    a_embeds = self.model(a_nodes, a_len, node_feat, edge_index, edge_attr)
                    p_embeds = self.model(p_nodes, p_len, node_feat, edge_index, edge_attr)
                    n_embeds = self.model(n_nodes, n_len, node_feat, edge_index, edge_attr)

                loss = self.compute_loss(a_embeds, p_embeds, n_embeds, pos_dist, neg_dist)
                loss.backward()
                self.optimizer.step()

            results = self.evaluate(node_feat=node_feat, edge_index=edge_index, edge_attr=edge_attr)
            if self.saved_dir is not None:
                if results[0] > best_hr10:
                    print("saving best model.")
                    best_hr10 = results[0]
                    self.save_model(save_optim=True)
            if (epoch + 1) % 5 == 0:
                print("valid hr10: {:.4f}".format(results[0]))

    def compute_metrics(self, embeds, true_dist):
        num_trajs, _ = embeds.shape
        dist_matrix = []
        for i in range(num_trajs):
            emb = np.repeat([embeds[i]], num_trajs, axis=0)
            matrix = np.linalg.norm(emb - embeds, ord=2, axis=1)
            dist_matrix.append(matrix)
            # dist_matrix.append(
            #     np.linalg.norm(embeds[i][np.newaxis, :] - embeds, ord=2, axis=-1)
            # )
        num_trajs, num_cands = true_dist.shape
        hr10, hr50, hr10_50, f_num = 0., 0., 0., 0
        for i in range(num_trajs):
            input_r = true_dist[i]
            cand_index = (input_r != -1)
            input_r = input_r[cand_index][:5000]
            input_r50 = np.argsort(input_r)[1:51]
            input_r10 = input_r50[:10]

            embed_r = dist_matrix[i]
            embed_r = embed_r[cand_index][:5000]
            embed_r50 = np.argsort(embed_r)[1:51]
            embed_r10 = embed_r50[:10]

            if cand_index.sum() >= 51:
                f_num += 1
                hr10 += len(np.intersect1d(input_r10, embed_r10))
                hr50 += len(np.intersect1d(input_r50, embed_r50))
                hr10_50 += len(np.intersect1d(input_r50, embed_r10))
        hr10 = hr10 / (10 * f_num)
        hr50 = hr50 / (50 * f_num)
        hr10_50 = hr10_50 / (10 * f_num)
        return hr10, hr50, hr10_50

    @torch.no_grad()
    def evaluate(
            self,
            test_dataset: Dataset = None,
            node_feat=None,
            edge_index=None,
            edge_attr=None
    ):
        if test_dataset is None:
            eval_loader = self.get_eval_dataloader()
            mode = "eval"
        else:
            eval_loader = self.get_test_dataloader(test_dataset)
            self.load_model()
            mode = "test"
        self.model.eval()
        edge_index = None if edge_index is None else edge_index.to(self.device)
        node_feat = None if node_feat is None else node_feat.to(self.device)
        edge_attr = None if edge_attr is None else edge_attr.to(self.device)
        embeddings, true_dist = [], []
        for batch in tqdm.tqdm(eval_loader, total=len(eval_loader), desc=mode):
            if self.train_dataset.model_name == "ST2Vec":
                nodes, time, seq_len, dist = batch
                nodes, time = nodes.to(self.device), time.to(self.device)
                batch_embeds = self.model(nodes, time, seq_len, node_feat, edge_index, edge_attr)
            else:
                nodes, _, seq_len, dist = batch
                nodes = nodes.to(self.device)
                batch_embeds = self.model(nodes, seq_len, node_feat, edge_index, edge_attr)
            embeddings.append(batch_embeds)
            true_dist.append(dist)
        embeddings = torch.cat(embeddings, dim=0)
        true_dist = torch.cat(true_dist, dim=0)
        hr10, hr50, hr10_50 = self.compute_metrics(embeddings.cpu().numpy(), true_dist[:5000].numpy())
        return hr10, hr50, hr10_50

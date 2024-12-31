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


class PredictTrainer(Trainer):
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
            mixup: bool = False,
            **kwargs
    ):
        super(PredictTrainer, self).__init__(model,
                                             train_dataset,
                                             eval_dataset,
                                             optimizer,
                                             num_epochs,
                                             data_collator,
                                             mixup,
                                             saved_dir,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             num_workers=num_workers,
                                             pin_memory=pin_memory,
                                             device=device)
        for key, val in kwargs.items():
            setattr(self, key, val)

    def compute_loss(self, pred_flow, true_flow):
        regression_criterion = nn.MSELoss(reduction="mean")

        loss = regression_criterion(pred_flow, true_flow)
        return loss
    
    def compute_mixup_loss(
            self,
            pred_flow,
            true_flow_a,
            true_flow_b,
            lam,
    ):
        regression_criterion = nn.MSELoss(reduction="mean")
        loss = lam * regression_criterion(pred_flow, true_flow_a) + (1 - lam) * regression_criterion(pred_flow, true_flow_b)
        return loss

    def forward_once(self, model_kwargs, batch, augment_fn=None):
        if self.train_dataset.model_name == "STMetaNet":
            flow, features, labels = batch
            flow, features, labels = flow.to(self.device), features.to(self.device), labels.to(self.device)
            if self.mixup:
                flow, labels_a, labels_b, lam = self.mixup_data(flow, labels)
                labels_a, labels_b = labels_a.permute(2, 0, 1, 3), labels_b.permute(2, 0, 1, 3)
                labels_a, labels_b = labels_a[..., :self.model.output_dim], labels_b[..., :self.model.output_dim]
            predicts, labels = self.model(flow, features, labels, model_kwargs["edge_index"], train_mode=True)
            if self.mixup:
                loss = self.compute_mixup_loss(predicts, labels_a, labels_b, lam)
            else:
                loss = self.compute_loss(predicts, labels)
        else:
            close_data, period_data, trend_data, features, labels = batch
            close_data, period_data, trend_data, features, labels = (
                close_data.to(self.device),
                period_data.to(self.device),
                trend_data.to(self.device),
                features.to(self.device),
                labels.to(self.device)
            )
            if self.mixup:
                data, labels_a, labels_b, lam = self.mixup_data(
                    (close_data, period_data, trend_data),
                    (labels, )
                )
                close_data, period_data, trend_data = data
                labels_a, labels_b = labels_a[0], labels_b[0]
            predicts = self.model(close_data, period_data, trend_data, features)
            if self.mixup:
                loss = self.compute_mixup_loss(predicts, labels_a, labels_b, lam)
            else:
                loss = self.compute_loss(predicts, labels)
        return loss
    
    def mixup_train(
            self, 
            edge_index=None, 
            scaler=None,
    ):
        train_loader = self.get_train_dataloader()
        model_kwargs = dict(
            edge_index=None if edge_index is None else edge_index.to(self.device)
        )
        min_rmse = float("inf")
        for epoch in range(self.num_epochs):
            print("[start {}-th training]".format(epoch + 1))
            self.model.train()
            for batch in tqdm.tqdm(train_loader, total=len(train_loader), desc="train"):
                self.optimizer.zero_grad()
                loss = self.forward_once(model_kwargs, batch)
                loss.backward()
                self.optimizer.step()
            results = self.evaluate(edge_index=edge_index, scaler=scaler)
            if self.saved_dir is not None:
                if results[0] < min_rmse:
                    print("saving best model.")
                    min_rmse = results[0]
                    self.save_model(save_optim=True)
            if (epoch + 1) % 5 == 0:
                print("valid rmse: {:.4f}".format(results[-1]))

    def train(self, edge_index, scaler):
        train_loader = self.get_train_dataloader()
        edge_index = None if edge_index is None else edge_index.to(self.device)
        min_rmse = float("inf")
        for epoch in range(self.num_epochs):
            print("[start {}-th training]".format(epoch + 1))
            self.model.train()
            for batch in tqdm.tqdm(train_loader, total=len(train_loader), desc="train"):
                self.optimizer.zero_grad()
                if self.train_dataset.model_name == "STMetaNet":
                    flow, features, labels = batch
                    flow, features, labels = flow.to(self.device), features.to(self.device), labels.to(self.device)
                    predicts, labels = self.model(flow, features, labels, edge_index, train_mode=True)
                    loss = self.compute_loss(predicts, labels)
                else:
                    close_data, period_data, trend_data, features, labels = batch
                    close_data, period_data, trend_data, features, labels = (
                        close_data.to(self.device),
                        period_data.to(self.device),
                        trend_data.to(self.device),
                        features.to(self.device),
                        labels.to(self.device)
                    )
                    predicts = self.model(close_data, period_data, trend_data, features)
                    loss = self.compute_loss(predicts, labels)
                loss.backward()
                self.optimizer.step()

            results = self.evaluate(edge_index=edge_index, scaler=scaler)
            if self.saved_dir is not None:
                if results[0] < min_rmse:
                    print("saving best model.")
                    min_rmse = results[0]
                    self.save_model(save_optim=True)
            if (epoch + 1) % 5 == 0:
                print("valid rmse: {:.4f}".format(results[-1]))

    @torch.no_grad()
    def evaluate(
            self,
            test_dataset: Dataset = None,
            edge_index=None,
            scaler=None
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
        cnt, rmse, mae = 0., 0., 0.
        for batch in tqdm.tqdm(eval_loader, total=len(eval_loader), desc=mode):
            if self.train_dataset.model_name == "STMetaNet":
                flow, features, labels = batch
                flow, features, labels = flow.to(self.device), features.to(self.device), labels.to(self.device)
                predicts, labels = self.model(flow, features, labels, edge_index, train_mode=False)
            else:
                close_data, period_data, trend_data, features, labels = batch
                close_data, period_data, trend_data, features, labels = (
                    close_data.to(self.device),
                    period_data.to(self.device),
                    trend_data.to(self.device),
                    features.to(self.device),
                    labels.to(self.device)
                )
                predicts = self.model(close_data, period_data, trend_data, features)

            preds, labels = scaler.inverse_transform(predicts), scaler.inverse_transform(labels)
            cnt += preds.numel()
            rmse += torch.sum((preds - labels) ** 2)
            mae += torch.sum(torch.abs(preds - labels))
        return torch.sqrt(rmse / (cnt + 1e-8)).item(), (mae / (cnt + 1e-8)).item()

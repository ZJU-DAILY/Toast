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


class ModeIdentifyTrainer(Trainer):
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
        super(ModeIdentifyTrainer, self).__init__(model,
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

    def compute_loss(self, recover_data, pred_logits, true_data, true_labels):
        classification_criterion = nn.CrossEntropyLoss(reduction="mean")
        if recover_data is None:
            loss = classification_criterion(pred_logits, true_labels)
        else:
            regression_criterion = nn.MSELoss(reduction="mean")
            recover_loss = regression_criterion(recover_data, true_data)
            predict_loss = classification_criterion(pred_logits, true_labels)
            loss = recover_loss * 100 + predict_loss
        return loss
    
    def compute_mixup_loss(
            self, 
            recover_data, 
            pred_logits, 
            true_data,
            batch_labels_a, 
            batch_labels_b, 
            lam
    ):
        cls_criterion = nn.CrossEntropyLoss(reduction="mean")
        if recover_data is None:
            loss = (
                lam * cls_criterion(pred_logits, batch_labels_a) + 
                (1 - lam) * cls_criterion(pred_logits, batch_labels_b)
            )
        else:
            reg_criterion = nn.MSELoss(reduction="mean")
            recover_loss = reg_criterion(recover_data, true_data)
            predict_loss = (
                lam * cls_criterion(pred_logits, batch_labels_a) + 
                (1 - lam) * cls_criterion(pred_logits, batch_labels_b)
            )
            loss = recover_loss * 100 + predict_loss
        return loss

    def forward_once(self, model_kwargs, batch, augment_fn=None):
        batch_data, seq_len, batch_labels = batch
        batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
        if augment_fn is not None:
            batch_data, _ = augment_fn(batch_data.squeeze(dim=1), seq_len)
            batch_data = batch_data.unsqueeze(dim=1)
        if self.mixup:
            batch_data, batch_labels_a, batch_labels_b, lam = self.mixup_data(batch_data, batch_labels)
        if self.train_dataset.model_name == "SECA":
            max_length = model_kwargs["max_length"]
            hidden_unlabel, pool_indices, hidden_label = self.model.encoding(
                batch_data, batch_data
            )
            recover_data = self.model.decoding(hidden_unlabel, pool_indices[::-1])
            pred_logits = self.model.cls_layer(hidden_label)
            recover_data = recover_data.permute(0, 2, 3, 1)[:, :max_length]
        else:
            hiddens, _ = self.model.encoding(batch_data)
            pred_logits = self.model.decoding(hiddens)
            recover_data = None
        if self.mixup:
            loss = self.compute_mixup_loss(recover_data, pred_logits, batch_data, batch_labels_a, batch_labels_b, lam)
        else:
            loss = self.compute_loss(recover_data, pred_logits, batch_data, batch_labels)
        return loss
    
    def mixup_train(self, max_length=None):
        best_F1 = float("-inf")
        train_loader = self.get_train_dataloader()
        model_kwargs = dict(max_length=max_length)
        for epoch in range(self.num_epochs):
            print("[start {}-th training]".format(epoch + 1))
            self.model.train()
            for batch in tqdm.tqdm(train_loader, total=len(train_loader), desc="train"):
                self.optimizer.zero_grad()
                loss = self.forward_once(model_kwargs, batch, augment_fn=None)
                loss.backward()
                self.optimizer.step()

            results = self.evaluate()
            if self.saved_dir is not None:
                if results[-1] > best_F1:
                    print("saving best model.")
                    best_F1 = results[-1]
                    self.save_model(save_optim=True)
            if (epoch + 1) % 5 == 0:
                print("valid f1: {:.4f}".format(results[-1]))

    def train(self):
        best_F1 = float("-inf")
        train_loader = self.get_train_dataloader()
        for epoch in range(self.num_epochs):
            print("[start {}-th training]".format(epoch + 1))
            self.model.train()
            for batch in tqdm.tqdm(train_loader, total=len(train_loader), desc="train"):
                batch_data, _, batch_labels = batch
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                self.optimizer.zero_grad()
                if self.train_dataset.model_name == "SECA":
                    recover_data, pred_logits = self.model(batch_data, batch_data)
                    recover_data = recover_data.permute(0, 2, 3, 1)
                else:
                    recover_data, pred_logits = None, self.model(batch_data)
                loss = self.compute_loss(recover_data, pred_logits, batch_data, batch_labels)
                loss.backward()
                self.optimizer.step()

            results = self.evaluate()
            if self.saved_dir is not None:
                if results[-1] > best_F1:
                    print("saving best model.")
                    best_F1 = results[-1]
                    self.save_model(save_optim=True)
            if (epoch + 1) % 5 == 0:
                print("valid f1: {:.4f}".format(results[-1]))

    @torch.no_grad()
    def compute_metrics(self, pred_labels, true_labels):
        acc = accuracy_score(true_labels.numpy(), pred_labels.numpy())
        prec = precision_score(true_labels.numpy(), pred_labels.numpy(), average="weighted")
        recall = recall_score(true_labels.numpy(), pred_labels.numpy(), average="weighted")
        macro_f1 = f1_score(true_labels.numpy(), pred_labels.numpy(), average="macro")
        weighted_f1 = f1_score(true_labels.numpy(), pred_labels.numpy(), average="weighted")
        return acc, prec, recall, macro_f1, weighted_f1

    @torch.no_grad()
    def evaluate(self, test_dataset: Dataset = None, *args):
        if test_dataset is None:
            eval_loader = self.get_eval_dataloader()
            mode = "eval"
        else:
            eval_loader = self.get_test_dataloader(test_dataset)
            self.load_model()
            mode = "test"
        self.model.eval()
        predictions, true_labels = [], []
        for batch in tqdm.tqdm(eval_loader, total=len(eval_loader), desc=mode):
            batch_data, _, batch_labels = batch
            batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
            if self.train_dataset.model_name == "SECA":
                recover_data, pred_logits = self.model(batch_data, batch_data)
            else:
                pred_logits = self.model(batch_data)

            pred_labels = torch.argmax(pred_logits, dim=-1)
            true_labels.append(batch_labels.cpu())
            predictions.append(pred_labels.cpu())
        true_labels, pred_labels = torch.cat(true_labels, dim=0), torch.cat(predictions, dim=0)
        acc, prec, recall, macro_f1, weighted_f1 = self.compute_metrics(pred_labels, true_labels)
        return acc, prec, recall, macro_f1, weighted_f1

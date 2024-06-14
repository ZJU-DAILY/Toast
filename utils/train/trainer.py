from typing import Callable
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer

from models.base_model import BaseModel


class Trainer:
    def __init__(
            self,
            model: BaseModel,
            train_dataset: Dataset,
            eval_dataset: Dataset,
            optimizer: Optimizer,
            num_epochs: int,
            data_collator: Callable,
            saved_path: str,
            **kwargs
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.saved_path = saved_path
        self.device = kwargs["device"]
        self.dataloader_params = {
            "batch_size": kwargs["batch_size"],
            "shuffle": kwargs["shuffle"],
            "collate_fn": data_collator,
            "num_workers": kwargs["num_workers"],
            "pin_memory": kwargs["pin_memory"]
        }

    def compute_loss(self, *args):
        raise NotImplementedError

    def train(self, *args):
        raise NotImplementedError

    def evaluate(
            self,
            test_dataset: Dataset = None,
            *args
    ):
        raise NotImplementedError

    def get_train_dataloader(self):
        return DataLoader(self.train_dataset, **self.dataloader_params)

    def get_eval_dataloader(self):
        return DataLoader(self.eval_dataset, **self.dataloader_params)

    def get_test_dataloader(self, dataset):
        return DataLoader(dataset,
                          batch_size=self.dataloader_params["batch_size"],
                          shuffle=False,
                          collate_fn=self.dataloader_params["collate_fn"],
                          num_workers=self.dataloader_params["num_workers"],
                          pin_memory=self.dataloader_params["pin_memory"])

    def freeze_model(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def unfreeze_model(self):
        for p in self.model.parameters():
            p.requires_grad = True

    def save_model(self):
        torch.save(self.model, self.saved_path)

    def load_model(self):
        device = torch.device(self.device) if isinstance(self.device, str) else self.device
        self.model = torch.load(self.saved_path, map_location=device)

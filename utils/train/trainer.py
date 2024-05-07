from typing import Callable
import os
import torch
from torch.utils.data import Dataset
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

    def save_state(self):
        torch.save(self.model.state_dict(), self.saved_path)

    def load_state(self):
        device = torch.device(self.device) if isinstance(self.device, str) else self.device
        model_state = torch.load(self.saved_path, map_location=device)
        self.model.load_state_dict(model_state)

from typing import Callable
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
            **kwargs
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.optimizer = optimizer
        self.num_epochs = num_epochs
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

    def evaluate(self, *args):
        raise NotImplementedError

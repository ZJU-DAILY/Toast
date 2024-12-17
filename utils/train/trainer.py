from typing import Callable, Union
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
            mixup: bool,
            saved_dir: str,
            **kwargs
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.saved_dir = saved_dir
        self.mixup = mixup
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

    def compute_metrics(self, *args):
        raise NotImplementedError

    def forward_once(self, *args):
        raise NotImplementedError

    def train(self, *args):
        raise NotImplementedError

    def evaluate(
            self,
            test_dataset: Dataset = None,
            *args
    ):
        raise NotImplementedError
    
    def mixup_data(
            self, 
            x: Union[torch.Tensor, tuple], 
            y: Union[torch.Tensor, tuple], 
            alpha=1.0
    ):
        """
        Applies the Mixup data augmentation technique to the input data.

        Mixup is a data augmentation technique that creates new training examples 
        by combining pairs of examples from the original dataset. This is done by 
        taking a weighted average of the input features and the corresponding labels.

        Args:
            x (Union[torch.Tensor, tuple]): Input data, can be a tensor or a tuple of tensors.
            y (Union[torch.Tensor, tuple]): Corresponding labels, can be a tensor or a tuple of tensors.
            alpha (float, optional): Parameter for the Beta distribution to sample the mixup ratio. Default is 1.0.

        Returns:
            mixed_x (Union[torch.Tensor, tuple]): Mixed input data.
            y_a (Union[torch.Tensor, tuple]): Original labels.
            y_b (Union[torch.Tensor, tuple]): Labels of the mixed examples.
            lam (float): Mixup ratio.
        """
        if alpha > 0:
            lam = torch.distributions.Beta(alpha, alpha).sample().item()
        else:
            lam = 1.0

        if isinstance(x, tuple):
            batch_size = x[0].shape[0]
            mixed_x, y_a, y_b = (), (), ()
        else:
            batch_size = x.shape[0]
            mixed_x, y_a, y_b = None, None, None
        index = torch.randperm(batch_size).to(x.device)
        if isinstance(x, tuple):
            for i in range(len(x)):
                mixed_x += (lam * x[i] + (1 - lam) * x[i][index],)
                y_a += (y[i],)
                y_b += (y[i][index],)
        else:
            mixed_x = lam * x + (1 - lam) * x[index]
            y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

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

    def save_model(self, save_optim=False):
        model_path = self.saved_dir + "/val-best-model.pt"
        torch.save(self.model, model_path)
        if save_optim:
            optim_path = self.saved_dir + "/optimizer.bin"
            torch.save(self.optimizer.state_dict(), optim_path)

    def load_model(self):
        device = torch.device(self.device) if isinstance(self.device, str) else self.device
        model_path = self.saved_dir + "/val-best-model.pt"
        self.model = torch.load(model_path, map_location=device)

import torch
from torch.utils.data import Dataset
from torch.optim import Optimizer
from typing import Union, Optional

from models.base_model import BaseModel


class Augmentor:
    def __init__(
            self,
            model: BaseModel,
            augment_dataset: Dataset,
            eval_dataset: Dataset,
            augment_type: Optional[str] = None,
            optimizer: Optional[Optimizer] = None,
            device: Union[torch.device, str] = "cpu",
            **kwargs
    ):
        self.model = model
        self.augment_dataset = augment_dataset
        self.eval_dataset = eval_dataset
        self.augment_type = augment_type
        self.optimizer = optimizer
        self.device = device

        for key, val in kwargs.items():
            setattr(self, key, val)

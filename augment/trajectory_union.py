import torch
from torch.utils.data import Dataset
from typing import Optional, Union

from augment.augment_config import TrajUnionConfig
from utils.train.trainer import Trainer


class TrajectoryUnion:
    def __init__(
            self,
            config: TrajUnionConfig,
            trainer: Trainer,
            augment_dataset: Optional[Dataset],
            device: Union[torch.device, str] = "cpu",
            **kwargs
    ):
        self.config = config
        self.trainer = trainer
        self.augment_dataset = augment_dataset
        self.device = torch.device(device) if isinstance(device, str) else device

        for key, val in kwargs.items():
            setattr(self, key, val)

    def augment(self):
        pass

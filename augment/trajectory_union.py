import torch
from torch.utils.data import Dataset
from typing import Optional, Union
from trak.projectors import BasicProjector, CudaProjector

from augment.augment_config import TrajUnionConfig
from utils.train.trainer import Trainer


class TrajUnion:
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

    def get_projector(self):
        def get_params_num():
            num_params = sum(
                [p.numel() for p in self.trainer.model.parameters() if p.requires_grad]
            )
            return num_params
        projector = BasicProjector(
            grad_dim=get_params_num(),
            proj_dim=self.config.proj_dim,
            proj_type=self.config.proj_type,
            seed=0,
            device=self.device,
            block_size=128
        )
        return projector

    def get_adam_gradients(self, inputs, labels, avg, avg_sq):
        loss = self.trainer.compute_loss(inputs, labels)
        loss.backward()
        vectorized_grads = torch.cat(
            [p.grad.view(-1) for p in self.trainer.model.parameters() if p.grad is not None]
        )
        updated_avg = self.config.beta1 * avg + (1 - self.config.beta1) * vectorized_grads
        updated_avg_sq = self.config.beta2 * avg_sq + (1 - self.config.beta2) * vectorized_grads ** 2
        vectorized_grads = updated_avg / torch.sqrt(updated_avg_sq + self.config.eps)
        return vectorized_grads

    def load_optimizer_state(self):
        optim_path = self.trainer.saved_dir + "/optimizer.bin"
        optim_state = torch.load(optim_path)["state"]
        names = [n for n, p in self.trainer.model.named_parameters() if p.requires_grad]
        avg = torch.cat([optim_state[n]["exp_avg"].view(-1) for n in names]).to(self.device)
        avg_sq = torch.cat([optim_state[n]["exp_avg_sq"].view(-1) for n in names]).to(self.device)
        return avg, avg_sq

    def collect_grads(self):
        augment_dataloader = self.trainer.get_test_dataloader(self.augment_dataset)
        projector = self.get_projector()
        prev_avg, prev_avg_sq = self.load_optimizer_state()

        proj_grads = []
        for batch in augment_dataloader:
            inputs, labels = batch.to(self.device)
            self.trainer.model.zero_grad()
            outputs = self.trainer.model(inputs)
            vectorized_grads = self.get_adam_gradients(outputs, labels, prev_avg, prev_avg_sq)
            proj_grad = projector.project(
                vectorized_grads,
                model_id=0
            )
            proj_grads.append(proj_grad.cpu())
        return proj_grads

    def augment(self):
        pass

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Optional, Union
from trak.projectors import BasicProjector, CudaProjector

from augment.augment_config import TrajUnionConfig
from utils.train.trainer import Trainer


class TrajSubset(Subset):
    model_name = None

    def __init__(self, dataset, indices, model_name):
        super(TrajSubset, self).__init__(dataset, indices)
        TrajSubset.model_name = model_name


def calculate_influence_score(augment_grads: torch.Tensor, valid_grads: torch.Tensor):
    """

    Args:
        augment_grads: [N_{aug}, num_params]
        valid_grads: [N_{valid}, num_params]

    Returns:
        influence_score: shape of [N_{aug}, N_{valid}]

    """
    influence_score = torch.matmul(
        augment_grads,
        valid_grads.T
    )
    return influence_score


def prepare_data(args, device):
    if isinstance(args, dict):
        return {k: v.to(device) for k, v in args.items()}
    elif isinstance(args, tuple):
        args = [v.to(device) for v in args]
        return tuple(args)
    elif isinstance(args, torch.Tensor):
        return args.to(device)
    else:
        raise ValueError(f"{type(args)} cannot be moved to GPU.")


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

    def get_projector(self, param_indices):
        def get_params_num():
            num_params = sum(
                [p.numel() for idx, (n, p) in enumerate(self.trainer.model.named_parameters())
                 if p.requires_grad and idx in param_indices]
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

    def obtain_gradients(self, optim_type="adam", param_indices=None, **kwargs):
        vectorized_grads = torch.cat(
            [p.grad.view(-1) for idx, (n, p) in enumerate(self.trainer.model.named_parameters())
             if p.requires_grad and idx in param_indices]
        )
        if optim_type == "adam":
            avg, avg_sq = kwargs["avg"], kwargs["avg_sq"]
            updated_avg = self.config.beta1 * avg + (1 - self.config.beta1) * vectorized_grads
            updated_avg_sq = self.config.beta2 * avg_sq + (1 - self.config.beta2) * vectorized_grads ** 2
            vectorized_grads = updated_avg / torch.sqrt(updated_avg_sq + self.config.eps)
            return vectorized_grads
        else:
            return vectorized_grads

    def load_optimizer_state(self, optim_type="adam"):
        optim_path = self.trainer.saved_dir + "/optimizer.bin"
        state_dict = torch.load(optim_path, map_location="cpu")["state"]
        param_indices = list(state_dict.keys())

        if optim_type == "adam":
            avg = torch.cat([state_dict[i]["exp_avg"].view(-1) for i in param_indices]).to(self.device)
            avg_sq = torch.cat([state_dict[i]["exp_avg_sq"].view(-1) for i in param_indices]).to(self.device)
            return avg, avg_sq, param_indices
        else:
            return param_indices

    # def load_optimizer_state(self, optim_type="adam"):
    #     optim_path = self.trainer.saved_dir + "/optimizer.bin"
    #     state_dict = torch.load(optim_path, map_location="cpu")
    #     opt_param_indices = list(state_dict["state"].keys())
    #     # opt_param_indices = state_dict["param_groups"][0]["params"]
    #
    #     if optim_type == "adam":
    #         model_param_name, model_params = zip(*[
    #             (n, p) for idx, (n, p) in enumerate(self.trainer.model.named_parameters())
    #             if p.requires_grad and idx in opt_param_indices
    #         ])
    #         model_param_size = {
    #             n: p.size() for idx, (n, p) in enumerate(self.trainer.model.named_parameters())
    #             if p.requires_grad and idx in opt_param_indices
    #         }
    #
    #         name_state_dict = {}
    #         for param_idx, param_name in zip(opt_param_indices, model_param_name):
    #             param_size = model_param_size[param_name]
    #             exp_avg = state_dict["state"][param_idx]["exp_avg"]
    #             exp_avg_sq = state_dict["state"][param_idx]["exp_avg_sq"]
    #             assert exp_avg.size() == param_size
    #             name_state_dict[param_name] = {"exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq}
    #
    #         names = [
    #             n for idx, (n, p) in enumerate(self.trainer.model.named_parameters())
    #             if p.requires_grad and idx in opt_param_indices
    #         ]
    #         avg = torch.cat([name_state_dict[n]["exp_avg"].view(-1) for n in names]).to(self.device)
    #         avg_sq = torch.cat([name_state_dict[n]["exp_avg_sq"].view(-1) for n in names]).to(self.device)
    #         return avg, avg_sq, opt_param_indices
    #     else:
    #         return opt_param_indices

    def merge_grads(self, output_dir, normalize=True):
        file_list = os.listdir(output_dir)
        file_list.sort(key=lambda x: int(x.split('.')[0].split('-')[-1]))
        merged_data = []
        for file in file_list:
            data = torch.load(os.path.join(output_dir, file))
            data = F.normalize(data, dim=1) if normalize else data
            merged_data.append(data)
        merged_data = torch.cat(merged_data, dim=0)
        if normalize:
            output_path = os.path.join(output_dir, "norm_grads.pt")
        else:
            output_path = os.path.join(output_dir, "unorm_grads.pt")
        torch.save(merged_data, output_path)

    def _collect_grads(
            self,
            dataloader,
            projector,
            phase,
            optim_type,
            model_kwargs,
            param_indices,
            prev_avg=None,
            prev_avg_sq=None
    ):
        def _project(full_grads):
            full_grads = torch.stack(full_grads, dim=0)
            projected_grads = projector.project(full_grads, model_id=0)
            return projected_grads

        def _save(proj_grads, cnt):
            proj_grads = torch.cat(proj_grads, dim=0)
            print(f"Saving gradient at step {cnt}: shape {proj_grads.shape}...", flush=True)
            output_path = os.path.join(output_dir, f"grads-{cnt}.pt")
            torch.save(proj_grads, output_path)

        output_dir = os.path.join(self.trainer.saved_dir, f"{phase}-{optim_type}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        count = 0
        full_gradients, projected_gradients = [], []
        for batch in dataloader:
            count += 1
            loss = self.trainer.forward_once(model_kwargs, batch)
            loss.backward()
            if optim_type == "adam":
                vectorized_grad = self.obtain_gradients(
                    optim_type,
                    param_indices,
                    avg=prev_avg,
                    avg_sq=prev_avg_sq
                )
            else:
                vectorized_grad = self.obtain_gradients(optim_type, param_indices)
            full_gradients.append(vectorized_grad)

            if count % self.config.proj_interval == 0:
                projected_grad = _project(full_gradients)
                full_gradients = []
                projected_gradients.append(projected_grad)

            if count % self.config.save_interval == 0:
                _save(projected_gradients, count)
                projected_gradients = []

        if len(full_gradients) > 0:
            projected_grad = _project(full_gradients)
            del full_gradients
            projected_gradients.append(projected_grad)
            _save(projected_gradients, count)
            del projected_gradients
        torch.cuda.empty_cache()
        self.merge_grads(output_dir, normalize=True)

    def select_augmentation(self, model_kwargs):
        self.trainer.model.train()
        augment_dataloader = DataLoader(self.augment_dataset, batch_size=1,
                                        collate_fn=self.trainer.dataloader_params["collate_fn"])
        valid_dataloader = DataLoader(self.trainer.eval_dataset, batch_size=1,
                                      collate_fn=self.trainer.dataloader_params["collate_fn"])
        # augment_dataloader = self.trainer.get_test_dataloader(self.augment_dataset)
        # valid_dataloader = self.trainer.get_eval_dataloader()

        prev_avg, prev_avg_sq, param_indices = self.load_optimizer_state("adam")
        projector = self.get_projector(param_indices)

        augment_grad_path = os.path.join(self.trainer.saved_dir, "train-adam", "norm_grads.pt")
        eval_grad_path = os.path.join(self.trainer.saved_dir, "eval-sgd", "norm_grads.pt")
        if not os.path.exists(augment_grad_path):
            self._collect_grads(augment_dataloader, projector, "train", "adam",
                                model_kwargs, param_indices, prev_avg, prev_avg_sq)
        if not os.path.exists(eval_grad_path):
            self._collect_grads(valid_dataloader, projector, "eval", "sgd", model_kwargs, param_indices)
        augment_grads, eval_grads = torch.load(augment_grad_path), torch.load(eval_grad_path)
        augment_grads, eval_grads = prepare_data((augment_grads, eval_grads), self.device)
        influence_score = calculate_influence_score(augment_grads, eval_grads)
        _, indices = torch.topk(influence_score.max(dim=-1)[0], k=self.config.num_augments, dim=-1, largest=True)
        subset = TrajSubset(self.augment_dataset, indices.cpu(), self.augment_dataset.model_name)
        return subset

    # def select_augment_data(self, model_kwargs):
    #     valid_dataloader = self.trainer.get_eval_dataloader()
    #     projector = self.get_projector()
    #     self.trainer.model.train()
    #     augment_data_grads = self.collect_grads(model_kwargs)
    #
    #     valid_data_grads = []
    #     for batch in valid_dataloader:
    #         self.trainer.model.zero_grad()
    #         loss = self.trainer.forward_once(model_kwargs, batch)
    #         vectorized_grads = self.get_gradients(loss, optim_type="sgd")
    #         proj_grad = projector.project(
    #             vectorized_grads, model_id=0
    #         )
    #         valid_data_grads.append(proj_grad.cpu())
    #     valid_data_grads = torch.stack(valid_data_grads, dim=0)
    #     torch.cuda.empty_cache()
    #     augment_data_grads, valid_data_grads = prepare_data((augment_data_grads, valid_data_grads), self.device)
    #     influence_score = calculate_influence_score(augment_data_grads, valid_data_grads)
    #     _, indices = torch.topk(influence_score, k=self.config.num_augments, dim=0, largest=True)
    #     return indices

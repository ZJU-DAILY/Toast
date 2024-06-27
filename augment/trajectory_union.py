import torch
from torch.utils.data import Dataset
from typing import Optional, Union
from trak.projectors import BasicProjector, CudaProjector

from augment.augment_config import TaskType, TrajUnionConfig
from utils.train.trainer import Trainer


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


def move2gpu(args, device):
    if isinstance(args, dict):
        return {k: v.to(device) for k, v in args.items()}
    elif isinstance(args, tuple):
        args = [v.to(device) for v in args]
        return tuple(args)
    elif isinstance(args, torch.Tensor):
        return args.to(device)


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

    def get_gradients(self, loss, optim_type="adam", **kwargs):
        loss.backward()
        vectorized_grads = torch.cat(
            [p.grad.view(-1) for p in self.trainer.model.parameters() if p.grad is not None]
        )
        if optim_type == "adam":
            avg, avg_sq = kwargs["avg"], kwargs["avg_sq"]
            updated_avg = self.config.beta1 * avg + (1 - self.config.beta1) * vectorized_grads
            updated_avg_sq = self.config.beta2 * avg_sq + (1 - self.config.beta2) * vectorized_grads ** 2
            vectorized_grads = updated_avg / torch.sqrt(updated_avg_sq + self.config.eps)
            return vectorized_grads
        else:
            return vectorized_grads

    def load_optimizer_state(self):
        optim_path = self.trainer.saved_dir + "/optimizer.bin"
        optim_state = torch.load(optim_path)["state"]
        names = [n for n, p in self.trainer.model.named_parameters() if p.requires_grad]
        avg = torch.cat([optim_state[n]["exp_avg"].view(-1) for n in names]).to(self.device)
        avg_sq = torch.cat([optim_state[n]["exp_avg_sq"].view(-1) for n in names]).to(self.device)
        return avg, avg_sq

    def forward_once(self, kwargs, args):
        if self.config.task_type == TaskType.TRAJ_REC:
            (src_grid_seq, src_gps_seq, feat_seq, src_seq_len, tgt_gps_seq,
             tgt_node_seq, tgt_rate_seq, tgt_seq_len, tgt_inf_scores,
             batched_graph, node_index) = move2gpu(args, self.device)
            traj_edge, tgt_node_seq = batched_graph.edge_index.long(), tgt_node_seq.long()
            traj_edge, tgt_node_seq = batched_graph.edge_index.long(), tgt_node_seq.long()
            output_node, output_rate, logits = self.trainer.model(
                *kwargs, src_grid_seq, src_seq_len, feat_seq, node_index,
                traj_edge, batched_graph.batch, batched_graph.weight,
                tgt_node_seq, tgt_rate_seq, tgt_seq_len, tgt_inf_scores, 0.
            )
            output_node_dim = output_node.size(2)
            output_node = output_node.permute(1, 0, 2)[1:]
            output_node = output_node.reshape(-1, output_node_dim)  # ((len - 1) * batch_size, node_size)
            output_rate = output_rate.permute(1, 0, 2).squeeze(-1)[1:]  # ((len - 1), batch_size)
            logits = logits.squeeze(-1)  # (num_nodes)

            tgt_node_seq = tgt_node_seq.permute(1, 0, 2)[1:].squeeze(-1)
            tgt_node_seq = tgt_node_seq.reshape(-1)  # ((len - 1) * batch_size)
            tgt_rate_seq = tgt_rate_seq.permute(1, 0, 2)[1:].squeeze(-1)  # ((len - 1), batch_size)
            loss = self.trainer.compute_loss(
                output_node, output_rate, logits,
                tgt_node_seq, tgt_rate_seq, batched_graph.gt,
                src_seq_len, tgt_seq_len, kwargs["weights"]
            )
            return loss
        else:
            raise NotImplementedError(f"No task {self.config.task_type}")

    def collect_grads(self, kwargs):
        augment_dataloader = self.trainer.get_test_dataloader(self.augment_dataset)
        projector = self.get_projector()
        prev_avg, prev_avg_sq = self.load_optimizer_state()

        proj_grads = []
        for batch in augment_dataloader:
            self.trainer.model.zero_grad()
            loss = self.forward_once(kwargs, batch)
            vectorized_grads = self.get_gradients(
                loss,
                optim_type="adam",
                avg=prev_avg,
                avg_sq=prev_avg_sq
            )
            proj_grad = projector.project(
                vectorized_grads,
                model_id=0
            )
            proj_grads.append(proj_grad)
        proj_grads = torch.stack(proj_grads, dim=0)
        return proj_grads

    def select_augment_data(self, **model_kwargs):
        valid_dataloader = self.trainer.get_eval_dataloader()
        projector = self.get_projector()
        augment_data_grads = self.collect_grads(model_kwargs)

        valid_data_grads = []
        for batch in valid_dataloader:
            self.trainer.model.zero_grad()
            loss = self.forward_once(model_kwargs, batch)
            vectorized_grads = self.get_gradients(loss, optim_type="sgd")
            proj_grad = projector.project(
                vectorized_grads, model_id=0
            )
            valid_data_grads.append(proj_grad)
        valid_data_grads = torch.stack(valid_data_grads, dim=0)
        influence_score = calculate_influence_score(augment_data_grads, valid_data_grads)
        _, indices = torch.topk(influence_score, k=self.config.num_augments, dim=0, largest=True)
        return indices

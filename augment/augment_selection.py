from typing import Dict, List, Any
import torch

from models.base_model import BaseModel
from augment.point_union import PointUnion
from augment.trajectory_union import TrajUnion


class AugmentOPSelection:
    def __init__(
            self,
            encoder: BaseModel,
            device,
    ):
        self.encoder = encoder
        self.device = device

    def compute_size_ratio(
            self, 
            augment_by_pointunion: Dict[Any, List[Any]] = None,
            augment_by_trajunion: Dict[Any, List[Any]] = None,
            augment_by_featjoin: Dict[Any, List[Any]] = None,
    ):
        def compute_single_size_ratio(data_dict):
            size_ratio = []
            for _, value in data_dict.items():
                size_ratio.append(len(value))
            size_ratio = torch.tensor(size_ratio) / torch.sum(torch.tensor(size_ratio))
            return size_ratio.to(self.device)
        size_ratio_pointunion = compute_single_size_ratio(augment_by_pointunion)
        size_ratio_trajunion = compute_single_size_ratio(augment_by_trajunion)
        size_ratio_featjoin = compute_single_size_ratio(augment_by_featjoin)
        return size_ratio_pointunion, size_ratio_trajunion, size_ratio_featjoin
    
    def estimate_loss_reduction(self, data, augment_data, size_ratio, model_kwargs):
        embeds = self.encoder.encoding(data, **model_kwargs)
        loss_reduction = []
        for idx, (_, aug_data) in enumerate(augment_data.items()):
            aug_embeds = self.encoder.encoding(aug_data, **model_kwargs)
            reduction_batch = torch.norm(
                embeds[idx][None, :] - aug_embeds,
                dim=-1
            )
            loss_reduction.append(reduction_batch.sum())
        loss_reduction = torch.sum(torch.stack(loss_reduction) / size_ratio)
        return loss_reduction

    def select_ops(
            self, 
            data: Any, 
            augment_by_pointunion: Dict[Any, List[Any]] = None, 
            augment_by_trajunion: Dict[Any, List[Any]] = None,
            augment_by_featjoin: Dict[Any, List[Any]] = None,
            time_by_pointunion: float = None, 
            time_by_trajunion: float = None, 
            time_by_featjoin: float = None,
            model_kwargs = None
    ):
        size_ratio_pointunion, size_ratio_trajunion, size_ratio_featjoin = self.compute_size_ratio(
            augment_by_pointunion=augment_by_pointunion,
            augment_by_trajunion=augment_by_trajunion,
            augment_by_featjoin=augment_by_featjoin,
        )
        estimated_reduction_pointunion = self.estimate_loss_reduction(
            data, 
            augment_by_pointunion,
            size_ratio_pointunion,
            model_kwargs
        )
        estimated_reduction_trajunion = self.estimate_loss_reduction(
            data,
            augment_by_trajunion,
            size_ratio_trajunion,
            model_kwargs
        )
        estimated_reduction_featjoin = self.estimate_loss_reduction(
            data,
            augment_by_featjoin,
            size_ratio_featjoin,
            model_kwargs
        )
        utility_pointunion = estimated_reduction_pointunion / time_by_pointunion
        utility_trajunion = estimated_reduction_trajunion / time_by_trajunion
        utility_featjoin = estimated_reduction_featjoin / time_by_featjoin
        return utility_pointunion, utility_trajunion, utility_featjoin
        
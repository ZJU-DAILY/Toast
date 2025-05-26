import enum
from typing import Union, Optional, List
from trak.projectors import ProjectionType


class TaskType(str, enum.Enum):
    TRAJ_RECOVERY = "trajectory_recovery"
    TRAJ_SIMILAR = "trajectory_similarity"
    TYPE_IDENTIFY = "transport_type_identification"
    FLOW_PREDICT = "traffic_flow_prediction"


class PointUnionConfig:
    def __init__(
            self,
            task_type: Optional[TaskType],
            model_name: str,
            num_virtual_tokens: int = 5,
            virtual_dim: int = 512,
            num_epochs: int = 10,
            learning_rate: float = 1e-3,
            projection: bool = True,
            project_hidden_dim: int = 256
    ):
        self.task_type = task_type
        self.model_name = model_name
        self.num_virtual_tokens = num_virtual_tokens
        self.virtual_dim = virtual_dim
        self.num_epochs = num_epochs

        self.lr = learning_rate
        self.projection = projection
        self.projection_hidden_dim = project_hidden_dim


class TrajUnionConfig:
    def __init__(
            self,
            task_type: Optional[TaskType],
            model_name: str,
            num_augments,
            proj_type=ProjectionType.rademacher,
            proj_dim=2048,
            gradient_type="adam",
            proj_interval=16,
            save_interval=50,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8
    ):
        self.task_type = task_type
        self.model_name = model_name
        self.num_augments = num_augments
        self.proj_type = proj_type
        self.proj_dim = proj_dim
        self.gradient_type = gradient_type
        self.proj_interval = proj_interval
        self.save_interval = save_interval
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

from typing import Union, Optional, List


class PointUnionConfig:
    def __init__(
            self,
            num_virtual_tokens: int = 5,
            augment_type: Optional[Union[str, List[str]]] = None,
            virtual_dim: int = 512,
            num_epochs: int = 10,
            learning_rate: float = 1e-3,
            projection: bool = True,
            project_hidden_dim: int = 256
    ):
        self.num_virtual_tokens = num_virtual_tokens
        self.augment_type = augment_type
        self.virtual_dim = virtual_dim
        self.num_epochs = num_epochs

        self.lr = learning_rate
        self.projection = projection
        self.projection_hidden_dim = project_hidden_dim


class TrajUnionConfig:
    def __init__(
            self
    ):
        pass

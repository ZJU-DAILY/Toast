import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel


class Encoder(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dims,
            kernel_size=(1, 3),
            pool_size=(1, 2)
    ):
        super(Encoder, self).__init__()
        self.data_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=input_dim if i == 0 else hidden_dims[i - 1],
                        out_channels=hidden_dim,
                        kernel_size=kernel_size,
                        padding="same"
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        kernel_size=kernel_size,
                        padding="same"
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(
                        kernel_size=pool_size,
                        stride=pool_size,
                        return_indices=True
                    )
                )
                for i, hidden_dim in enumerate(hidden_dims)
            ]
        )
        self.label_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=input_dim if i == 0 else hidden_dims[i - 1],
                        out_channels=hidden_dim,
                        kernel_size=kernel_size,
                        padding="same"
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        kernel_size=kernel_size,
                        padding="same"
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(
                        kernel_size=pool_size,
                        stride=pool_size
                    )
                )
                for i, hidden_dim in enumerate(hidden_dims)
            ]
        )

    def forward(self, unlabeled_data: torch.Tensor, labeled_data: torch.Tensor = None):
        """

        Args:
            unlabeled_data: [batch_size, height, width, channels]
            labeled_data: [batch_size, height, width, channels]

        Returns:

        """
        hidden_unlabel = unlabeled_data.permute(0, 3, 1, 2)
        pool_indices = []
        for i, layer in enumerate(self.data_convs):
            hidden_unlabel, indices = layer(hidden_unlabel)
            pool_indices.append(indices)
        if labeled_data is None:
            return hidden_unlabel, pool_indices
        else:
            hidden_labeled = labeled_data.permute(0, 3, 1, 2)
            for layer in self.label_convs:
                hidden_labeled = layer(hidden_labeled)
            return hidden_unlabel, pool_indices, hidden_labeled


class Decoder(nn.Module):
    def __init__(
            self,
            output_dim,
            hidden_dims,
            kernel_size=(1, 3),
            pool_size=(1, 2)
    ):
        super(Decoder, self).__init__()
        self.upsample = nn.ModuleList(
            [
                nn.MaxUnpool2d(
                    kernel_size=pool_size,
                    stride=pool_size
                )
                for _ in range(len(hidden_dims))
            ]
        )
        self.deconvs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        kernel_size=kernel_size,
                        padding=(0, 1)
                    ),
                    nn.ReLU(),
                    nn.ConvTranspose2d(
                        in_channels=hidden_dim,
                        out_channels=hidden_dims[i + 1] if i != len(hidden_dims) - 1 else output_dim,
                        kernel_size=kernel_size,
                        padding=(0, 1)
                    ),
                    nn.ReLU() if i != len(hidden_dims) - 1 else nn.Sigmoid(),
                )
                for i, hidden_dim in enumerate(hidden_dims)
            ]
        )

    def forward(self, hidden_repr, pool_indices):
        """

        Args:
            hidden_repr: [batch_size, channels, height, width]

        Returns:
            outputs: [batch, channels, height, width]
        """
        outputs = hidden_repr
        for i, (upsample, deconv) in enumerate(zip(self.upsample, self.deconvs)):
            outputs = upsample(outputs, pool_indices[i])
            outputs = deconv(outputs)
        return outputs


class Classifier(nn.Module):
    def __init__(
            self,
            hidden_dim,
            num_classes
    ):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(in_features=hidden_dim, out_features=num_classes)

    def forward(self, hidden_repr: torch.Tensor):
        """

        Args:
            hidden_repr: [batch, channels, height, width]

        Returns:

        """
        flatten_repr = hidden_repr.permute(0, 2, 3, 1).flatten(start_dim=1)
        logits = self.linear(self.dropout(flatten_repr))
        return logits


class SECA(BaseModel):
    def __init__(self, input_dim, num_classes, max_length, hidden_dims):
        super(SECA, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dims)
        self.decoder = Decoder(input_dim, hidden_dims[::-1])
        cls_size = hidden_dims[-1] * max_length // (2 ** len(hidden_dims))
        self.cls_layer = Classifier(cls_size, num_classes)

    def extract_feat(self, *args):
        pass

    def encoding(self, unlabeled_data, labeled_data):
        return self.encoder(unlabeled_data, labeled_data)

    def decoding(self, hidden_repr, pool_indices):
        return self.decoder(hidden_repr, pool_indices)

    def forward(self, unlabeled_data, labeled_data):
        hidden_unlabeled, pool_indices, hidden_labeled = self.encoding(unlabeled_data, labeled_data)
        decoded_data = self.decoding(hidden_unlabeled, pool_indices[::-1])
        logits = self.cls_layer(hidden_labeled)
        return decoded_data, logits


class CNNSECA(BaseModel):
    def __init__(self, input_dim, num_classes, max_length, hidden_dims):
        super(CNNSECA, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dims)
        cls_size = hidden_dims[-1] * max_length // (2 ** len(hidden_dims))
        self.decoder = nn.Sequential(
            nn.Linear(cls_size, cls_size // 4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(cls_size // 4, num_classes)
        )

    def extract_feat(self, *args):
        pass

    def encoding(self, data):
        return self.encoder(data)

    def decoding(self, hidden_repr):
        flatten_repr = hidden_repr.permute(0, 2, 3, 1).flatten(start_dim=1)
        return self.decoder(flatten_repr)

    def forward(self, inputs):
        hiddens, _ = self.encoding(inputs)
        logits = self.decoding(hiddens)
        return logits

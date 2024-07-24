import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

from models.base_model import BaseModel


class MetaLearner(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(MetaLearner, self).__init__()
        self.node_learner = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, inputs):
        hiddens = self.node_learner(inputs)
        return hiddens


class MetaGAT(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MetaGAT, self).__init__(aggr="add", flow="source_to_target", node_dim=0)
        self.out_channels = out_channels
        self.linears = nn.Sequential(
            nn.Linear(in_channels, 16),
            nn.Sigmoid(),
            nn.Linear(16, 2),
            nn.Sigmoid(),
            nn.Linear(2, out_channels * out_channels * 2)
        )

    def message(self, states_j, states_i, features_j, features_i):
        # states_j: [num_nodes, batch_size, length, dim]
        states = torch.cat((states_i, states_j), dim=-1)
        features = torch.cat((features_i, features_j), dim=-1)

        weights = self.linears(features)
        weights = weights.reshape(-1, self.out_channels * 2, self.out_channels)

        num_nodes, batch_size, length, dim = states.shape
        # shape [num_nodes, batch_size * length, dim]
        states = states.reshape(num_nodes, -1, dim)

        alpha = F.leaky_relu(torch.bmm(states, weights))
        # shape [num_nodes, batch_size, length, dim]
        alpha = alpha.reshape(num_nodes, batch_size, length, -1)
        alpha = torch.softmax(alpha, dim=1)
        return alpha * states_j

    def forward(
            self,
            states: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
            edge_index: torch.Tensor,
            features: torch.Tensor = None
    ):
        edge_index, _ = add_self_loops(edge_index, num_nodes=states.shape[0])
        new_states = self.propagate(edge_index, states=states, features=features)
        new_states = F.relu(new_states)
        return new_states


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.rnn_layer = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.meta_gat = MetaGAT(in_channels=hidden_dim * 2, out_channels=hidden_dim)
        self.meta_rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, data, features, edge_index):
        """

        Args:
            data: shape [num_nodes, batch_size, length, dim]
            features: [num_nodes, dim]

        Returns:

        """
        num_nodes, batch_size, length, _ = data.shape
        data = data.reshape(num_nodes * batch_size, length, -1)
        hidden_states = []
        outputs, hiddens = self.rnn_layer(data)
        outputs = outputs.reshape(num_nodes, batch_size, length, -1)
        hiddens = hiddens.squeeze().reshape(num_nodes, batch_size, -1)
        hidden_states.append(hiddens)

        outputs = self.meta_gat(outputs, edge_index, features)

        # shape [num_nodes, batch_size, 1, dim]
        outputs = outputs.reshape(num_nodes * batch_size, length, -1)
        outputs, hiddens = self.meta_rnn(outputs)
        hiddens = hiddens.squeeze().reshape(num_nodes, batch_size, -1)
        hidden_states.append(hiddens)
        return hidden_states


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, cl_decay_steps):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rnn_layer = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.meta_gat = MetaGAT(in_channels=hidden_dim * 2, out_channels=hidden_dim)
        self.meta_rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim * 2, output_dim)
        self.cl_decay_steps = cl_decay_steps

    def sampling(self):
        threshold = self.cl_decay_steps / (self.cl_decay_steps + 1)
        return random.random() < threshold

    def forward(self, hidden_embeds, features, labels, edge_index, train_mode=True):
        num_nodes, batch_size, length, _ = labels.shape
        auxiliary = labels[:, :, :, self.output_dim:]
        labels = labels[:, :, :, :self.output_dim]
        start = torch.zeros(num_nodes, batch_size, self.input_dim, device=labels.device)
        pred_logits = []
        for idx in range(length):
            if idx == 0:
                data = start.unsqueeze(dim=2).reshape(num_nodes * batch_size, 1, -1)
            else:
                prev = torch.cat((pred_logits[idx - 1], auxiliary[:, :, idx - 1]), dim=-1)
                truth = torch.cat((labels[:, :, idx - 1], auxiliary[:, :, idx - 1]), dim=-1)
                if train_mode and self.sampling():
                    data = truth
                else:
                    data = prev
                data = data.unsqueeze(dim=2).reshape(num_nodes * batch_size, 1, -1)

            initial_hidden = hidden_embeds[0].unsqueeze(dim=0).reshape(1, num_nodes * batch_size, -1)
            outputs, hiddens = self.rnn_layer(data, initial_hidden)
            outputs = outputs.reshape(num_nodes, batch_size, 1, -1)
            hidden_embeds[0] = hiddens.squeeze().reshape(num_nodes, batch_size, -1)

            outputs = self.meta_gat(outputs, edge_index, features)

            initial_hidden = hidden_embeds[1].unsqueeze(dim=0).reshape(1, num_nodes * batch_size, -1)
            outputs = outputs.reshape(num_nodes * batch_size, 1, -1)
            outputs, hiddens = self.meta_rnn(outputs, initial_hidden)
            outputs = outputs.reshape(num_nodes, batch_size, 1, -1)
            hidden_embeds[1] = hiddens.squeeze().reshape(num_nodes, batch_size, -1)

            inputs = torch.cat((outputs.squeeze(), features.unsqueeze(dim=1).repeat(1, batch_size, 1)), dim=-1)
            pred = self.output_layer(inputs)
            pred_logits.append(pred)
        # shape [num_nodes, batch_size, length, dim]
        pred_logits = torch.stack(pred_logits, dim=2)
        return pred_logits


class STMetaNet(BaseModel):
    def __init__(self, input_dim, output_dim, feat_dim, hidden_dim):
        super(STMetaNet, self).__init__()
        self.output_dim = output_dim
        self.meta_learner = MetaLearner(feat_dim, hidden_dim)
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(input_dim, hidden_dim, output_dim, 2000)

    def extract_feat(self, features):
        return self.meta_learner(features)

    def encoding(self, data, features, edge_index):
        return self.encoder(data, features, edge_index)

    def decoding(self, hidden_embeds, features, labels, edge_index, train_mode=True):
        return self.decoder(hidden_embeds, features, labels, edge_index, train_mode=True)

    def forward(self, data, features, labels, edge_index, train_mode):
        data, labels = data.permute(2, 0, 1, 3), labels.permute(2, 0, 1, 3)
        features = self.extract_feat(torch.mean(features, dim=0))
        encoded_embeds = self.encoding(data, features, edge_index)
        predicts = self.decoding(encoded_embeds, features, labels, edge_index, train_mode)
        return predicts, labels[:, :, :, :self.output_dim]


class ResUnit(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResUnit, self).__init__()
        self.unit = nn.Sequential(
            nn.BatchNorm2d(num_features=input_dim),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=input_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=1,
                padding="same"
            ),
            nn.BatchNorm2d(num_features=input_dim),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=input_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=1,
                padding="same"
            )
        )

    def forward(self, inputs):
        outputs = self.unit(inputs)
        return inputs + outputs
    
    
class ExtNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ExtNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, inputs):
        outputs = self.net(inputs)
        return outputs


class STResNet(BaseModel):
    def __init__(self, num_flows, width, height, closeness_len, period_len, trend_len, feat_dim, num_layers):
        super(STResNet, self).__init__()
        self.num_flows = num_flows
        self.width = width
        self.height = height
        feat_hidden_dim = num_flows * (closeness_len + period_len + trend_len)
        closeness_dim = closeness_len * num_flows
        period_dim = period_len * num_flows
        trend_dim = trend_len * num_flows
        self.ext_net = self._build_ext_channel(feat_dim, 64, num_flows)
        self.c_net = self._build_conv_channel(closeness_dim, 2, num_layers)
        self.p_net = self._build_conv_channel(period_dim, 2, num_layers)
        self.t_net = self._build_conv_channel(trend_dim, 2, num_layers)
        self.w_c = nn.Parameter(torch.randn(1, num_flows, width, height))
        self.w_p = nn.Parameter(torch.randn(1, num_flows, width, height))
        self.w_t = nn.Parameter(torch.randn(1, num_flows, width, height))

    def _build_ext_channel(self, input_dim, hidden_dim, output_dim):
        return ExtNet(input_dim, hidden_dim, output_dim)

    def _build_conv_channel(self, input_dim, output_dim, num_layers):
        net = nn.Sequential()
        net.append(
            nn.Conv2d(
                in_channels=input_dim,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding="same"
            )
        )
        for lid in range(num_layers):
            net.append(ResUnit(64, 64))
        net.append(
            nn.Conv2d(
                in_channels=64,
                out_channels=output_dim,
                kernel_size=3,
                stride=1,
                padding="same"
            )
        )
        return net

    def extract_feat(self, features):
        return self.ext_net(features)

    def encoding(self, close_inputs, period_inputs, trend_inputs, feat_embeds):
        close_embeds = self.c_net(close_inputs)
        period_embeds = self.p_net(period_inputs)
        trend_embeds = self.t_net(trend_inputs)
        feat_embeds = feat_embeds.permute(0, 3, 1, 2)
        fused_embeds = (
            self.w_c * close_embeds +
            self.w_p * period_embeds +
            self.w_t * trend_embeds +
            feat_embeds
        )
        return torch.tanh(fused_embeds)

    def decoding(self, encoded_embeds):
        return encoded_embeds

    def forward(self, close_inputs, period_inputs, trend_inputs, features):
        feat_embeds = self.extract_feat(features)
        encoded_embeds = self.encoding(close_inputs, period_inputs, trend_inputs, feat_embeds)
        return encoded_embeds

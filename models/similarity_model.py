import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_geometric.nn.conv import GCNConv

from models.base_model import BaseModel


class LocationEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LocationEmbedding, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv = GCNConv(input_dim, hidden_dim, cached=True)
        self.dropout = nn.Dropout()

    def forward(self, traj_seqs, seq_len, node_feat, edge_index, edge_feat=None):
        batch_size, max_len, _ = traj_seqs.shape
        road_embed = self.conv(node_feat, edge_index, edge_feat)
        road_embed = self.dropout(torch.relu(road_embed))
        spatial_embeds = torch.zeros(batch_size, max_len, self.hidden_dim,
                                     dtype=torch.float, device=traj_seqs.device)
        for bs in range(batch_size):
            length = seq_len[bs]
            spatial_embeds[bs, :length] = torch.index_select(road_embed, dim=0, index=traj_seqs[bs, :length].squeeze())
        return spatial_embeds


class TimeEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, pretrain=False, pretrain_path=None):
        super(TimeEmbedding, self).__init__()
        self.pretrain = pretrain
        if pretrain:
            self.date2vec = torch.load(pretrain_path)
        else:
            self.linear_1 = nn.Linear(input_dim, hidden_dim // 2)
            self.linear_2 = nn.Linear(input_dim, hidden_dim // 2)

    def forward(self, time_seqs):
        """
        padding and timestamp series embedding
        :param time_seqs: list [batch,timestamp_seq]
        :return: packed_input
        """
        if self.pretrain:
            with torch.no_grad():
                time_embeds = self.date2vec.encode(time_seqs)
        else:
            time_embeds_1 = self.linear_1(time_seqs)
            time_embeds_2 = torch.cos(self.linear_2(time_seqs))
            time_embeds = torch.cat((time_embeds_1, time_embeds_2), dim=-1)
        return time_embeds


class FeatureExtractor(nn.Module):
    def __init__(self, spatial_dim, temporal_dim, hidden_dim, pretrain, pretrain_model):
        super(FeatureExtractor, self).__init__()
        self.pos_encoder = LocationEmbedding(spatial_dim, hidden_dim)
        # self.time_encoder = TimeEmbedding(
        #     temporal_dim,
        #     hidden_dim,
        #     pretrain,
        #     pretrain_model
        # )

    def forward(self, traj_seqs, time_seqs, seq_len, node_feat, edge_index, edge_feat=None):
        spatial_embeds = self.pos_encoder(traj_seqs, seq_len, node_feat, edge_index, edge_feat)
        # time_embeds = self.time_encoder(time_seqs)
        time_embeds = time_seqs
        return spatial_embeds, time_embeds


class CoAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(CoAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.temperature = hidden_dim ** 0.5
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Dropout(0.1)
        )
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, spatial_embeds, time_embeds):
        hiddens = torch.stack((spatial_embeds, time_embeds), dim=2)
        query = self.query(hiddens)
        key = self.key(hiddens)
        value = self.value(hiddens)
        attn = torch.matmul(query / self.temperature, key.transpose(2, 3))
        attn_out = torch.matmul(F.softmax(attn, dim=-1), value)
        attn_out = self.ffn(attn_out) + attn_out
        attn_out = self.layer_norm(attn_out)
        spatial_out, time_out = attn_out[:, :, 0], attn_out[:, :, 1]
        return spatial_out, time_out


class ST2VecEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate):
        super(ST2VecEncoder, self).__init__()
        self.attn_layer = CoAttention(input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True
        )
        # self-attention weights
        self.w_omega = nn.Parameter(torch.Tensor(hidden_dim * 2, hidden_dim * 2))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_dim * 2, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def get_mask(self, tensor, length):
        """
        create mask based on the sentence lengths
        :param seq_lengths: sequence length after `pad_packed_sequence`
        :return: mask (batch_size, max_seq_len)
        """
        batch_size, max_len, _ = tensor.shape
        mask = torch.ones(batch_size, max_len, device=tensor.device)

        for i, l in enumerate(length):
            if l < max_len:
                mask[i, l:] = 0
        return mask

    def forward(self, spatial_embeds, time_embeds, seq_len):
        # output features (h_t) from the last layer of the LSTM, for each t
        # (batch_size, seq_len, 2 * num_hiddens)
        spatial_attn, time_attn = self.attn_layer(spatial_embeds, time_embeds)
        attn_outputs = torch.cat((spatial_attn, time_attn), dim=-1)
        lstm_inputs = pack_padded_sequence(attn_outputs, seq_len, batch_first=True, enforce_sorted=False)

        packed_output, _ = self.lstm(lstm_inputs)  # output, (h, c)
        outputs, seq_lengths = pad_packed_sequence(packed_output, batch_first=True)

        # get sequence mask
        mask = self.get_mask(outputs, seq_lengths)

        # Attention...
        # (batch_size, seq_len, 2 * num_hiddens)
        u = torch.tanh(torch.matmul(outputs, self.w_omega))
        # (batch_size, seq_len)
        att = torch.matmul(u, self.u_omega).squeeze()

        # add mask
        att = att.masked_fill(mask == 0, -1e10)

        # (batch_size, seq_len,1)
        att_score = F.softmax(att, dim=1).unsqueeze(2)
        # normalization attention weight
        # (batch_size, seq_len, 2 * num_hiddens)
        scored_outputs = outputs * att_score

        # weighted sum as output
        # (batch_size, 2 * num_hiddens)
        out = torch.sum(scored_outputs, dim=1)
        return out


class ST2Vec(BaseModel):
    def __init__(
            self,
            spatial_dim,
            temporal_dim,
            feature_dim,
            hidden_dim,
            num_layers,
            dropout_rate,
            pretrain=True,
            pretrain_model=None
    ):
        super(ST2Vec, self).__init__()
        self.extractor = FeatureExtractor(spatial_dim, temporal_dim, feature_dim, pretrain, pretrain_model)
        self.encoder = ST2VecEncoder(feature_dim, hidden_dim, num_layers, dropout_rate)

    def extract_feat(self, traj_seqs, time_seqs, seq_len, node_feat, edge_index, edge_feat=None):
        spatial_embeds, time_embeds = self.extractor(traj_seqs, time_seqs, seq_len, node_feat, edge_index, edge_feat)
        return spatial_embeds, time_embeds

    def encoding(self, spatial_embeds, time_embeds, seq_len):
        return self.encoder(spatial_embeds, time_embeds, seq_len)

    def decoding(self, inputs):
        return inputs

    def forward(self, traj_seqs, time_seqs, seq_len, node_feat, edge_index, edge_feat=None):
        features = self.extract_feat(traj_seqs, time_seqs, seq_len, node_feat, edge_index, edge_feat)
        hiddens = self.encoding(*features, seq_len)
        return hiddens


class GTSEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate):
        super(GTSEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True
        )
        # self-attention weights
        self.w_omega = nn.Parameter(torch.Tensor(hidden_dim * 2, hidden_dim * 2))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_dim * 2, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def get_mask(self, tensor, length):
        """
        create mask based on the sentence lengths
        :param seq_lengths: sequence length after `pad_packed_sequence`
        :return: mask (batch_size, max_seq_len)
        """
        batch_size, max_len, _ = tensor.shape
        mask = torch.ones(batch_size, max_len, device=tensor.device)

        for i, l in enumerate(length):
            if l < max_len:
                mask[i, l:] = 0
        return mask

    def forward(self, spatial_embeds, seq_len):
        # output features (h_t) from the last layer of the LSTM, for each t
        # (batch_size, seq_len, 2 * num_hiddens)
        lstm_inputs = pack_padded_sequence(spatial_embeds, seq_len, batch_first=True, enforce_sorted=False)

        packed_output, _ = self.lstm(lstm_inputs)  # output, (h, c)
        outputs, seq_lengths = pad_packed_sequence(packed_output, batch_first=True)

        # get sequence mask
        mask = self.get_mask(outputs, seq_lengths)

        # Attention...
        # (batch_size, seq_len, 2 * num_hiddens)
        u = torch.tanh(torch.matmul(outputs, self.w_omega))
        # (batch_size, seq_len)
        att = torch.matmul(u, self.u_omega).squeeze()

        # add mask
        att = att.masked_fill(mask == 0, -1e10)

        # (batch_size, seq_len,1)
        att_score = F.softmax(att, dim=1).unsqueeze(2)
        # normalization attention weight
        # (batch_size, seq_len, 2 * num_hiddens)
        scored_outputs = outputs * att_score

        # weighted sum as output
        # (batch_size, 2 * num_hiddens)
        out = torch.sum(scored_outputs, dim=1)
        return out


class GTS(BaseModel):
    def __init__(
            self,
            spatial_dim,
            feature_dim,
            hidden_dim,
            num_layers,
            dropout_rate
    ):
        super(GTS, self).__init__()
        self.extractor = LocationEmbedding(spatial_dim, feature_dim)
        self.encoder = GTSEncoder(feature_dim, hidden_dim, num_layers, dropout_rate)

    def extract_feat(self, traj_seqs, seq_len, node_feat, edge_index, edge_feat=None):
        spatial_embeds = self.extractor(traj_seqs, seq_len, node_feat, edge_index, edge_feat)
        return spatial_embeds

    def encoding(self, spatial_embeds, seq_len):
        return self.encoder(spatial_embeds, seq_len)

    def decoding(self, inputs):
        return inputs

    def forward(self, traj_seqs, seq_len, node_feat, edge_index, edge_feat=None):
        s_embeds = self.extract_feat(traj_seqs, seq_len, node_feat, edge_index, edge_feat)
        hiddens = self.encoding(s_embeds, seq_len)
        return hiddens

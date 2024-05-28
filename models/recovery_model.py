import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.pool import global_mean_pool, global_max_pool, global_add_pool

from models.base_model import BaseModel


class RoadGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, num_heads=8):
        super(RoadGNN, self).__init__()
        assert hidden_dim % num_heads == 0
        self.num_layers = num_layers
        self.gnn_layer = nn.ModuleList(
            [
                GATConv(in_channels=input_dim if i == 0 else hidden_dim,
                        out_channels=hidden_dim // num_heads,
                        heads=num_heads,
                        add_self_loops=False)
                for i in range(num_layers)
            ]
        )
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, edge_index, node_feat, readout=True, batch=None, weight=None):
        for idx, layer in enumerate(self.gnn_layer):
            node_feat = layer(node_feat, edge_index)
            if idx < self.num_layers - 1:
                node_feat = F.leaky_relu(node_feat)
        node_embed = self.dropout_layer(node_feat)
        if not readout:
            return node_embed
        if weight is None:
            graph_embed = global_mean_pool(node_embed, batch)
        else:
            weighted_node_embed = node_embed * weight.unsqueeze(1)
            graph_embed = global_mean_pool(weighted_node_embed, batch)
        return graph_embed


class PositionEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=150):
        super(PositionEncoding, self).__init__()
        self.hidden_dim = hidden_dim

        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x * math.sqrt(self.hidden_dim)
        x = x + self.pe[:, :x.size(1)]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim_per_head = hidden_dim // num_heads

        self.q_layer = nn.Linear(hidden_dim, hidden_dim)
        self.k_layer = nn.Linear(hidden_dim, hidden_dim)
        self.v_layer = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)

    def attention(self, q, k, v, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dim_per_head)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = self.dropout_layer(scores)
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask=None):
        q = self.q_layer(q).view(q.size(0), -1, self.num_heads, self.dim_per_head)
        k = self.k_layer(k).view(q.size(0), -1, self.num_heads, self.dim_per_head)
        v = self.v_layer(v).view(q.size(0), -1, self.num_heads, self.dim_per_head)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = self.attention(q, k, v, mask, self.dropout_layer)

        concat = scores.transpose(1, 2).contiguous().view(q.size(0), -1, self.hidden_dim)
        output = self.output_layer(concat)
        return output


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(self.dropout_layer(x))
        return x


class Norm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-6):
        super(Norm, self).__init__()
        self.eps = eps

        self.alpha = nn.Parameter(torch.ones(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / \
               (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.attn_layer = MultiHeadAttention(hidden_dim, num_heads)
        self.feed_forward = FeedForwardNetwork(hidden_dim)
        self.norm_layer_1 = Norm(hidden_dim)
        self.norm_layer_2 = Norm(hidden_dim)
        self.norm_layer_3 = Norm(hidden_dim)
        self.dropout_layer_1 = nn.Dropout(dropout)
        self.dropout_layer_2 = nn.Dropout(dropout)

    def forward(self, x, mask, norm=False):
        x2 = self.norm_layer_1(x)
        x = x + self.dropout_layer_1(self.attn_layer(x2, x2, x2, mask))
        x2 = self.norm_layer_2(x)
        x = x + self.dropout_layer_2(self.feed_forward(x2))
        return x if not norm else self.norm_layer_3(x)


class GraphNorm(nn.Module):
    def __init__(self, input_dim):
        super(GraphNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, input_dim))
        self.beta = nn.Parameter(torch.zeros(1, input_dim))
        self.moving_mean = torch.zeros(1, input_dim)
        self.moving_var = torch.ones(1, input_dim)

    def graph_norm(self, input_embed, batch, eps=1e-5, momentum=0.9, mask2d=None):
        if self.moving_mean.device != input_embed.device:
            self.moving_mean, self.moving_var = self.moving_mean.to(input_embed.device), \
                                                self.moving_var.to(input_embed.device)
        if not torch.is_grad_enabled():
            embed_hat = (input_embed - self.moving_mean) / torch.sqrt(self.moving_var + eps)
        else:
            mean_embed = global_mean_pool(input_embed, batch)
            if mask2d is not None:
                mean_embed = mean_embed.reshape(mask2d.size(0), mask2d.size(1), -1)
                mask2d = mask2d.unsqueeze(-1)
                mean_embed = (mean_embed * mask2d).sum(dim=(0, 1)) / mask2d.sum()
                mean_embed = mean_embed.reshape(1, -1)
            else:
                mean_embed = mean_embed.mean(dim=0, keepdim=True)
            var = ((input_embed - mean_embed) ** 2).mean(dim=0, keepdim=True)
            embed_hat = (input_embed - mean_embed) / torch.sqrt(var + eps)
            self.moving_mean = momentum * self.moving_mean + (1.0 - momentum) * mean_embed
            self.moving_var = momentum * self.moving_var + (1.0 - momentum) * var
        output = self.gamma * embed_hat + self.beta
        return output

    def forward(self, node_embed, batch, mask2d=None):
        output = self.graph_norm(node_embed, batch, mask2d=mask2d)
        return output


class GatedFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GatedFusion, self).__init__()
        self.spatio_linear = nn.Linear(input_dim, hidden_dim)
        self.temporal_linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, spatio_feat, temporal_feat):
        spatio_feat = F.leaky_relu(self.spatio_linear(spatio_feat))
        temporal_feat = F.leaky_relu(self.temporal_linear(temporal_feat))
        z = torch.sigmoid(spatio_feat + temporal_feat)
        fused_feat = z * spatio_feat + (1. - z) * temporal_feat
        return fused_feat


class GraphRefinementLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=8, dropout=0.1):
        super(GraphRefinementLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.graph_norm_1 = GraphNorm(input_dim)
        self.graph_norm_2 = GraphNorm(hidden_dim)
        self.attn_layer = GatedFusion(input_dim, hidden_dim)
        self.graph_forward = GATConv(in_channels=hidden_dim,
                                     out_channels=hidden_dim // num_heads,
                                     heads=num_heads,
                                     add_self_loops=False)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, seq_embed, edge_index, node_feats, batch, mask2d=None):
        batch_size, max_len = seq_embed.size(0), seq_embed.size(1)
        seq_embed = seq_embed.reshape(-1, self.hidden_dim)  # (batch_size * src_len->graph num, dim)
        seq2node_embed = torch.index_select(seq_embed, dim=0, index=batch)

        norm_feats = self.graph_norm_1(node_feats, batch, mask2d)
        node_feats = node_feats + self.dropout_layer(self.attn_layer(norm_feats, seq2node_embed))
        norm_feats = self.graph_norm_2(node_feats, batch, mask2d)
        node_feats = node_feats + self.dropout_layer(self.graph_forward(norm_feats, edge_index))
        seq_embed = global_mean_pool(node_feats, batch).reshape(batch_size, max_len, -1)
        return seq_embed, node_feats


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, extra_input_dim, extra_output_dim, device):
        super(Encoder, self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.position_encoder = PositionEncoding(hidden_dim)
        self.transformer = nn.ModuleList(
            [
                TransformerEncoder(hidden_dim) for _ in range(num_layers)
            ]
        )
        self.graph_refiner = nn.ModuleList(
            [
                GraphRefinementLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
            ]
        )
        self.norm_layer = Norm(hidden_dim)
        self.extra_linear = nn.Linear(extra_input_dim, extra_output_dim)
        self.extra_layer = nn.Linear(hidden_dim + extra_output_dim, hidden_dim)

    def mask_log_softmax(self, logits, batch, weight):
        maxes = global_max_pool(logits, batch)
        maxes = torch.index_select(maxes, dim=0, index=batch)
        exp_nodes = torch.exp(logits - maxes) * weight.unsqueeze(-1)

        sums = global_add_pool(exp_nodes, batch)
        sums = torch.index_select(sums, dim=0, index=batch)
        pred = exp_nodes / (sums + 1e-6)
        return torch.log(torch.clip(pred, 1e-6, 1))

    def forward(self, traj_seq, traj_seq_len, feat_seq, node_feats, edge_index, batch, weight):
        """

        Args:
            traj_seq (torch.Tensor): shape (batch_size, length, hidden_dim + 3)
            traj_seq_len (torch.Tensor): shape (batch_size, length)
            feat_seq (torch.Tensor): shape (batch_size, 25)
            node_feats:
            edge_index:
            batch:
            weight:

        Returns:
            encode_seq (torch.Tensor): shape (batch_size, length, hidden_dim)
            encode_embed (torch.Tensor): shape (batch_size, hidden_dim)
            logits (torch.Tensor): shape (num_node, 1)

        """
        batch_size, max_len = traj_seq.size(0), traj_seq.size(1)
        mask3d = torch.zeros(batch_size, max_len, max_len).to(self.device)
        mask2d = torch.zeros(batch_size, max_len).to(self.device)
        for bs in range(batch_size):
            mask3d[bs, :traj_seq_len[bs], :traj_seq_len[bs]] = 1
            mask2d[bs, :traj_seq_len[bs]] = 1
        seq_embed = self.input_layer(traj_seq)
        seq_embed = self.position_encoder(seq_embed)
        for l in range(self.num_layers):
            seq_embed = self.transformer[l](seq_embed, mask3d)
            seq_embed, node_feats = self.graph_refiner[l](seq_embed, edge_index, node_feats, batch, mask2d)
        encode_seq = self.norm_layer(seq_embed) * mask2d.unsqueeze(-1)
        encode_embed = torch.mean(encode_seq, dim=1)
        logits = self.output_layer(node_feats)
        logits = self.mask_log_softmax(logits, batch, weight)
        feat_embed = torch.tanh(self.extra_linear(feat_seq))
        encode_embed = torch.tanh(self.extra_layer(torch.cat((feat_embed, encode_embed), dim=-1)))
        return encode_seq, encode_embed, logits


class DecoderAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DecoderAttention, self).__init__()
        self.attn_layer = nn.Linear(input_dim, hidden_dim)
        self.trans_weight = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, encode_embed, hidden_embed, attn_mask):
        """

        Args:
            encode_embed (torch.Tensor): shape (batch_size, length, hidden_dim)
            hidden_embed (torch.Tensor): shape (batch_size, hidden_dim)
            attn_mask (torch.Tensor): shape (batch_size, length)

        Returns:
            (torch.Tensor): shape (batch_size, length)

        """
        seq_len = encode_embed.size(1)
        hidden_embed = hidden_embed.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn_layer(torch.cat((hidden_embed, encode_embed), dim=-1)))
        attention = self.trans_weight(energy).squeeze(-1)
        attention = attention.masked_fill(attn_mask == 0, -1e6)
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, node_size, hidden_dim, node_feat_dim, dropout, device):
        super(Decoder, self).__init__()
        self.node_size = node_size
        self.device = device
        self.tandem_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        self.attn_layer = DecoderAttention(hidden_dim * 2, hidden_dim)
        self.rnn = nn.GRU(hidden_dim * 2 + 1, hidden_dim)
        self.node_layer = nn.Linear(hidden_dim, node_size)
        self.rate_layer = nn.Linear(hidden_dim + node_feat_dim, 1)
        self.dropout_layer = nn.Dropout(dropout)

    def masked_log_softmax(self, x, mask):
        maxes = torch.max(x, dim=1, keepdim=True)[0]
        x_exp = torch.exp(x - maxes) * mask
        pred = x_exp / (x_exp.sum(dim=1, keepdim=True) + 1e-6)
        pred = torch.clip(pred, 1e-6, 1)
        return torch.log(pred)

    def forward(self, encode_seq, hidden_embed, node_embed, rn_node_feat,
                traj_seq_len, node_idx, rate_seq, tgt_seq_len,
                influence_score, tf_ratio=0.5):
        """

        Args:
            encode_seq (torch.Tensor): shape (batch_size, src_length, hidden_dim)
            hidden_embed (torch.Tensor): shape (batch_size, hidden_dim)
            node_embed (torch.Tensor): shape (num_road_segment, hidden_dim)
            rn_node_feat:
            traj_seq_len:
            node_idx (torch.Tensor): shape (batch_size, tgt_length, 1)
            rate_seq (torch.Tensor): shape (batch_size, tgt_length, 1)
            tgt_seq_len:
            influence_score:
            tf_ratio (float): default: `0.5`

        Returns:
            output_node_idx (torch.Tensor): shape (batch_size, tgt_length, num_node)
            output_rate_seq (torch.Tensor): shape (batch_size, tgt_length, 1)

        """
        batch_size, max_len, max_tgt_len = encode_seq.size(0), encode_seq.size(1), node_idx.size(1)
        attn_mask = torch.zeros(batch_size, max_len).to(self.device)
        for bs in range(batch_size):
            attn_mask[bs][:traj_seq_len[bs]] = 1.
        output_node_idx = torch.zeros(batch_size, max_tgt_len, self.node_size).to(self.device)
        output_rate_seq = torch.zeros(rate_seq.size()).to(self.device)
        input_node = node_idx[:, 0, :].squeeze(-1)  # (batch_size)
        input_rate = rate_seq[:, 0, :].unsqueeze(0)  # (1, batch_size, 1)
        for t in range(1, max_tgt_len):
            segment_feat = torch.index_select(rn_node_feat, dim=0, index=input_node)

            node_feat = self.dropout_layer(torch.index_select(node_embed, dim=0, index=input_node)).unsqueeze(0)  # (1, batch_size, dim)
            attn = self.attn_layer(encode_seq, hidden_embed, attn_mask)
            weighted_out = torch.bmm(attn.unsqueeze(1), encode_seq).permute(1, 0, 2)  # (1, batch_size, dim)
            rnn_input = torch.cat((weighted_out, node_feat, input_rate), dim=-1)
            rnn_output, hidden_embed = self.rnn(rnn_input, hidden_embed.unsqueeze(0))
            hidden_embed = hidden_embed.squeeze(0)

            node_prob = self.masked_log_softmax(self.node_layer(rnn_output.squeeze(0)), influence_score[:, t, :])
            pred_node = torch.argmax(node_prob, dim=-1)
            pred_node_embed = self.dropout_layer(torch.index_select(node_embed, dim=0, index=pred_node))
            rate_input = self.tandem_layer(torch.cat((pred_node_embed, hidden_embed), dim=-1))
            pred_rate = torch.sigmoid(self.rate_layer(torch.cat((rate_input, segment_feat), dim=-1)))

            output_node_idx[:, t, :] = node_prob
            output_rate_seq[:, t] = pred_rate
            teacher_force = random.random() < tf_ratio

            input_node = node_idx[:, t, :].squeeze(-1) if teacher_force else pred_node
            input_rate = rate_seq[:, t, :].unsqueeze(0) if teacher_force else pred_rate.unsqueeze(0)
        for bs in range(batch_size):
            seq_len = tgt_seq_len[bs]
            output_node_idx[bs][seq_len:] = -100
            output_node_idx[bs][seq_len:, 0] = 0
            output_rate_seq[bs][seq_len:] = 0
        return output_node_idx, output_rate_seq


class RNTrajRec(BaseModel):
    def __init__(self, num_rn_node, hidden_dim, num_gnn_layers, num_encoder_layers, extra_input_dim, extra_output_dim, road_feat_dim, grid_shape, grid_embed_size, batch_size, device, dropout=0.1):
        super(RNTrajRec, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.num_rn_node = num_rn_node
        grid_embed_table_shape = grid_shape + [grid_embed_size]
        self.grid_embed_table = nn.Parameter(torch.rand(grid_embed_table_shape))
        self.road_node_embed_table = nn.Parameter(torch.rand(num_rn_node, grid_embed_size))
        self.grid_gru = nn.GRU(grid_embed_size, grid_embed_size)

        self.feat_extractor = RoadGNN(input_dim=grid_embed_size,
                                      hidden_dim=hidden_dim,
                                      num_layers=num_gnn_layers,
                                      dropout=dropout)
        self.encoder = Encoder(hidden_dim + 3, hidden_dim, num_encoder_layers, extra_input_dim, extra_output_dim, device)
        self.decoder = Decoder(num_rn_node, hidden_dim, road_feat_dim, dropout, device)

    def extract_feat(self, road_grid, road_len, road_node_idx, road_edge_index, batch):
        num_node, max_seqlen = road_grid.shape[0], road_grid.shape[1]
        road_grids = road_grid.reshape(-1, 2)
        grid_input = self.grid_embed_table[road_grids[:, 0], road_grids[:, 1]]
        grid_input = grid_input.reshape(num_node, max_seqlen, -1).transpose(0, 1)  # (seq_len, num_node, feat_dim)
        packed_grid_input = nn.utils.rnn.pack_padded_sequence(grid_input, road_len,
                                                              batch_first=False, enforce_sorted=False)
        _, gru_output = self.grid_gru(packed_grid_input)
        grid_embed = gru_output.squeeze(0)

        road_node_embed = torch.index_select(self.road_node_embed_table, dim=0, index=road_node_idx)
        grid_node_embed = torch.index_select(grid_embed, dim=0, index=road_node_idx)
        road_node_embed = F.leaky_relu(road_node_embed + grid_node_embed)

        road_embed = self.feat_extractor(road_edge_index, road_node_embed, True, batch)
        return road_embed

    def encoding(self, road_embed, traj_seq, src_seq_len, feat_seq, traj_node_idx, traj_edge, batch, weight):
        node_embed = torch.index_select(road_embed, dim=0, index=traj_node_idx)
        traj_embed = global_add_pool(node_embed * weight.unsqueeze(1), batch) / \
                     global_add_pool(weight.unsqueeze(1), batch)
        traj_embed = traj_embed.reshape(traj_seq.size(0), traj_seq.size(1), -1)
        encode_input = torch.cat((traj_embed, traj_seq), dim=-1)
        encode_seq, hidden_embed, logits = self.encoder(encode_input, src_seq_len, feat_seq, node_embed, traj_edge, batch, weight)
        return encode_seq, hidden_embed, logits

    def decoding(self, encode_seq, hidden_embed, road_embed, rn_node_feat,
                 src_seq_len, tgt_rid_seq, rate_seq, tgt_seq_len,
                 influence_score, tf_ratio=0.5):
        output_node_seq, output_rate_seq = self.decoder(encode_seq, hidden_embed, road_embed, rn_node_feat,
                                                        src_seq_len, tgt_rid_seq, rate_seq, tgt_seq_len,
                                                        influence_score, tf_ratio)
        return output_node_seq, output_rate_seq

    def forward(self, road_grid, road_len, road_nodes, road_edge, road_batch, road_feat,
                src_grid_seq, src_seq_len, feat_seq, traj_node, traj_edge, traj_batch, weight,
                tgt_rid_seq, tgt_rate_seq, tgt_seq_len, influence_score, tf_ratio=0.5):
        road_embed = self.extract_feat(road_grid, road_len, road_nodes, road_edge, road_batch)
        encoded_seq, hidden_embed, node_logits = self.encoding(road_embed, src_grid_seq, src_seq_len, feat_seq,
                                                               traj_node, traj_edge, traj_batch, weight)
        output_node, output_rate = self.decoding(encoded_seq, hidden_embed, road_embed, road_feat,
                                                 src_seq_len, tgt_rid_seq, tgt_rate_seq, tgt_seq_len,
                                                 influence_score, tf_ratio)
        return output_node, output_rate, node_logits

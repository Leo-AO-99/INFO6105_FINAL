import math
import torch
from torch import nn
import numpy as np

from config import ModelConfig


def PositionEmbedding(seq_len, d, n):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in range(d):

            denominator = np.power(n, 2 * i / d)
            if i % 2 == 0:
                P[k, i] = np.sin(k / denominator)
            else:
                P[k, i] = np.cos(k / denominator)
    return torch.tensor(P, dtype=torch.float32).unsqueeze(0)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    # MLP
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.dim, config.hidden_dim, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(config.hidden_dim, config.dim, bias=True)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(self.dropout(x))
        return x


class EncoderBlock(nn.Module):
    # or we can call it as Transformer Block
    # according to the paper
    # Multi-Head Attention -> Add & Norm -> Feed Forward -> Add & Norm
    # Position Embedding is outside the Transformer Block
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = Attention(config)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_dropout = nn.Dropout(config.dropout_rate)

        self.feed_forward = FeedForward(config)
        self.feed_forward_norm = RMSNorm(config.dim, config.norm_eps)
        self.feed_forward_dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x, e_mask):
        # we can choose whether to apply normalization before or after attention
        # while llama3 apply normalization before attention
        x_norm = self.attention_norm(x)
        x = x + self.attention_dropout(self.attention(x_norm, x_norm, x_norm, e_mask))

        x_norm = self.feed_forward_norm(x)
        x = x + self.feed_forward_dropout(self.feed_forward(x_norm))
        return x


class DecoderBlock(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attention = Attention(config)
        self.self_attention_dropout = nn.Dropout(config.dropout_rate)
        self.self_attention_norm = RMSNorm(config.dim, config.norm_eps)

        self.cross_attention = Attention(config)
        self.cross_attention_dropout = nn.Dropout(config.dropout_rate)
        self.cross_attention_norm = RMSNorm(config.dim, config.norm_eps)

        self.feed_forward = FeedForward(config)
        self.feed_forward_dropout = nn.Dropout(config.dropout_rate)
        self.feed_forward_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x, y, d_mask, padding_mask):
        """
        x: [batch_size, seq_len, dim]
        y: [batch_size, seq_len, dim], encoder output, for cross-attention
        d_mask: [batch_size, 1, seq_len, seq_len], lookahead mask for self-attention, ignore future tokens
        padding_mask: [batch_size, 1, 1, seq_len], mask for cross-attention, ignore padding tokens
        """

        # Masked Self-Attention
        x_norm = self.self_attention_norm(x)
        self_attention_output = x + self.self_attention_dropout(self.self_attention(x_norm, x_norm, x_norm, d_mask))

        # Cross-Attention
        x_norm = self.cross_attention_norm(self_attention_output)
        cross_attention_output = self_attention_output + self.cross_attention_dropout(self.cross_attention(x_norm, y, y, padding_mask))

        # FeedForward
        x_norm = self.feed_forward_norm(cross_attention_output)
        output = cross_attention_output + self.feed_forward_dropout(self.feed_forward(x_norm))

        return output


class Encoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layers = config.n_encoder_layers
        self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(self.layers)])
        self.norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x, e_mask):
        for i in range(self.layers):
            x = self.blocks[i](x, e_mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layers = config.n_decoder_layers
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(self.layers)])
        self.norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, tgt, encoder_output, d_mask, padding_mask):
        for i in range(self.layers):
            tgt = self.blocks[i](tgt, encoder_output, d_mask, padding_mask)

        return self.norm(tgt)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads

        # W_q, W_k, W_v
        self.wq = nn.Linear(config.dim, config.dim)
        self.wk = nn.Linear(config.dim, config.dim)
        self.wv = nn.Linear(config.dim, config.dim)

        self.wo = nn.Linear(config.dim, config.dim)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, q, k, v, mask=None):
        """
        q: [batch_size, seq_len, dim]
        k: [batch_size, seq_len, dim]
        v: [batch_size, seq_len, dim]
        """
        batch_size = q.shape[0]
        q, k, v = self.wq(q), self.wk(k), self.wv(v)

        # [batch_size, seq_len, dim] -> [batch_size, seq_len, n_heads, head_dim]
        # according to below comment, we want [batch_size, n_heads, seq_len, head_dim], so why can't we just view as this shape?
        # TODO
        q = q.view(batch_size, -1, self.n_heads, self.head_dim)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim)

        # For llama3, RoPE here, and without Tensor.transpose()

        # [batch_size, seq_len, n_heads, head_dim] -> [batch_size, n_heads, seq_len, head_dim]
        # transpose is hard to understand, we can think in this way:
        # the dimension of tensor has its own meaning
        # like q[a][b][c] means a-th sequence, b-th query vector in this sequence, in c-th head
        # now we transpose 1-th dimension and 2-th dimension, q[a][b][c] -> q[a][c][b]
        # so it is easy to understand, new tensor will be [batch_size, n_heads, seq_len, head_dim]
        # because we care the seq_len * head_dim as a whole
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_values = self.attention(
            q, k, v, mask
        )  # [batch_size, n_heads, seq_len, head_dim]
        # concat all the heads
        # since we want to concat, it is a good idea that arrange the sub-tensor one by one in memory
        # that is why we use contiguous()
        # assuming i-th token, we want its sub-tensor in all heads arranged one by one in memory
        # so sub-tensor of i-th token from different heads are contiguous in memory
        concat_output = (
            attn_values.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        )  # [batch_size, seq_len, dim]

        return self.wo(concat_output)

    def attention(self, q, k, v, mask=None):
        attn_scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )  # Q * K^T / sqrt(d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill_(mask == 0, -1e9)
        attn_dist = self.softmax(attn_scores)  # softmax(Q * K^T / sqrt(d_k))
        attn_dist = self.dropout(attn_dist)
        attn_values = torch.matmul(attn_dist, v)  # softmax(Q * K^T / sqrt(d_k)) * V
        return attn_values

class Transformer(nn.Module):

    def __init__(self, config: ModelConfig, src_pad_id, tgt_pad_id, device):
        super().__init__()
        self.src_vocab_size = config.src_vocab_size
        self.tgt_vocab_size = config.tgt_vocab_size
        self.device = device
        self.dim = config.dim
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id
        self.max_seq_len = config.max_seq_len

        # transformer architecture
        # transformer architecture
        self.src_embedding = nn.Embedding(self.src_vocab_size, self.dim)
        self.tgt_embedding = nn.Embedding(self.tgt_vocab_size, self.dim)

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.output_layer = nn.Linear(self.dim, self.tgt_vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, tgt):
        """
        src: [batch_size, seq_len], e.g. [[1, 2, 3, 4], [5, 6, 7, 8]] `1` can refer to token `I`
        tgt: [batch_size, seq_len], e.g. [[1, 2, 3, 4], [5, 6, 7, 8]] `1` can refer to token `我`
        """
        batch_size = src.size(0)
        
        # padding mask
        src_padding_mask = (src == self.src_pad_id).unsqueeze(1).unsqueeze(2)
        tgt_padding_mask = (tgt == self.tgt_pad_id).unsqueeze(1).unsqueeze(2)
        
        # embedding
        src_input = self.src_embedding(src)
        tgt_input = self.tgt_embedding(tgt)

        # positional embedding
        # TODO
        
        # lookahead mask
        tgt_seq_len = self.max_seq_len # we need to set the seq_len to max_seq_len
        tgt_lookahead_mask = torch.full((tgt_seq_len, tgt_seq_len), float("-inf"))
        tgt_lookahead_mask = torch.triu(tgt_lookahead_mask, diagonal=1)
        tgt_lookahead_mask = tgt_lookahead_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 8, -1, -1).to(self.device) 

        # load to device
        src_input = src_input.to(self.device)
        tgt_input = tgt_input.to(self.device)
        src_padding_mask = src_padding_mask.to(self.device)
        tgt_padding_mask = tgt_padding_mask.to(self.device)
        tgt_lookahead_mask = tgt_lookahead_mask.to(self.device)

        # Encoder output
        encoder_output = self.encoder(src_input, src_padding_mask)
        
        # Decoder output
        decoder_output = self.decoder(tgt_input, encoder_output, tgt_lookahead_mask, tgt_padding_mask)
        
        # output
        output = self.softmax(self.output_layer(decoder_output))
        
        return output

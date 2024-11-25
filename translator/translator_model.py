import math
import torch
from torch import nn
import numpy as np

from config import ModelConfig



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
class DecoderLayer(nn.Module):

  def __init__(self, config: ModelConfig):
        super().__init__()
        self.multihead_attention1 = Attention(config)
        self.dropout1 = nn.Dropout(config.dropout_rate)
        self.add_norm1 =   RMSNorm(config.dim, config.norm_eps)
        self.multihead_attention2 = Attention(config)
        self.dropout2 = nn.Dropout(config.dropout_rate)
        self.add_norm2 =  RMSNorm(config.dim, config.norm_eps)
        self.feed_forward =FeedForward(config)
        self.dropout3 = nn.Dropout(config.dropout_rate)
        self.add_norm3 =  RMSNorm(config.dim, config.norm_eps)
  def forward(self,x,y,d_mask,padding_mask):
       # Masked Self-Attention
        multihead_output1 = self.multihead_attention1(x, x, x, d_mask)
        multihead_output1 = self.dropout1(multihead_output1)
        addnorm_output1 = self.add_norm1(x + multihead_output1)
        # Cross-Attention
        multihead_output2 = self.multihead_attention2(addnorm_output1,y, y, padding_mask)
        multihead_output2 = self.dropout2(multihead_output2)
        addnorm_output2 = self.add_norm2(addnorm_output1 + multihead_output2)
        # FeedForward
        feedforward_output = self.feed_forward(addnorm_output2)
        feedforward_output = self.dropout3(feedforward_output)
        output = self.add_norm3(addnorm_output2 + feedforward_output)
        return output
def PositionEmbedding(seq_len, d,n):

    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in range(d):
            
            denominator = np.power(n, 2*i/d)
            if i%2==0:
                P[k, i] = np.sin(k/denominator)
            else:
                P[k, i] = np.cos(k/denominator)
    return torch.tensor(P, dtype=torch.float32).unsqueeze(0)  # 添加批量维度
class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.tgt_vocab_size, config.dim)
        self.time=config.dim
        self.position_encoding = nn.Parameter(
            torch.randn(1, config.max_seq_len, config.dim), requires_grad=False
        )
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = RMSNorm(config.dim, config.norm_eps)
     

    def forward(self, tgt, encoder_output, padding_mask):
        batch_size, tgt_seq_len = tgt.size()

        # Embedding + Positional Encoding
        tgt_embedded = self.embedding(tgt) + PositionEmbedding(tgt_seq_len,self.time,10000).to(tgt.device)
        print("PositionEmbedding shape:", PositionEmbedding(tgt_seq_len, self.time, 10000).shape)
        print("tgt_embedded shape:", tgt_embedded.shape)
        # Lookahead Mask
        d_mask= torch.full([tgt_seq_len,tgt_seq_len],float('-inf'))
        d_mask = torch.triu( d_mask,diagonal=1)
        d_mask = d_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 8, -1, -1).to(tgt.device)
        print("d_mask shape:", d_mask.shape)
     





        for layer in self.layers:
            tgt_embedded = layer(tgt_embedded, encoder_output, d_mask, padding_mask)

        return self.norm(tgt_embedded)



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

        # TODO 做什么用的
        self.wo = nn.Linear(config.dim, config.dim)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, q, k, v, mask=None):
        '''
        q: [batch_size, seq_len, dim]
        k: [batch_size, seq_len, dim]
        v: [batch_size, seq_len, dim]
        '''
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

        attn_values = self.attention(q, k, v, mask) # [batch_size, n_heads, seq_len, head_dim]
        # concat all the heads
        # since we want to concat, it is a good idea that arrange the sub-tensor one by one in memory
        # that is why we use contiguous()
        # assuming i-th token, we want its sub-tensor in all heads arranged one by one in memory
        # so sub-tensor of i-th token from different heads are contiguous in memory
        concat_output = attn_values.transpose(1, 2).contiguous().view(batch_size, -1, self.dim) # [batch_size, seq_len, dim]

        return self.wo(concat_output)
        
    def attention(self, q, k, v, mask=None):
        attn_scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim) # Q * K^T / sqrt(d_k)
        if mask is not None:
            #mask = mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill_(mask == 0, -1e9)
            print("attn_scores shape:", attn_scores.shape)
        attn_dist = self.softmax(attn_scores) # softmax(Q * K^T / sqrt(d_k))
        attn_dist = self.dropout(attn_dist)
        attn_values = torch.matmul(attn_dist, v) # softmax(Q * K^T / sqrt(d_k)) * V
        return attn_values

class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.src_vocab_size = config.src_vocab_size
        self.tgt_vocab_size = config.tgt_vocab_size
        
        self.src_embedding = nn.Embedding(self.src_vocab_size, config.dim)
        self.tgt_embedding = nn.Embedding(self.tgt_vocab_size, config.dim)
        
        self.encoder = Encoder(config)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, tgt):
        '''
        src: [batch_size, seq_len], e.g. [[1, 2, 3, 4], [5, 6, 7, 8]] `1` can refer to token `I`
        tgt: [batch_size, seq_len], e.g. [[1, 2, 3, 4], [5, 6, 7, 8]] `1` can refer to token `我`
        '''
        src = self.src_embedding(src) # [batch_size, seq_len, dim]
        tgt = self.tgt_embedding(tgt) # [batch_size, seq_len, dim]
def test_decoder():
    config = ModelConfig()
    decoder = Decoder(config)
    encoder=Encoder(config)

    # 模拟输入
    batch_size = config.max_batch_size
    tgt_seq_len = 15
    src_seq_len = 15

    tgt = torch.randint(0, config.tgt_vocab_size, (batch_size, tgt_seq_len))
    

    src = torch.randint(0, config.src_vocab_size, (batch_size, src_seq_len))
    src_embedding = nn.Embedding(config.src_vocab_size, config.dim)
    padding_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    print("padding_mask shape:", padding_mask.shape)
    encoder_output = encoder(src_embedding(src), padding_mask)
    output = decoder(tgt, encoder_output, padding_mask)
    print("Source Input Shape:", src.shape)
    print("Target Input Shape:", tgt.shape)
    print("Encoder Output Shape:", encoder_output.shape)
    print("Decoder Output Shape:", output.shape)

# 调用测试函数
test_decoder()
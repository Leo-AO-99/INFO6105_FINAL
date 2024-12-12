from typing import Literal
from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    dim: int = Field(default=512, description="dimension of the vector embeddings")
    hidden_dim: int = Field(default=2048, description="dimension of the hidden layer in MLP")
    n_heads: int = Field(default=8, description="number of attention heads")
    n_encoder_layers: int = Field(default=6, description="number of encoder layers")
    n_decoder_layers: int = Field(default=6, description="number of decoder layers")
    dropout_rate: float = Field(default=0.1, description="dropout rate")
    norm_eps: float = Field(default=1e-6, description="epsilon for normalization")
    optim_lr: float = Field(default=1e-4, description="learning rate")
    src_vocab_size: int = Field(description="source vocabulary size")
    tgt_vocab_size: int = Field(description="target vocabulary size")
    max_seq_len: int = Field(default=128, description="maximum sequence length")
    max_batch_size: int = Field(default=128, description="maximum batch size")
    norm_way: Literal["LN", "RMS"] = Field(description="normalization way")

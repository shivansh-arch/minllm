import torch
import torch.nn as nn

from attention import MultiHeadAttention
from rmsnorm import RMSNorm
from swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = RMSNorm(d_model)

        self.ffn = SwiGLU(d_model, hidden_dim)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x):
        # Pre-Norm Multi-Head Attention
        x = x + self.attention(self.norm1(x))

        # Pre-Norm Feed Forward
        x = x + self.ffn(self.norm2(x))

        return x
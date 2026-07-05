from torch import nn
import torch

from src.rope import RoPE


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

        # RoPE works on each head separately
        self.rope = RoPE(self.depth)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Linear projections
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # Split into heads
        q = q.view(batch_size, seq_len, self.num_heads, self.depth).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.depth).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.depth).transpose(1, 2)

        # Apply RoPE
        q = self.rope(q)
        k = self.rope(k)

        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / (self.depth ** 0.5)

        # Causal Mask
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device),
            diagonal=1
        ).bool()

        scores = scores.masked_fill(mask, float("-inf"))

        # Attention weights
        weights = torch.softmax(scores, dim=-1)

        # Weighted sum of values
        output = torch.matmul(weights, v)

        # Merge heads
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )

        # Final projection
        output = self.dense(output)

        return output
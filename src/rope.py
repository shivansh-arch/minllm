import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2).float() / dim)
        )

        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_table", emb.cos())
        self.register_buffer("sin_table", emb.sin())

    def rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x):
        seq_len = x.shape[-2]

        # Shape: (1, 1, seq_len, head_dim)
        cos = self.cos_table[:seq_len].unsqueeze(0).unsqueeze(0).to(x.dtype)
        sin = self.sin_table[:seq_len].unsqueeze(0).unsqueeze(0).to(x.dtype)

        return x * cos + self.rotate_half(x) * sin
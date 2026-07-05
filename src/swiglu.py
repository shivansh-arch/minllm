import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()

        self.w1 = nn.Linear(d_model, hidden_dim)  # Gate projection
        self.w2 = nn.Linear(d_model, hidden_dim)  # Up projection
        self.w3 = nn.Linear(hidden_dim, d_model)  # Down projection

    def forward(self, x):
        gate = F.silu(self.w1(x))
        up = self.w2(x)

        out = gate * up
        out = self.w3(out)

        return out
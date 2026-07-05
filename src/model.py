import torch
import torch.nn as nn

from src.block import TransformerBlock
from src.rmsnorm import RMSNorm


class MiniLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        hidden_dim,
        num_layers,
    ):
        super().__init__()

        # Token Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Transformer Blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    hidden_dim,
                )
                for _ in range(num_layers)
            ]
        )

        # Final RMSNorm
        self.norm = RMSNorm(d_model)

        # Language Modeling Head
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # Token embeddings
        x = self.embedding(x)

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Final normalization
        x = self.norm(x)

        # Output logits
        logits = self.output_layer(x)

        return logits
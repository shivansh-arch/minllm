import torch
import torch.nn as nn


class Trainer:
    def __init__(self, model, dataset, lr=3e-4, max_grad_norm=1.0):
        self.model = model
        self.dataset = dataset
        self.max_grad_norm = max_grad_norm

        # Device the model is on
        self.device = next(model.parameters()).device

        # Put model in training mode
        self.model.train()

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr
        )

    def train_step(self):
        # Get batch
        x, y = self.dataset.get_batch()

        # Forward pass
        logits = self.model(x)

        # Reshape for CrossEntropyLoss
        batch_size, seq_len, vocab_size = logits.shape

        logits = logits.view(batch_size * seq_len, vocab_size)
        y = y.view(batch_size * seq_len)

        # Compute loss
        loss = self.criterion(logits, y)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.max_grad_norm
        )

        # Update parameters
        self.optimizer.step()

        return loss.item()

    def train(self, steps, eval_every=100):
        loss_history = []

        for step in range(1, steps + 1):
            loss = self.train_step()
            loss_history.append(loss)

            if step % eval_every == 0:
                print(f"Step {step}: Loss = {loss:.4f}")

        return loss_history
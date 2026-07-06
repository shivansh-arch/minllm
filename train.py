import torch

from src.dataset import TextDataset
from src.model import MiniLM
from src.trainer import Trainer


config = {
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "hidden_dim": 512,
    "context_length": 128,
    "batch_size": 32,
    "lr": 3e-4,
    "steps": 5000,
}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset
    dataset = TextDataset(
        file_path="holmes.txt",
        split="train",
        context_length=config["context_length"],
        batch_size=config["batch_size"],
    )

    # Model
    model = MiniLM(
        vocab_size=dataset.vocab_size,
        d_model=config["d_model"],
        num_heads=config["n_heads"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["n_layers"],
    ).to(device)

    # Trainer
    trainer = Trainer(
        model=model,
        dataset=dataset,
        lr=config["lr"],
    )

    # Train
    
    
    # Generate
    loss_history = trainer.train(config["steps"])

    import json

    with open("loss_history.json", "w") as f:
        json.dump(loss_history, f)

    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    main()
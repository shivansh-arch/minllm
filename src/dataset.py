import os
import torch
import urllib.request


class TextDataset:
    def __init__(
        self,
        file_path="holmes.txt",
        split="train",
        context_length=128,
        batch_size=32,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.context_length = context_length
        self.batch_size = batch_size

        # Download dataset if missing
        if not os.path.exists(file_path):
            print("Downloading dataset...")
            urllib.request.urlretrieve(
                "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
                file_path,
            )

        # Load text
        with open(file_path, "r", encoding="utf-8") as f:
            self.text = f.read()

        print(f"Loaded {len(self.text)} characters")

        # Build vocabulary
        self.characters = sorted(list(set(self.text)))
        self.vocab_size = len(self.characters)

        self.char_to_idx = {
            ch: i for i, ch in enumerate(self.characters)
        }

        self.idx_to_char = {
            i: ch for i, ch in enumerate(self.characters)
        }

        # Encode entire dataset
        data = torch.tensor(
            self.encode(self.text),
            dtype=torch.long
        )

        # Train / validation split
        n = int(0.9 * len(data))

        if split == "train":
            self.data = data[:n]
        elif split == "val":
            self.data = data[n:]
        else:
            raise ValueError("split must be 'train' or 'val'")

    def encode(self, text):
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices):
        return "".join(self.idx_to_char[i] for i in indices)

    def get_batch(self):
        ix = torch.randint(
            len(self.data) - self.context_length,
            (self.batch_size,)
        )

        x = torch.stack([
            self.data[i:i + self.context_length]
            for i in ix
        ])

        y = torch.stack([
            self.data[i + 1:i + self.context_length + 1]
            for i in ix
        ])

        # Move tensors to GPU if available
        x = x.to(self.device)
        y = y.to(self.device)

        return x, y
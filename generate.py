import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(
    model,
    dataset,
    prompt,
    max_new_tokens=100,
    temperature=1.0,
):
    model.eval()

    device = next(model.parameters()).device

    # Encode prompt
    tokens = dataset.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):

        # Keep only the last context_length tokens
        x = tokens[:, -dataset.context_length :]

        # Forward pass
        logits = model(x)

        # Take logits for the last token
        logits = logits[:, -1, :]

        # Temperature scaling
        logits = logits / temperature

        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)

        # Sample next token
        next_token = torch.multinomial(probs, num_samples=1)

        # Append token
        tokens = torch.cat([tokens, next_token], dim=1)

    # Decode
    return dataset.decode(tokens[0].tolist())
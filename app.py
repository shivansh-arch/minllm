import json
import matplotlib.pyplot as plt
import streamlit as st
import torch

from dataset import TextDataset
from generate import generate
from model import MiniLM


# -------------------------
# Config
# -------------------------

config = {
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "hidden_dim": 512,
    "context_length": 128,
    "batch_size": 32,
}


# -------------------------
# Load Model
# -------------------------

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = TextDataset(
        file_path="holmes.txt",
        split="train",
        context_length=config["context_length"],
        batch_size=config["batch_size"],
    )

    model = MiniLM(
        vocab_size=dataset.vocab_size,
        d_model=config["d_model"],
        num_heads=config["n_heads"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["n_layers"],
    )

    model.load_state_dict(
        torch.load(
            "model.pth",
            map_location=device,
        )
    )

    model.to(device)
    model.eval()

    return model, dataset


model, dataset = load_model()


# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(
    page_title="MiniLLM",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 MiniLLM")

tab1, tab2 = st.tabs(
    [
        "Text Generation",
        "Training Loss",
    ]
)

# =====================================================
# Tab 1
# =====================================================

with tab1:

    st.header("Generate Sherlock Holmes Text")

    prompt = st.text_input(
        "Prompt",
        value="Sherlock Holmes",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1,
    )

    max_tokens = st.slider(
        "Max New Tokens",
        20,
        500,
        200,
    )

    if st.button("Generate"):

        text = generate(
            model=model,
            dataset=dataset,
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )

        st.subheader("Generated Text")

        st.write(text)

# =====================================================
# Tab 2
# =====================================================

with tab2:

    st.header("Training Loss")

    try:

        with open("loss_history.json", "r") as f:
            losses = json.load(f)

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(losses)

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")

        ax.grid(True)

        st.pyplot(fig)

    except FileNotFoundError:

        st.warning(
            "loss_history.json not found. Train the model first."
        )
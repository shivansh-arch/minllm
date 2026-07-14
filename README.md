# 🤖 MiniLLM: GPT-Style Language Model from Scratch

A lightweight GPT-style language model built entirely in **PyTorch** to understand how modern Large Language Models work under the hood.

Instead of relying on high-level libraries, this project implements the core transformer components from scratch, including **Multi-Head Self-Attention, Rotary Positional Embeddings (RoPE), RMSNorm, and SwiGLU**. The trained model is served through an interactive **Streamlit** interface for text generation and training visualization.

---

## 🚀 Features

* GPT-style autoregressive language model
* Multi-Head Self-Attention
* Rotary Positional Embeddings (RoPE)
* RMSNorm instead of LayerNorm
* SwiGLU Feed Forward Network
* Causal Masking for next-token prediction
* Character-level tokenizer
* Custom training pipeline
* Temperature-based text generation
* Interactive Streamlit web application
* Training loss visualization

---

# 📂 Project Structure

```text
MiniLLM/
│
├── src/
│   ├── attention.py
│   ├── block.py
│   ├── dataset.py
│   ├── model.py
│   ├── rmsnorm.py
│   ├── rope.py
│   ├── swiglu.py
│   ├── tokenizer.py
│   └── trainer.py
│
├── train.py
├── generate.py
├── app.py
├── download_data.py
├── requirements.txt
├── model.pth
└── loss_history.json
```

---

# 🏗️ Model Architecture

The model follows a simplified GPT architecture.

```text
Input Text
      │
Character Tokenizer
      │
Embedding Layer
      │
┌───────────────────────────────┐
│ Transformer Block × N         │
│ ├── RoPE                      │
│ ├── Multi-Head Attention      │
│ ├── RMSNorm                   │
│ ├── SwiGLU Feed Forward        │
│ └── Residual Connections      │
└───────────────────────────────┘
      │
Final RMSNorm
      │
Linear Projection
      │
Next Token Prediction
```

---

# 🧠 Technologies Used

* Python
* PyTorch
* Streamlit
* Matplotlib

---

# 📊 Training Configuration

| Parameter           | Value |
| ------------------- | ----- |
| Embedding Dimension | 256   |
| Transformer Layers  | 6     |
| Attention Heads     | 8     |
| Hidden Dimension    | 512   |
| Context Length      | 128   |
| Batch Size          | 32    |
| Learning Rate       | 3e-4  |
| Training Steps      | 5000  |

---

# 📚 Dataset

The model is trained on **Sherlock Holmes** text.

The dataset is converted into character-level tokens and trained using next-token prediction.

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/shivansh-arch/minllm.git
cd minllm
```

Install dependencies

```bash
pip install -r requirements.txt
```

Download the dataset

```bash
python download_data.py
```

---

# 🏋️ Training

Train the language model

```bash
python train.py
```

Training will generate:

* `model.pth`
* `loss_history.json`

---

# 💬 Generate Text

Generate text from the trained model

```bash
python generate.py
```

---

# 🌐 Run the Web App

Launch the Streamlit interface

```bash
streamlit run app.py
```

The application provides:

* Prompt-based text generation
* Temperature control
* Maximum token selection
* Training loss visualization

---

# 🧩 Implemented Transformer Components

✅ Token Embeddings

✅ Multi-Head Self-Attention

✅ Rotary Positional Embeddings (RoPE)

✅ RMSNorm

✅ SwiGLU Feed Forward Network

✅ Residual Connections

✅ Causal Language Modeling

---

# 📈 Future Improvements

* Byte Pair Encoding (BPE) tokenizer
* Mixed precision training (AMP)
* Validation perplexity
* Checkpoint management
* TensorBoard / Weights & Biases integration
* Top-k and Top-p sampling
* Quantized inference
* FastAPI inference service
* Multi-GPU training
* Docker deployment

---

# 📖 Learning Objectives

This project was built to gain a practical understanding of:

* Transformer architecture
* Self-attention mechanism
* Positional encoding with RoPE
* Modern normalization techniques
* Decoder-only language models
* Training autoregressive neural networks
* End-to-end LLM development in PyTorch

---

# ⚠️ Limitations

* Character-level tokenizer (slower and less efficient than BPE)
* Trained on a relatively small corpus
* Intended for educational purposes
* Not designed for production-scale language generation

---

# 👨‍💻 Author

**Shivansh Gupta**

Computer Science Student | Machine Learning & AI Enthusiast

GitHub: https://github.com/shivansh-arch

---

## ⭐ If you found this project useful, consider giving it a star!

# Learning-DL: 100M Transformer Pre-training

A high-performance implementation and research repository focused on pre-training a 100M-parameter Transformer. The primary objective is to exceed the performance of GPT-2 (124M) while achieving competitive benchmarks with modern on-device models such as **MobileLLM** and **SmolLM**.

## 🚀 Project Objective
The goal is to engineer a parameter-efficient language model optimized for on-device inference. This project bridges the gap between fundamental deep learning research and modern LLM scaling laws.

- **Target:** 100M Parameters.
- **Benchmark Goal:** Outperform GPT-2 (124M) on standard zero-shot evaluations.
- **Focus:** Parameter efficiency, architectural optimization, and stable pre-training at scale.

## 🛠️ Technical Stack & Architecture
The core implementation (`model.py`, `train.py`) utilizes a modern Llama-style architecture:

- **Architecture:** Decoder-only Transformer with RMSNorm and RoPE (Rotary Positional Embeddings).
- **Activation:** SwiGLU (optimized for representation density).
- **Tokenizer:** GPT-2 Byte-level BPE via `tiktoken`.
- **Optimization:** 
    - AdamW with decoupled weight decay.
    - Linear Learning Rate Warmup and Cosine Decay.
    - Automatic Mixed Precision (AMP) for high-throughput training.
    - Gradient Clipping to ensure training stability.

## 📂 Repository Structure

### Core Engine (Root)
- **`model.py`:** Production-grade Transformer implementation with modular attention blocks.
- **`train.py`:** Scalable training pipeline featuring checkpointing and real-time loss tracking.
- **`utils.py`:** High-performance data processing and visualization utilities.

### Research Lab (`notebooks/`)
A collection of exploratory experiments and architectural prototyping:
- **Transformers/Attention:** Deep dives into the mechanics of self-attention.
- **CNNs & Computer Vision:** Experiments in spatial feature extraction.
- **Optimization Prelims:** Mathematical foundations of backpropagation and tensor calculus.

## 📊 Comparative Targets

| Model | Parameters | Architecture | Target Performance |
| :--- | :--- | :--- | :--- |
| GPT-2 (Small) | 124M | Classic Transformer | Baseline |
| **Project 100M** | **100M** | **Llama-Style** | **> GPT-2** |
| MobileLLM | 125M | Modern On-Device | Competitive |
| SmolLM | 135M | Modern On-Device | Competitive |

## ⚙️ Setup

This project uses `pyproject.toml` for streamlined dependency management.

```bash
# Clone the repository
git clone https://github.com/your-username/Learning-DL.git
cd Learning-DL

# Install dependencies
pip install -e .
```

---
*This repository is a continuous experiment in deep learning engineering and model scaling.*

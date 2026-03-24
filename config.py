import torch
from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 4096
    context_length: int = 128
    embedding_dim: int = 256
    num_heads: int = 8
    num_blocks: int = 6
    dropout: float = 0.1
    use_bias: bool = False

    @property
    def head_size(self) -> int:
        return self.embedding_dim // self.num_heads

@dataclass
class TrainerConfig:
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    batch_size: int = 32
    learning_rate: float = 5e-4
    max_steps: int = 10000
    eval_interval: int = 200
    val_steps: int = 50
    show_samples: bool = False
    save_path: str = "model.pt"

import torch
from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 4096
    context_length: int = 512
    embedding_dim: int = 192
    num_heads: int = 6
    num_blocks: int = 24
    dropout: float = 0.1
    use_bias: bool = False

    @property
    def head_size(self) -> int:
        return self.embedding_dim // self.num_heads

@dataclass
class TrainerConfig:
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    batch_size: int = 256
    learning_rate: float = 8e-4
    max_steps: int = 50000
    eval_interval: int = 1000
    val_steps: int = 100
    save_path: str = "model.safetensors"
    
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_ratio: float = 0.05

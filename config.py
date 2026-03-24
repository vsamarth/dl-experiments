import torch
from dataclasses import dataclass

@dataclass
class ModelConfig:
    # "Deep but Thin" Configuration (~45M parameters)
    # Shifts complexity from width (capacity) to depth (logic/composition)
    vocab_size: int = 4096
    context_length: int = 256    # Requested: 256
    embedding_dim: int = 384     # "Thin" width
    num_heads: int = 6           # 384 / 6 = 64 head_size (Hardware friendly)
    num_blocks: int = 24         # "Deep" stack for higher-order logic
    dropout: float = 0.1
    use_bias: bool = False

    @property
    def head_size(self) -> int:
        return self.embedding_dim // self.num_heads

@dataclass
class TrainerConfig:
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # 4090 can handle very large batches with a thin model and short context
    batch_size: int = 256        # Doubled since model is thinner and context is shorter
    learning_rate: float = 6e-4  # Thinner models can often handle slightly higher LRs
    max_steps: int = 50000       
    eval_interval: int = 500
    val_steps: int = 100
    save_path: str = "model.safetensors"
    
    # Optimizer settings
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_ratio: float = 0.05

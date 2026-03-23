import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Centralized configuration for the Tiny-Llama model."""
    vocab_size: int = 50257        # Default for GPT-2
    context_length: int = 8     # Known as BLOCK_SIZE
    embedding_dim: int = 128      # Known as N_EMBED
    num_heads: int = 4            # Number of parallel attention heads
    head_size: int = 32           # embedding_dim // num_heads
    use_bias: bool = False        # Modern LLMs (Llama) prefer bias=False
    dropout: float = 0.1          # Regularization for training
    learning_rate: float = 1e-3
    num_blocks = 3
    batch_size = 4

class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor):
        raise NotImplementedError("This method should be implemented by subclasses.")

class AttentionHead(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.key = nn.Linear(config.embedding_dim, config.head_size, bias=config.use_bias)
        self.query = nn.Linear(config.embedding_dim, config.head_size, bias=config.use_bias)
        self.value = nn.Linear(config.embedding_dim, config.head_size, bias=config.use_bias)
        
        # Buffer registration for device-agnostic masking
        self.register_buffer("mask", torch.tril(torch.ones(config.context_length, config.context_length)))
        self.head_size = config.head_size
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        q = self.query(x) 
        k = self.key(x) 
        v = self.value(x) 

        # Scaled dot-product attention
        attn_scores = q @ k.transpose(-1, -2) / (self.head_size ** 0.5) 
        attn_scores = attn_scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))  # type: ignore
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = attn_weights @ v 
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig): 
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(config) for _ in range(config.num_heads)])
        self.proj = nn.Linear(config.num_heads * config.head_size, config.embedding_dim, bias=config.use_bias)

    
    def forward(self, x: torch.Tensor):
        attn = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.proj(attn)
        return out

class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig): 
        super().__init__()
        hidden_dim = 4 * config.embedding_dim
        self.net = nn.Sequential(
            nn.Linear(config.embedding_dim, hidden_dim, bias=config.use_bias),
            nn.ReLU(),
            nn.Linear(hidden_dim, config.embedding_dim),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config: ModelConfig): 
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embedding_dim)
        self.attn = MultiHeadAttention(config)
        self.ffn = FeedForward(config)

    def forward(self, x):
        x = self.attn(x)
        x = self.ffn(x)
        return x

class Transformer(LanguageModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.num_blocks)])
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size)

    def forward(self, x):
        tok = self.token_embedding(x)
        out = self.blocks(tok)
        return self.lm_head(tok)



class BigramLanguageModel(LanguageModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding_table = nn.Embedding(config.context_length, config.embedding_dim)
        self.attention = MultiHeadAttention(config)
        
        self.lm_head = nn.Linear(config.head_size, config.vocab_size)

    def forward(self, x):
        B, T = x.shape
        
        # Standard token + position embedding logic
        tok_emb = self.token_embedding_table(x) # (B, T, C)
        logits = self.lm_head(tok_emb) # (B, T, vocab_size)
        
        return logits

class Sampler:
    def __init__(self, model: LanguageModel, temperature: float = 1.0):
        self.model = model
        self.temperature = temperature
        self.block_size = model.config.context_length

    def sample(self, context, max_length=100):
        generated = context 
        for _ in range(max_length):
            # 1. FIX: Added cropping logic to prevent index errors
            idx_cond = generated[:, -self.block_size:]
            
            logits = self.model(idx_cond) 
            logits = logits[:, -1, :] / self.temperature 
            
            probs = F.softmax(logits, dim=-1) 
            next_token = torch.multinomial(probs, num_samples=1) 
            generated = torch.cat((generated, next_token), dim=1) 
        return generated
import math
import torch
from torch import nn
from torch.nn import functional as F
from config import ModelConfig

class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        assert config.embedding_dim % config.num_heads == 0
        
        # Key, Query, Value projections in a single batch
        self.c_attn = nn.Linear(config.embedding_dim, 3 * config.embedding_dim, bias=config.use_bias)
        # Output projection
        self.c_proj = nn.Linear(config.embedding_dim, config.embedding_dim, bias=config.use_bias)
        
        # Regularization
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.num_heads
        self.n_embd = config.embedding_dim

    def forward(self, x):
        B, T, C = x.size() # Batch size, Sequence length, Embedding dim

        # Calculate Query, Key, Value for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape to (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.config.dropout if self.training else 0, 
            is_causal=True
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C) # Re-assemble head outputs
        return self.resid_dropout(self.c_proj(y))

class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.embedding_dim, 4 * config.embedding_dim, bias=config.use_bias),
            nn.GELU(),
            nn.Linear(4 * config.embedding_dim, config.embedding_dim, bias=config.use_bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embedding_dim, eps=1e-5)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.embedding_dim, eps=1e-5)
        self.ffn = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_dim)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.num_blocks)])
        self.ln_f = nn.LayerNorm(config.embedding_dim, eps=1e-5)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

        # Weight tying
        self.token_embedding.weight = self.lm_head.weight

        # Professional initialization
        self.apply(self._init_weights)
        # Apply special scaling to residual projections (GPT-2 style)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.num_blocks))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = self.drop(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

class Sampler:
    def __init__(self, model: nn.Module, temperature: float = 1.0):
        self.model = model
        self.temperature = temperature

    @torch.no_grad()
    def sample(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.model.config.context_length:]
            logits, _ = self.model(idx_cond)
            logits = logits[:, -1, :] / self.temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

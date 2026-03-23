import torch
from tqdm.auto import tqdm
import time
from dataclasses import dataclass
from torch import nn, optim
import torch.nn.functional as F
import torchmetrics
from model import ModelConfig, Sampler, LanguageModel
from torch.utils.data import DataLoader

@dataclass
class TrainerConfig:
    """Hyperparameters for the optimization process."""
    learning_rate: float = 5e-4  
    weight_decay: float = 0.1    
    max_grad_norm: float = 1.0  
    total_steps: int = 10000  
    warmup_steps: int = 500       
    eval_interval: int = 1000     
    sample_interval: int = 2000    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True           # Use Automatic Mixed Precision for 5090 speeds


def compute_loss(logits: torch.Tensor, targets: torch.Tensor):
    B, T, C = logits.shape
    return F.cross_entropy(logits.reshape(B * T, C), targets.reshape(B * T))

def train_step(model: nn.Module, batch: dict[str, torch.Tensor], optimizer: optim.Optimizer, metrics: torchmetrics.MetricCollection, device = 'cpu'):
    data = batch["encoded"].to(device)
    x = data[:, :-1]
    y = data[:, 1:]

    optimizer.zero_grad(set_to_none=True)

    logits = model(x)  
    loss = compute_loss(logits, y)

    loss.backward()
    optimizer.step()
    metrics.update(value=loss.item())

def train_model(model: LanguageModel, train_loader: DataLoader, val_loader: DataLoader, m_config: ModelConfig, t_config: TrainerConfig, sampler: Sampler, enc):
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=t_config.learning_rate, 
        weight_decay=t_config.weight_decay
    )
    
    model.to(t_config.device)
    pbar = tqdm(enumerate(train_loader), total=t_config.total_steps, desc="Training")

    for step, batch in pbar:
        if step >= t_config.total_steps: break
        
        x, y = batch[0].to(t_config.device), batch[1].to(t_config.device)
        
        # --- The Training Step ---
        model.train()
        optimizer.zero_grad(set_to_none=True) # More efficient than zero_grad()

        with torch.cuda.amp.autocast(enabled=t_config.use_amp):
            logits = model(x)
            # Reshape for CrossEntropy: (B*T, Vocab)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        # Backprop with Scaler
        scaler.scale(loss).backward()
        
        # Unscale before clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), t_config.max_grad_norm)
        
        # Step optimizer and scaler
        scaler.step(optimizer)
        scaler.update()

        # --- Logging & Maintenance ---
        if step % 100 == 0:
            loss_val = loss.item()
            pbar.set_postfix(loss=f"{loss_val:.4f}")
            
            if torch.isnan(loss):
                print(f"\n💥 Explosion! NaN detected at step {step}")
                break


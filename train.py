import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm
from torchmetrics.aggregation import MeanMetric
from model import Transformer, Sampler
from config import ModelConfig, TrainerConfig
import argparse
import math

class StreamingStoryDataset(IterableDataset):
    """Infinite token stream with specified split."""
    def __init__(self, tokenizer, context_length, split="train"):
        self.ds = load_dataset("roneneldan/TinyStories", streaming=True, split=split)
        self.tokenizer = tokenizer
        self.context_length = context_length

    def __iter__(self):
        buffer = []
        for example in self.ds:
            buffer.extend(self.tokenizer.encode(example["text"]).ids)
            while len(buffer) >= self.context_length + 1:
                chunk = buffer[:self.context_length + 1]
                yield torch.tensor(chunk[:-1]), torch.tensor(chunk[1:])
                buffer = buffer[self.context_length:]

def train_step(model, optimizer, x, y):
    """Executes a single training optimization step."""
    optimizer.zero_grad(set_to_none=True)
    logits, loss = model(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss

@torch.no_grad()
def eval_step(model, x, y):
    """Executes a single evaluation forward pass."""
    _, loss = model(x, y)
    return loss

def validate(model, loader, steps, device):
    """Runs evaluation over a fixed number of validation batches."""
    model.eval()
    loss_metric = MeanMetric().to(device)
    for i, (x, y) in enumerate(loader):
        if i >= steps: break
        loss = eval_step(model, x.to(device), y.to(device))
        loss_metric.update(loss)
    model.train()
    return loss_metric.compute().item()

def train_model(t_config: TrainerConfig):
    # Setup
    tokenizer = Tokenizer.from_file("tokenizer.json")
    m_config = ModelConfig(vocab_size=tokenizer.get_vocab_size())
    model = Transformer(m_config).to(t_config.device)
    if t_config.device == "cuda":
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=t_config.learning_rate,
        weight_decay=0.1,
        betas=(0.9, 0.95)
    )
    train_loader = DataLoader(StreamingStoryDataset(tokenizer, m_config.context_length, "train"), batch_size=t_config.batch_size)
    val_loader = DataLoader(StreamingStoryDataset(tokenizer, m_config.context_length, "validation"), batch_size=t_config.batch_size)
    sampler = Sampler(model)
    
    # Metrics
    train_loss_metric = MeanMetric().to(t_config.device)
    
    print(f"Training on {t_config.device} ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
    
    model.train()
    pbar = tqdm(enumerate(train_loader), total=t_config.max_steps, dynamic_ncols=True)
    
    for step, (x, y) in pbar:
        if step >= t_config.max_steps: break
        
        loss = train_step(model, optimizer, x.to(t_config.device), y.to(t_config.device))
        train_loss_metric.update(loss)
        
        # Logging
        cur_loss = train_loss_metric.compute().item()
        pbar.set_description(f"Loss: {cur_loss:.4f}")
        
        # Periodic Evaluation & Sampling
        if step > 0 and step % t_config.eval_interval == 0:
            val_loss = validate(model, val_loader, t_config.val_steps, t_config.device)
            print(f"\n[Step {step}] Val Loss: {val_loss:.4f}")
            
            if t_config.show_samples:
                model.eval()
                ctx = torch.zeros((1, 1), dtype=torch.long, device=t_config.device)
                out = sampler.sample(ctx, max_new_tokens=50)
                print(f"Sample: {tokenizer.decode(out[0].cpu().numpy())}\n")
                model.train()
            
            train_loss_metric.reset()

    torch.save(model.state_dict(), t_config.save_path)
    print(f"Saved to {t_config.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--samples", action="store_true")
    args = parser.parse_args()
    
    config = TrainerConfig(max_steps=args.steps, show_samples=args.samples)
    train_model(config)

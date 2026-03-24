import torch, argparse, math
from torch import nn, optim
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm
from model import Transformer, Sampler
from config import ModelConfig, TrainerConfig

class StreamingStoryDataset(IterableDataset):
    """Infinite token stream yielding (x, y) pairs."""
    def __init__(self, tokenizer, context_length, split="train"):
        self.ds = load_dataset("roneneldan/TinyStories", streaming=True, split=split)
        self.tok, self.ctx = tokenizer, context_length

    def __iter__(self):
        tokens = []
        eot = self.tok.token_to_id("<|endoftext|>")
        for ex in self.ds:
            tokens.extend(self.tok.encode(ex["text"]).ids + [eot])
            while len(tokens) > self.ctx:
                yield torch.tensor(tokens[:self.ctx]), torch.tensor(tokens[1:self.ctx+1])
                tokens = tokens[self.ctx:]

@torch.no_grad()
def validate(model, loader, steps, device):
    model.eval()
    losses = [model(x.to(device), y.to(device))[1].item() for i, (x, y) in enumerate(loader) if i < steps]
    model.train()
    return sum(losses) / len(losses) if losses else 0

def train(cfg: TrainerConfig):
    # Setup
    tok = Tokenizer.from_file("tokenizer.json")
    model = Transformer(ModelConfig(vocab_size=tok.get_vocab_size())).to(cfg.device)
    if cfg.device == "cuda": model = torch.compile(model)
    
    opt = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=(cfg.beta1, cfg.beta2))
    
    # Scheduler logic
    warmup = int(cfg.warmup_ratio * cfg.max_steps)
    def lr_fn(s):
        if s < warmup: return s / max(1, warmup)
        pct = (s - warmup) / max(1, cfg.max_steps - warmup)
        return 0.1 + 0.9 * (0.5 * (1 + math.cos(math.pi * pct)))
    sched = optim.lr_scheduler.LambdaLR(opt, lr_fn)

    # Data
    train_loader = DataLoader(StreamingStoryDataset(tok, model.config.context_length), batch_size=cfg.batch_size)
    val_loader = DataLoader(StreamingStoryDataset(tok, model.config.context_length, "validation"), batch_size=cfg.batch_size)
    
    print(f"Training {sum(p.numel() for p in model.parameters())/1e6:.1f}M params on {cfg.device}")
    pbar = tqdm(enumerate(train_loader), total=cfg.max_steps, dynamic_ncols=True)
    loss_history, val_loss = [], None

    for step, (x, y) in pbar:
        if step > cfg.max_steps: break
        
        # Train step
        opt.zero_grad(set_to_none=True)
        _, loss = model(x.to(cfg.device), y.to(cfg.device))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        
        loss_history.append(loss.item())
        
        # Eval & Logging
        if step > 0 and (step % cfg.eval_interval == 0 or step == cfg.max_steps):
            val_loss = validate(model, val_loader, cfg.val_steps, cfg.device)
            loss_history = []

        if loss_history:
            desc = f"Loss: {sum(loss_history)/len(loss_history):.4f} | LR: {sched.get_last_lr()[0]:.2e}"
            if val_loss: desc += f" | Val: {val_loss:.4f}"
            pbar.set_description(desc)

    # Wrap up
    print("\n--- Final Sample (500 tokens) ---\n")
    model.eval()
    out = Sampler(model).sample(torch.zeros((1, 1), dtype=torch.long, device=cfg.device), 500)
    print(f"{tok.decode(out[0].cpu().numpy())}\n")
    
    from safetensors.torch import save_file
    save_file(model.state_dict(), cfg.save_path)
    print(f"Saved to {cfg.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    train(TrainerConfig(max_steps=args.steps))

import torch, argparse, math, wandb, os
from torch import nn, optim
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm
from safetensors.torch import save_model, load_model
from model import Transformer, Sampler
from config import ModelConfig, TrainerConfig

class StreamingStoryDataset(IterableDataset):
    """Infinite token stream with shuffling and repetition."""
    def __init__(self, tok, ctx, split="train", epochs=1):
        self.ds = load_dataset("roneneldan/TinyStories", streaming=True, split=split)
        self.ds = self.ds.shuffle(buffer_size=10000, seed=42)
        if epochs > 1: self.ds = self.ds.repeat(epochs)
        self.tok, self.ctx = tok, ctx

    def __iter__(self):
        tokens, eot = [], self.tok.token_to_id("<|endoftext|>")
        for ex in self.ds:
            tokens.extend(self.tok.encode(ex["text"]).ids + [eot])
            while len(tokens) > self.ctx:
                yield torch.tensor(tokens[:self.ctx]), torch.tensor(tokens[1:self.ctx+1])
                tokens = tokens[self.ctx:]

@torch.no_grad()
def validate(model, loader, steps, device):
    model.eval()
    losses, pbar = [], tqdm(enumerate(loader), desc="Eval", leave=False)
    for i, (x, y) in pbar:
        if steps and i >= steps: break
        losses.append(model(x.to(device), y.to(device))[1].item())
    model.train()
    return sum(losses) / len(losses) if losses else 0

def train(cfg: TrainerConfig):
    # Initialization
    tok = Tokenizer.from_file("tokenizer.json")
    m_cfg = ModelConfig(vocab_size=tok.get_vocab_size())
    model = Transformer(m_cfg).to(cfg.device)
    
    if os.path.exists(cfg.save_path):
        print(f"Resuming from {cfg.save_path}...")
        load_model(model, cfg.save_path)
    
    if cfg.device == "cuda": model = torch.compile(model)
    
    opt = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=(cfg.beta1, cfg.beta2))
    warmup = int(cfg.warmup_ratio * cfg.max_steps)
    lr_fn = lambda s: s/warmup if s < warmup else 0.1 + 0.9*(0.5*(1 + math.cos(math.pi*(s-warmup)/(cfg.max_steps-warmup))))
    sched = optim.lr_scheduler.LambdaLR(opt, lr_fn)

    train_loader = DataLoader(StreamingStoryDataset(tok, m_cfg.context_length), batch_size=cfg.batch_size)
    val_loader = DataLoader(StreamingStoryDataset(tok, m_cfg.context_length, "validation"), batch_size=cfg.batch_size)
    
    wandb.init(project="tiny-stories", config={**cfg.__dict__, **m_cfg.__dict__})
    # Log tokenizer as an artifact
    if os.path.exists("tokenizer.json"):
        art = wandb.Artifact("tokenizer", type="model")
        art.add_file("tokenizer.json")
        wandb.log_artifact(art)
    
    print(f"Training {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    pbar = tqdm(range(cfg.max_steps), dynamic_ncols=True)
    train_iter, loss_hist = iter(train_loader), []

    try:
        for step in pbar:
            x, y = next(train_iter)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=cfg.device, dtype=torch.bfloat16):
                _, loss = model(x.to(cfg.device), y.to(cfg.device))
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            loss_hist.append(loss.item())
            metrics = {"train/loss": loss.item(), "train/lr": sched.get_last_lr()[0]}

            if step > 0 and step % cfg.eval_interval == 0:
                val_l = validate(model, val_loader, cfg.val_steps, cfg.device)
                metrics["val/loss"] = val_l
                
                model.eval()
                sample = Sampler(model).sample(torch.zeros((1, 1), dtype=torch.long, device=cfg.device), 100)
                decoded = tok.decode(sample[0].cpu().numpy())
                wandb.log({"samples": wandb.Html(decoded)})
                print(f"\nStep {step} | Val: {val_l:.4f} | Sample: {decoded[:70]}...")
                model.train()
                loss_hist = []

            wandb.log(metrics)
            if loss_hist: pbar.set_description(f"Loss: {sum(loss_hist)/len(loss_hist):.4f}")

        print("\n--- Final Evaluation ---")
        final_loss = validate(model, val_loader, None, cfg.device)
        wandb.run.summary["final_val_loss"] = final_loss
        print(f"Final Val Loss: {final_loss:.4f}")

    finally:
        save_model(model, cfg.save_path)
        # Log model as an artifact
        if os.path.exists(cfg.save_path):
            model_art = wandb.Artifact("model", type="model")
            model_art.add_file(cfg.save_path)
            wandb.log_artifact(model_art)
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    train(TrainerConfig(max_steps=args.steps))

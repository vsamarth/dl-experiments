import os
import argparse
from itertools import islice
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

def train(args):
    """Loads a dataset and trains a BPE tokenizer on a specific column."""
    print(f"Loading '{args.dataset}' (split: {args.split})...")
    ds = load_dataset(args.dataset, streaming=True, split=args.split)
    
    # Initialize Tokenizer & Trainer (GPT-style ByteLevel BPE)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|endoftext|>", "<|pad|>"]
    )
    
    # Training
    corpus = (ex[args.text_column] for ex in islice(ds, args.limit))
    limit_label = args.limit if args.limit else "all"
    print(f"Training on {limit_label} samples (vocab_size={args.vocab_size})...")
    
    try:
        tokenizer.train_from_iterator(corpus, trainer=trainer)
    except Exception as e:
        print(f"Warning: Training interrupted or network error: {e}")
        print("Attempting to save partial results...")
    
    # Save Output
    save_path = os.path.abspath(args.save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    print(f"Tokenizer successfully saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streamlined BPE tokenizer trainer.")
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--vocab_size", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save_path", type=str, default="tokenizer.json")
    
    train(parser.parse_args())

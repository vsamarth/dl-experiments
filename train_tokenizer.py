import argparse
from itertools import islice
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

def train(args):
    print(f"Loading '{args.dataset}' (split: {args.split})...")
    ds = load_dataset(args.dataset, streaming=True, split=args.split)
    
    # Initialize BPE Tokenizer with ByteLevel (GPT-style)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|endoftext|>", "<|pad|>"]
    )
    
    # Train using a streaming generator
    corpus = (ex[args.text_column] for ex in islice(ds, args.limit))
    print(f"Training on up to {args.limit} samples with vocab size {args.vocab_size}...")
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    
    tokenizer.save(args.save_path)
    print(f"Tokenizer successfully saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streamlined BPE tokenizer trainer.")
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories", help="Dataset name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (default: train)")
    parser.add_argument("--text_column", type=str, default="text", help="Column name (default: text)")
    parser.add_argument("--vocab_size", type=int, default=4096, help="Vocab size (default: 4096)")
    parser.add_argument("--limit", type=int, default=100000, help="Max samples (default: 100000)")
    parser.add_argument("--save_path", type=str, default="tokenizer-bpe.json", help="Output path")
    
    train(parser.parse_args())

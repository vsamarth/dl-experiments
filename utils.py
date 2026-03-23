import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import clear_output
import itertools
import tiktoken

def plot_metrics(history):
    sns.set_theme(style="whitegrid")
    prefixes = ['train_', 'test_']
    unique_metrics = sorted(list(set(k.replace(p, '') for k in history for p in prefixes if k.startswith(p))))
    
    num_metrics = len(unique_metrics)
    if num_metrics == 0: return

    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))
    if num_metrics == 1: axes = [axes]
    
    epochs = range(1, len(next(iter(history.values()))) + 1)

    for i, metric_name in enumerate(unique_metrics):
        ax = axes[i]
        for p in prefixes:
            key = f"{p}{metric_name}"
            if key in history and len(history[key]) > 0:
                sns.lineplot(x=epochs, y=history[key], ax=ax, label=p.replace('_', '').capitalize(), marker='o')
        
        ax.set_title(f'Model {metric_name.capitalize()}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name.capitalize())
        ax.legend()

    clear_output(wait=True)
    plt.tight_layout()
    plt.show()



enc = tiktoken.get_encoding("gpt2")

def tokenize(examples):
    texts = [text + "<|endoftext|>" for text in examples["text"]]
    encoded = enc.encode_batch(texts, allowed_special={"<|endoftext|>"})
    return {"encoded": encoded}

def group(examples, block_size=256):
    concat = list(itertools.chain.from_iterable(examples["encoded"]))
    chunk_size = block_size + 1
    total = len(concat)
    if total >= chunk_size:
        total = (total // chunk_size) * chunk_size
    result = [concat[i : i + chunk_size] for i in range(0, total, chunk_size)]
    return {"encoded": result}
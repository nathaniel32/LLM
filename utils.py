import torch
import config
from torch.nn import functional as F

def prepare_data():
    with open('data/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Train and test splits
    n = int(0.9*len(text)) # first 90% will be train, rest val
    train_data = text[:n]
    val_data = text[n:]

    return train_data, val_data

# data loading
def get_batch(data):
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    x, y = x.to(config.device), y.to(config.device)
    return x, y

@torch.no_grad()
def estimate_loss(model, data):
    model.eval()
    losses = torch.zeros(config.eval_iters)
    for k in range(config.eval_iters):
        X, Y = get_batch(data)
        logits, loss = model(X, Y)
        losses[k] = loss.item()
    loss_avg = losses.mean()
    model.train()
    return loss_avg

def generate(model, idx, max_new_tokens, tokenizer=None):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # crop idx to the last block_size tokens
        idx_cond = idx[:, -config.block_size:]
        # get the predictions
        logits, loss = model(idx_cond)
        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1) # (B, C)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        if tokenizer:
            text_so_far = tokenizer.detokenize(idx[0].tolist())
            print(text_so_far, end="\r", flush=True)
    return idx
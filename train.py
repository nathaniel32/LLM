import os
import requests
import tiktoken
import numpy as np
import torch
from contextlib import nullcontext
from model import GPTConfig, GPT
import math
import time

class Train:
    def __init__(self, config:GPTConfig, data_url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", data_dir="datasets/shakespeare"):
        seed = 1337
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = 'bfloat16'
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        self.ctx = nullcontext() if self.device == 'cpu' else torch.amp.autocast(device_type=self.device, dtype=ptdtype)

        self.out_dir = 'out'
        self.data_dir = data_dir
        self.config = config
        self.batch_size = 1
        self.learning_rate = 6e-4
        self.eval_iters = 200

        self.prepare_dataset(data_url=data_url)

    def prepare_dataset(self, data_url):
        if os.path.exists(os.path.join(self.data_dir, 'train.bin')):
            print("Datasets found!")
            return
        
        os.makedirs(self.data_dir, exist_ok=True)
        input_file_path = os.path.join(self.data_dir, 'input.txt')
        if not os.path.exists(input_file_path):
            with open(input_file_path, 'w', encoding='utf-8') as f:
                f.write(requests.get(data_url).text)

        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = f.read()
        n = len(data)
        train_data = data[:int(n*0.9)]
        val_data = data[int(n*0.9):]

        # encode with tiktoken gpt2 bpe
        enc = tiktoken.get_encoding("gpt2")
        train_ids = enc.encode_ordinary(train_data)
        val_ids = enc.encode_ordinary(val_data)
        print(f"train has {len(train_ids):,} tokens")
        print(f"val has {len(val_ids):,} tokens")

        # export to bin files
        train_ids = np.array(train_ids, dtype=np.uint16)
        val_ids = np.array(val_ids, dtype=np.uint16)

        train_ids.tofile(os.path.join(self.data_dir, 'train.bin'))
        val_ids.tofile(os.path.join(self.data_dir, 'val.bin'))

    def get_batch(self, split):
        # np.memmap every batch to avoid a memory leak
        if split == 'train':
            data = np.memmap(os.path.join(self.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(self.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - self.config.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+self.config.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+self.config.block_size]).astype(np.int64)) for i in ix])
        if self.device == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y
    
    def get_model(self, attn_type="mha", resume=False):
        weight_decay = 1e-1
        beta1 = 0.9
        beta2 = 0.95

        ckpt_path = os.path.join(self.out_dir, attn_type, 'ckpt.pt')
        if not os.path.exists(ckpt_path):
            print("Checkpoint not found!")
            resume = False

        if resume:
            print(f"Resuming training from {self.out_dir}")
            
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            checkpoint_model_args = checkpoint['model_args']
            self.config = GPTConfig(
                block_size=checkpoint_model_args["block_size"],
                vocab_size=checkpoint_model_args["vocab_size"],
                n_layer=checkpoint_model_args["n_layer"],
                n_head=checkpoint_model_args["n_head"],
                n_embd=checkpoint_model_args["n_embd"],
                dropout=checkpoint_model_args["dropout"],
                bias=checkpoint_model_args["bias"])
            state_dict = checkpoint['model']
        else:
            print("Initializing a new model from scratch")
        
        model = GPT(self.config, attn_type)
        model.to(self.device)
        optimizer = model.configure_optimizers(weight_decay, self.learning_rate, (beta1, beta2), self.device)
        
        if resume:
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
            iter_num = checkpoint['iter_num']
            best_val_loss = checkpoint['best_val_loss']

            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            iter_num = 0
            best_val_loss = float('inf')
        
        return model, optimizer, iter_num, best_val_loss
        
    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self, it):
        warmup_iters = 2000
        lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
        min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return self.learning_rate * (it + 1) / (warmup_iters + 1)
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (self.learning_rate - min_lr)
    
    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(self, model):
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self.get_batch(split)
                with self.ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
    
    def train(self, attn_type="mha", resume=True):
        model, optimizer, iter_num, best_val_loss = self.get_model(attn_type=attn_type, resume=resume)

        X, Y = self.get_batch('train')
        
        t0 = time.time()
        local_iter_num = 0
        running_mfu = -1.0
        decay_lr = True # whether to decay the learning rate
        eval_interval = 10
        always_save_checkpoint = True
        eval_only = False # if True, script exits right after the first eval
        gradient_accumulation_steps = 5 * 8
        scaler = torch.amp.GradScaler(enabled=(self.dtype == 'float16'))
        grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
        log_interval = 1
        max_iters = 2000 # total number of training iterations

        while iter_num < max_iters:
            # determine and set the learning rate for this iteration
            lr = self.get_lr(iter_num) if decay_lr else self.learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            if iter_num % eval_interval == 0:
                losses = self.estimate_loss(model)
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

                print({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100, # convert to percentage
                })
                if losses['val'] < best_val_loss or always_save_checkpoint:
                    best_val_loss = losses['val']
                    if iter_num > 0 or always_save_checkpoint:
                        checkpoint = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'model_args': self.config.to_dict(),
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss
                        }
                        save_dir = os.path.join(self.out_dir, attn_type)
                        print(f"saving checkpoint to {save_dir}")
                        
                        os.makedirs(save_dir, exist_ok=True)
                        torch.save(checkpoint, os.path.join(save_dir, 'ckpt.pt'))
                        print("Checkpoint saved successfully.")

            if iter_num == 0 and eval_only:
                break

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(gradient_accumulation_steps):
                with self.ctx:
                    logits, loss = model(X, Y)
                    loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y = self.get_batch('train')
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()

            # clip the gradient
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()

            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % log_interval == 0:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * gradient_accumulation_steps
                if local_iter_num >= 5: # let the training loop settle a bit
                    mfu = model.estimate_mfu(self.batch_size * gradient_accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

            iter_num += 1
            local_iter_num += 1

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=bool, default=True)
parser.add_argument("--attn_type", type=str, default="mha")
args = parser.parse_args()
print(vars(args))

config = GPTConfig(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768, dropout=0.0, bias=True)
#config = GPTConfig(block_size=1024, vocab_size=50257, n_layer=24, n_head=16, n_embd=1536, dropout=0.0, bias=True)

train = Train(config=config, data_url="https://raw.githubusercontent.com/uwgraphics/VEP2_TCP_SimpleText/refs/heads/main/N3/N37535.txt", data_dir="datasets/simple_text")
train.train(attn_type=args.attn_type, resume=args.resume)
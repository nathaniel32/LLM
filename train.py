import os
import requests
import tiktoken
import numpy as np
import torch
from contextlib import nullcontext
from model import ModelConfig, GPT
import math
import time
import env
from env import AttnType, PosType
from logger import Logger
from dataclasses import asdict

class Train:
    def __init__(self, model_type, attn_type:AttnType, dataset_type, pos_type:PosType):
        seed = 1337
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = 'float32'
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        self.ctx = nullcontext() if self.device == 'cpu' else torch.amp.autocast(device_type=self.device, dtype=ptdtype)

        self.attn_type:AttnType = attn_type
        self.pos_type:PosType = pos_type

        self.out_dir = os.path.join('out', model_type, attn_type.value, pos_type.value)
        self.data_dir = os.path.join('datasets', dataset_type)
        
        self.logger = Logger(out_dir=self.out_dir)
        self.config = ModelConfig(**env.model_configs[model_type])
        
        self.batch_size = 1
        self.learning_rate = 6e-4
        self.eval_iters = 200
        
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        self.prepare_dataset(data_url=env.dataset_configs[dataset_type])

    def prepare_dataset(self, data_url):
        if os.path.exists(os.path.join(self.data_dir, 'train.bin')):
            print("Datasets found!")
            return
        
        os.makedirs(self.data_dir, exist_ok=True)
        input_file_path = os.path.join(self.data_dir, 'input.txt')
        if not os.path.exists(input_file_path):
            with open(input_file_path, 'w', encoding='utf-8') as f:
                f.write(requests.get(data_url, headers=self.headers).text)

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
    
    def get_model(self, resume=False):
        weight_decay = 1e-1
        beta1 = 0.9
        beta2 = 0.95

        ckpt_path = os.path.join(self.out_dir, 'ckpt.pt')
        if not os.path.exists(ckpt_path):
            print("Checkpoint not found!")
            resume = False

        if resume:
            print(f"Resuming training from {self.out_dir}")
            
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            checkpoint_model_args = checkpoint['model_args']
            self.config = ModelConfig(
                block_size=checkpoint_model_args["block_size"],
                vocab_size=checkpoint_model_args["vocab_size"],
                n_layer=checkpoint_model_args["n_layer"],
                n_head=checkpoint_model_args["n_head"],
                n_embd=checkpoint_model_args["n_embd"],
                dropout=checkpoint_model_args["dropout"],
                bias=checkpoint_model_args["bias"],
                gqa_kv_head=checkpoint_model_args["gqa_kv_head"])
            state_dict = checkpoint['model']
            self.attn_type = checkpoint['attn_type']
            self.pos_type = checkpoint['pos_type']
            
            print({'attn_type': self.attn_type.value, 'pos_type': self.pos_type.value})
        else:
            print("Initializing a new model from scratch")
        
        model = GPT(self.config, self.attn_type, self.pos_type)
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
        
        self.logger.set_meta({
            "attn_type": self.attn_type.value,
            "pos_type": self.pos_type.value,
            "param": model.get_num_params(),
            **asdict(self.config)
        })

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
    
    def train(self, resume=True):
        model, optimizer, iter_num, best_val_loss = self.get_model(resume=resume)

        X, Y = self.get_batch('train')
        
        t0 = time.time()
        
        decay_lr = True # whether to decay the learning rate
        gradient_accumulation_steps = 5 * 8
        scaler = torch.amp.GradScaler(enabled=(self.dtype == 'float16'))
        grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
        
        log_interval = 1
        eval_interval = 10
        max_iters = 2000 # total number of training iterations
        patience = 20

        patience_counter = 0
        local_iter_num = 0
        running_mfu = -1.0
        
        while iter_num < max_iters:
            # determine and set the learning rate for this iteration
            lr = self.get_lr(iter_num) if decay_lr else self.learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            if iter_num % eval_interval == 0:
                losses = self.estimate_loss(model)
                
                if losses['val'] < best_val_loss:
                    best_val_loss = losses['val']
                    patience_counter = 0
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': asdict(self.config),
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'attn_type': self.attn_type,
                        'pos_type': self.pos_type
                    }
                    save_dir = os.path.join(self.out_dir)
                    print(f"saving checkpoint to {save_dir}")
                    
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(checkpoint, os.path.join(save_dir, 'ckpt.pt'))
                    print("Checkpoint saved successfully.")
                else:
                    patience_counter += 1

                self.logger.log(category="val_log", key=iter_num, metrics={
                    "iter": iter_num,
                    "patience": patience_counter,
                    "train_loss": float(losses['train']),
                    "val_loss": float(losses['val']),
                    'best_val_loss': float(best_val_loss) if best_val_loss != float('inf') else None,
                    "lr": lr,
                    "mfu_percent": running_mfu*100
                })

                if patience_counter >= patience:
                    print(f"Early stopping triggered at iter {iter_num} after {patience_counter} evaluations without improvement.")
                    break
                else:
                    print(f"Patience: {patience_counter}/{patience}")

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
                
                self.logger.log(category="train_log", key=iter_num, metrics={
                    "iter": iter_num,
                    "train_loss": float(lossf),
                    "time_ms": dt*1000,
                    "mfu_percent": running_mfu*100
                })

            iter_num += 1
            local_iter_num += 1

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="small")
parser.add_argument("--attn_type", type=str, default="mha")
parser.add_argument("--pos_type", type=str, default="wpe")
parser.add_argument("--dataset_type", type=str, default="shakespeare")
parser.add_argument("--no-resume", action="store_false", dest="resume", default=True)
args = parser.parse_args()
print(vars(args))

train = Train(model_type=args.model_type, attn_type=AttnType(args.attn_type), dataset_type=args.dataset_type, pos_type=PosType(args.pos_type))
train.train(resume=args.resume)
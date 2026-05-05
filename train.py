import os
import numpy as np
import torch
from contextlib import nullcontext
from model import Transformer
import math
import time
from config import Configs, args, args_configs
from logger import Logger
from dataclasses import asdict
from dataclasses import dataclass
import random
from utils import set_seed

@dataclass
class TrainState:
    iter_num: int = 0
    best_val_loss: float = float('inf')
    patience_counter: int = 0

    def info(self):
        return {"iter_num": self.iter_num, "best_val_loss": self.best_val_loss, "patience_counter": self.patience_counter}

class Train:
    def __init__(self, configs:Configs):
        set_seed()
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

        self.configs = configs
        self.train_state = TrainState()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.configs.train_type.value.dtype]
        self.ctx = nullcontext() if self.device == 'cpu' else torch.amp.autocast(device_type=self.device, dtype=ptdtype)
        
        self.logger = Logger(out_dir=configs.out_dir)
        
        self.configs.dataset_type.value.prepare_dataset()
    
    def get_batch(self, split):
        # np.memmap every batch to avoid a memory leak
        if split == 'train':
            data = np.memmap(os.path.join(self.configs.dataset_type.value.root_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(self.configs.dataset_type.value.root_dir, 'val.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - self.configs.model_type.value.block_size, (self.configs.train_type.value.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+self.configs.model_type.value.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+self.configs.model_type.value.block_size]).astype(np.int64)) for i in ix])
        if self.device == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y
    
    def get_model(self, resume=False, filename='last_checkpoint.pt'):
        ckpt_path = os.path.join(self.configs.out_dir, filename)
        if not os.path.exists(ckpt_path):
            print("Checkpoint not found!")
            resume = False

        scaler = torch.amp.GradScaler(enabled=(self.configs.train_type.value.dtype == 'float16'))

        if resume:
            print(f"Resuming training from {self.configs.out_dir}")
            
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.configs = Configs(**checkpoint['args'])
            self.train_state = TrainState(**checkpoint['state'])
            
            print(self.configs.info())
            print(self.train_state.info())
        else:
            print("Initializing a new model from scratch")
        
        model = Transformer(self.configs)
        model.to(self.device)
        optimizer = model.configure_optimizers(self.configs.train_type.value.weight_decay, self.configs.train_type.value.learning_rate, (self.configs.train_type.value.beta1, self.configs.train_type.value.beta2), self.device)
        
        if resume:
            state_dict = checkpoint['model']

            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
            model.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            
            torch.set_rng_state(checkpoint['rng_state_torch'].cpu())
            np.random.set_state(checkpoint['rng_state_numpy'])
            random.setstate(checkpoint['rng_state_python'])
            if checkpoint['rng_state_cuda'] is not None and self.device == 'cuda' and torch.cuda.is_available():
                rng_states = [s.cpu() for s in checkpoint['rng_state_cuda']]
                torch.cuda.set_rng_state_all(rng_states)
        
        self.logger.set_meta({"params": model.get_num_params(), **self.configs.to_dict()})
        
        print(self.configs.info())
        print(f"Total Params: {model.get_num_params()/1e6:.2f}M")

        return model, optimizer, scaler
        
    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.configs.train_type.value.warmup_iters:
            return self.configs.train_type.value.learning_rate * (it + 1) / (self.configs.train_type.value.warmup_iters + 1)
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.configs.train_type.value.lr_decay_iters:
            return self.configs.train_type.value.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.configs.train_type.value.warmup_iters) / (self.configs.train_type.value.lr_decay_iters - self.configs.train_type.value.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return self.configs.train_type.value.min_lr + coeff * (self.configs.train_type.value.learning_rate - self.configs.train_type.value.min_lr)
    
    def calculate_diagnostics(self, model):
        diag = {}
        
        # 1. L2 Weight Norm
        total_w_norm = 0.0
        for p in model.parameters():
            total_w_norm += p.data.norm(2).item() ** 2
        diag['weight_norm'] = total_w_norm ** 0.5
        
        # Global Grad Norm
        total_g_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_g_norm += p.grad.detach().data.norm(2).item() ** 2
        diag['grad_norm'] = total_g_norm ** 0.5

        # WPE Specific Grad Norm (Penting untuk melihat seberapa aktif posisi dipelajari)
        wpe_g_norm = 0.0
        if hasattr(model.transformer, 'wpe') and model.transformer.wpe is not None:
            for p in model.transformer.wpe.parameters():
                if p.grad is not None:
                    wpe_g_norm += p.grad.detach().data.norm(2).item() ** 2
        diag['wpe_grad_norm'] = wpe_g_norm ** 0.5

        # 3. VRAM Usage (GB)
        if self.device == 'cuda':
            diag['vram_gb'] = torch.cuda.max_memory_allocated() / (1024**3)
            
        return diag

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_metrics(self, model):
        out = {}
        model.eval()
        
        for split in ['train', 'val']:
            # tensor untuk menyimpan metrik per iterasi
            losses = torch.zeros(self.configs.train_type.value.eval_iters)
            accuracies = torch.zeros(self.configs.train_type.value.eval_iters)
            
            for k in range(self.configs.train_type.value.eval_iters):
                X, Y = self.get_batch(split)
                
                with self.ctx:
                    logits, loss = model(X, Y)
                    
                # Menyimpan nilai loss
                losses[k] = loss.item()
                
                # probabilitas tertinggi
                predictions = torch.argmax(logits, dim=-1)
                
                # Menghitung akurasi batch
                correct = (predictions == Y).sum().item()
                total = Y.numel() 
                accuracies[k] = correct / total

            # rata-rata
            mean_loss = losses.mean().item()
            mean_acc = accuracies.mean().item()
            
            out[f'{split}_loss'] = mean_loss
            out[f'{split}_perplexity'] = torch.exp(torch.tensor(mean_loss)).item()
            out[f'{split}_accuracy'] = mean_acc
            
        model.train()
        return out
    
    def save_model(self, model, optimizer, scaler, filename):
        if self.configs.train_type.value.save_ckpt:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'args': asdict(self.configs),
                'state': asdict(self.train_state),
                'rng_state_torch': torch.get_rng_state(),
                'rng_state_numpy': np.random.get_state(),
                'rng_state_python': random.getstate(),
                'rng_state_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }

            os.makedirs(self.configs.out_dir, exist_ok=True)

            path = os.path.join(self.configs.out_dir, filename)
            print(f"saving checkpoint to {path}")
            
            torch.save(checkpoint, path)
            print("Checkpoint saved successfully.")
        else:
            print("save_ckpt:", self.configs.train_type.value.save_ckpt)
    
    def train(self, resume=True):
        model, optimizer, scaler = self.get_model(resume=resume)
        self.configs.in_training = True
        
        if not resume:
            set_seed()

        X, Y = self.get_batch('train')
        
        local_iter_num = 0
        running_mfu = -1.0
        
        while True:
            # determine and set the learning rate for this iteration
            lr = self.get_lr(self.train_state.iter_num) if self.configs.train_type.value.decay_lr else self.configs.train_type.value.learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            if self.train_state.iter_num % self.configs.train_type.value.eval_interval == 0 or self.train_state.iter_num == self.configs.train_type.value.max_iters:
                metrics = self.estimate_metrics(model)

                self.save_model(model, optimizer, scaler, filename='last_checkpoint.pt')
                
                if metrics['val_loss'] < self.train_state.best_val_loss:
                    self.train_state.best_val_loss = metrics['val_loss']
                    self.train_state.patience_counter = 0
                    self.save_model(model, optimizer, scaler, filename='best_checkpoint.pt')
                else:
                    self.train_state.patience_counter += 1

                self.logger.log(category="val_log", key="iter", metrics={
                    "iter": self.train_state.iter_num,
                    "patience": self.train_state.patience_counter,
                    'best_val_loss': float(self.train_state.best_val_loss) if self.train_state.best_val_loss != float('inf') else None,
                    "lr": lr,
                    **metrics
                })

                if self.configs.train_type.value.patience is not None:
                    if self.train_state.patience_counter >= self.configs.train_type.value.patience:
                        print(f"Early stopping triggered at iter {self.train_state.iter_num} after {self.train_state.patience_counter} evaluations without improvement.")
                        break
                    else:
                        print(f"Patience: {self.train_state.patience_counter}/{self.configs.train_type.value.patience}")

                if self.train_state.iter_num == self.configs.train_type.value.max_iters:
                    print(f"Reached max iterations: {self.train_state.iter_num}. Stopping training!")
                    break

            previous_time = time.time()

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(self.configs.train_type.value.gradient_accumulation_steps):
                with self.ctx:
                    logits, loss = model(X, Y)
                    loss = loss / self.configs.train_type.value.gradient_accumulation_steps # scale the loss to account for gradient accumulation
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y = self.get_batch('train')
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            diagnostics = self.calculate_diagnostics(model)

            # clip the gradient
            if self.configs.train_type.value.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.configs.train_type.value.grad_clip)                

            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()

            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            # timing and logging
            current_time = time.time()
            delta_time = current_time - previous_time
            
            if self.train_state.iter_num % self.configs.train_type.value.log_interval == 0:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * self.configs.train_type.value.gradient_accumulation_steps
                if local_iter_num >= 5: # let the training loop settle a bit
                    mfu = model.estimate_mfu(self.configs.train_type.value.batch_size * self.configs.train_type.value.gradient_accumulation_steps, delta_time)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                
                self.logger.log(category="train_log", key="iter", metrics={
                    "iter": self.train_state.iter_num,
                    "train_loss": float(lossf),
                    "time_ms": delta_time*1000,
                    "mfu_percent": running_mfu * 100 if running_mfu >= 0 else None,
                    **diagnostics
                })

            self.train_state.iter_num += 1
            local_iter_num += 1

train = Train(configs=args_configs)
train.train(resume=args.resume)
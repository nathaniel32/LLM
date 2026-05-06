import os
import numpy as np
import torch
from contextlib import nullcontext
import math
import time
from config import Configs, args, args_configs
from utils import set_seed, ModelContext

class Train:
    def __init__(self, configs:Configs):
        set_seed()
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

        self.model_context = ModelContext(configs=configs)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.model_context.configs.train_type.value.dtype]
        self.ctx = nullcontext() if self.device == 'cpu' else torch.amp.autocast(device_type=self.device, dtype=ptdtype)
             
        self.model_context.configs.dataset_type.value.prepare_dataset()
    
    def get_batch(self, split):
        # np.memmap every batch to avoid a memory leak
        if split == 'train':
            data = np.memmap(os.path.join(self.model_context.configs.dataset_type.value.root_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(self.model_context.configs.dataset_type.value.root_dir, 'val.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - self.model_context.configs.model_type.value.block_size, (self.model_context.configs.train_type.value.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+self.model_context.configs.model_type.value.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+self.model_context.configs.model_type.value.block_size]).astype(np.int64)) for i in ix])
        if self.device == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.model_context.configs.train_type.value.warmup_iters:
            return self.model_context.configs.train_type.value.learning_rate * (it + 1) / (self.model_context.configs.train_type.value.warmup_iters + 1)
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.model_context.configs.train_type.value.lr_decay_iters:
            return self.model_context.configs.train_type.value.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.model_context.configs.train_type.value.warmup_iters) / (self.model_context.configs.train_type.value.lr_decay_iters - self.model_context.configs.train_type.value.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return self.model_context.configs.train_type.value.min_lr + coeff * (self.model_context.configs.train_type.value.learning_rate - self.model_context.configs.train_type.value.min_lr)
    
    def calculate_diagnostics(self):
        diag = {}
        
        # 1. L2 Weight Norm
        total_w_norm = 0.0
        for p in self.model_context.model.parameters():
            total_w_norm += p.data.norm(2).item() ** 2
        diag['weight_norm'] = total_w_norm ** 0.5
        
        # Global Grad Norm
        total_g_norm = 0.0
        for p in self.model_context.model.parameters():
            if p.grad is not None:
                total_g_norm += p.grad.detach().data.norm(2).item() ** 2
        diag['grad_norm'] = total_g_norm ** 0.5

        # WPE Specific Grad Norm
        wpe_g_norm = 0.0
        if hasattr(self.model_context.model.transformer, 'wpe') and self.model_context.model.transformer.wpe is not None:
            for p in self.model_context.model.transformer.wpe.parameters():
                if p.grad is not None:
                    wpe_g_norm += p.grad.detach().data.norm(2).item() ** 2
        diag['wpe_grad_norm'] = wpe_g_norm ** 0.5

        # 3. VRAM Usage (GB)
        if self.device == 'cuda':
            diag['vram_gb'] = torch.cuda.max_memory_allocated() / (1024**3)
            
        return diag

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_metrics(self):
        out = {}
        self.model_context.model.eval()
        
        for split in ['train', 'val']:
            losses = torch.zeros(self.model_context.configs.train_type.value.eval_iters)
            accuracies = torch.zeros(self.model_context.configs.train_type.value.eval_iters)
            
            for k in range(self.model_context.configs.train_type.value.eval_iters):
                X, Y = self.get_batch(split)
                
                with self.ctx:
                    logits, loss = self.model_context.model(X, Y)
                    
                losses[k] = loss.item()
                
                predictions = torch.argmax(logits, dim=-1)
                correct = (predictions == Y).sum().item()
                total = Y.numel()
                accuracies[k] = correct / total

            mean_loss = losses.mean().item()
            mean_acc = accuracies.mean().item()
            
            out[f'{split}_loss'] = mean_loss
            out[f'{split}_perplexity'] = torch.exp(torch.tensor(mean_loss)).item()
            out[f'{split}_accuracy'] = mean_acc
            
        self.model_context.model.train()
        return out
    
    def train(self, resume=True):
        self.model_context.load_model(resume=resume, device=self.device)
        
        self.model_context.configs.in_training = True
        
        X, Y = self.get_batch('train')
        
        local_iter_num = 0
        running_mfu = -1.0
        
        while True:
            # determine and set the learning rate for this iteration
            lr = self.get_lr(self.model_context.train_state.iter_num) if self.model_context.configs.train_type.value.decay_lr else self.model_context.configs.train_type.value.learning_rate
            for param_group in self.model_context.optimizer.param_groups:
                param_group['lr'] = lr

            if self.model_context.train_state.iter_num % self.model_context.configs.train_type.value.eval_interval == 0 or self.model_context.train_state.iter_num == self.model_context.configs.train_type.value.max_iters:
                metrics = self.estimate_metrics()

                self.model_context.save_model(filename='last_checkpoint.pt')

                if metrics['val_loss'] < self.model_context.train_state.best_val_loss:
                    self.model_context.train_state.best_val_loss = metrics['val_loss']
                    self.model_context.train_state.patience_counter = 0
                    self.model_context.save_model(filename='best_checkpoint.pt')
                else:
                    self.model_context.train_state.patience_counter += 1

                self.model_context.logger.log(category="val_log", key="iter", metrics={
                    "iter": self.model_context.train_state.iter_num,
                    "patience": self.model_context.train_state.patience_counter,
                    'best_val_loss': float(self.model_context.train_state.best_val_loss) if self.model_context.train_state.best_val_loss != float('inf') else None,
                    "lr": lr,
                    **metrics
                })

                if self.model_context.configs.train_type.value.patience is not None:
                    if self.model_context.train_state.patience_counter >= self.model_context.configs.train_type.value.patience:
                        print(f"Early stopping triggered at iter {self.model_context.train_state.iter_num} after {self.model_context.train_state.patience_counter} evaluations without improvement.")
                        break
                    else:
                        print(f"Patience: {self.model_context.train_state.patience_counter}/{self.model_context.configs.train_type.value.patience}")

                if self.model_context.train_state.iter_num == self.model_context.configs.train_type.value.max_iters:
                    print(f"Reached max iterations: {self.model_context.train_state.iter_num}. Stopping training!")
                    break

            previous_time = time.time()

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(self.model_context.configs.train_type.value.gradient_accumulation_steps):
                with self.ctx:
                    logits, loss = self.model_context.model(X, Y)
                    loss = loss / self.model_context.configs.train_type.value.gradient_accumulation_steps
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y = self.get_batch('train')
                # backward pass, with gradient scaling if training in fp16
                self.model_context.scaler.scale(loss).backward()

            self.model_context.scaler.unscale_(self.model_context.optimizer)
            diagnostics = self.calculate_diagnostics()

            # clip the gradient
            if self.model_context.configs.train_type.value.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model_context.model.parameters(), self.model_context.configs.train_type.value.grad_clip)

            # step the optimizer and scaler if training in fp16
            self.model_context.scaler.step(self.model_context.optimizer)
            self.model_context.scaler.update()

            # flush the gradients as soon as we can, no need for this memory anymore
            self.model_context.optimizer.zero_grad(set_to_none=True)

            # timing and logging
            current_time = time.time()
            delta_time = current_time - previous_time
            
            if self.model_context.train_state.iter_num % self.model_context.configs.train_type.value.log_interval == 0:
                lossf = loss.item() * self.model_context.configs.train_type.value.gradient_accumulation_steps
                if local_iter_num >= 5:
                    mfu = self.model_context.model.estimate_mfu(self.model_context.configs.train_type.value.batch_size * self.model_context.configs.train_type.value.gradient_accumulation_steps, delta_time)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                
                self.model_context.logger.log(category="train_log", key="iter", metrics={
                    "iter": self.model_context.train_state.iter_num,
                    "train_loss": float(lossf),
                    "time_ms": delta_time*1000,
                    "mfu_percent": running_mfu * 100 if running_mfu >= 0 else None,
                    **diagnostics
                })

            self.model_context.train_state.iter_num += 1
            local_iter_num += 1

train = Train(configs=args_configs)
train.train(resume=args.resume)
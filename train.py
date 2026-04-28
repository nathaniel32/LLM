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

class Train:
    def __init__(self, configs:Configs):
        self.set_seed(1337)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

        self.configs = configs

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.configs.train_type.value.dtype]
        self.ctx = nullcontext() if self.device == 'cpu' else torch.amp.autocast(device_type=self.device, dtype=ptdtype)
        
        self.logger = Logger(out_dir=configs.out_dir)
        
        self.configs.dataset_type.value.prepare_dataset()

    def set_seed(self, seed=1337):
        import random
        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
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
    
    def get_model(self, resume=False):
        ckpt_path = os.path.join(self.configs.out_dir, 'ckpt.pt')
        if not os.path.exists(ckpt_path):
            print("Checkpoint not found!")
            resume = False

        if resume:
            print(f"Resuming training from {self.configs.out_dir}")
            
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.configs = Configs(**checkpoint['args'])
            state_dict = checkpoint['model']
            
            print(self.configs.to_dict())
        else:
            print("Initializing a new model from scratch")
        
        model = Transformer(self.configs)
        model.to(self.device)
        optimizer = model.configure_optimizers(self.configs.train_type.value.weight_decay, self.configs.train_type.value.learning_rate, (self.configs.train_type.value.beta1, self.configs.train_type.value.beta2), self.device)
        
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
            "param": model.get_num_params(),
            **self.configs.to_dict()
        })

        print({'iter_num':iter_num, 'best_val_loss':best_val_loss})

        return model, optimizer, iter_num, best_val_loss
        
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
    
    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(self, model):
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.configs.train_type.value.eval_iters)
            for k in range(self.configs.train_type.value.eval_iters):
                X, Y = self.get_batch(split)
                with self.ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
    
    def train(self, resume=True):
        model, optimizer, iter_num, best_val_loss = self.get_model(resume=resume)

        if not resume:
            self.set_seed(1337)

        X, Y = self.get_batch('train')
        
        scaler = torch.amp.GradScaler(enabled=(self.configs.train_type.value.dtype == 'float16'))
        
        patience_counter = 0
        local_iter_num = 0
        running_mfu = -1.0
        
        while iter_num < self.configs.train_type.value.max_iters:
            # determine and set the learning rate for this iteration
            lr = self.get_lr(iter_num) if self.configs.train_type.value.decay_lr else self.configs.train_type.value.learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            if iter_num % self.configs.train_type.value.eval_interval == 0:
                losses = self.estimate_loss(model)
                
                if losses['val'] < best_val_loss:
                    best_val_loss = losses['val']
                    patience_counter = 0
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'args': asdict(self.configs),
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss
                    }

                    print(f"saving checkpoint to {self.configs.out_dir}")
                    
                    os.makedirs(self.configs.out_dir, exist_ok=True)
                    torch.save(checkpoint, os.path.join(self.configs.out_dir, 'ckpt.pt'))
                    print("Checkpoint saved successfully.")
                else:
                    patience_counter += 1

                self.logger.log(category="val_log", key="iter", metrics={
                    "iter": iter_num,
                    "patience": patience_counter,
                    "train_loss": float(losses['train']),
                    "val_loss": float(losses['val']),
                    'best_val_loss': float(best_val_loss) if best_val_loss != float('inf') else None,
                    "lr": lr
                })

                if self.configs.train_type.value.patience is not None:
                    if patience_counter >= self.configs.train_type.value.patience:
                        print(f"Early stopping triggered at iter {iter_num} after {patience_counter} evaluations without improvement.")
                        break
                    else:
                        print(f"Patience: {patience_counter}/{self.configs.train_type.value.patience}")

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

            # clip the gradient
            if self.configs.train_type.value.grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.configs.train_type.value.grad_clip)

            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()

            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            # timing and logging
            current_time = time.time()
            delta_time = current_time - previous_time
            
            if iter_num % self.configs.train_type.value.log_interval == 0:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * self.configs.train_type.value.gradient_accumulation_steps
                if local_iter_num >= 5: # let the training loop settle a bit
                    mfu = model.estimate_mfu(self.configs.train_type.value.batch_size * self.configs.train_type.value.gradient_accumulation_steps, delta_time)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                
                self.logger.log(category="train_log", key="iter", metrics={
                    "iter": iter_num,
                    "train_loss": float(lossf),
                    "time_ms": delta_time*1000,
                    "mfu_percent": running_mfu * 100 if running_mfu >= 0 else None
                })

            iter_num += 1
            local_iter_num += 1

print(vars(args))

train = Train(configs=args_configs)
train.train(resume=args.resume)
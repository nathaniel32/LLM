import random
import torch
import numpy as np
import os
from dataclasses import asdict
from dataclasses import dataclass
import random
from typing import Optional
from model import Transformer
from config import Configs, AttnType, PosType, NormType

def set_seed(seed=1234):        
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

@dataclass
class TrainState:
    iter_num: int = 0
    best_val_loss: float = float('inf')
    patience_counter: int = 0

    def info(self):
        return {"iter_num": self.iter_num, "best_val_loss": self.best_val_loss, "patience_counter": self.patience_counter}

@dataclass
class ModelContext:
    configs: Configs
    model: Optional[Transformer] = None
    optimizer: Optional[torch.optim.AdamW] = None
    scaler: Optional[torch.amp.GradScaler] = None
    train_state: Optional[TrainState] = None

    def from_pretrained(self):
        from transformers import GPT2LMHeadModel

        # create a from-scratch initialized minGPT model
        self.configs = Configs(flash=True, model_type=self.configs.model_type, attn_type=AttnType.MHA, pos_type=PosType.WPE, norm_type=NormType.LAYER)

        self.model = Transformer(self.configs)
        sd = self.model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(self.configs.model_type.value.name)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

    def get_model(self, resume, device, filename='last_checkpoint.pt'):
        ckpt_path = os.path.join(self.configs.out_dir, filename)
        if not os.path.exists(ckpt_path):
            print("Checkpoint not found!")
            resume = False

        self.scaler = torch.amp.GradScaler(enabled=(self.configs.train_type.value.dtype == 'float16'))

        if resume:
            print(f"Resuming training from {self.configs.out_dir}")
            
            checkpoint = torch.load(ckpt_path, map_location=device)
            self.configs = Configs(**checkpoint['configs'])
            self.train_state = TrainState(**checkpoint['train_state'])
            
            print(self.configs.info())
            print(self.train_state.info())
        else:
            print("Initializing a new model from scratch")
            self.train_state = TrainState()
        
        self.model = Transformer(self.configs)
        self.model.to(device)
        self.optimizer = self.model.configure_optimizers(
            self.configs.train_type.value.weight_decay,
            self.configs.train_type.value.learning_rate,
            (self.configs.train_type.value.beta1, self.configs.train_type.value.beta2),
            device
        )
        
        if resume:
            state_dict = checkpoint['model']

            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
            self.model.load_state_dict(state_dict)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scaler.load_state_dict(checkpoint['scaler'])
            
            torch.set_rng_state(checkpoint['rng_state_torch'].cpu())
            np.random.set_state(checkpoint['rng_state_numpy'])
            random.setstate(checkpoint['rng_state_python'])
            if checkpoint['rng_state_cuda'] is not None and device == 'cuda' and torch.cuda.is_available():
                rng_states = [s.cpu() for s in checkpoint['rng_state_cuda']]
                torch.cuda.set_rng_state_all(rng_states)
        
        print(self.configs.info())
        print(f"Total Params: {self.model.get_num_params()/1e6:.2f}M")

    def save_model(self, filename):
        if self.configs.train_type.value.save_ckpt:
            checkpoint = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scaler': self.scaler.state_dict(),
                'configs': asdict(self.configs),
                'train_state': asdict(self.train_state),
                'rng_state_torch': torch.get_rng_state(),
                'rng_state_numpy': np.random.get_state(),
                'rng_state_python': random.getstate(),
                'rng_state_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            }

            os.makedirs(self.configs.out_dir, exist_ok=True)

            path = os.path.join(self.configs.out_dir, filename)
            print(f"saving checkpoint to {path}")
            
            torch.save(checkpoint, path)
            print("Checkpoint saved successfully.")
        else:
            print("save_ckpt:", self.configs.train_type.value.save_ckpt)
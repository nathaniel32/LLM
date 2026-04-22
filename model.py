import math
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from abc import ABC, abstractmethod

@dataclass
class ModelConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float
    bias: bool
    gqa_kv_head: Optional[int] = None

    def to_dict(self):
        return dict(
            block_size=self.block_size,
            vocab_size=self.vocab_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            dropout=self.dropout,
            bias=self.bias,
            gqa_kv_head=self.gqa_kv_head
        )

class BaseKVCache(ABC):
    def __init__(self, block_size):
        self.pos = 0
        self.block_size = block_size

    @abstractmethod
    def update(self, *args):
        pass

    def reset(self):
        self.pos = 0

class KVCache(BaseKVCache):
    def __init__(self, block_size):
        super().__init__(block_size)
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    def update(self, k: torch.Tensor, v: torch.Tensor):
        B, nh, T, hs = k.shape

        if self.k is None or self.k.shape[0] != B or self.k.shape[1] != nh:
            self.k = torch.zeros((B, nh, self.block_size, hs), device=k.device, dtype=k.dtype)
            self.v = torch.zeros((B, nh, self.block_size, hs), device=v.device, dtype=v.dtype)
            self.pos = 0

        self.k[:, :, self.pos:self.pos + T, :] = k
        self.v[:, :, self.pos:self.pos + T, :] = v
        self.pos += T

        return self.k[:, :, :self.pos, :], self.v[:, :, :self.pos, :]

class LatentKVCache(BaseKVCache):
    def __init__(self, block_size):
        super().__init__(block_size)
        self.c_kv: torch.Tensor | None = None

    def update(self, c_kv: torch.Tensor):
        B, T, latent_dim = c_kv.shape

        if self.c_kv is None or self.c_kv.shape[0] != B:
            self.c_kv = torch.zeros((B, self.block_size, latent_dim), device=c_kv.device, dtype=c_kv.dtype)
            self.pos = 0

        self.c_kv[:, self.pos:self.pos + T, :] = c_kv
        self.pos += T

        return self.c_kv[:, :self.pos, :]

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim)) # torch.Size([768])
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        #mu = input.mean(-1, keepdim=True)
        #var = ((input - mu) ** 2).mean(-1, keepdim=True)
        #xhat = (input - mu) / torch.sqrt(var + 1e-5)
        #return xhat * self.weight + self.bias

        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

class BaseSelfAttention(nn.Module):
    def __init__(self, config:ModelConfig, is_rope):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.is_rope = is_rope

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head # 64

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias) # torch.Size([768]) -> torch.Size([768])

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.config = config
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
            #self.bias = torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size).to('cuda') # torch.Size([1, 1, 1024, 1024])

    def _causal_attention(self, q, k, v, B, T, C):
        if self.flash:
            is_causal = T > 1
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.config.dropout if self.training else 0, is_causal=is_causal)
        else:
            # Attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim)) # torch.Size([1, 12, 7, 7])

            if T > 1:
                T_full = k.size(2)
                t_start = T_full - T
                att = att.masked_fill(self.bias[:, :, t_start:t_start + T, :T_full] == 0, float('-inf'))

            att = F.softmax(att, dim=-1) # torch.Size([1, 12, 7, 7])
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs) | torch.Size([1, 12, 7, 7]) * torch.Size([1, 12, 7, 64]) -> torch.Size([1, 12, 7, 64])
        
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # 1. torch.Size([1, 12, 7, 64]) -> torch.Size([1, 7, 12, 64])
        # 2. torch.Size([1, 7, 12, 64]) -> torch.Size([1, 7, 768])

        # output projection
        y = self.resid_dropout(self.c_proj(y)) # torch.Size([1, 7, 768])
        return y
    
    @staticmethod
    def _frequencies_calculation(head_dim: int, base: int = 10000):
        # Shape: [head_dim//2]
        i = torch.arange(0, head_dim // 2, dtype=torch.float32)
        return base ** (-2 * i / head_dim)

    @classmethod
    def apply_rope(cls, x: torch.Tensor, pos, is_before_cache):
        """
        x:   [batch, heads, seq_len, head_dim]
        pos: int — posisi awal (default 0, saat KV-cache bisa > 0)
        """
        head_dim = x.shape[-1]
        seq_len  = x.shape[-2]
        device   = x.device
        
        # after cache --> full token seq
        # before cache or seq_len > 1 --> start token
        is_full_seq = seq_len > 1
        pos_start = 0 if is_full_seq else (pos if is_before_cache else pos-1)
        pos_end = pos_start + seq_len

        # Buat [seq_len, head_dim//2], langsung di device yang benar
        freqs   = cls._frequencies_calculation(head_dim).to(device)         # [d//2]
        pos_ids = torch.arange(pos_start, pos_end, device=device).float()   # [seq_len]
        angles  = torch.outer(pos_ids, freqs)                               # [seq_len, d//2]
        
        #print(x.shape[-2])
        #print(pos_ids)
        #print("...."*10)

        cos_a = torch.cos(angles)  # [seq_len, d//2]
        sin_a = torch.sin(angles)

        # Split half — broadcast otomatis ke [B, H, seq_len, d//2]
        x1, x2 = x[..., :head_dim//2], x[..., head_dim//2:]
        x_rot = torch.cat([
            x1 * cos_a - x2 * sin_a,
            x1 * sin_a + x2 * cos_a,
        ], dim=-1)

        return x_rot

class CausalSelfAttention(BaseSelfAttention):
    
    def __init__(self, config:ModelConfig, is_rope, n_kv_head):
        super().__init__(config, is_rope)

        self.cache = KVCache(config.block_size)

        assert config.n_head % n_kv_head == 0
        self.n_kv_head = n_kv_head

        self.kv_repeat = config.n_head // n_kv_head # 3
        self.kv_dim = n_kv_head * self.head_dim

        self.c_attn = nn.Linear(config.n_embd, config.n_embd + 2 * self.kv_dim, bias=config.bias)
        
    def forward(self, x:torch.Tensor, use_cache: bool = False):
        B, T, C = x.size() # torch.Size([1, 7, 768])

        combined = self.c_attn(x)
        q, k, v = combined.split([C, self.kv_dim, self.kv_dim], dim=2)

        # Reshape
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs) -> torch.Size([1, 12, 7, 64])
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2) # (B, n_kv_head, T, hs) -> torch.Size([1, 4, 7, 64])
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2) # (B, n_kv_head, T, hs) -> torch.Size([1, 4, 7, 64])

        if self.is_rope:
            pos = self.cache.pos
            q = self.apply_rope(q, pos, is_before_cache=True)
            k = self.apply_rope(k, pos, is_before_cache=True)

        if use_cache:
            k, v = self.cache.update(k, v)
        
        # K, V
        k = k.repeat_interleave(self.kv_repeat, dim=1)  # (B, n_head, T_full, head_dim)
        v = v.repeat_interleave(self.kv_repeat, dim=1)

        return self._causal_attention(q, k, v, B, T, C)

class MultiHeadLatentAttention(BaseSelfAttention):
    def __init__(self, config:ModelConfig, is_rope):
        super().__init__(config, is_rope)

        self.cache = LatentKVCache(config.block_size)

        self.q_latent_dim = 128
        self.kv_latent_dim = 64

        # Query compression
        self.q_down = nn.Linear(config.n_embd, self.q_latent_dim, bias=config.bias)
        self.q_norm = RMSNorm(self.q_latent_dim)
        self.q_up = nn.Linear(self.q_latent_dim, config.n_embd, bias=config.bias)

        # KV compression
        self.kv_down = nn.Linear(config.n_embd, self.kv_latent_dim, bias=config.bias)
        self.kv_norm = RMSNorm(self.kv_latent_dim)
        self.kv_up = nn.Linear(self.kv_latent_dim, config.n_embd * 2, bias=config.bias)

    def forward(self, x:torch.Tensor, use_cache: bool = False):
        B, T, C = x.size()

        # Q: down-project -> normalize -> up-project
        c_q = self.q_norm(self.q_down(x))
        q = self.q_up(c_q).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # KV: down-project -> normalize -> up-project
        c_kv = self.kv_norm(self.kv_down(x))

        if use_cache:
            c_kv = self.cache.update(c_kv)
        
        k, v = self.kv_up(c_kv).split(self.config.n_embd, dim=2)
        k = k.view(B, -1, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.n_head, self.head_dim).transpose(1, 2)

        if self.is_rope:
            pos = self.cache.pos
            q = self.apply_rope(q, pos, is_before_cache=False)
            k = self.apply_rope(k, pos, is_before_cache=False)

        return self._causal_attention(q, k, v, B, T, C)

class MLP(nn.Module):

    def __init__(self, config:ModelConfig):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x) # torch.Size([1, 7, 768]) -> torch.Size([1, 7, 3072])
        x = self.gelu(x)
        x = self.c_proj(x) # torch.Size([1, 7, 3072]) -> torch.Size([1, 7, 768])
        x = self.dropout(x)
        return x
    
class Block(nn.Module):
    "attention + MLP + LayerNorm"

    def __init__(self, config:ModelConfig, attn_type, is_rope):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        
        if attn_type == "mla":
            self.attn = MultiHeadLatentAttention(config, is_rope)
        else:
            if attn_type == "mha":
                n_kv_head = config.n_head
            elif attn_type == "gqa":
                n_kv_head=config.gqa_kv_head
            elif attn_type == "mqa":
                n_kv_head = 1
            
            self.attn = CausalSelfAttention(config, is_rope, n_kv_head=n_kv_head)
        
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x:torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), use_cache=use_cache)
        x = x + self.mlp(self.ln_2(x))
        return x
    
class GPT(nn.Module):
    "Embedding → Block → Block → Block → lm_head"

    def __init__(self, config:ModelConfig, attn_type, is_pos_emb):
        super().__init__()
        self.config = config
        self.is_pos_emb = is_pos_emb
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # Weight Token Embedding -> torch.Size([50257, 768])
            wpe = nn.Embedding(config.block_size, config.n_embd) if is_pos_emb else None, # Weight Position Embedding -> torch.Size([1024, 768])
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, attn_type, is_rope=not is_pos_emb) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # torch.Size([50257, 768]) | 768 input features & 50257 output features
        self.transformer.wte.weight = self.lm_head.weight # weight tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.is_pos_emb:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        import inspect

        # start with all of the candidate parameters
        #param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        #param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
    
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        return flops_achieved / flops_promised

    def reset_cache(self):
        for block in self.transformer.h:
            block.attn.cache.reset()

    def forward(self, idx, targets=None, use_cache: bool = False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd) -> torch.Size([1, 7, 768])
        
        if self.is_pos_emb:
            if use_cache:
                cache_len = self.transformer.h[0].attn.cache.pos
                pos = torch.arange(cache_len, cache_len + t, dtype=torch.long, device=device)
            else:
                pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

            # idx = tensor([[1169, 7577,  286,  262, 2679, 6056,  389]], device='cuda:0')
            # pos = tensor([0, 1, 2, 3, 4, 5, 6], device='cuda:0')

            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd) -> torch.Size([7, 768]) -> gk perlu batch, karena weight posisi antar batch selalu sama
            x = tok_emb + pos_emb
            #print(f"{tok_emb[0][1][2]} + {pos_emb[1][2]} = {tok_emb[0][1][2] + pos_emb[1][2]} == {x[0][1][2]}")
        else:
            x = tok_emb
        
        x = self.transformer.drop(x) # torch.Size([1, 7, 768])

        # hidden layer
        for block in self.transformer.h:
            x = block(x, use_cache=use_cache) # torch.Size([1, 7, 768])
        
        # Layer Normalization
        x = self.transformer.ln_f(x) # torch.Size([1, 7, 768])

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim | torch.Size([1, 1, 768]) -> torch.Size([1, 1, 50257])
            loss = None

        return logits, loss
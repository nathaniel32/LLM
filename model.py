import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from abc import ABC, abstractmethod
from config import Configs, AttnType, PosType, NormType

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
        B, nh, T, khs = k.shape
        _, _, _, vhs = v.shape

        if self.k is None or self.k.shape[0] != B or self.k.shape[1] != nh:
            self.k = torch.zeros((B, nh, self.block_size, khs), device=k.device, dtype=k.dtype)
            self.v = torch.zeros((B, nh, self.block_size, vhs), device=v.device, dtype=v.dtype)
            self.pos = 0

        self.k[:, :, self.pos:self.pos + T, :] = k
        self.v[:, :, self.pos:self.pos + T, :] = v
        self.pos += T

        return self.k[:, :, :self.pos, :], self.v[:, :, :self.pos, :]

class MLAKVCache(BaseKVCache):
    """Cache untuk efficient MLA — menyimpan kv_latent dan k_pe, bukan full k,v"""
    def __init__(self, block_size, kv_lora_dim, qk_rope_head_dim):
        super().__init__(block_size)
        self.kv_lora_dim = kv_lora_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_cache: torch.Tensor | None = None  # (B, T, kv_lora_dim)
        self.pe_cache: torch.Tensor | None = None  # (B, T, qk_rope_head_dim)
        self.pos = 0

    def update(self, kv_latent: torch.Tensor, k_pe: torch.Tensor):
        """
        kv_latent: (B, T, kv_lora_dim)
        k_pe:      (B, T, qk_rope_head_dim)  -- sudah di-RoPE, squeeze dari (B,1,T,rope_dim)
        """
        B, T, _ = kv_latent.shape

        if self.kv_cache is None or self.kv_cache.shape[0] != B:
            self.kv_cache = torch.zeros(B, self.block_size, self.kv_lora_dim,
                                        device=kv_latent.device, dtype=kv_latent.dtype)
            self.pe_cache = torch.zeros(B, self.block_size, self.qk_rope_head_dim,
                                        device=k_pe.device, dtype=k_pe.dtype)
            self.pos = 0

        self.kv_cache[:, self.pos:self.pos + T, :] = kv_latent
        self.pe_cache[:, self.pos:self.pos + T, :] = k_pe
        self.pos += T

        return self.kv_cache[:, :self.pos, :], self.pe_cache[:, :self.pos, :]

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
    def __init__(self, configs:Configs):
        super().__init__()
        self.configs = configs
        self.model_config = configs.model_type.value

        assert self.model_config.n_embd % self.model_config.n_head == 0

        self.is_rope = True if configs.pos_type == PosType.ROPE else False

        self.n_head = self.model_config.n_head
        self.head_dim = self.model_config.n_embd // self.model_config.n_head # 64

        # regularization
        self.attn_dropout = nn.Dropout(self.model_config.dropout)
        self.resid_dropout = nn.Dropout(self.model_config.dropout)
        
        if not configs.flash:
            print("WARNING: using slow attention")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(self.model_config.block_size, self.model_config.block_size)).view(1, 1, self.model_config.block_size, self.model_config.block_size))
            #self.bias = torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size).to('cuda') # torch.Size([1, 1, 1024, 1024])

    def _causal_attention(self, q, k, v, T, head_dim):
        if self.configs.flash:
            is_causal = T > 1
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.model_config.dropout if self.training else 0, is_causal=is_causal)
        else:
            # Attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim)) # torch.Size([1, 12, 7, 7])

            if T > 1:
                T_full = k.size(2)
                t_start = T_full - T
                att = att.masked_fill(self.bias[:, :, t_start:t_start + T, :T_full] == 0, float('-inf'))

            att = F.softmax(att, dim=-1) # torch.Size([1, 12, 7, 7])
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs) | torch.Size([1, 12, 7, 7]) * torch.Size([1, 12, 7, 64]) -> torch.Size([1, 12, 7, 64])
        
        return y
    
    @staticmethod
    def _frequencies_calculation(head_dim: int, base: int = 10000):
        # Shape: [head_dim//2]
        i = torch.arange(0, head_dim // 2, dtype=torch.float32)
        return base ** (-2 * i / head_dim)

    @classmethod
    def apply_rope(cls, x: torch.Tensor, pos):
        """
        x:   [batch, heads, seq_len, head_dim]
        pos: int — posisi awal (default 0, saat KV-cache bisa > 0)
        """
        head_dim = x.shape[-1]
        seq_len  = x.shape[-2]
        device   = x.device
        
        # seq_len > 1 --> full token
        is_full_seq = seq_len > 1
        pos_start = 0 if is_full_seq else pos
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
    
    def __init__(self, configs:Configs, n_kv_head):
        super().__init__(configs)

        self.cache = KVCache(self.model_config.block_size)

        assert self.model_config.n_head % n_kv_head == 0
        self.n_kv_head = n_kv_head

        self.kv_repeat = self.model_config.n_head // n_kv_head # 3
        self.kv_dim = n_kv_head * self.head_dim

        self.c_attn = nn.Linear(self.model_config.n_embd, self.model_config.n_embd + 2 * self.kv_dim, bias=self.model_config.bias)

        # output projection
        self.c_proj = nn.Linear(self.model_config.n_embd, self.model_config.n_embd, bias=self.model_config.bias) # torch.Size([768]) -> torch.Size([768])
        
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
            q = self.apply_rope(q, pos)
            k = self.apply_rope(k, pos)

        if use_cache:
            k, v = self.cache.update(k, v)
        
        # K, V
        k = k.repeat_interleave(self.kv_repeat, dim=1)  # (B, n_head, T_full, head_dim)
        v = v.repeat_interleave(self.kv_repeat, dim=1)

        y = self._causal_attention(q, k, v, T, self.head_dim)

        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # 1. torch.Size([1, 12, 7, 64]) -> torch.Size([1, 7, 12, 64])
        # 2. torch.Size([1, 7, 12, 64]) -> torch.Size([1, 7, 768])

        # output projection
        y = self.resid_dropout(self.c_proj(y)) # torch.Size([1, 7, 768])
        return y

class MultiHeadLatentAttention(BaseSelfAttention):
    def __init__(self, configs:Configs, efficient=True):
        super().__init__(configs)
        self.efficient = efficient
        
        self.qk_nope_head_dim = self.head_dim // 4
        self.qk_rope_head_dim = self.head_dim // 2
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = self.head_dim // 2

        self.q_lora_dim = 0
        self.kv_lora_dim = self.head_dim * 2 # config.n_embd // 8

        self.cache = KVCache(self.model_config.block_size) if not efficient else MLAKVCache(self.model_config.block_size, self.kv_lora_dim, self.qk_rope_head_dim)

        # Query compression
        if self.q_lora_dim == 0:
            self.wq = nn.Linear(self.model_config.n_embd, self.n_head * self.qk_head_dim, bias=self.model_config.bias)
        else:
            self.q_down = nn.Linear(self.model_config.n_embd, self.q_lora_dim, bias=self.model_config.bias)
            self.q_norm = RMSNorm(self.q_lora_dim)
            self.q_up = nn.Linear(self.q_lora_dim, self.n_head * self.qk_head_dim, bias=self.model_config.bias)

        # KV compression
        self.kv_down = nn.Linear(self.model_config.n_embd, self.kv_lora_dim + self.qk_rope_head_dim, bias=self.model_config.bias)
        self.kv_norm = RMSNorm(self.kv_lora_dim)
        self.kv_up = nn.Linear(self.kv_lora_dim, self.n_head * (self.qk_nope_head_dim+self.v_head_dim), bias=self.model_config.bias)

        self.c_proj = nn.Linear(self.n_head * self.v_head_dim, self.model_config.n_embd, bias=self.model_config.bias)

    def forward(self, x: torch.Tensor, use_cache: bool = False):
        B, T, C = x.size()
        pos = self.cache.pos

        if self.q_lora_dim == 0:
            q = self.wq(x)
        else:
            # Q: down -> norm -> up
            c_q = self.q_norm(self.q_down(x))
            q = self.q_up(c_q)

        q = q.view(B, T, self.n_head, self.qk_head_dim).transpose(1, 2)
        # q: (B, n_head, T, qk_head_dim)
        
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        if self.is_rope:
            q_pe = self.apply_rope(q_pe, pos)
            # q_pe: (B, n_head, T, rope_dim)

        # KV: down -> split rope vs lora
        c_kv = self.kv_down(x)  # (B, T, kv_lora_dim + rope_dim)
        kv_latent, k_pe = torch.split(c_kv, [self.kv_lora_dim, self.qk_rope_head_dim], dim=-1)
        # k_pe: (B, T, rope_dim) -> (B, 1, T, rope_dim) -> apply rope -> (B, 1, T, rope_dim)
        if self.is_rope:
            k_pe = self.apply_rope(k_pe.unsqueeze(1), pos)
        else:
            k_pe = k_pe.unsqueeze(1)
        
        if not self.efficient:
            # expand ke semua head: (B, n_head, T, rope_dim)
            k_pe = k_pe.expand(-1, self.n_head, -1, -1)

            # KV up-project
            kv = self.kv_up(self.kv_norm(kv_latent))
            kv = kv.view(B, T, self.n_head, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
            # kv: (B, n_head, T, nope_dim + v_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

            # Gabungkan q dan k
            q = torch.cat([q_nope, q_pe], dim=-1)          # (B, n_head, T, qk_head_dim)
            k = torch.cat([k_nope, k_pe], dim=-1)          # (B, n_head, T, qk_head_dim)

            if use_cache:
                #print(q.shape, k.shape, v.shape)
                k, v = self.cache.update(k, v)

            y = self._causal_attention(q, k, v, T, self.qk_head_dim)
        else:
            # ── EFFICIENT PATH ────────────────────────────────────────────────
            kv_latent_normed = self.kv_norm(kv_latent)          # (B, T, kv_lora_dim)
            k_pe_2d = k_pe.squeeze(1)                      # (B, T, rope_dim)

            if use_cache:
                kv_latent_normed, k_pe_2d = self.cache.update(kv_latent_normed, k_pe_2d)
            # kv_latent_normed: (B, T_full, kv_lora_dim)
            # k_pe_2d:          (B, T_full, rope_dim)

            # Absorb wkv_b ke q_nope
            wkv_b = self.kv_up.weight  # (n_head*(nope+v_dim), kv_lora_dim)
            wkv_b = wkv_b.view(self.n_head, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_dim)
            w_k = wkv_b[:, :self.qk_nope_head_dim, :]   # (n_head, nope_dim, kv_lora_dim)
            w_v = wkv_b[:, self.qk_nope_head_dim:, :]   # (n_head, v_dim,   kv_lora_dim)

            # q_nope: (B, n_head, T, nope_dim) → projected ke latent space
            q_latent = torch.einsum("bnsd,hdc->bnsc", q_nope, w_k)
            # (B, n_head, T, kv_lora_dim)

            softmax_scale = self.qk_head_dim ** -0.5
            scores_nope = torch.einsum("bnsc,btc->bnst", q_latent, kv_latent_normed)
            scores_rope = torch.einsum("bnsr,btr->bnst", q_pe,     k_pe_2d)
            scores = (scores_nope + scores_rope) * softmax_scale
            # scores: (B, n_head, T, T_full)

            # Causal mask
            T_full = scores.shape[-1]
            if T > 1:
                mask = torch.tril(torch.ones(T, T_full, device=x.device)).unsqueeze(0).unsqueeze(0)
                scores = scores.masked_fill(mask == 0, float('-inf'))

            attn = torch.softmax(scores, dim=-1)
            attn = self.attn_dropout(attn)

            out_latent = torch.einsum("bnst,btc->bnsc", attn,       kv_latent_normed)
            y           = torch.einsum("bnsc,hdc->bnsd", out_latent, w_v)
            # y: (B, n_head, T, v_head_dim)
        
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.v_head_dim)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, configs:Configs):
        super().__init__()
        hidden_dim = int(configs.attn_type.value.mlp_ratio * configs.model_type.value.n_embd)
        self.c_fc    = nn.Linear(configs.model_type.value.n_embd, hidden_dim, bias=configs.model_type.value.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(hidden_dim, configs.model_type.value.n_embd, bias=configs.model_type.value.bias)
        self.dropout = nn.Dropout(configs.model_type.value.dropout)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x) # torch.Size([1, 7, 768]) -> torch.Size([1, 7, 3072])
        x = self.gelu(x)
        x = self.c_proj(x) # torch.Size([1, 7, 3072]) -> torch.Size([1, 7, 768])
        x = self.dropout(x)
        return x
    
class Block(nn.Module):
    "attention + MLP + Norm"

    def __init__(self, configs:Configs):
        super().__init__()

        if configs.norm_type == NormType.RMS:
            self.ln_1 = RMSNorm(configs.model_type.value.n_embd)
            self.ln_2 = RMSNorm(configs.model_type.value.n_embd)
        elif configs.norm_type == NormType.LAYER:
            print("Using LayerNorm!")
            self.ln_1 = LayerNorm(configs.model_type.value.n_embd, bias=configs.model_type.value.bias)
            self.ln_2 = LayerNorm(configs.model_type.value.n_embd, bias=configs.model_type.value.bias)
        
        if configs.attn_type in (AttnType.MLA_STD, AttnType.MLA_ISO):
            self.attn = MultiHeadLatentAttention(configs)
        else:
            if configs.attn_type.value.kv_head is not None: # MQA, GQA
                n_kv_head = configs.attn_type.value.kv_head
            else: # MHA
                n_kv_head = configs.model_type.value.n_head
            
            #print("KV_HEAD:", n_kv_head)
            self.attn = CausalSelfAttention(configs, n_kv_head=n_kv_head)

        self.mlp = MLP(configs)

    def forward(self, x:torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), use_cache=use_cache)
        x = x + self.mlp(self.ln_2(x))
        return x
    
class Transformer(nn.Module):
    "Embedding → Block → Block → Block → lm_head"

    def __init__(self, configs:Configs):
        super().__init__()
        self.configs = configs
        self.is_wpe = True if configs.pos_type == PosType.WPE else False
        #print({'is_wpe': self.is_wpe, 'pos_type': configs.pos_type.value})

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(configs.model_type.value.vocab_size, configs.model_type.value.n_embd), # Weight Token Embedding -> torch.Size([50257, 768])
            wpe = nn.Embedding(configs.model_type.value.block_size, configs.model_type.value.n_embd) if self.is_wpe else None, # Weight Position Embedding -> torch.Size([1024, 768])
            drop = nn.Dropout(configs.model_type.value.dropout),
            h = nn.ModuleList([Block(configs) for _ in range(configs.model_type.value.n_layer)]),
            ln_f = RMSNorm(configs.model_type.value.n_embd) if configs.norm_type == NormType.RMS else LayerNorm(configs.model_type.value.n_embd, bias=configs.model_type.value.bias),
        ))
        self.lm_head = nn.Linear(configs.model_type.value.n_embd, configs.model_type.value.vocab_size, bias=False) # torch.Size([50257, 768]) | 768 input features & 50257 output features
        self.transformer.wte.weight = self.lm_head.weight # weight tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * configs.model_type.value.n_layer))

        # report number of parameters
        #print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.is_wpe:
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
    
    def estimate_mfu(self, fwdbwd_per_iter, dt, flops_promised=6.45e12):
        N = self.get_num_params()
        model_config = self.configs.model_type.value
        L, H, Q, T = model_config.n_layer, model_config.n_head, model_config.n_embd//model_config.n_head, model_config.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        return flops_achieved / flops_promised

    def reset_cache(self):
        for block in self.transformer.h:
            block.attn.cache.reset()

    def forward(self, idx, targets=None, use_cache: bool = False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.configs.model_type.value.block_size, f"Cannot forward sequence of length {t}, block size is only {self.configs.model_type.value.block_size}"
        
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd) -> torch.Size([1, 7, 768])
        
        if self.is_wpe:
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
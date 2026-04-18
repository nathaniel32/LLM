import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float
    bias: bool

    def to_dict(self):
        return dict(
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            block_size=self.block_size,
            bias=self.bias,
            vocab_size=self.vocab_size,
            dropout=self.dropout
        )

class KVCache:
    def __init__(self):
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    def update(self, k: torch.Tensor, v: torch.Tensor):
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat([self.k, k], dim=2)  # concat di dim T
            self.v = torch.cat([self.v, v], dim=2)
        return self.k, self.v

    def reset(self):
        self.k = None
        self.v = None

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
    
class CausalSelfAttention(nn.Module):

    def __init__(self, config:GPTConfig):
        super().__init__()

        self.cache = KVCache()

        # 768/12 = 64 -> head_dim
        assert config.n_embd % config.n_head == 0
        
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias) # w = torch.Size([2304, 768]), b = torch.Size([2304])
        
        # output projection -> mencampur dan mengintegrasikan informasi dari semua head
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias) # torch.Size([768, 768])
        
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
                
        # causal mask to ensure that attention is only applied to the left in the input sequence
        #self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        
        self.bias = torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size).to('cuda') # torch.Size([1, 1, 1024, 1024])
        """ 
        [1,0,0,0]
        [1,1,0,0]
        [1,1,1,0]
        [1,1,1,1] 
        """

    def forward(self, x:torch.Tensor, use_cache: bool = False):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd) | torch.Size([1, 7, 768])

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        x = self.c_attn(x)
        q, k, v  = x.split(self.n_embd, dim=2)
        # 1. torch.Size([1, 7, 768]) -> torch.Size([1, 7, 2304])
        # 2. torch.Size([1, 7, 2304]) -> [torch.Size([1, 7, 768]), torch.Size([1, 7, 768]), torch.Size([1, 7, 768])]

        # Multi-Head-Attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) -> torch.Size([1, 7, 12, 64]) -> torch.Size([1, 12, 7, 64])
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) -> torch.Size([1, 12, 7, 64])
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) -> torch.Size([1, 12, 7, 64])

        if use_cache:
            k, v = self.cache.update(k, v)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # manual implementation of attention
        T_q    = q.size(2)
        T_full = k.size(2)
        t_start = T_full - T_q

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # torch.Size([1, 12, 7, 7])
        att = att.masked_fill(self.bias[:, :, t_start:t_start + T_q, :T_full] == 0, float('-inf'))
        #print(att.shape)

        att = F.softmax(att, dim=-1) # torch.Size([1, 12, 7, 7])
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        #  torch.Size([1, 12, 7, 7]) * torch.Size([1, 12, 7, 64]) -> torch.Size([1, 12, 7, 64])
        
        y = y.transpose(1, 2).contiguous().view(B, T_q, C) # re-assemble all head outputs side by side
        # 1. torch.Size([1, 12, 7, 64]) -> torch.Size([1, 7, 12, 64])
        # 2. torch.Size([1, 7, 12, 64]) -> torch.Size([1, 7, 768])

        # output projection
        y = self.resid_dropout(self.c_proj(y)) # torch.Size([1, 7, 768])
        return y
    
class MLP(nn.Module):

    def __init__(self, config:GPTConfig):
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

    def __init__(self, config:GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x:torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), use_cache=use_cache)
        x = x + self.mlp(self.ln_2(x))
        return x
    
class GPT(nn.Module):
    "Embedding → Block → Block → Block → lm_head"

    def __init__(self, config:GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # Weight Token Embedding -> torch.Size([50257, 768])
            wpe = nn.Embedding(config.block_size, config.n_embd), # Weight Position Embedding -> torch.Size([1024, 768])
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # torch.Size([50257, 768]) | 768 input features & 50257 output features
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

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
        if non_embedding:
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
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
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
        mfu = flops_achieved / flops_promised
        return mfu

    def reset_cache(self):
        for block in self.transformer.h:
            block.attn.cache.reset()

    def forward(self, idx, targets=None, use_cache: bool = False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        if use_cache:
            # ambil posisi saat ini dari panjang cache yang sudah ada
            cache_len = self.transformer.h[0].attn.cache.k.size(2) if self.transformer.h[0].attn.cache.k is not None else 0
            pos = torch.arange(cache_len, cache_len + t, dtype=torch.long, device=device)
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # idx = tensor([[1169, 7577,  286,  262, 2679, 6056,  389]], device='cuda:0')
        # pos = tensor([0, 1, 2, 3, 4, 5, 6], device='cuda:0')

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd) -> torch.Size([1, 7, 768])
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd) -> torch.Size([7, 768]) -> gk perlu batch, karena weight posisi antar batch selalu sama
        
        x = self.transformer.drop(tok_emb + pos_emb) # tok_emb + pos_emb -> dropout | torch.Size([1, 7, 768])

        #print(f"{tok_emb[0][1][2]} + {pos_emb[1][2]} = {tok_emb[0][1][2] + pos_emb[1][2]} == {x[0][1][2]}")

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
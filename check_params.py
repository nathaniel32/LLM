from config import AttnType, args_configs
from model import Transformer

configs = args_configs

def check_config(attn_type):
    configs.attn_type = attn_type
    model = Transformer(configs)
    
    m = configs.model_type.value
    a = configs.attn_type.value
    
    # Calculate KV Cache size per token (in elements)
    if configs.attn_type.value.is_mla:
        # In model.py:
        # self.qk_rope_head_dim = self.head_dim // 2
        # self.kv_lora_dim = configs.model_type.value.n_embd // 4
        head_dim = m.n_embd // m.n_head
        kv_lora_dim = m.n_embd // 4 # head_dim * 2
        qk_rope_head_dim = head_dim // 2
        kv_size_per_layer = kv_lora_dim + qk_rope_head_dim
    else:
        n_kv_head = a.kv_head if a.kv_head is not None else m.n_head
        head_dim = m.n_embd // m.n_head
        kv_size_per_layer = 2 * n_kv_head * head_dim
        
    total_kv_size = kv_size_per_layer * m.n_layer
    
    params = model.get_num_params()
    print(configs.info())
    print(f"Total Params: {params/1e6:.2f}M - {params}")
    print(f"KV Cache Size per token (elements): {total_kv_size}")
    print(f"KV Cache Size for {m.block_size} tokens (MB, float16): {total_kv_size * m.block_size * 2 / 1024**2:.4f} MB")
    print("=" * 100)

for at in AttnType:
    check_config(at)

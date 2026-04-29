
import torch
from config import Configs, ModelType, AttnType, PosType, NormType, TrainType
from model import Transformer

def check_config(attn_type):
    configs = Configs(
        flash=True,
        model_type=ModelType.RESEARCH,
        attn_type=attn_type,
        pos_type=PosType.ROPE,
        norm_type=NormType.RMS,
        train_type=TrainType.RESEARCH,
        dataset_type=None
    )
    model = Transformer(configs)
    params = model.get_num_params()
    
    m = configs.model_type.value
    a = configs.attn_type.value
    
    # Calculate KV Cache size per token (in elements)
    if configs.attn_type == AttnType.MLA:
        # In model.py:
        # self.qk_rope_head_dim = self.head_dim // 2
        # self.kv_lora_dim = self.head_dim * 2
        head_dim = m.n_embd // m.n_head
        kv_lora_dim = head_dim * 2
        qk_rope_head_dim = head_dim // 2
        kv_size_per_layer = kv_lora_dim + qk_rope_head_dim
    else:
        n_kv_head = a.kv_head if a.kv_head is not None else m.n_head
        head_dim = m.n_embd // m.n_head
        kv_size_per_layer = 2 * n_kv_head * head_dim
        
    total_kv_size = kv_size_per_layer * m.n_layer
    
    print(f"AttnType: {attn_type.name}")
    print(f"Total Params: {params/1e6:.2f}M")
    print(f"KV Cache Size per token (elements): {total_kv_size}")
    print(f"KV Cache Size for {m.block_size} tokens (MB, float16): {total_kv_size * m.block_size * 2 / 1024**2:.4f} MB")
    print("-" * 30)

for at in AttnType:
    check_config(at)

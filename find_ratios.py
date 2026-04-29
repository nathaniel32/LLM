
import torch
import torch.nn as nn
from config import Configs, ModelType, AttnType, PosType, NormType, TrainType
from model import Transformer

def get_params_for_ratio(attn_type, mlp_ratio):
    configs = Configs(
        flash=True,
        model_type=ModelType.RESEARCH,
        attn_type=attn_type,
        pos_type=PosType.ROPE,
        norm_type=NormType.RMS,
        train_type=TrainType.RESEARCH,
        dataset_type=None
    )
    # Manually override mlp_ratio for this test
    configs.attn_type.value.mlp_ratio = mlp_ratio
    
    model = Transformer(configs)
    return model.get_num_params()

# Target: MHA params with mlp_ratio=4
target_params = get_params_for_ratio(AttnType.MHA, 4)
print(f"Target Params (MHA @ ratio 4): {target_params/1e6:.4f}M")
print("-" * 30)

for at in [AttnType.GQA, AttnType.MQA, AttnType.MLA]:
    # Simple search for the best ratio to match target_params
    best_ratio = 4.0
    best_diff = float('inf')
    
    # Try ratios from 4.0 to 5.5 with small steps
    for r in [x * 0.1 for x in range(40, 60)]:
        # Note: In the actual model, mlp_ratio is used in:
        # nn.Linear(n_embd, mlp_ratio * n_embd)
        # Since Linear weights must be integers, we simulate the nearest integer dimension
        
        configs = Configs(
            flash=True,
            model_type=ModelType.RESEARCH,
            attn_type=at,
            pos_type=PosType.ROPE,
            norm_type=NormType.RMS,
            train_type=TrainType.RESEARCH,
            dataset_type=None
        )
        m_cfg = configs.model_type.value
        hidden_dim = int(r * m_cfg.n_embd)
        
        # Manually calculate params for this hypothetical model
        # 1. Embeddings
        params = m_cfg.vocab_size * m_cfg.n_embd # wte
        # 2. Layers
        for _ in range(m_cfg.n_layer):
            # Norms
            params += m_cfg.n_embd * 2 # ln1, ln2
            # Attn (specific to type)
            if at == AttnType.MLA:
                kv_lora_dim = (m_cfg.n_embd // m_cfg.n_head) * 2
                qk_rope_head_dim = (m_cfg.n_embd // m_cfg.n_head) // 2
                qk_nope_head_dim = (m_cfg.n_embd // m_cfg.n_head) // 4
                v_head_dim = (m_cfg.n_embd // m_cfg.n_head) // 2
                # wq: n_embd -> n_head * qk_head_dim
                params += m_cfg.n_embd * m_cfg.n_head * (qk_nope_head_dim + qk_rope_head_dim)
                # kv_down: n_embd -> kv_lora_dim + rope_dim
                params += m_cfg.n_embd * (kv_lora_dim + qk_rope_head_dim)
                # kv_up: kv_lora_dim -> n_head * (nope + v)
                params += kv_lora_dim * m_cfg.n_head * (qk_nope_head_dim + v_head_dim)
                # c_proj: n_head * v_dim -> n_embd
                params += (m_cfg.n_head * v_head_dim) * m_cfg.n_embd
            else:
                n_kv_head = at.value.kv_head if at.value.kv_head is not None else m_cfg.n_head
                kv_dim = n_kv_head * (m_cfg.n_embd // m_cfg.n_head)
                # c_attn: n_embd -> n_embd + 2*kv_dim
                params += m_cfg.n_embd * (m_cfg.n_embd + 2 * kv_dim)
                # c_proj: n_embd -> n_embd
                params += m_cfg.n_embd * m_cfg.n_embd
            
            # MLP: n_embd -> hidden_dim -> n_embd
            params += m_cfg.n_embd * hidden_dim # c_fc
            params += hidden_dim * m_cfg.n_embd # c_proj
        
        # 3. Final Head
        params += m_cfg.n_embd * m_cfg.vocab_size # lm_head (shared with wte)
        
        diff = abs(params - target_params)
        if diff < best_diff:
            best_diff = diff
            best_ratio = r
            
    print(f"AttnType: {at.name}")
    print(f"Best Ratio to match MHA: {best_ratio:.1f}")
    # Calculate actual params with this ratio
    configs.attn_type.value.mlp_ratio = int(best_ratio) if best_ratio % 1 == 0 else best_ratio
    print(f"Resulting Params: {params/1e6:.4f}M")

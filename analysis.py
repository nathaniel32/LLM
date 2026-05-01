import os
import json
import matplotlib.pyplot as plt
from config import Configs, AttnType, train_type, model_type, pos_type, norm_type, dataset_type

def load_metrics(config):
    log_path = os.path.join(config.out_dir, 'metrics.json')
    
    if not os.path.exists(log_path):
        print(f"Warning: File not found at {log_path}")
        return []
        
    with open(log_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

def plot_attention_comparison(configs_list, save_path):
    plt.figure(figsize=(12, 7))
    cmap = plt.get_cmap('tab10')

    for i, conf in enumerate(configs_list):
        conf:Configs

        log_data = load_metrics(conf)

        if not log_data:
            continue
        
        val_data = log_data.get('val_log')
            
        label_name = f"{conf.attn_type.name} - {log_data['params']}"
        col_iters = [d['iter'] for d in val_data]
        col_val_loss = [d['val_loss'] for d in val_data]
        
        plt.plot(col_iters, col_val_loss, '-', 
                 label=label_name, 
                 linewidth=2.5, 
                 color=cmap(i))
        
        if 'train_loss' in val_data[0]:
            col_train_loss = [d['train_loss'] for d in val_data]
            plt.plot(col_iters, col_train_loss, '--', 
                     alpha=0.3, 
                     color=cmap(i))

    plt.yscale('log')
    plt.title('Attention Variants Performance Comparison', fontsize=14, pad=15)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Loss (Log Scale)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, which="both", ls="-", alpha=0.15)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

configs_list = [
    Configs(
        flash=True,
        train_type=train_type,
        dataset_type=dataset_type,
        model_type=model_type,
        attn_type=attn_type,
        pos_type=pos_type,
        norm_type=norm_type
    )
    for attn_type in AttnType
]

plot_attention_comparison(configs_list, save_path="out/mqa_vs_mha_loss.png")
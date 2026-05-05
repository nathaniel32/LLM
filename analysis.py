import os
import json
import matplotlib.pyplot as plt
from config import Configs, AttnType, PosType, TrainType, NormType, ModelType, DatasetType, train_type, dataset_type, norm_type, model_type, attn_type, pos_type

def load_data(config:Configs):
    data = {}

    if config.out_dir is not None:
        log_path = os.path.join(config.out_dir, 'metrics.json')
        
        if not os.path.exists(log_path):
            print(f"Warning: File not found at {log_path}")
            return data
            
        with open(log_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    return data

def plot_validation_comparison(configs_list, save_dir, metric_types=['loss', 'perplexity', 'accuracy']):
    
    for metric_type in metric_types:
        plt.figure(figsize=(12, 7))
        cmap = plt.get_cmap('tab10')

        for i, conf in enumerate(configs_list):
            conf:Configs

            log_data = load_data(conf)

            if not log_data:
                continue
            
            val_data = log_data.get('val_log')
                
            label_name = f"{conf.attn_type.name} | {conf.pos_type.name} | {log_data['params']/1e6:.2f}M"
            col_iters = [d['iter'] for d in val_data]
            
            col_val_loss = [d['metrics']['val'][metric_type] for d in val_data]
            plt.plot(col_iters, col_val_loss, '-', label=label_name, linewidth=2.5, color=cmap(i))
            
            col_train_loss = [d['metrics']['train'][metric_type] for d in val_data]
            plt.plot(col_iters, col_train_loss, '--', alpha=0.3, color=cmap(i))

        plt.yscale('log')
        plt.title(f'Attention Variants Performance Comparison - {metric_type}', fontsize=14, pad=15)
        plt.xlabel('Iterations', fontsize=12)
        plt.ylabel(f'{metric_type} (Log Scale)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.grid(True, which="both", ls="-", alpha=0.15)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f'{metric_type}.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def plot_train_comparison(configs_list, save_dir, metric_types=['train_loss', 'time_ms', 'wpe_grad_norm', 'vram_gb']):
    
    for metric_type in metric_types:
        plt.figure(figsize=(12, 7))
        cmap = plt.get_cmap('tab10')

        for i, conf in enumerate(configs_list):
            conf:Configs

            log_data = load_data(conf)

            if not log_data:
                continue
            
            train_data = log_data.get('train_log')
                
            label_name = f"{conf.attn_type.name} | {conf.pos_type.name} | {log_data['params']/1e6:.2f}M"
            col_iters = [d['iter'] for d in train_data]
            
            col_val_loss = [d[metric_type] for d in train_data]
            plt.plot(col_iters, col_val_loss, '-', label=label_name, linewidth=2.5, color=cmap(i))
        
        plt.yscale('log')
        plt.title(f'Attention Variants Performance Comparison - {metric_type}', fontsize=14, pad=15)
        plt.xlabel('Iterations', fontsize=12)
        plt.ylabel(f'{metric_type} (Log Scale)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.grid(True, which="both", ls="-", alpha=0.15)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f'{metric_type}.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

configs_list = [
    Configs(
        flash=True,
        train_type=tt,
        dataset_type=dt,
        model_type=mt,
        attn_type=at,
        pos_type=pt,
        norm_type=nt
    )
    for tt in [train_type]
    for dt in [dataset_type]
    for mt in [model_type]
    for at in AttnType
    for pt in PosType
    for nt in [norm_type]
]

plot_validation_comparison(configs_list, save_dir="out", metric_types=['perplexity'])
plot_train_comparison(configs_list, save_dir="out")
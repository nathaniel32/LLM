import os
import json
import matplotlib.pyplot as plt
from config import Configs, AttnType, PosType, TrainType, NormType, ModelType, DatasetType, train_type, dataset_type, norm_type, model_type, attn_type, pos_type
from utils import ModelContext

def plot_comparison(configs_list, log_type, cols, save_dir='out/plot'):
    
    for col in cols:
        plt.figure(figsize=(12, 7))
        cmap = plt.get_cmap('tab10')

        for i, conf in enumerate(configs_list):
            model_context = ModelContext(configs=conf)

            metrics_data = model_context.logger.data

            if not metrics_data:
                continue
            
            log_data = metrics_data.get(log_type)
                
            label_name = f"{model_context.configs.attn_type.name} | {model_context.configs.pos_type.name} | {metrics_data['params']/1e6:.2f}M"
            col_iters = [d['iter'] for d in log_data]
            
            col_value = [d[col] for d in log_data]
            plt.plot(col_iters, col_value, '-', label=label_name, linewidth=2.5, color=cmap(i))

        plt.yscale('log')
        plt.title(f'Comparison - {col}', fontsize=14, pad=15)
        plt.xlabel('Iterations', fontsize=12)
        plt.ylabel(col, fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.grid(True, which="both", ls="-", alpha=0.15)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f'{col}.png')
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

plot_comparison(configs_list, log_type='val_log', cols=['val_loss', 'val_perplexity', 'val_accuracy'])
plot_comparison(configs_list, log_type='train_log', cols=['train_loss', 'time_ms', 'mfu_percent', 'weight_norm', 'grad_norm', 'wpe_grad_norm', 'vram_gb'])
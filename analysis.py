import os
import json
import matplotlib.pyplot as plt
from config import Configs, TrainType, DatasetType, ModelType, AttnType, PosType, NormType

def load_metrics(config):
    log_path = os.path.join(config.out_dir, 'metrics.json')
    
    if not os.path.exists(log_path):
        print(f"Warning: File not found at {log_path}")
        return []
        
    with open(log_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    return data.get('val_log', [])

def plot_attention_comparison(mqa_log, mha_log, save_path):
    if not mqa_log or not mha_log:
        print("Incomplete log data. Ensure both models are trained.")
        return

    mqa_iters = [log['iter'] for log in mqa_log]
    mqa_train_loss = [log['train_loss'] for log in mqa_log]
    mqa_val_loss = [log['val_loss'] for log in mqa_log]

    mha_iters = [log['iter'] for log in mha_log]
    mha_train_loss = [log['train_loss'] for log in mha_log]
    mha_val_loss = [log['val_loss'] for log in mha_log]

    plt.figure(figsize=(10, 6))

    plt.plot(mqa_iters, mqa_train_loss, '--', label='MQA Train Loss', color='blue')
    plt.plot(mqa_iters, mqa_val_loss, '-', label='MQA Val Loss', color='blue', linewidth=2)

    plt.plot(mha_iters, mha_train_loss, '--', label='MHA Train Loss', color='red')
    plt.plot(mha_iters, mha_val_loss, '-', label='MHA Val Loss', color='red', linewidth=2)

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('MQA vs MHA Training and Validation Loss Comparison')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()

    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(save_path, dpi=300)
    plt.show()

train_type = TrainType.RESEARCH
dataset_type = DatasetType.OPENWEBTEXT
model_type = ModelType.RESEARCH
pos_type = PosType.ROPE
norm_type = NormType.RMS

mqa_conf = Configs(
    flash=True,
    train_type=train_type,
    dataset_type=dataset_type,
    model_type=model_type,
    attn_type=AttnType.MQA,
    pos_type=pos_type,
    norm_type=norm_type
)

mha_conf = Configs(
    flash=True,
    train_type=train_type,
    dataset_type=dataset_type,
    model_type=model_type,
    attn_type=AttnType.MHA,
    pos_type=pos_type,
    norm_type=norm_type
)

mqa_data = load_metrics(mqa_conf)
mha_data = load_metrics(mha_conf)

plot_attention_comparison(mqa_data[3:], mha_data[3:], save_path="out/mqa_vs_mha_loss.png")
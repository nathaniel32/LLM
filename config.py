from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from abc import ABC
import os

project_root_global = os.path.dirname(os.path.abspath(__file__))

@dataclass
class BaseConfig(ABC):
    name: str

@dataclass
class DatasetConfig(BaseConfig):
    root_dir: str = field(init=False)
    url: Optional[str]
    dir_path: Optional[str]

    def __post_init__(self):
        self.root_dir = os.path.join(project_root_global, 'datasets', self.name)

    def prepare_dataset(self):
        import requests

        if os.path.exists(os.path.join(self.root_dir, 'train.bin')):
            print("Datasets found!")
            return
        
        if self.dir_path is not None:
            if os.path.exists(os.path.join(self.dir_path, 'train.bin')):
                self.root_dir = self.dir_path
                print("Datasets found in dir_path!:", self.root_dir)
                return
        
        if self.url is not None:
            import tiktoken
            import numpy as np

            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

            print("Downloading text from the internet:", self.url)

            os.makedirs(self.root_dir, exist_ok=True)
            input_file_path = os.path.join(self.root_dir, 'input.txt')
            if not os.path.exists(input_file_path):
                with open(input_file_path, 'w', encoding='utf-8') as f:
                    f.write(requests.get(self.url, headers=headers).text)

            with open(input_file_path, 'r', encoding='utf-8') as f:
                data = f.read()
            n = len(data)
            train_data = data[:int(n*0.9)]
            val_data = data[int(n*0.9):]

            # encode with tiktoken gpt2 bpe
            enc = tiktoken.get_encoding("gpt2")
            train_ids = enc.encode_ordinary(train_data)
            val_ids = enc.encode_ordinary(val_data)
            print(f"train has {len(train_ids):,} tokens")
            print(f"val has {len(val_ids):,} tokens")

            # export to bin files
            train_ids = np.array(train_ids, dtype=np.uint16)
            val_ids = np.array(val_ids, dtype=np.uint16)

            train_ids.tofile(os.path.join(self.root_dir, 'train.bin'))
            val_ids.tofile(os.path.join(self.root_dir, 'val.bin'))
        else:
            raise Exception("Datasets not found!")

@dataclass
class ModelConfig(BaseConfig):
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float
    bias: bool

@dataclass
class AttnConfig(BaseConfig):
    mlp_ratio: int
    kv_head: Optional[int] = None
    is_mla:bool = False

@dataclass
class PosConfig(BaseConfig):
    pass

@dataclass
class NormConfig(BaseConfig):
    pass

@dataclass
class TrainConfig(BaseConfig):
    dtype: str
    batch_size: int
    gradient_accumulation_steps: int
    max_iters: int
    lr_decay_iters: int
    warmup_iters: int
    eval_iters: int
    eval_interval: int
    log_interval: int
    min_lr: float
    learning_rate: float
    grad_clip: Optional[float]
    weight_decay: float
    beta1: float
    beta2: float
    patience: Optional[int]
    decay_lr: bool
    save_ckpt: bool = True

#########################################################################################

class AttnType(Enum):
    MHA = AttnConfig(name="attn_mha", mlp_ratio=4)
    
    GQA_STD = AttnConfig(name="attn_gqa_std", mlp_ratio=4, kv_head=2)
    MQA_STD = AttnConfig(name="attn_mqa_std", mlp_ratio=4, kv_head=1)
    MLA_STD = AttnConfig(name="attn_mla_std", mlp_ratio=4, is_mla=True)

    GQA_ISO = AttnConfig(name="attn_gqa_iso", mlp_ratio=4.667, kv_head=4)
    MQA_ISO = AttnConfig(name="attn_mqa_iso", mlp_ratio=4.917, kv_head=1)
    MLA_ISO = AttnConfig(name="attn_mla_iso", mlp_ratio=4.728, is_mla=True)

class PosType(Enum):
    WPE = PosConfig(name="pos_wpe")
    ROPE = PosConfig(name="pos_rope")

class NormType(Enum):
    RMS = NormConfig(name="norm_rms")
    LAYER = NormConfig(name="norm_layer")

class DatasetType(Enum):
    TINY_STORIES = DatasetConfig(name="TinyStories", url=None, dir_path="D:/Datasets/LLM/TinyStories_2m_row")
    THE_STACK = DatasetConfig(name="the-stack", url=None, dir_path="D:/Datasets/LLM/the-stack-2m-row")
    FINEWEB_EDU = DatasetConfig(name="fineweb-edu", url=None, dir_path="D:/Datasets/LLM/fineweb-edu-2m-row")
    OPENWEBTEXT = DatasetConfig(name="openwebtext", url=None, dir_path="D:/Datasets/LLM/openwebtext")
    SHAKESPEARE = DatasetConfig(name="shakespeare", url='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt', dir_path=None)

class ModelType(Enum):
    TINY_RESEARCH = ModelConfig(name="tiny_research", block_size=256, vocab_size=50257, n_layer=2, n_head=4, n_embd=32, dropout=0.0, bias=False)
    ATTN_RESEARCH = ModelConfig(name="attn_research", block_size=1024, vocab_size=50257, n_layer=2, n_head=64, n_embd=512, dropout=0.0, bias=False)
    POS_RESEARCH = ModelConfig(name="pos_research", block_size=2048, vocab_size=50257, n_layer=6, n_head=8, n_embd=768, dropout=0.0, bias=False)

    GPT2 = ModelConfig(name="gpt2", block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768, dropout=0.0, bias=True)  # 124M
    GPT2_MEDIUM = ModelConfig(name="gpt2-medium", block_size=1024, vocab_size=50257, n_layer=24, n_head=16, n_embd=1024, dropout=0.0, bias=True)  # 350M
    GPT2_LARGE = ModelConfig(name="gpt2-large", block_size=1024, vocab_size=50257, n_layer=36, n_head=20, n_embd=1280, dropout=0.0, bias=True)  # 774M
    GPT2_XL = ModelConfig(name="gpt2-xl", block_size=1024, vocab_size=50257, n_layer=48, n_head=25, n_embd=1600, dropout=0.0, bias=True)  # 1558M

class TrainType(Enum):
    DEFAULT = TrainConfig(
        name="train_default",
        dtype='float16',

        batch_size=2,
        gradient_accumulation_steps=4,

        max_iters=2000,
        lr_decay_iters=2000,
        warmup_iters=200,
        eval_iters=50,

        eval_interval=50,
        log_interval=10,

        min_lr=6e-5,
        learning_rate=1e-3,

        grad_clip=1.0,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.95,

        patience=None,
        decay_lr=True,
        save_ckpt=False
    )

#########################################################################################

@dataclass
class Configs:
    flash: bool
    model_type: ModelType
    attn_type: AttnType
    pos_type: PosType
    norm_type: NormType
    train_type: Optional[TrainType] = None
    dataset_type: Optional[DatasetType] = None
    out_root: str = os.path.join(project_root_global, "out")
    in_training: bool = False
    
    @property
    def out_dir(self) -> str:
        if self.train_type is None or self.dataset_type is None:
            return None
        
        import os
        flash_str = "flash" if self.flash else "no_flash"
        return os.path.join(self.out_root, self.dataset_type.value.name, self.model_type.value.name, self.norm_type.value.name, self.train_type.value.name, self.pos_type.value.name, flash_str, self.attn_type.value.name)
    
    def info(self):
        m = self.model_type.value
        t = self.train_type.value if self.train_type else None
        d = self.dataset_type.value if self.dataset_type else None

        info_str = [
            f"{'='*50}",
            f" CONFIGURATION: {self.model_type.name} on {self.dataset_type.name if self.dataset_type else 'None'} ",
            f"{'='*50}",
            f" Arsitektur  : {self.attn_type.name} | {self.pos_type.name} | {self.norm_type.name}",
            f" Flash Attn  : {'ACTIVE' if self.flash else 'INACTIVE'}",
            f" Params      : L={m.n_layer}, H={m.n_head}, E={m.n_embd}, B={m.block_size}",
            f" Training    : LR={t.learning_rate if t else '-'}, Batch={t.batch_size if t else '-'}, Accum={t.gradient_accumulation_steps if t else '-'}",
            f" Precision   : {t.dtype if t else '-'}",
            f" Output Dir  : {self.out_dir}",
            f"{'='*50}"
        ]
        return "\n".join(info_str)

    def to_dict(self):
        def _to_dict(obj):
            if isinstance(obj, Enum):
                return _to_dict(obj.value)
            elif hasattr(obj, '__dataclass_fields__'):
                return {k: _to_dict(getattr(obj, k)) for k in obj.__dataclass_fields__}
            elif isinstance(obj, dict):
                return {k: _to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return type(obj)(_to_dict(v) for v in obj)
            else:
                return obj
        return _to_dict(self)

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--no_flash", action="store_false", dest="flash")
parser.add_argument("--train_type", type=str, default="DEFAULT")
parser.add_argument("--dataset_type", type=str, default="FINEWEB_EDU")
parser.add_argument("--model_type", type=str, default="GPT2")
parser.add_argument("--attn_type", type=str, default="MHA")
parser.add_argument("--pos_type", type=str, default="ROPE")
parser.add_argument("--norm_type", type=str, default="RMS")

parser.add_argument("--max_new_tokens", type=int, default=100000)
parser.add_argument("--no_cache", action="store_false", dest="use_cache")
parser.add_argument("--start", type=str, default="the colors of the German flag are")
parser.add_argument("--print_out", action="store_true")
parser.add_argument("--pretrained", action="store_true")
parser.add_argument("--compare", action="store_true")

parser.add_argument("--no_resume", action="store_false", dest="resume")

args = parser.parse_args()

train_type = TrainType[args.train_type.upper()]
dataset_type = DatasetType[args.dataset_type.upper()]
model_type = ModelType[args.model_type.upper()]
attn_type = AttnType[args.attn_type.upper()]
pos_type = PosType[args.pos_type.upper()]
norm_type = NormType[args.norm_type.upper()]

args_configs = Configs(flash=args.flash, train_type=train_type, dataset_type=dataset_type, model_type=model_type, attn_type=attn_type, pos_type=pos_type, norm_type=norm_type)
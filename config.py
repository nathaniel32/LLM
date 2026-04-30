from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from abc import ABC
import os

@dataclass
class BaseConfig(ABC):
    name: str

@dataclass
class DatasetConfig(BaseConfig):
    root_dir: str = field(init=False)
    url: Optional[str]
    dir_path: Optional[str]

    def __post_init__(self):
        project_root = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.join(project_root, 'datasets', self.name)

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

@dataclass
class PosConfig(BaseConfig):
    pass

@dataclass
class NormConfig(BaseConfig):
    pass

@dataclass
class TrainConfig(BaseConfig):
    batch_size: int
    max_iters: int
    gradient_accumulation_steps: int
    eval_interval: int
    eval_iters: int
    learning_rate: float
    patience: Optional[int]
    dtype: str
    grad_clip: Optional[float]
    warmup_iters: int
    lr_decay_iters: int
    weight_decay: float
    min_lr: float
    log_interval: int
    decay_lr: bool
    beta1: float
    beta2: float

#########################################################################################

class DatasetType(Enum):
    THE_STACK = DatasetConfig(name="the-stack", url=None, dir_path="D:/Datasets/LLM/the-stack-2m-row")
    FINEWEB_EDU = DatasetConfig(name="fineweb-edu", url=None, dir_path="D:/Datasets/LLM/fineweb-edu-2m-row")
    OPENWEBTEXT = DatasetConfig(name="openwebtext", url=None, dir_path="D:/Datasets/LLM/openwebtext")
    SHAKESPEARE = DatasetConfig(name="shakespeare", url='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt', dir_path=None)

class ModelType(Enum):
    RESEARCH = ModelConfig(name="model_research", block_size=512, vocab_size=50257, n_layer=6, n_head=8, n_embd=512, dropout=0.0, bias=False)
    SMALL = ModelConfig(name="model_small", block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768, dropout=0.0, bias=False)
    MEDIUM = ModelConfig(name="model_medium", block_size=1024, vocab_size=50257, n_layer=24, n_head=16, n_embd=1024, dropout=0.0, bias=False)
    LARGE = ModelConfig(name="model_large", block_size=1024, vocab_size=50257, n_layer=36, n_head=20, n_embd=1280, dropout=0.0, bias=False)
    XL = ModelConfig(name="model_xl", block_size=1024, vocab_size=50257, n_layer=48, n_head=25, n_embd=1600, dropout=0.0, bias=False)
    
    GPT2 = ModelConfig(name="gpt2", block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768,  dropout=0.0, bias=True)  # 124M
    GPT2_MEDIUM = ModelConfig(name="gpt2-medium", block_size=1024, vocab_size=50257, n_layer=24, n_head=16, n_embd=1024, dropout=0.0, bias=True)  # 350M
    GPT2_LARGE = ModelConfig(name="gpt2-large", block_size=1024, vocab_size=50257, n_layer=36, n_head=20, n_embd=1280, dropout=0.0, bias=True)  # 774M
    GPT2_XL = ModelConfig(name="gpt2-xl", block_size=1024, vocab_size=50257, n_layer=48, n_head=25, n_embd=1600, dropout=0.0, bias=True)  # 1558M

class AttnType(Enum):
    MHA = AttnConfig(name="attn_mha", mlp_ratio=4)
    
    GQA_ISO = AttnConfig(name="attn_gqa_iso", mlp_ratio=4.75, kv_head=2)
    MQA_ISO = AttnConfig(name="attn_mqa_iso", mlp_ratio=4.875, kv_head=1)
    MLA_ISO = AttnConfig(name="attn_mla_iso", mlp_ratio=5.125)
    
    GQA_STD = AttnConfig(name="attn_gqa_std", mlp_ratio=4, kv_head=2)
    MQA_STD = AttnConfig(name="attn_mqa_std", mlp_ratio=4, kv_head=1)
    MLA_STD = AttnConfig(name="attn_mla_std", mlp_ratio=4)

class PosType(Enum):
    WPE = PosConfig(name="pos_wpe")
    ROPE = PosConfig(name="pos_rope")

class NormType(Enum):
    RMS = NormConfig(name="norm_rms")
    LAYER = NormConfig(name="norm_layer")

class TrainType(Enum):
    DEFAULT = TrainConfig(
        name="train_default",
        batch_size=2,
        max_iters=600000,
        gradient_accumulation_steps=5,
        eval_interval=500,
        eval_iters=200,
        learning_rate=6e-4,
        patience=20,
        dtype='float16',
        grad_clip=1.0,
        warmup_iters=2000,
        lr_decay_iters=600000,
        weight_decay=1e-1,
        min_lr=6e-5,
        log_interval=1,
        decay_lr=True,
        beta1=0.9,
        beta2=0.95
    )
    RESEARCH = TrainConfig(
        name="train_research",
        batch_size=6,
        gradient_accumulation_steps=8,       # Effective batch size = 48
        max_iters=15_000,                    # 48*15000*512 = ~368M Token
        eval_interval=250,
        eval_iters=100,
        learning_rate=5e-4,
        patience=None,
        dtype='float32',
        grad_clip=1.0,
        warmup_iters=1000,
        lr_decay_iters=15_000,
        weight_decay=1e-1,
        min_lr=5e-5,
        log_interval=10,
        decay_lr=True,
        beta1=0.9,
        beta2=0.95
    )
    FAST_RESEARCH = TrainConfig(
        name="train_fast_research",
        batch_size=4,
        gradient_accumulation_steps=4,
        max_iters=3000,
        eval_interval=100,
        eval_iters=50,
        learning_rate=3e-4,
        patience=None,
        dtype='float16',
        grad_clip=1.0,
        warmup_iters=200,
        lr_decay_iters=3000,
        weight_decay=0.05,
        min_lr=1e-5,
        log_interval=10,
        decay_lr=True,
        beta1=0.9,
        beta2=0.95
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
    
    @property
    def out_dir(self) -> str:
        import os
        flash_str = "flash" if self.flash else "no_flash"
        return os.path.join('out', self.dataset_type.value.name, self.model_type.value.name, self.norm_type.value.name, self.train_type.value.name, self.pos_type.value.name, self.attn_type.value.name, flash_str)
    
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
    
    def info(self):
        m = self.model_type.value
        t = self.train_type.value
        
        info_str = [
            f"{'='*50}",
            f" CONFIGURATION: {self.model_type.name} on {self.dataset_type.name} ",
            f"{'='*50}",
            f" Arsitektur  : {self.attn_type.name} | {self.pos_type.name} | {self.norm_type.name}",
            f" Flash Attn  : {'ACTIVE' if self.flash else 'INACTIVE'}",
            f" Params      : L={m.n_layer}, H={m.n_head}, E={m.n_embd}, B={m.block_size}",
            f" Training    : LR={t.learning_rate}, Batch={t.batch_size}, Accum={t.gradient_accumulation_steps}",
            f" Precision   : {t.dtype}",
            f" Output Dir  : {self.out_dir}",
            f"{'='*50}"
        ]
        return "\n".join(info_str)

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--no_flash", action="store_false", dest="flash")
parser.add_argument("--train_type", type=str, default="RESEARCH")
parser.add_argument("--dataset_type", type=str, default="OPENWEBTEXT")
parser.add_argument("--model_type", type=str, default="RESEARCH")
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

args_configs = Configs(flash=args.flash, train_type=TrainType[args.train_type.upper()], dataset_type=DatasetType[args.dataset_type.upper()], model_type=ModelType[args.model_type.upper()], attn_type=AttnType[args.attn_type.upper()], pos_type=PosType[args.pos_type.upper()], norm_type=NormType[args.norm_type.upper()])
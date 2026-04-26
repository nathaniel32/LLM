from enum import Enum

class AttnType(Enum):
    MHA = "mha"
    GQA = "gqa"
    MQA = "mqa"
    MLA = "mla"

class PosType(Enum):
    WPE = "wpe"
    ROPE = "rope"

class NormType(Enum):
    RMS = "rms"
    LAYER = "layer"

model_configs = {
    'small': dict(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768,  dropout=0.0, bias=False, gqa_kv_head=4),
    'medium': dict(block_size=1024, vocab_size=50257, n_layer=24, n_head=16, n_embd=1024, dropout=0.0, bias=False, gqa_kv_head=4),
    'large': dict(block_size=1024, vocab_size=50257, n_layer=36, n_head=20, n_embd=1280, dropout=0.0, bias=False, gqa_kv_head=6),
    'xl': dict(block_size=1024, vocab_size=50257, n_layer=48, n_head=25, n_embd=1600, dropout=0.0, bias=False, gqa_kv_head=6),
}

dataset_configs = {
    'openwebtext': {
        'url': None,
        'dir_path': None
    },
    'shakespeare': {
        'url': 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
        'dir_path': None
    },
    'simple_text': {
        'url': 'https://raw.githubusercontent.com/uwgraphics/VEP2_TCP_SimpleText/refs/heads/main/N3/N37535.txt',
        'dir_path': None
    },
    'wiki_indo': {
        'url': 'https://id.wikipedia.org/w/index.php?title=Indonesia&action=raw',
        'dir_path': None
    },
    'wiki_indo_json': {
        'url': 'https://id.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&titles=Indonesia&explaintext=1',
        'dir_path': None
    }
}
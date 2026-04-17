import os
import requests
import tiktoken
import numpy as np
import torch
from contextlib import nullcontext

class Train:
    def __init__(self):
        seed = 1337
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = 'bfloat16'
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        self.ctx = nullcontext() if self.device == 'cpu' else torch.amp.autocast(device_type=self.device, dtype=ptdtype)

        self.out_dir = 'out'
        os.makedirs(self.out_dir, exist_ok=True)

        self.data_dir = "datasets/shakespeare"

    def prepare_dataset(self, data_url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", data_dir="datasets/shakespeare"):
        os.makedirs(data_dir, exist_ok=True)
        input_file_path = os.path.join(data_dir, 'input.txt')
        if not os.path.exists(input_file_path):
            with open(input_file_path, 'w', encoding='utf-8') as f:
                f.write(requests.get(data_url).text)

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

        train_ids.tofile(os.path.join(data_dir, 'train.bin'))
        val_ids.tofile(os.path.join(data_dir, 'val.bin'))

train = Train()
train.prepare_dataset()
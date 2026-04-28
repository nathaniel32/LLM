import os

root_dir = "D:/Datasets/LLM"
huggingface_dir = os.path.join(root_dir, "huggingface")
huggingface_cache_dir = os.path.join(huggingface_dir, "cache")

os.environ["HF_HOME"] = huggingface_dir
os.environ["HF_DATASETS_CACHE"] = huggingface_cache_dir

from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

num_proc = 4
num_proc_load_dataset = num_proc
output_dir = os.path.join(root_dir, "openwebtext")
os.makedirs(output_dir, exist_ok=True)

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset)

    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')

    def process(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)
        return {'ids': ids, 'len': len(ids)}

    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(output_dir, f'{split}.bin')
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)

        arr.flush()
        assert idx == arr_len
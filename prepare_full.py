import os

os.environ["HF_HOME"]        = "D:/Datasets/LLM/huggingface"
os.environ["HF_DATASETS_CACHE"] = "D:/Datasets/LLM/huggingface/cache"

from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

OUTPUT_DIR = "D:/Datasets/LLM/openwebtext"
os.makedirs(OUTPUT_DIR, exist_ok=True)

num_proc = 4
num_proc_load_dataset = 8
batch_size = 1024 

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    dataset = load_dataset("Skylion007/openwebtext", num_proc=num_proc_load_dataset, trust_remote_code=True)

    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')

    def process(examples):
        outputs = {"ids": [], "len": []}
        for text in examples["text"]:
            ids = enc.encode_ordinary(text)
            ids.append(enc.eot_token)
            outputs["ids"].append(ids)
            outputs["len"].append(len(ids))
        return outputs

    tokenized = split_dataset.map(
        process,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        keep_in_memory=False,
        remove_columns=['text'],
        desc="tokenizing the splits"
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)

        filename = os.path.join(OUTPUT_DIR, f'{split}.bin')

        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        idx = 0

        for batch in tqdm(dset.iter(batch_size=batch_size), desc=f'writing {filename}'):
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)

        arr.flush()
        print(f"- Saved: {filename}")
import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

def prepare_data(root_dir="datasets", max_samples=10000, datasets="openwebtext", val_ratio=0.01):
    out_dir = os.path.join(root_dir, datasets)
    
    os.makedirs(out_dir, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")
    dataset = load_dataset(datasets, split="train", streaming=True)

    train_ids = []
    val_ids = []

    val_step = int(1 / val_ratio) if val_ratio > 0 else 0
    print({"val_step": val_step})

    for i, example in enumerate(tqdm(dataset, desc="Processing")):
        if i >= max_samples:
            break

        ids = enc.encode_ordinary(example["text"])
        ids.append(enc.eot_token)

        if val_step > 0 and i % val_step == 0:
            val_ids.extend(ids)
        else:
            train_ids.extend(ids)

    train_arr = np.array(train_ids, dtype=np.uint16)
    val_arr = np.array(val_ids, dtype=np.uint16)

    train_arr.tofile(os.path.join(out_dir, "train.bin"))
    val_arr.tofile(os.path.join(out_dir, "val.bin"))

    print(f"Train tokens: {len(train_arr):,}")
    print(f"Val tokens: {len(val_arr):,}")

def read_bin_file(file_path, num_tokens=50):
    enc = tiktoken.get_encoding("gpt2")
    tokens_arr = np.fromfile(file_path, dtype=np.uint16)
    sample_tokens = tokens_arr[:num_tokens]
    text = enc.decode(sample_tokens)
    return len(tokens_arr), text

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=1000000)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--check_data", action="store_true")
    args = parser.parse_args()

    if not args.check_data:
        prepare_data(max_samples=args.max_samples, val_ratio=args.val_ratio)
    else:
        total_tokens, decoded_text = read_bin_file("datasets/openwebtext/val.bin", 500)
        print(f"- Total: {total_tokens}")
        print(f"- Text: \n{decoded_text}")
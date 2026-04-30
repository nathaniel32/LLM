import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

def prepare_data(max_samples, val_ratio, hs_path, hs_name, out_root_dir, col):
    data_name = input("Dataname: ")
    out_dir = os.path.join(out_root_dir, data_name)
    os.makedirs(out_dir, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")
    dataset = load_dataset(hs_path, name=hs_name, split="train", streaming=True)
    dataset = dataset.shuffle(seed=42, buffer_size=10_000)

    val_size = int(max_samples * val_ratio)
    train_size = max_samples - val_size
    print(f"Train samples: {train_size:,}, Val samples: {val_size:,}")

    train_tokens_count = 0
    val_tokens_count = 0

    with open(os.path.join(out_dir, "train.bin"), "wb") as f_train, \
         open(os.path.join(out_dir, "val.bin"), "wb") as f_val:

        for i, example in enumerate(tqdm(dataset, total=max_samples, desc="Processing")):
            if i >= max_samples:
                break

            ids = enc.encode_ordinary(example[col])
            ids.append(enc.eot_token)
            arr = np.array(ids, dtype=np.uint16)

            if i >= train_size:
                arr.tofile(f_val)
                val_tokens_count += len(ids)
            else:
                arr.tofile(f_train)
                train_tokens_count += len(ids)

    print(f"Train tokens: {train_tokens_count:,}")
    print(f"Val tokens: {val_tokens_count:,}")
    print(f"Out dir: {out_dir}")

def read_bin_file(file_path, num_tokens=50):
    enc = tiktoken.get_encoding("gpt2")
    tokens_arr = np.fromfile(file_path, dtype=np.uint16)
    sample_tokens = tokens_arr[:num_tokens]
    text = enc.decode(sample_tokens)
    return len(tokens_arr), text

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, required=True)
    parser.add_argument("--val_ratio", type=float, default=0.01)
    parser.add_argument("--hs_path", type=str, default='HuggingFaceFW/fineweb-edu')
    parser.add_argument("--hs_name", type=str)
    parser.add_argument("--out_root_dir", type=str, default='datasets')
    parser.add_argument("--check_path", type=str)
    parser.add_argument("--col", type=str, default='text')

    args = parser.parse_args()

    if args.check_path is None:
        prepare_data(max_samples=args.max_samples, val_ratio=args.val_ratio, hs_path=args.hs_path, hs_name=args.hs_name, out_root_dir=args.out_root_dir, col=args.col)
    else:
        total_tokens, decoded_text = read_bin_file(args.check_path, args.max_samples)
        print(f"- Text: \n{decoded_text}")
        print(f"- Total: {total_tokens}")

# py prepare_stream.py --max_samples 10000 --hs_path "HuggingFaceFW/fineweb-edu" --out_root_dir D:\Datasets\LLM
# py prepare_stream.py --max_samples 10000 --hs_path "allenai/c4" --hs_name id
# py prepare_stream.py --max_samples 10000 --hs_path "bigcode/the-stack" --out_root_dir D:\Datasets\LLM --col content
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


class OpenWebTextProcessor:
    """
    Download, tokenize, and save the OpenWebText dataset
    into binary files suitable for LLM training.
    """

    def __init__(self, root_dir: str = "D:/Datasets/LLM", num_proc: int = 4, num_proc_load: int = 8, batch_size: int = 1024, val_ratio: float = 0.0005, seed: int = 2357, encoding: str = "gpt2"):
        self.root_dir = root_dir
        self.num_proc = num_proc
        self.num_proc_load = num_proc_load
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.seed = seed

        self.huggingface_dir = os.path.join(root_dir, "huggingface")
        self.cache_dir = os.path.join(self.huggingface_dir, "cache")
        self.tokenized_cache_dir = os.path.join(self.cache_dir, "tokenized")
        self.output_dir = os.path.join(root_dir, "openwebtext")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.tokenized_cache_dir, exist_ok=True)

        self.enc = tiktoken.get_encoding(encoding)

    @property
    def cache_file_names(self) -> dict:
        return {
            "train": os.path.join(self.tokenized_cache_dir, "train_tokenized.arrow"),
            "val":   os.path.join(self.tokenized_cache_dir, "val_tokenized.arrow"),
        }

    def load(self):
        """Load the OpenWebText dataset from HuggingFace."""
        print("Loading dataset...")
        dataset = load_dataset("openwebtext", num_proc=self.num_proc_load, trust_remote_code=True)
        split = dataset["train"].train_test_split(test_size=self.val_ratio, seed=self.seed, shuffle=True)
        split["val"] = split.pop("test")
        return split

    def _process_batch(self, examples: dict) -> dict:
        """Tokenize a batch of text examples."""
        outputs = {"ids": [], "len": []}
        for text in examples["text"]:
            ids = self.enc.encode_ordinary(text)
            ids.append(self.enc.eot_token)
            outputs["ids"].append(ids)
            outputs["len"].append(len(ids))
        return outputs

    def tokenize(self, split_dataset):
        """Tokenize the dataset splits."""
        print("Tokenizing...")
        return split_dataset.map(self._process_batch, batched=True, batch_size=1000, num_proc=self.num_proc, keep_in_memory=False, remove_columns=["text"], desc="tokenizing the splits", cache_file_names=self.cache_file_names)

    def save(self, tokenized):
        """Write tokenized splits to binary .bin files."""
        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"], dtype=np.uint64)
            filename = os.path.join(self.output_dir, f"{split}.bin")
            arr = np.memmap(filename, dtype=np.uint16, mode="w+", shape=(int(arr_len),))
            idx = 0

            for batch in tqdm(dset.iter(batch_size=self.batch_size), desc=f"writing {split}.bin"):
                arr_batch = np.concatenate(batch["ids"])
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)

            arr.flush()
            print(f"  Saved: {filename}  ({arr_len:,} tokens)")

    def run(self):
        """Run the full pipeline: load → tokenize → save."""
        split_dataset = self.load()
        tokenized = self.tokenize(split_dataset)
        self.save(tokenized)
        print("Done!")


if __name__ == "__main__":
    processor = OpenWebTextProcessor(root_dir="D:/Datasets/LLM", num_proc=4, num_proc_load=8, batch_size=1024)
    processor.run()
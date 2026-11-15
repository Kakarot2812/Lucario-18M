import os
from datasets import load_dataset
import numpy as np
import tiktoken
import torch
from tqdm import tqdm
from config import (
    CACHE_DIR,
    DATASET_CONFIG,
    DATASET_NAME,
    MAX_TRAIN_TOKENS,
    MAX_VAL_TOKENS,
    device,
)

bpe_tokenizer = tiktoken.get_encoding("gpt2")
vocab_size = bpe_tokenizer.n_vocab
EOT_TOKEN = bpe_tokenizer.eot_token  # 50256 (<|endoftext|>)


def encode(s):
    return bpe_tokenizer.encode(s, allowed_special={"<|endoftext|>"})


def decode(ids):
    return bpe_tokenizer.decode(ids)


os.makedirs(CACHE_DIR, exist_ok=True)
train_bin = os.path.join(CACHE_DIR, "train_eot_tokens.bin")
val_bin = os.path.join(CACHE_DIR, "val_eot_tokens.bin")


def _build_fineweb_memmap():
    print(f"Streaming {DATASET_NAME} ({DATASET_CONFIG}) into uint16 memmap shards...")
    stream = load_dataset(
        DATASET_NAME,
        name=DATASET_CONFIG,
        split="train",
        streaming=True,
    )

    val_mm = np.memmap(val_bin, dtype=np.uint16, mode="w+", shape=(MAX_VAL_TOKENS,))
    train_mm = np.memmap(train_bin, dtype=np.uint16, mode="w+", shape=(MAX_TRAIN_TOKENS,))

    val_idx = 0
    train_idx = 0
    pbar = tqdm(total=MAX_VAL_TOKENS + MAX_TRAIN_TOKENS, unit="tok", desc="Tokenizing FineWeb-Edu")

    for sample in stream:
        # encode_ordinary avoids crashes on literal special-token text inside web pages
        tokens = bpe_tokenizer.encode_ordinary(sample["text"])
        tokens.append(EOT_TOKEN)
        arr = np.asarray(tokens, dtype=np.uint16)

        if val_idx < MAX_VAL_TOKENS:
            take = min(len(arr), MAX_VAL_TOKENS - val_idx)
            val_mm[val_idx : val_idx + take] = arr[:take]
            val_idx += take
            pbar.update(take)
            arr = arr[take:]

        if len(arr) > 0 and train_idx < MAX_TRAIN_TOKENS:
            take = min(len(arr), MAX_TRAIN_TOKENS - train_idx)
            train_mm[train_idx : train_idx + take] = arr[:take]
            train_idx += take
            pbar.update(take)

        if val_idx >= MAX_VAL_TOKENS and train_idx >= MAX_TRAIN_TOKENS:
            break

    val_mm.flush()
    train_mm.flush()
    pbar.close()


if not (os.path.exists(train_bin) and os.path.exists(val_bin)):
    _build_fineweb_memmap()
else:
    print("Loading cached FineWeb-Edu memmap shards...")

train_data = np.memmap(train_bin, dtype=np.uint16, mode="r")
val_data = np.memmap(val_bin, dtype=np.uint16, mode="r")
print(f"Train tokens: {len(train_data):,} | Validation tokens: {len(val_data):,}")


def get_batch(split, batch_size, context_length):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - context_length - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(d[i : i + context_length].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(d[i + 1 : i + 1 + context_length].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


if __name__ == "__main__":
    xb, yb = get_batch("train", 4, 64)
    print("Sample batch shapes:", xb.shape, yb.shape)

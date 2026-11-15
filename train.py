import glob
import inspect
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import tiktoken
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from config import (
    BATCH_SIZE,
    CACHE_DIR,
    CONTEXT_LENGTH,
    DIM,
    DROPOUT,
    EVAL_INTERVAL,
    GRAD_ACCUM_STEPS,
    GRAD_CLIP,
    LR,
    MAX_ITERS,
    MIN_LR,
    N_HEADS,
    N_KV_HEADS,
    N_LAYERS,
    TIE_WEIGHTS,
    WARMUP_ITERS,
)
from model import ModernLLM


def setup_ddp():
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Different seed per rank so each GPU samples distinct random batches
    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)
    return ddp, rank, local_rank, world_size, device


def find_shard(cache_dir: str, split: str) -> str:
    direct = os.path.join(cache_dir, f"{split}.bin")
    if os.path.exists(direct):
        return direct
    matches = sorted(glob.glob(os.path.join(cache_dir, f"*{split}*.bin")))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find {split}.bin shard in '{cache_dir}'. Run data.py first.")


class MemmapBatchLoader:
    def __init__(self, bin_path: str, batch_size: int, seq_len: int, device: torch.device):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device

    def __len__(self) -> int:
        return len(self.data)

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.data) - self.seq_len - 1
        starts = torch.randint(0, max_start, (self.batch_size,))
        x = torch.stack(
            [torch.from_numpy(self.data[s : s + self.seq_len].astype(np.int64)) for s in starts]
        )
        y = torch.stack(
            [torch.from_numpy(self.data[s + 1 : s + 1 + self.seq_len].astype(np.int64)) for s in starts]
        )
        return (
            x.to(self.device, non_blocking=True),
            y.to(self.device, non_blocking=True),
        )


def get_lr(step: int) -> float:
    if step < WARMUP_ITERS:
        return LR * (step + 1) / WARMUP_ITERS
    if step >= MAX_ITERS:
        return MIN_LR
    progress = (step - WARMUP_ITERS) / max(1, MAX_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR + coeff * (LR - MIN_LR)


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    val_loader: MemmapBatchLoader,
    eval_iters: int,
    amp_ctx,
    ddp: bool,
    device: torch.device,
) -> float:
    model.eval()
    losses = torch.zeros(eval_iters, device=device)
    for k in range(eval_iters):
        x, y = val_loader.get_batch()
        with amp_ctx:
            _, loss = model(x, y)
        losses[k] = loss.detach()
    mean_loss = losses.mean()
    if ddp:
        dist.all_reduce(mean_loss, op=dist.ReduceOp.AVG)
    model.train()
    return mean_loss.item()


def save_checkpoint(
    path: str,
    raw_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    best_val_loss: float,
    vocab_size: int,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    torch.save(
        {
            "step": step,
            "best_val_loss": best_val_loss,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": {
                "vocab_size": vocab_size,
                "dim": DIM,
                "n_layers": N_LAYERS,
                "n_heads": N_HEADS,
                "n_kv_heads": N_KV_HEADS,
                "context_length": CONTEXT_LENGTH,
                "dropout": DROPOUT,
                "tie_weights": TIE_WEIGHTS,
            },
        },
        tmp_path,
    )
    os.replace(tmp_path, path)


def build_model(vocab_size: int, device: torch.device) -> torch.nn.Module:
    init_params = inspect.signature(ModernLLM.__init__).parameters
    candidate_args = {
        "vocab_size": vocab_size,
        "dim": DIM,
        "d_model": DIM,
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "n_kv_heads": N_KV_HEADS,
        "context_length": CONTEXT_LENGTH,
        "max_seq_len": CONTEXT_LENGTH,
        "dropout": DROPOUT,
        "tie_weights": TIE_WEIGHTS,
    }
    kwargs = {k: v for k, v in candidate_args.items() if k in init_params}
    return ModernLLM(**kwargs).to(device)


def main():
    ddp, rank, local_rank, world_size, device = setup_ddp()
    is_master = rank == 0

    try:
        train_path = find_shard(CACHE_DIR, "train")
        val_path = find_shard(CACHE_DIR, "val")

        train_loader = MemmapBatchLoader(train_path, BATCH_SIZE, CONTEXT_LENGTH, device)
        val_loader = MemmapBatchLoader(val_path, BATCH_SIZE, CONTEXT_LENGTH, device)

        enc = tiktoken.get_encoding("gpt2")
        vocab_size = enc.n_vocab

        raw_model = build_model(vocab_size, device)
        model = DDP(raw_model, device_ids=[local_rank]) if ddp else raw_model

        use_cuda = device.type == "cuda"
        major, _ = torch.cuda.get_device_capability(device) if use_cuda else (0, 0)
        amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
            if use_cuda
            else nullcontext()
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=(use_cuda and amp_dtype == torch.float16)
        )

        # Apply weight decay only to 2D+ weight matrices (skip norms and embeddings)
        decay_params = [p for p in raw_model.parameters() if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for p in raw_model.parameters() if p.requires_grad and p.dim() < 2]
        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": 0.1},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=LR,
            betas=(0.9, 0.95),
            fused=use_cuda,
        )

        ckpt_dir = "/kaggle/working/checkpoints"
        latest_ckpt = os.path.join(ckpt_dir, "latest.pt")
        best_ckpt = os.path.join(ckpt_dir, "best_model.pt")
        start_step = 0
        best_val_loss = float("inf")

        if os.path.exists(latest_ckpt):
            ckpt = torch.load(latest_ckpt, map_location=device)
            raw_model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_step = ckpt["step"] + 1
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            if is_master:
                print(f"Resumed from step {start_step} (best val loss: {best_val_loss:.4f})")

        if is_master:
            n_params = sum(p.numel() for p in raw_model.parameters())
            tokens_per_step = BATCH_SIZE * CONTEXT_LENGTH * GRAD_ACCUM_STEPS * world_size
            print(f"DDP World Size: {world_size} GPU(s) | Device: {torch.cuda.get_device_name(device)}")
            print(f"Train tokens: {len(train_loader):,} | Val tokens: {len(val_loader):,}")
            print(f"ModernLLM | GQA {N_HEADS}Q/{N_KV_HEADS}KV | Params: {n_params / 1e6:.2f}M")
            print(f"Mixed precision: {amp_dtype} | Effective tokens/step: {tokens_per_step:,}")

        model.train()
        t0 = time.time()
        log_interval = 50
        eval_iters = 50

        for step in range(start_step, MAX_ITERS):
            lr = get_lr(step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            accum_loss = torch.zeros(1, device=device)

            for micro_step in range(GRAD_ACCUM_STEPS):
                x, y = train_loader.get_batch()
                sync_ctx = (
                    model.no_sync()
                    if (ddp and micro_step < GRAD_ACCUM_STEPS - 1)
                    else nullcontext()
                )
                with sync_ctx:
                    with amp_ctx:
                        _, loss = model(x, y)
                        scaled_loss = loss / GRAD_ACCUM_STEPS
                    accum_loss += scaled_loss.detach()
                    scaler.scale(scaled_loss).backward()

            if ddp:
                dist.all_reduce(accum_loss, op=dist.ReduceOp.AVG)

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            if is_master and (step % log_interval == 0 or step == MAX_ITERS - 1):
                dt = time.time() - t0
                t0 = time.time()
                tok_sec = (
                    BATCH_SIZE * CONTEXT_LENGTH * GRAD_ACCUM_STEPS * world_size * log_interval
                ) / max(dt, 1e-6)
                print(
                    f"step {step:5d}/{MAX_ITERS} | "
                    f"train loss {accum_loss.item():.4f} | "
                    f"lr {lr:.2e} | "
                    f"{tok_sec:,.0f} tok/s"
                )

            if (step > 0 and step % EVAL_INTERVAL == 0) or step == MAX_ITERS - 1:
                val_loss = estimate_loss(model, val_loader, eval_iters, amp_ctx, ddp, device)
                if is_master:
                    print(f"--- step {step:5d} | val loss {val_loss:.4f} ---")
                    save_checkpoint(
                        latest_ckpt, raw_model, optimizer, scaler, step, best_val_loss, vocab_size
                    )
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(
                            best_ckpt, raw_model, optimizer, scaler, step, best_val_loss, vocab_size
                        )
                        print(f"    Saved new best checkpoint -> {best_ckpt}")

        if is_master:
            print(f"Training complete. Best validation loss: {best_val_loss:.4f}")

    finally:
        if ddp and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

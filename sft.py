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
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP

from model import ModernLLM

# Portable paths (works on both local Mac/Linux and Kaggle)
CKPT_DIR = "/kaggle/working/checkpoints" if os.path.exists("/kaggle/working") else "checkpoints"
BASE_CKPT_PATH = (
    os.path.join(CKPT_DIR, "best_model.pt")
    if os.path.exists(os.path.join(CKPT_DIR, "best_model.pt"))
    else os.path.join(CKPT_DIR, "modern_llm_18m_gqa_base_fp16.pt")
)
SFT_CKPT_PATH = os.path.join(CKPT_DIR, "sft_best_model.pt")
SFT_DATASET = "yahma/alpaca-cleaned"

# Anti-overfitting SFT Hyperparameters
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2
EPOCHS = 2
LR = 3e-5
MIN_LR = 3e-6
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.10
GRAD_CLIP = 1.0
SFT_DROPOUT = 0.05
EVAL_INTERVAL = 100
REPLAY_RATIO = 0.15  # 15% FineWeb-Edu replay to prevent catastrophic forgetting


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
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)
    return ddp, rank, local_rank, world_size, device


def build_model_from_config(cfg: dict, device: torch.device) -> torch.nn.Module:
    init_params = inspect.signature(ModernLLM.__init__).parameters
    dim = cfg.get("dim", 256)
    ctx_len = cfg.get("context_length", 512)
    candidate_args = {
        "vocab_size": cfg.get("vocab_size", 50257),
        "dim": dim,
        "d_model": dim,
        "n_layers": cfg.get("n_layers", 6),
        "n_heads": cfg.get("n_heads", 8),
        "n_kv_heads": cfg.get("n_kv_heads", 2),
        "context_length": ctx_len,
        "max_seq_len": ctx_len,
        "dropout": SFT_DROPOUT,
        "tie_weights": cfg.get("tie_weights", True),
    }
    kwargs = {k: v for k, v in candidate_args.items() if k in init_params}
    return ModernLLM(**kwargs).to(device)


def format_prompt(instruction: str, user_input: str = "") -> str:
    instruction = instruction.strip()
    user_input = user_input.strip()
    if user_input:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{user_input}\n\n"
            f"### Response:\n"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def encode_sft_pair(
    enc: tiktoken.Encoding, instruction: str, response: str, max_seq_len: int, user_input: str = ""
) -> tuple[list[int], list[int]] | None:
    prompt_text = format_prompt(instruction, user_input)
    prompt_ids = enc.encode_ordinary(prompt_text)
    response_ids = enc.encode_ordinary(response.strip()) + [enc.eot_token]

    if len(prompt_ids) + len(response_ids) > max_seq_len + 1:
        return None

    full_ids = prompt_ids + response_ids
    labels = ([-100] * len(prompt_ids)) + response_ids
    x = full_ids[:-1]
    y = labels[1:]
    assert y[-1] == enc.eot_token, "Target sequence must end with EOS"
    return (x, y)


def prepare_alpaca_examples(
    enc: tiktoken.Encoding, max_seq_len: int
) -> tuple[list[tuple[list[int], list[int]]], int]:
    ds = load_dataset(SFT_DATASET, split="train")
    examples = []
    seen = set()
    dropped_truncated = 0

    for row in ds:
        inst = row["instruction"].strip()
        inp = row.get("input", "").strip()
        response_text = row["output"].strip()
        if not response_text:
            continue

        key = (inst, inp)
        if key in seen:
            continue
        seen.add(key)

        pair = encode_sft_pair(enc, inst, response_text, max_seq_len, inp)
        if pair is None:
            dropped_truncated += 1
            continue
        examples.append(pair)

    return examples, dropped_truncated


def build_concise_curriculum(
    enc: tiktoken.Encoding, max_seq_len: int
) -> list[tuple[list[int], list[int]]]:
    """2,500 leak-free arithmetic and concise instruction-following pairs."""
    rng = np.random.default_rng(999)
    curriculum = []

    # Exclude exact pairs used in eval_suite_250.py to keep evaluation strictly held-out
    eval_blocked = set()
    for i in range(15):
        eval_blocked.add(("add", 7 + i * 3, 4 + i * 2))
        eval_blocked.add(("sub", 45 + i * 4, 12 + i * 2))
    for i in range(10):
        eval_blocked.add(("mul", 3 + i, 4 + (i % 6)))
        b = 2 + (i % 7)
        ans = 3 + i
        eval_blocked.add(("div", b * ans, b))

    # 1. 2,000 leak-free arithmetic pairs (+, -, *, /)
    ops = ["add", "sub", "mul", "div"]
    while len(curriculum) < 2000:
        op = ops[len(curriculum) % 4]
        if op == "add":
            a, b = int(rng.integers(2, 95)), int(rng.integers(2, 95))
            if ("add", a, b) in eval_blocked:
                continue
            ans = a + b
            q = f"What is {a} + {b}?" if len(curriculum) % 2 == 0 else f"Calculate {a} plus {b}."
            r = f"{ans}" if len(curriculum) % 3 == 0 else f"{a} + {b} = {ans}."
        elif op == "sub":
            b = int(rng.integers(2, 60))
            ans = int(rng.integers(1, 80))
            a = b + ans
            if ("sub", a, b) in eval_blocked:
                continue
            q = f"What is {a} - {b}?" if len(curriculum) % 2 == 0 else f"Subtract {b} from {a}."
            r = f"{ans}" if len(curriculum) % 3 == 0 else f"{a} - {b} = {ans}."
        elif op == "mul":
            a, b = int(rng.integers(2, 16)), int(rng.integers(2, 16))
            if ("mul", a, b) in eval_blocked:
                continue
            ans = a * b
            q = (
                f"What is {a} multiplied by {b}?"
                if len(curriculum) % 2 == 0
                else f"What is {a} x {b}?"
            )
            r = f"{ans}" if len(curriculum) % 3 == 0 else f"{a} multiplied by {b} is {ans}."
        else:
            b = int(rng.integers(2, 13))
            ans = int(rng.integers(2, 20))
            a = b * ans
            if ("div", a, b) in eval_blocked:
                continue
            q = f"What is {a} divided by {b}?"
            r = f"{ans}" if len(curriculum) % 3 == 0 else f"{a} divided by {b} is {ans}."

        pair = encode_sft_pair(enc, q, r, max_seq_len)
        if pair is not None:
            curriculum.append(pair)

    # 2. 500 concise constraint, classification, and short-answer pairs
    templates = [
        ("Answer with only Yes or No: Is water a liquid at room temperature?", "Yes"),
        ("Answer with only Yes or No: Can birds fly?", "Yes"),
        ("Answer with only Yes or No: Is 100 smaller than 15?", "No"),
        ("Answer with only Yes or No: Do squares have five sides?", "No"),
        ("Answer with only Yes or No: Is snow cold?", "Yes"),
        ("Write the word 'cloud' in all capital letters.", "CLOUD"),
        ("Write the word 'ocean' in all capital letters.", "OCEAN"),
        ("Write the word 'forest' in all capital letters.", "FOREST"),
        ("Repeat the exact word: mango", "mango"),
        ("Repeat the exact word: silver", "silver"),
        ("Repeat the exact word: planet", "planet"),
        ("Name the opposite of tall.", "Short"),
        ("Name the opposite of heavy.", "Light"),
        ("Name the opposite of early.", "Late"),
        ("Name the opposite of loud.", "Quiet"),
        ("Name three common shapes separated by commas.", "Circle, square, triangle"),
        ("Name three domestic animals separated by commas.", "Dog, cat, horse"),
        ("Name two sources of renewable energy.", "Solar energy and wind energy."),
        ("Give a one-sentence definition of gravity.", "Gravity is the natural force that pulls objects with mass toward one another."),
        ("Give a one-sentence definition of evaporation.", "Evaporation is the process by which a liquid changes into a gas when heated."),
        ("Give a one-sentence definition of an island.", "An island is a piece of land completely surrounded by water."),
        ("Tom is older than Sam, and Sam is older than Leo. Who is the oldest?", "Tom"),
        ("Tom is older than Sam, and Sam is older than Leo. Who is the youngest?", "Leo"),
        ("If yesterday was Wednesday, what day is today?", "Thursday"),
        ("If tomorrow is Sunday, what day is today?", "Saturday"),
    ]
    for q, r in templates * 20:
        pair = encode_sft_pair(enc, q, r, max_seq_len)
        if pair is not None:
            curriculum.append(pair)

    return curriculum


def load_replay_examples(
    max_seq_len: int, target_count: int
) -> list[tuple[list[int], list[int]]]:
    shard_paths = sorted(glob.glob("data_cache/*train*.bin"))
    if not shard_paths or target_count <= 0:
        return []

    tokens = np.memmap(shard_paths[0], dtype=np.uint16, mode="r")
    max_start = len(tokens) - (max_seq_len + 1)
    if max_start <= 0:
        return []

    rng = np.random.default_rng(123)
    starts = rng.integers(0, max_start, size=target_count)
    replay = []
    for s in starts:
        chunk = tokens[int(s) : int(s) + max_seq_len + 1].astype(np.int64).tolist()
        replay.append((chunk[:-1], chunk[1:]))
    return replay


class SFTBatchSampler:
    def __init__(
        self,
        examples: list[tuple[list[int], list[int]]],
        batch_size: int,
        max_seq_len: int,
        pad_token_id: int,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.examples = examples[rank::world_size]
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.device = device
        self.rng = np.random.default_rng(100 + rank)
        self.indices = np.arange(len(self.examples))
        self.pos = 0
        self.shuffle()

    def shuffle(self):
        self.rng.shuffle(self.indices)
        self.pos = 0

    def __len__(self) -> int:
        return max(1, len(self.examples) // self.batch_size)

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pos + self.batch_size > len(self.indices):
            self.shuffle()

        batch_idx = self.indices[self.pos : self.pos + self.batch_size]
        self.pos += self.batch_size
        batch = [self.examples[i] for i in batch_idx]

        raw_max = max(len(x) for x, _ in batch)
        seq_len = next((b for b in (128, 256, 384, 512) if b >= raw_max and b <= self.max_seq_len), self.max_seq_len)

        x_tensor = torch.full((self.batch_size, seq_len), self.pad_token_id, dtype=torch.long)
        y_tensor = torch.full((self.batch_size, seq_len), -100, dtype=torch.long)

        for i, (x, y) in enumerate(batch):
            length = min(len(x), seq_len)
            x_tensor[i, :length] = torch.tensor(x[:length], dtype=torch.long)
            y_tensor[i, :length] = torch.tensor(y[:length], dtype=torch.long)

        return (
            x_tensor.to(self.device, non_blocking=True),
            y_tensor.to(self.device, non_blocking=True),
        )


def get_lr(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return LR * (step + 1) / max(1, warmup_steps)
    if step >= total_steps:
        return MIN_LR
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR + coeff * (LR - MIN_LR)


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    val_loader: SFTBatchSampler,
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


@torch.no_grad()
def generate_response(
    model: torch.nn.Module,
    enc: tiktoken.Encoding,
    prompt_text: str,
    max_seq_len: int,
    device: torch.device,
    use_sft_template: bool = True,
    max_new_tokens: int = 80,
) -> str:
    model.eval()
    formatted = format_prompt(prompt_text) if use_sft_template else prompt_text
    prompt_ids = enc.encode_ordinary(formatted)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -max_seq_len:])
        next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if next_tok.item() == enc.eot_token:
            break
        idx = torch.cat([idx, next_tok], dim=1)

    generated_ids = idx[0, len(prompt_ids) :].tolist()
    return enc.decode(generated_ids).strip()


def main():
    ddp, rank, local_rank, world_size, device = setup_ddp()
    is_master = rank == 0

    try:
        if not os.path.exists(BASE_CKPT_PATH):
            raise FileNotFoundError(f"Base checkpoint not found at {BASE_CKPT_PATH}")

        ckpt = torch.load(BASE_CKPT_PATH, map_location=device)
        model_cfg = ckpt["config"]
        max_seq_len = model_cfg.get("context_length", 512)
        enc = tiktoken.get_encoding("gpt2")

        if is_master:
            print(f"Loading and auditing {SFT_DATASET}...")
            alpaca_examples, dropped = prepare_alpaca_examples(enc, max_seq_len)
        if ddp:
            dist.barrier()
        if not is_master:
            alpaca_examples, dropped = prepare_alpaca_examples(enc, max_seq_len)

        # Held-out Alpaca-only validation split (5% = ~2,540 examples)
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(alpaca_examples))
        val_size = max(500, int(0.05 * len(alpaca_examples)))
        val_examples = [alpaca_examples[i] for i in perm[:val_size]]
        train_alpaca = [alpaca_examples[i] for i in perm[val_size:]]

        # Add leak-free concise/arithmetic curriculum + 15% FineWeb-Edu replay to train split
        concise_examples = build_concise_curriculum(enc, max_seq_len)
        base_train_count = len(train_alpaca) + len(concise_examples)
        replay_count = int(base_train_count * REPLAY_RATIO)
        replay_examples = load_replay_examples(max_seq_len, replay_count)
        train_examples = train_alpaca + concise_examples + replay_examples

        train_loader = SFTBatchSampler(
            train_examples, BATCH_SIZE, max_seq_len, enc.eot_token, device, rank, world_size
        )
        val_loader = SFTBatchSampler(
            val_examples, BATCH_SIZE, max_seq_len, enc.eot_token, device, rank, world_size
        )

        raw_model = build_model_from_config(model_cfg, device)
        raw_state = {
            k.replace("_orig_mod.", ""): v.float() for k, v in ckpt["model_state_dict"].items()
        }
        raw_model.load_state_dict(raw_state)
        model = DDP(raw_model, device_ids=[local_rank]) if ddp else raw_model

        use_cuda = device.type == "cuda"
        major, _ = torch.cuda.get_device_capability(device) if use_cuda else (0, 0)
        amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
            if use_cuda
            else (torch.amp.autocast(device_type="mps", dtype=torch.bfloat16) if device.type == "mps" else nullcontext())
        )
        scaler = torch.amp.GradScaler("cuda", enabled=(use_cuda and amp_dtype == torch.float16))

        decay_params = [p for p in raw_model.parameters() if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for p in raw_model.parameters() if p.requires_grad and p.dim() < 2]
        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": WEIGHT_DECAY},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=LR,
            betas=(0.9, 0.95),
            fused=use_cuda,
        )

        steps_per_epoch = max(1, len(train_loader) // GRAD_ACCUM_STEPS)
        total_steps = steps_per_epoch * EPOCHS
        warmup_steps = max(20, int(total_steps * WARMUP_RATIO))

        if is_master:
            os.makedirs(CKPT_DIR, exist_ok=True)
            print(f"Loaded base checkpoint : {BASE_CKPT_PATH} on {device}")
            print(f"Full-EOS Alpaca pairs  : {len(alpaca_examples):,} (dropped {dropped} truncated > {max_seq_len} tok)")
            print(f"Train mix              : {len(train_alpaca):,} Alpaca + {len(concise_examples):,} Concise/Math + {len(replay_examples):,} Replay")
            print(f"Held-out Alpaca Val    : {len(val_examples):,}")
            print(f"Total steps            : {total_steps} ({EPOCHS} epochs) | Peak LR: {LR:.1e} -> Min LR: {MIN_LR:.1e}")

        model.train()
        best_val_loss = float("inf")
        t0 = time.time()

        for step in range(total_steps):
            lr = get_lr(step, total_steps, warmup_steps)
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

            if is_master and (step % 10 == 0 or step == total_steps - 1):
                dt = time.time() - t0
                t0 = time.time()
                print(
                    f"sft step {step:4d}/{total_steps} | "
                    f"train loss {accum_loss.item():.4f} | "
                    f"lr {lr:.2e} | {dt:.1f}s"
                )

            if (step > 0 and step % EVAL_INTERVAL == 0) or step == total_steps - 1:
                val_loss = estimate_loss(model, val_loader, 40, amp_ctx, ddp, device)
                if is_master:
                    print(f"--- sft step {step:4d} | held-out SFT val loss {val_loss:.4f} ---")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        tmp_path = f"{SFT_CKPT_PATH}.tmp"
                        torch.save(
                            {
                                "step": step,
                                "best_val_loss": best_val_loss,
                                "model_state_dict": raw_model.state_dict(),
                                "config": model_cfg,
                                "prompt_format": "### Instruction:\n{instruction}\n\n### Response:\n",
                            },
                            tmp_path,
                        )
                        os.replace(tmp_path, SFT_CKPT_PATH)
                        print(f"    Saved new best SFT checkpoint -> {SFT_CKPT_PATH}")

        if is_master:
            print(f"\nSFT Complete! Best held-out SFT val loss: {best_val_loss:.4f}")
            best_sft = torch.load(SFT_CKPT_PATH, map_location=device)
            raw_model.load_state_dict(best_sft["model_state_dict"])

            print("\n=== Post-SFT Deterministic Verification (temp=0) ===")
            checks = [
                ("What is the chemical formula of water?", True),
                ("What is 9 multiplied by 6?", True),
                ("Answer with only Yes or No: Is ice frozen water?", True),
                ("Once upon a time, a young clockmaker found a brass key hidden inside an old pocket watch.", False),
            ]
            for prompt_str, is_inst in checks:
                mode_tag = "SFT" if is_inst else "RAW_STORY"
                print(f"\n[{mode_tag}] {prompt_str}")
                print("-" * 60)
                print(generate_response(raw_model, enc, prompt_str, max_seq_len, device, is_inst))

    finally:
        if ddp and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

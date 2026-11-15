import argparse
import inspect
import re
import time
from collections import Counter

import tiktoken
import torch
from model import ModernLLM


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    state = {k.replace("_orig_mod.", ""): v.float() for k, v in ckpt["model_state_dict"].items()}

    init_params = inspect.signature(ModernLLM.__init__).parameters
    dim, ctx = cfg.get("dim", 256), cfg.get("context_length", 512)
    kwargs = {
        k: v
        for k, v in {
            "vocab_size": cfg.get("vocab_size", 50257),
            "dim": dim,
            "d_model": dim,
            "n_layers": cfg.get("n_layers", 6),
            "n_heads": cfg.get("n_heads", 8),
            "n_kv_heads": cfg.get("n_kv_heads", 2),
            "context_length": ctx,
            "max_seq_len": ctx,
            "dropout": 0.0,
            "tie_weights": cfg.get("tie_weights", True),
        }.items()
        if k in init_params
    }
    model = ModernLLM(**kwargs).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, ctx


def trim_to_last_sentence(text: str) -> str:
    """Avoid mid-sentence truncation if generation hits max_tokens before EOS."""
    text = text.strip()
    if not text or text[-1] in ".!?":
        return text
    matches = list(re.finditer(r"[.!?](?=\s|$)", text))
    if matches:
        return text[: matches[-1].end()].strip()
    return text


@torch.no_grad()
def generate_text(
    model,
    enc: tiktoken.Encoding,
    ctx: int,
    device: torch.device,
    prompt: str,
    mode: str = "sft",
    max_tokens: int = 64,
    temperature: float = 0.25,
    top_k: int = 30,
    top_p: float = 0.85,
    rep_penalty: float = 1.22,
    freq_penalty: float = 0.35,
    no_repeat_ngram: int = 3,
    seed: int = 42,
) -> tuple[str, int, float]:
    torch.manual_seed(seed)
    formatted = f"### Instruction:\n{prompt.strip()}\n\n### Response:\n" if mode == "sft" else prompt
    p_ids = enc.encode_ordinary(formatted)[-(ctx - max_tokens) :]
    idx = torch.tensor([p_ids], dtype=torch.long, device=device)

    gen = []
    hit_eos = False
    t0 = time.time()

    for _ in range(max_tokens):
        out = model(idx[:, -ctx:])
        logits = (out[0] if isinstance(out, tuple) else out)[:, -1, :].float()

        # 1. Repetition & linear frequency penalties BEFORE top-k/top-p truncation
        if gen:
            counts = Counter(gen)
            unique_ids = torch.tensor(list(counts.keys()), dtype=torch.long, device=device)
            freq_counts = torch.tensor(list(counts.values()), dtype=torch.float32, device=device)

            if rep_penalty != 1.0:
                sl = logits[:, unique_ids]
                logits[:, unique_ids] = torch.where(sl < 0, sl * rep_penalty, sl / rep_penalty)

            if freq_penalty > 0.0:
                logits[:, unique_ids] -= freq_penalty * freq_counts

        # 2. Block exact n-gram loops
        if no_repeat_ngram >= 2 and len(gen) >= no_repeat_ngram - 1:
            n = no_repeat_ngram
            prefix = tuple(gen[-(n - 1) :])
            banned = {
                gen[i + n - 1]
                for i in range(len(gen) - (n - 1))
                if tuple(gen[i : i + n - 1]) == prefix
            }
            if banned:
                logits[:, torch.tensor(list(banned), dtype=torch.long, device=device)] = -float("inf")

        # 3. Temperature + Top-K + Top-P nucleus filtering
        if temperature <= 0.0:
            next_tok = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            scaled = logits / temperature
            if top_k > 0:
                k = min(top_k, scaled.size(-1))
                kth_vals = torch.topk(scaled, k).values[:, -1, None]
                scaled = torch.where(scaled < kth_vals, -float("inf"), scaled)

            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
                cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cum_probs > top_p
                sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                sorted_mask[:, 0] = False
                scaled.scatter_(1, sorted_indices, torch.where(sorted_mask, -float("inf"), sorted_logits))

            probs = torch.softmax(scaled, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)

        tid = next_tok.item()
        if tid == enc.eot_token:
            hit_eos = True
            break
        gen.append(tid)
        idx = torch.cat([idx, next_tok], dim=1)

        if mode == "sft" and len(gen) >= 3 and "\n###" in enc.decode(gen[-6:]):
            hit_eos = True
            break

    elapsed = max(1e-6, time.time() - t0)
    text = enc.decode(gen).split("\n### Instruction:")[0].split("\n\n###")[0].strip()
    if not hit_eos:
        text = trim_to_last_sentence(text)
    return text, len(gen), len(gen) / elapsed


def main():
    parser = argparse.ArgumentParser(description="Generate text with ModernLLM-GQA (18.57M)")
    parser.add_argument("--prompt", type=str, default="Explain photosynthesis in simple terms.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/modern_llm_18m_gqa_sft_fp16.pt")
    parser.add_argument("--mode", type=str, choices=["sft", "base"], default="sft")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temp", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--top-p", type=float, default=0.85)
    parser.add_argument("--rep-penalty", type=float, default=1.22)
    parser.add_argument("--freq-penalty", type=float, default=0.35)
    parser.add_argument("--no-repeat-ngram", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = get_device()
    enc = tiktoken.get_encoding("gpt2")
    model, ctx = load_model(args.checkpoint, device)

    output, n_toks, tok_per_sec = generate_text(
        model=model,
        enc=enc,
        ctx=ctx,
        device=device,
        prompt=args.prompt,
        mode=args.mode,
        max_tokens=args.max_tokens,
        temperature=args.temp,
        top_k=args.top_k,
        top_p=args.top_p,
        rep_penalty=args.rep_penalty,
        freq_penalty=args.freq_penalty,
        no_repeat_ngram=args.no_repeat_ngram,
        seed=args.seed,
    )

    print(f"Checkpoint : {args.checkpoint} ({args.mode.upper()} | {device})")
    print(f"Hyperparams: temp={args.temp}, top_k={args.top_k}, top_p={args.top_p}, rep={args.rep_penalty}, freq={args.freq_penalty}, max_tok={args.max_tokens}")
    print(f"Prompt     : {args.prompt}")
    print("-" * 60)
    print(output)
    print("-" * 60)
    print(f"Generated {n_toks} tokens ({tok_per_sec:.1f} tok/s)")


if __name__ == "__main__":
    main()

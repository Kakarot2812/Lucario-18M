import argparse
import inspect
import re
import time
from collections import Counter

import tiktoken
import torch
from model import ModernLLM

STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "her",
    "was", "one", "our", "out", "has", "had", "that", "this", "with", "from",
    "they", "will", "would", "there", "their", "what", "about", "which", "when",
    "make", "like", "time", "just", "know", "take", "into", "year", "your",
    "some", "could", "them", "other", "than", "then", "now", "look", "only",
    "come", "its", "over", "think", "also", "back", "after", "use", "two",
    "how", "work", "first", "well", "way", "even", "new", "want", "because",
    "any", "these", "give", "most", "into", "such", "called", "process",
}


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
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, ctx


def _build_word_token_groups(enc: tiktoken.Encoding) -> dict[str, list[int]]:
    """Map normalized lowercase alpha words (>=3 chars) to all BPE token variants (with/without leading space/caps)."""
    groups: dict[str, list[int]] = {}
    for tid in range(enc.n_vocab):
        if tid == enc.eot_token:
            continue
        piece = enc.decode([tid]).strip().lower()
        if len(piece) >= 3 and piece.isalpha() and piece not in STOPWORDS:
            groups.setdefault(piece, []).append(tid)
    return groups


def trim_to_sentences(text: str, max_sentences: int = 0) -> str:
    text = text.strip()
    if not text:
        return text
    matches = list(re.finditer(r"[.!?](?=\s|$)", text))
    if max_sentences > 0 and len(matches) >= max_sentences:
        return text[: matches[max_sentences - 1].end()].strip()
    if text[-1] not in ".!?" and matches:
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
    temperature: float = 0.30,
    top_k: int = 35,
    top_p: float = 0.88,
    rep_penalty: float = 1.25,
    freq_penalty: float = 0.45,
    presence_penalty: float = 0.60,
    no_repeat_ngram: int = 3,
    max_sentences: int = 2,
    seed: int = 42,
) -> tuple[str, int, float]:
    torch.manual_seed(seed)
    formatted = f"### Instruction:\n{prompt.strip()}\n\n### Response:\n" if mode == "sft" else prompt
    p_ids = enc.encode_ordinary(formatted)[-(ctx - max_tokens) :]
    idx = torch.tensor([p_ids], dtype=torch.long, device=device)

    word_groups = _build_word_token_groups(enc)
    gen: list[int] = []
    hit_eos = False
    t0 = time.time()

    for _ in range(max_tokens):
        out = model(idx[:, -ctx:])
        logits = (out[0] if isinstance(out, tuple) else out)[:, -1, :].float()

        if gen:
            # 1. Exact token ID repetition, frequency & presence penalties
            counts = Counter(gen)
            unique_ids = torch.tensor(list(counts.keys()), dtype=torch.long, device=device)
            freq_counts = torch.tensor(list(counts.values()), dtype=torch.float32, device=device)

            if rep_penalty != 1.0:
                sl = logits[:, unique_ids]
                logits[:, unique_ids] = torch.where(sl < 0, sl * rep_penalty, sl / rep_penalty)

            if freq_penalty > 0.0:
                logits[:, unique_ids] -= freq_penalty * freq_counts

            if presence_penalty > 0.0:
                logits[:, unique_ids] -= presence_penalty

            # 2. Case/space-insensitive content-word penalty (prevents "glucose ... glucose ... glucose")
            decoded_so_far = enc.decode(gen).lower()
            used_words = Counter(w for w in re.findall(r"[a-z]{3,}", decoded_so_far) if w not in STOPWORDS)
            if used_words and (freq_penalty > 0.0 or presence_penalty > 0.0):
                variant_ids = []
                variant_penalties = []
                for word, w_cnt in used_words.items():
                    tids = word_groups.get(word)
                    if tids:
                        pen = (presence_penalty * 0.85) + (freq_penalty * 0.85 * w_cnt)
                        for t in tids:
                            variant_ids.append(t)
                            variant_penalties.append(pen)
                if variant_ids:
                    v_idx = torch.tensor(variant_ids, dtype=torch.long, device=device)
                    v_pen = torch.tensor(variant_penalties, dtype=torch.float32, device=device)
                    logits[:, v_idx] -= v_pen

        # 3. Exact n-gram blocking
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

        # 4. Temperature + Top-K + Top-P sampling
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

        cur_text = enc.decode(gen)
        if mode == "sft" and "\n###" in cur_text:
            hit_eos = True
            break
        if max_sentences > 0 and len(re.findall(r"[.!?](?=\s|$)", cur_text)) >= max_sentences:
            hit_eos = True
            break

    elapsed = max(1e-6, time.time() - t0)
    text = enc.decode(gen).split("\n### Instruction:")[0].split("\n\n###")[0].strip()
    text = trim_to_sentences(text, max_sentences=max_sentences if not hit_eos else 0)
    return text, len(gen), len(gen) / elapsed


def main():
    parser = argparse.ArgumentParser(description="Generate text with Lucario-18M (18.57M GQA)")
    parser.add_argument("--prompt", type=str, default="Explain photosynthesis in simple terms.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/modern_llm_18m_gqa_sft_fp16.pt")
    parser.add_argument("--mode", type=str, choices=["sft", "base"], default="sft")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temp", type=float, default=0.30)
    parser.add_argument("--top-k", type=int, default=35)
    parser.add_argument("--top-p", type=float, default=0.88)
    parser.add_argument("--rep-penalty", type=float, default=1.25)
    parser.add_argument("--freq-penalty", type=float, default=0.45)
    parser.add_argument("--presence-penalty", type=float, default=0.60)
    parser.add_argument("--no-repeat-ngram", type=int, default=3)
    parser.add_argument("--max-sentences", type=int, default=2)
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
        presence_penalty=args.presence_penalty,
        no_repeat_ngram=args.no_repeat_ngram,
        max_sentences=args.max_sentences,
        seed=args.seed,
    )

    print(f"Checkpoint : {args.checkpoint} ({args.mode.upper()} | {device})")
    print(f"Prompt     : {args.prompt}")
    print("-" * 60)
    print(output)
    print("-" * 60)
    print(f"Generated {n_toks} tokens ({tok_per_sec:.1f} tok/s)")


if __name__ == "__main__":
    main()

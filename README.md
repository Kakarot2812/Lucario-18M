# ModernLLM-GQA (18.57M Decoder-Only Transformer from Scratch)

A compact, modern decoder-only Transformer built from scratch in PyTorch—featuring **Grouped-Query Attention (8Q / 2KV)**, **Rotary Position Embeddings (RoPE)**, **RMSNorm**, **SwiGLU**, **weight tying**, and **compact KV caching**—pretrained on **FineWeb-Edu** and supervised fine-tuned (SFT) on **Alpaca-Cleaned** with **15% pretraining replay** and **2,500 concise/arithmetic curriculum pairs**.

**Live Demo:** [Hugging Face Space (`kakarot2812/modern-llm-from-scratch`)](https://kakarot2812-modern-llm-from-scratch.hf.space)

---

## Architecture & Training Summary

| Specification | Details |
| :--- | :--- |
| **Parameters** | **18.57M** unique (`12.87M` tied GPT-2 BPE embeddings + `5.70M` 6-layer Transformer backbone) |
| **Dimensions** | `6` blocks, `dim = 256`, `context_length = 512`, `8` Query heads / `2` KV heads (**75% KV-cache reduction**) |
| **Pretraining** | `HuggingFaceFW/fineweb-edu` (`sample-10BT`) — `val_loss = 3.8602` |
| **SFT Alignment** | `58,384` sequences (`48,269` Alpaca-Cleaned + `2,500` concise/math + `15%` replay) — `val_loss = 2.6129` (`step 1600`) |

---

## Strict 5-Axis Evaluation Matrix (195 Unique Tasks, Apple M4 MPS)

Evaluated deterministically (`temp = 0.0`, `rep_penalty = 1.15`, `no_repeat_ngram = 3`) with zero output repair or substring leniency:

| Model (`Total / Backbone`) | MC-LogProb (50) | Strict Gen-QA (100) | Strict IFEval (25) | Strict Rubric (40 pts) | **Composite** | 512-Ctx KV Cache | EOS Stop |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GQA SFT (`18.57M / 5.70M`)** | 58.0% | **6.0%** | **16.0%** | **27.5%** (11/40) | **26.9** | **`768 KB`** | **80.0%** |
| **GPT-2 Base (`124.4M / 85.8M`)** | **72.0%** | 0.0% | 0.0% | 20.0% (8/40) | **23.0** | `18,432 KB` | 0.7% |
| **Pythia-70M (`70.40M / 18.9M`)** | 70.0% | 0.0% | 0.0% | 2.5% (1/40) | **18.1** | `6,144 KB` | 39.3% |
| **GQA Base (`18.57M / 5.70M`)** | **64.0%** | 0.0% | 0.0% | 2.5% (1/40) | **16.6** | **`768 KB`** | 2.1% |
| **Pythia-31M (`30.50M / 4.74M`)** | 56.0% | 0.0% | 0.0% | 5.0% (2/40) | **15.2** | `3,072 KB` | 0.0% |

---

## Quickstart Guide

### 1. Clone & Install
```bash
git clone [https://github.com/Kakarot2812/Modern_llm_from_scratch.git](https://github.com/Kakarot2812/Modern_llm_from_scratch.git)
cd Modern_llm_from_scratch
python3 -m venv .venv && source .venv/bin/activate
pip install torch tiktoken datasets transformers gradio huggingface_hub

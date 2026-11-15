# Strict 5-Axis Evaluation Matrix Report (195 Unique Tasks)

- **Timestamp (UTC):** 2026-11-13 10:04:22 UTC
- **Device:** `mps` | **Generative Decoding:** `temp=0.0`, `rep_penalty=1.15`, `no_repeat_ngram=3`
- **Grading Hygiene:** Zero output repair; multi-line/runaway answers rejected in Axis 2; exact list cardinality & single-token `Yes/No` enforced in Axis 3; contradictory distractors penalized with `0 pts` in Axis 4.

## 1. Capability & Alignment Matrix (Axes 1–4)

| Model (`Total / Backbone`) | Axis 1: MC-LogProb (50) | Axis 2: Strict Gen-QA (100) *(Fact / Math)* | Axis 3: Strict IFEval (25) | Axis 4: Strict Rubric (40 pts) | **Composite Score** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GQA Base (18.57M)** (`18.57M / 5.70M`) | **64.0%** | **0.0%** *(0% / 0%)* | **0.0%** | **2.5%** (1/40) | **16.6** |
| **GQA SFT (18.57M)** (`18.57M / 5.70M`) | **58.0%** | **6.0%** *(10% / 2%)* | **16.0%** | **27.5%** (11/40) | **26.9** |
| **Pythia-31M (30.5M)** (`30.50M / 4.74M`) | **56.0%** | **0.0%** *(0% / 0%)* | **0.0%** | **5.0%** (2/40) | **15.2** |
| **Pythia-70M (70.4M)** (`70.40M / 18.92M`) | **70.0%** | **0.0%** *(0% / 0%)* | **0.0%** | **2.5%** (1/40) | **18.1** |
| **GPT-2 Base (124.4M)** (`124.4M / 85.80M`) | **72.0%** | **0.0%** *(0% / 0%)* | **0.0%** | **20.0%** (8/40) | **23.0** |

## 2. Systems, GQA Memory Efficiency & Generation Hygiene (Axis 5)

| Model | 512-Ctx KV Cache (FP16) | Throughput (`tok/s`) | EOS Stop Rate | 4-Gram Diversity | Eval Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GQA Base (18.57M)** | **`768 KB`** | `432.6 tok/s` | 2.1% | 80.1% | 10.1s |
| **GQA SFT (18.57M)** | **`768 KB`** | `260.3 tok/s` | 80.0% | 94.1% | 8.9s |
| **Pythia-31M (30.5M)** | **`3072 KB`** | `507.5 tok/s` | 0.0% | 62.0% | 8.6s |
| **Pythia-70M (70.4M)** | **`6144 KB`** | `285.3 tok/s` | 39.3% | 74.4% | 13.2s |
| **GPT-2 Base (124.4M)** | **`18432 KB`** | `127.1 tok/s` | 0.7% | 84.4% | 34.4s |

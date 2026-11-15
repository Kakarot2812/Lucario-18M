import spaces
import gradio as gr
import tiktoken
import torch
from generate import generate_text, load_model

CPU_DEVICE = torch.device("cpu")
ENC = tiktoken.get_encoding("gpt2")

MODEL_SFT, CTX_SFT = load_model("checkpoints/modern_llm_18m_gqa_sft_fp16.pt", CPU_DEVICE)
MODEL_BASE, CTX_BASE = load_model("checkpoints/modern_llm_18m_gqa_base_fp16.pt", CPU_DEVICE)


@spaces.GPU(duration=30)
def run_inference(
    prompt: str,
    model_choice: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    rep_penalty: float,
    freq_penalty: float,
    presence_penalty: float,
    no_repeat_ngram: int,
    max_sentences: int,
    seed: int,
):
    if not prompt.strip():
        return "Please enter a prompt."

    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_sft = model_choice.startswith("Lucario-18M SFT")
    model = (MODEL_SFT if is_sft else MODEL_BASE).to(target_device)
    ctx = CTX_SFT if is_sft else CTX_BASE

    text, n_toks, tps = generate_text(
        model=model,
        enc=ENC,
        ctx=ctx,
        device=target_device,
        prompt=prompt,
        mode="sft" if is_sft else "base",
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
        rep_penalty=float(rep_penalty),
        freq_penalty=float(freq_penalty),
        presence_penalty=float(presence_penalty),
        no_repeat_ngram=int(no_repeat_ngram),
        max_sentences=int(max_sentences),
        seed=int(seed),
    )
    return f"{text}\n\n---\n*{n_toks} tokens • {tps:.1f} tok/s ({target_device})*"


with gr.Blocks(title="Lucario-18M (18.57M GQA Transformer from Scratch)") as demo:
    gr.Markdown(
        "# Lucario-18M (18.57M GQA Transformer from Scratch)\n"
        "Grouped-Query Attention (`8Q / 2KV`) + RoPE + RMSNorm + SwiGLU trained from scratch "
        "on FineWeb-Edu and SFT-aligned on Alpaca-Cleaned."
    )
    with gr.Row():
        with gr.Column(scale=1):
            prompt_in = gr.Textbox(lines=3, label="Prompt", value="Explain photosynthesis in simple terms.")
            ckpt_in = gr.Radio(
                ["Lucario-18M SFT (Aligned)", "Lucario-18M Base (Pretrained)"],
                value="Lucario-18M SFT (Aligned)",
                label="Checkpoint",
            )
            with gr.Row():
                max_tok_in = gr.Slider(16, 128, value=64, step=8, label="Max Tokens")
                max_sent_in = gr.Slider(0, 5, value=2, step=1, label="Max Sentences (0 = unlimited)")
            with gr.Row():
                temp_in = gr.Slider(0.0, 1.2, value=0.30, step=0.05, label="Temperature")
                top_k_in = gr.Slider(1, 100, value=35, step=1, label="Top-K")
                top_p_in = gr.Slider(0.1, 1.0, value=0.88, step=0.02, label="Top-P")
            with gr.Accordion("Anti-Repetition & Reproducibility Controls", open=True):
                with gr.Row():
                    rep_in = gr.Slider(1.0, 1.5, value=1.25, step=0.02, label="Repetition Penalty")
                    freq_in = gr.Slider(0.0, 1.2, value=0.45, step=0.05, label="Frequency Penalty")
                    pres_in = gr.Slider(0.0, 1.5, value=0.60, step=0.05, label="Presence Penalty")
                with gr.Row():
                    ngram_in = gr.Slider(0, 5, value=3, step=1, label="No-Repeat N-Gram")
                    seed_in = gr.Number(value=42, precision=0, label="Random Seed")
            with gr.Row():
                clear_btn = gr.Button("Clear")
                submit_btn = gr.Button("Generate", variant="primary")
        with gr.Column(scale=1):
            out_box = gr.Textbox(lines=14, label="Generated Response")

    inputs = [
        prompt_in, ckpt_in, max_tok_in, temp_in, top_k_in, top_p_in,
        rep_in, freq_in, pres_in, ngram_in, max_sent_in, seed_in,
    ]
    submit_btn.click(fn=run_inference, inputs=inputs, outputs=out_box)
    clear_btn.click(fn=lambda: ("", ""), outputs=[prompt_in, out_box])

    gr.Examples(
        examples=[
            ["Explain photosynthesis in simple terms.", "Lucario-18M SFT (Aligned)", 64, 0.30, 35, 0.88, 1.25, 0.45, 0.60, 3, 2, 42],
            ["Summarize in one sentence: Grouped-Query Attention reduces KV-cache memory by sharing key and value heads across multiple query heads.", "Lucario-18M SFT (Aligned)", 48, 0.0, 35, 0.88, 1.15, 0.25, 0.30, 3, 1, 42],
            ["Photosynthesis is the biological process by which", "Lucario-18M Base (Pretrained)", 64, 0.30, 35, 0.88, 1.25, 0.45, 0.60, 3, 2, 42],
        ],
        inputs=inputs,
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.launch()

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
    rep_penalty: float,
    freq_penalty: float,
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
        rep_penalty=float(rep_penalty),
        freq_penalty=float(freq_penalty),
    )
    return f"{text}\n\n---\n*{n_toks} tokens • {tps:.1f} tok/s ({target_device})*"


demo = gr.Interface(
    fn=run_inference,
    inputs=[
        gr.Textbox(lines=3, label="Prompt", value="Explain photosynthesis in simple terms."),
        gr.Radio(
            ["Lucario-18M SFT (Aligned)", "Lucario-18M Base (Pretrained)"],
            value="Lucario-18M SFT (Aligned)",
            label="Checkpoint",
        ),
        gr.Slider(16, 128, value=64, step=8, label="Max Tokens"),
        gr.Slider(0.0, 1.0, value=0.25, step=0.05, label="Temperature"),
        gr.Slider(1.0, 1.5, value=1.22, step=0.02, label="Repetition Penalty"),
        gr.Slider(0.0, 1.0, value=0.35, step=0.05, label="Frequency Penalty"),
    ],
    outputs=gr.Textbox(lines=8, label="Generated Response"),
    title="Lucario-18M (18.57M GQA Transformer from Scratch)",
    description=(
        "Grouped-Query Attention (8Q / 2KV) + RoPE + RMSNorm + SwiGLU trained from scratch "
        "on FineWeb-Edu and SFT-aligned on Alpaca-Cleaned."
    ),
    examples=[
        ["Explain photosynthesis in simple terms.", "Lucario-18M SFT (Aligned)", 64, 0.25, 1.22, 0.35],
        ["Summarize in one sentence: Grouped-Query Attention reduces KV-cache memory by sharing key and value heads across multiple query heads.", "Lucario-18M SFT (Aligned)", 48, 0.0, 1.15, 0.15],
        ["Photosynthesis is the biological process by which", "Lucario-18M Base (Pretrained)", 64, 0.25, 1.22, 0.35],
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()

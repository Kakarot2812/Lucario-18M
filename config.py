import torch

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Architecture Hyperparameters
DROPOUT = 0.0
CONTEXT_LENGTH = 512
DIM = 256
N_HEADS = 8
N_KV_HEADS = 2  # Grouped-Query Attention: 8 Q heads share 2 KV heads (4x smaller KV cache)
N_LAYERS = 6
TIE_WEIGHTS = True

assert DIM % N_HEADS == 0, "DIM must be divisible by N_HEADS"
assert N_HEADS % N_KV_HEADS == 0, "N_HEADS must be divisible by N_KV_HEADS"

# FineWeb-Edu (10BT sample) Dataset Settings
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
CACHE_DIR = "data_cache"
MAX_TRAIN_TOKENS = 1_000_000_000  # 200M tokens (~400 MB uint16 .bin file)
MAX_VAL_TOKENS = 5_000_000      # 2M tokens (~4 MB uint16 .bin file)

# Training & Cosine Schedule
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 8            # Effective batch = 64 seqs = 32,768 tokens/step
MAX_ITERS = 30518
EVAL_INTERVAL = 1000
LR = 5e-4
MIN_LR = 5e-5
WARMUP_ITERS = 600
GRAD_CLIP = 1.0

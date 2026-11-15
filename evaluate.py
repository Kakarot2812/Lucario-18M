import gc
import inspect
import json
import os
import re
import time
from datetime import datetime, timezone

import tiktoken
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import ModernLLM

BASE_PATH = "checkpoints/modern_llm_18m_gqa_base_fp16.pt"
SFT_PATH = "checkpoints/modern_llm_18m_gqa_sft_fp16.pt"
BENCHMARK_DIR = "benchmark"

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

gpt2_enc = tiktoken.get_encoding("gpt2")


def clear_device_cache():
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def load_modern_llm(path: str):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    state = {k.replace("_orig_mod.", ""): v.float() for k, v in ckpt["model_state_dict"].items()}
    init_params = inspect.signature(ModernLLM.__init__).parameters
    dim, ctx = cfg.get("dim", 256), cfg.get("context_length", 512)
    kwargs = {
        k: v
        for k, v in {
            "vocab_size": cfg.get("vocab_size", 50257),
            "dim": dim,
            "n_layers": cfg.get("n_layers", 6),
            "n_heads": cfg.get("n_heads", 8),
            "n_kv_heads": cfg.get("n_kv_heads", 2),
            "context_length": ctx,
            "dropout": 0.0,
            "tie_weights": cfg.get("tie_weights", True),
        }.items()
        if k in init_params
    }
    model = ModernLLM(**kwargs).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, ctx


# ==============================================================================
# AXIS 1: 4-Way Multiple-Choice Log-Likelihood Benchmark (50 Unique Questions)
# ==============================================================================

def build_axis1_mc() -> list[dict]:
    """25 Science/Factual + 25 Commonsense/Reasoning 4-way MC items (gold is index 0)."""
    raw = [
        # 25 Science & Factual
        ("The chemical formula of water is", [" H2O", " CO2", " NaCl", " O2"]),
        ("The planet closest to the Sun in our solar system is", [" Mercury", " Venus", " Mars", " Jupiter"]),
        ("The largest planet in our solar system is", [" Jupiter", " Saturn", " Earth", " Neptune"]),
        ("During photosynthesis, green plants absorb", [" carbon dioxide", " nitrogen", " hydrogen", " argon"]),
        ("To survive, humans need to breathe in", [" oxygen", " carbon monoxide", " helium", " methane"]),
        ("At standard atmospheric pressure, water boils at", [" 100 degrees Celsius", " 0 degrees Celsius", " 50 degrees Celsius", " 212 degrees Celsius"]),
        ("At standard atmospheric pressure, water freezes at", [" 0 degrees Celsius", " 100 degrees Celsius", " 32 degrees Celsius", " -100 degrees Celsius"]),
        ("The capital city of France is", [" Paris", " London", " Berlin", " Rome"]),
        ("The capital city of Japan is", [" Tokyo", " Beijing", " Seoul", " Bangkok"]),
        ("The chemical symbol for gold is", [" Au", " Ag", " Fe", " Cu"]),
        ("The organ that pumps blood throughout the human body is the", [" heart", " liver", " kidney", " stomach"]),
        ("The organelle known as the powerhouse of the cell is the", [" mitochondrion", " ribosome", " golgi", " vacuole"]),
        ("Planets are kept in orbit around the Sun by", [" gravity", " magnetism", " friction", " electricity"]),
        ("The largest ocean on Earth is the", [" Pacific Ocean", " Atlantic Ocean", " Indian Ocean", " Arctic Ocean"]),
        ("The play Romeo and Juliet was written by", [" William Shakespeare", " Charles Dickens", " Mark Twain", " Homer"]),
        ("The theory of general relativity was developed by", [" Albert Einstein", " Isaac Newton", " Galileo Galilei", " Niels Bohr"]),
        ("The primary language spoken in Brazil is", [" Portuguese", " Spanish", " French", " Italian"]),
        ("The hardest natural substance found on Earth is", [" diamond", " quartz", " iron", " graphite"]),
        ("The planet known as the Red Planet is", [" Mars", " Venus", " Saturn", " Mercury"]),
        ("The most abundant gas in Earth's atmosphere is", [" nitrogen", " oxygen", " argon", " carbon dioxide"]),
        ("The official currency used in Japan is the", [" yen", " euro", " dollar", " pound"]),
        ("An instrument used to measure temperature is a", [" thermometer", " barometer", " compass", " microscope"]),
        ("An electron carries a", [" negative charge", " positive charge", " neutral charge", " magnetic pole"]),
        ("A proton carries a", [" positive charge", " negative charge", " neutral charge", " thermal charge"]),
        ("In computer hardware, CPU stands for", [" Central Processing Unit", " Computer Personal Unit", " Control Power Utility", " Core Programming Unit"]),
        # 25 Commonsense & Reasoning
        ("If you drop a glass cup onto a concrete floor, it will most likely", [" shatter", " bounce", " melt", " float"]),
        ("When water is placed in a freezer overnight, it turns into", [" ice", " steam", " smoke", " oil"]),
        ("To unlock a locked wooden door, you typically use a", [" key", " spoon", " pillow", " leaf"]),
        ("If it is pouring rain outside and you walk without an umbrella, your clothes will get", [" wet", " dry", " dusty", " warm"]),
        ("People wear heavy coats and gloves in the winter to stay", [" warm", " cold", " wet", " hungry"]),
        ("If Alice is taller than Bob and Bob is taller than Charlie, the tallest person is", [" Alice", " Charlie", " Bob", " nobody"]),
        ("If today is Monday, then the day after tomorrow will be", [" Wednesday", " Tuesday", " Friday", " Sunday"]),
        ("If today is Friday, then yesterday was", [" Thursday", " Saturday", " Monday", " Wednesday"]),
        ("A baker puts bread dough into a hot oven so that it will", [" bake", " freeze", " dissolve", " shrink"]),
        ("When the Sun sets in the evening, the sky becomes", [" dark", " bright", " green", " loud"]),
        ("You use a pair of scissors primarily to", [" cut paper", " boil water", " paint walls", " measure weight"]),
        ("Fish use their gills to extract oxygen from", [" water", " sand", " rocks", " sunlight"]),
        ("Birds build nests in trees primarily to", [" lay eggs", " catch fish", " store snow", " hide rocks"]),
        ("If you have ten apples and give four away, the number of apples you have left is", [" six", " fourteen", " four", " ten"]),
        ("A library is a quiet building where people go to borrow and read", [" books", " shoes", " vegetables", " bicycles"]),
        ("Before crossing a busy street, pedestrians should look", [" both ways", " straight up", " at their shoes", " behind a tree"]),
        ("Plants grow toward a window because they need", [" sunlight", " darkness", " carpet", " glass"]),
        ("When a person is very tired at night, they usually go to", [" sleep", " run", " swim", " shout"]),
        ("To see tiny bacteria that are invisible to the naked eye, a scientist uses a", [" microscope", " telescope", " stethoscope", " periscope"]),
        ("To observe distant galaxies and stars at night, an astronomer uses a", [" telescope", " microscope", " thermometer", " ruler"]),
        ("Ice cream left out in the hot summer sun will quickly", [" melt", " freeze", " boil", " crystallize"]),
        ("The opposite of the word empty is", [" full", " hollow", " dark", " slow"]),
        ("The opposite of the word fast is", [" slow", " quick", " loud", " bright"]),
        ("A bicycle has two wheels, whereas a tricycle has", [" three wheels", " four wheels", " one wheel", " six wheels"]),
        ("If a train travels 60 miles in one hour at constant speed, in two hours it travels", [" 120 miles", " 30 miles", " 60 miles", " 180 miles"]),
    ]
    return [{"stem": s, "choices": c, "answer_idx": 0} for s, c in raw]


@torch.no_grad()
def eval_axis1_mc_logprob(model, encode_fn, mc_items: list[dict]) -> float:
    """Evaluate 4-way multiple-choice via length-normalized conditional log-likelihood."""
    correct = 0
    for item in mc_items:
        stem_ids = encode_fn(item["stem"])
        choice_scores = []
        for ch in item["choices"]:
            ch_ids = encode_fn(ch)
            full_ids = stem_ids + ch_ids
            x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
            y = torch.tensor(full_ids[1:], dtype=torch.long, device=device)
            out = model(x)
            logits = out.logits if hasattr(out, "logits") else (out[0] if isinstance(out, tuple) else out)
            log_probs = F.log_softmax(logits[0].float(), dim=-1)
            start = len(stem_ids) - 1
            cont_lp = log_probs[start:, :].gather(1, y[start:].unsqueeze(1)).squeeze(1)
            choice_scores.append(cont_lp.mean().item())
        pred_idx = int(torch.tensor(choice_scores).argmax().item())
        if pred_idx == item["answer_idx"]:
            correct += 1
    return round(100.0 * correct / len(mc_items), 1)


# ==============================================================================
# AXIS 2: Strict Single-Turn Generative QA (50 Factual + 50 Arithmetic = 100)
# ==============================================================================

def _strict_single_line(text: str) -> str:
    """Reject multi-line / runaway outputs by inspecting only line 1 and requiring <= 18 words."""
    clean = text.strip()
    if not clean:
        return ""
    lines = [ln.strip() for ln in clean.splitlines() if ln.strip()]
    if len(lines) > 1:
        return ""
    line = lines[0]
    if len(line.split()) > 18:
        return ""
    return line


def grade_generative_factual(response: str, acceptable: list[str]) -> bool:
    """Require a concise single-sentence answer ending with or equal to the target entity."""
    line = _strict_single_line(response)
    if not line:
        return False
    norm = re.sub(r"(?<=\d),(?=\d)", "", line.lower()).strip(" .!?;:\"'()")
    for ans in acceptable:
        target = re.sub(r"(?<=\d),(?=\d)", "", ans.lower()).strip()
        # Must either BE the entity or end as a direct declarative answer ("... is <entity>")
        if norm == target:
            return True
        if re.search(rf"(?:^|\b(?:is|are|was|were|called|equals)\s+)(?:the\s+|a\s+|an\s+)?{re.escape(target)}(?:\s+degrees\s+celsius|\s*°c|\s+km/s)?$", norm):
            return True
    return False


def grade_generative_arithmetic(response: str, expected_num: str) -> bool:
    """Require either the bare number or a single valid terminal equation '= N' / 'is N'."""
    line = _strict_single_line(response)
    if not line:
        return False
    norm = re.sub(r"(?<=\d),(?=\d)", "", line.lower()).strip(" .!?;:\"'()")
    if norm == str(expected_num):
        return True
    m = re.search(r"(?:=|is|equals)\s*(-?\d+(?:\.\d+)?)$", norm)
    return bool(m and m.group(1) == str(expected_num))


# ==============================================================================
# AXIS 3: IFEval-Lite Programmatic Format & Constraint Validators (25 Prompts)
# ==============================================================================

def _extract_strict_list(text: str) -> list[str] | None:
    """Parse explicit bullet/numbered/comma list; reject prose paragraphs."""
    clean = text.strip()
    if not clean or len(clean.split()) > 20:
        return None
    lines = [ln.strip() for ln in clean.splitlines() if ln.strip()]
    if len(lines) > 1:
        items = [re.sub(r"^(?:[-*•]|\d+[.)])\s*", "", ln).strip(" .,!?;:\"'").lower() for ln in lines]
        return [x for x in items if x]
    # Single-line list separated by commas or 'and'
    if ":" in clean:
        clean = clean.split(":", 1)[1].strip()
    parts = re.split(r"\s*,\s*(?:and\s+)?|\s+and\s+", clean.strip(" .,!?;:\"'"))
    items = [p.strip(" .,!?;:\"'").lower() for p in parts if p.strip(" .,!?;:\"'")]
    return items if len(items) >= 2 else None


def _count_sentences(text: str) -> int:
    clean = text.strip()
    if not clean or "\n" in clean:
        return 0 if not clean else len([ln for ln in clean.splitlines() if ln.strip()])
    sents = [s for s in re.split(r"[.!?]+(?:\s+|$)", clean) if s.strip()]
    return len(sents)


def build_axis3_constraints() -> list[dict]:
    prog_langs = {"python", "java", "c", "c++", "c#", "javascript", "typescript", "rust", "go", "golang", "ruby", "swift", "kotlin", "php", "r", "scala", "sql"}
    instruments = {"piano", "guitar", "violin", "drum", "drums", "flute", "trumpet", "saxophone", "cello", "clarinet", "harp"}
    ringed = {"saturn", "jupiter", "uranus", "neptune"}
    marine = {"whale", "blue whale", "dolphin", "porpoise", "seal", "sea lion", "walrus", "manatee", "otter", "sea otter", "orca"}
    citrus = {"orange", "lemon", "lime", "grapefruit", "tangerine", "mandarin", "pomelo", "clementine"}
    weekdays = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
    seasons = {"spring", "summer", "autumn", "fall", "winter"}

    def exact_yn(expected: str):
        return lambda r: r.strip().rstrip(".!").lower() == expected

    def exact_list(count: int, allowed: set[str]):
        def _fn(r: str) -> bool:
            items = _extract_strict_list(r)
            if not items or len(items) != count:
                return False
            return len(set(items)) == count and set(items).issubset(allowed)
        return _fn

    def strict_one_sentence(keywords: set[str]):
        def _fn(r: str) -> bool:
            clean = r.strip()
            return (
                "\n" not in clean
                and _count_sentences(clean) == 1
                and clean.endswith((".", "!", "?"))
                and 4 <= len(clean.split()) <= 30
                and any(re.search(rf"\b{re.escape(k)}\b", clean.lower()) for k in keywords)
            )
        return _fn

    def strict_entity(allowed: set[str]):
        def _fn(r: str) -> bool:
            clean = re.sub(r"^(?:a|an|the)\s+", "", r.strip().strip(" .,!?;:\"'").lower())
            return "\n" not in r.strip() and 1 <= len(clean.split()) <= 3 and clean in allowed
        return _fn

    return [
        {"prompt": "Answer with only Yes or No: Is fire hot?", "validator": exact_yn("yes")},
        {"prompt": "Answer with only Yes or No: Can fish breathe underwater?", "validator": exact_yn("yes")},
        {"prompt": "Answer with only Yes or No: Is 10 greater than 25?", "validator": exact_yn("no")},
        {"prompt": "Answer with only Yes or No: Are triangles four-sided shapes?", "validator": exact_yn("no")},
        {"prompt": "Answer with only Yes or No: Is the Sun a star?", "validator": exact_yn("yes")},
        {"prompt": "List three primary colors.", "validator": exact_list(3, {"red", "blue", "yellow"})},
        {"prompt": "Name two seasons of the year.", "validator": exact_list(2, seasons)},
        {"prompt": "Name three days of the week.", "validator": exact_list(3, weekdays)},
        {"prompt": "Write the word 'apple' in all capital letters.", "validator": lambda r: r.strip().strip("\"'.,!?") == "APPLE"},
        {"prompt": "Repeat the exact word: banana", "validator": lambda r: r.strip().strip("\"'.,!?") == "banana"},
        {"prompt": "Give a one-sentence definition of a river.", "validator": strict_one_sentence({"water", "stream", "flow", "flowing", "land"})},
        {"prompt": "Give a one-sentence definition of a library.", "validator": strict_one_sentence({"book", "books", "read", "reading", "collection"})},
        {"prompt": "Give a one-sentence definition of a bicycle.", "validator": strict_one_sentence({"wheel", "wheels", "pedal", "pedals", "ride", "vehicle"})},
        {"prompt": "Name one programming language.", "validator": strict_entity(prog_langs)},
        {"prompt": "Name one musical instrument.", "validator": strict_entity(instruments)},
        {"prompt": "Name one planet with rings.", "validator": strict_entity(ringed)},
        {"prompt": "Name one mammal that lives in the ocean.", "validator": strict_entity(marine)},
        {"prompt": "Name one citrus fruit.", "validator": strict_entity(citrus)},
        {"prompt": "Write a short greeting to a friend.", "validator": lambda r: "\n" not in r.strip() and 2 <= len(r.strip().split()) <= 15 and bool(re.match(r"^(?:hi|hello|hey|greetings|good\s+(?:morning|afternoon|evening)|how\s+are\s+you)\b", r.strip(), re.I))},
        {"prompt": "State one benefit of regular exercise.", "validator": strict_one_sentence({"health", "healthy", "heart", "strength", "muscle", "energy", "fitness", "weight"})},
        {"prompt": "State one use of electricity.", "validator": strict_one_sentence({"light", "lighting", "power", "heat", "heating", "energy", "computer", "machine", "appliance", "device"})},
        {"prompt": "Name the opposite of hot.", "validator": strict_entity({"cold", "cool", "freezing"})},
        {"prompt": "Name the opposite of fast.", "validator": strict_entity({"slow", "sluggish"})},
        {"prompt": "Name the opposite of dark.", "validator": strict_entity({"light", "bright"})},
        {"prompt": "Name the opposite of empty.", "validator": strict_entity({"full", "filled"})},
    ]


# ==============================================================================
# AXIS 4: Grounded & Open-Ended Rubric (20 Prompts / 40 Points Max)
# ==============================================================================

def build_axis4_rubric() -> list[dict]:
    return [
        # 5 Summarization (require >=2 core concepts + strict 1-sentence limit for full 2 pts)
        {"subtask": "Summarization", "prompt": "Summarize in one sentence: Photosynthesis allows green plants to use sunlight, water, and carbon dioxide to produce glucose and oxygen.", "concepts": ["plant", "sunlight", "glucose", "oxygen"], "min_hits": 2, "one_sent": True},
        {"subtask": "Summarization", "prompt": "Summarize in one sentence: Grouped-Query Attention reduces KV-cache memory by sharing key and value heads across multiple query heads.", "concepts": ["memory", "head", "attention"], "min_hits": 2, "one_sent": True},
        {"subtask": "Summarization", "prompt": "Summarize in one sentence: Regular aerobic exercise strengthens the heart muscle, improves blood circulation, and increases lung capacity.", "concepts": ["exercise", "heart", "circulation", "lung"], "min_hits": 2, "one_sent": True},
        {"subtask": "Summarization", "prompt": "Summarize in one sentence: Cloud computing lets organizations rent servers and storage over the internet instead of buying physical hardware.", "concepts": ["cloud", "internet", "server", "storage"], "min_hits": 2, "one_sent": True},
        {"subtask": "Summarization", "prompt": "Summarize in one sentence: Recycling aluminum cans saves up to 95 percent of the energy required to make new cans from raw ore.", "concepts": ["recycling", "aluminum", "energy"], "min_hits": 2, "one_sent": True},
        # 5 Context Comprehension (must NOT mention distractor entities)
        {"subtask": "Context QA", "prompt": "Text: Atlas Robotics was founded in 2018 by Elena Vance in Boston to build warehouse drones.\nQuestion: In what city was Atlas Robotics founded?", "concepts": ["boston"], "forbidden": [], "min_hits": 1, "one_sent": False},
        {"subtask": "Context QA", "prompt": "Text: Atlas Robotics was founded in 2018 by Elena Vance in Boston to build warehouse drones.\nQuestion: Who founded Atlas Robotics?", "concepts": ["elena", "vance"], "forbidden": [], "min_hits": 2, "one_sent": False},
        {"subtask": "Context QA", "prompt": "Text: The red kayak belongs to Liam, while the blue canoe belongs to Noah.\nQuestion: What color is Liam's boat?", "concepts": ["red"], "forbidden": ["blue"], "min_hits": 1, "one_sent": False},
        {"subtask": "Context QA", "prompt": "Text: Every Tuesday at 9 AM, the bakery sells fresh sourdough bread and cinnamon rolls.\nQuestion: On which day of the week does the bakery sell sourdough bread?", "concepts": ["tuesday"], "forbidden": [], "min_hits": 1, "one_sent": False},
        {"subtask": "Context QA", "prompt": "Text: Mount Kestrel is 4,200 meters tall and lies on the northern border of the valley.\nQuestion: How tall is Mount Kestrel in meters?", "concepts": ["4,200", "4200"], "forbidden": [], "min_hits": 1, "one_sent": False},
        # 10 Reasoning (must NOT contain contradictory opposites)
        {"subtask": "Reasoning", "prompt": "If all roses are flowers and some flowers fade quickly, are all roses flowers?", "concepts": ["yes"], "forbidden": ["no"], "min_hits": 1, "one_sent": False},
        {"subtask": "Reasoning", "prompt": "Alice is taller than Bob, and Bob is taller than Charlie. Who is the tallest?", "concepts": ["alice"], "forbidden": ["charlie"], "min_hits": 1, "one_sent": False},
        {"subtask": "Reasoning", "prompt": "Alice is taller than Bob, and Bob is taller than Charlie. Who is the shortest?", "concepts": ["charlie"], "forbidden": ["alice"], "min_hits": 1, "one_sent": False},
        {"subtask": "Reasoning", "prompt": "If today is Monday, what day will it be in two days?", "concepts": ["wednesday"], "forbidden": [], "min_hits": 1, "one_sent": False},
        {"subtask": "Reasoning", "prompt": "If today is Friday, what day was yesterday?", "concepts": ["thursday"], "forbidden": ["saturday"], "min_hits": 1, "one_sent": False},
        {"subtask": "Reasoning", "prompt": "A box contains 3 red balls and 5 blue balls. How many balls are in the box in total?", "concepts": ["8", "eight"], "forbidden": [], "min_hits": 1, "one_sent": False},
        {"subtask": "Reasoning", "prompt": "Which is heavier: one kilogram of iron or one kilogram of feathers?", "concepts": ["same", "equal", "neither", "both"], "forbidden": [], "min_hits": 1, "one_sent": False},
        {"subtask": "Reasoning", "prompt": "If a car travels 60 kilometers in 1 hour, how far does it travel in 2 hours?", "concepts": ["120"], "forbidden": [], "min_hits": 1, "one_sent": False},
        {"subtask": "Reasoning", "prompt": "Sarah has 10 dollars and buys a book for 4 dollars. How much money does she have left?", "concepts": ["6", "six"], "forbidden": [], "min_hits": 1, "one_sent": False},
        {"subtask": "Reasoning", "prompt": "If it is raining outside, will the grass get wet or stay dry?", "concepts": ["wet"], "forbidden": ["dry"], "min_hits": 1, "one_sent": False},
    ]


def score_axis4_item(response: str, item: dict) -> int:
    clean = response.strip()
    if not clean:
        return 0
    low = clean.lower()
    # Immediate 0 if response hedges with contradictory distractor (e.g., "red and blue" or "wet or dry")
    for bad in item.get("forbidden", []):
        if re.search(rf"\b{re.escape(bad)}\b", low):
            return 0
    hits = sum(1 for kw in item["concepts"] if re.search(rf"\b{re.escape(kw)}\b", low))
    if hits < item["min_hits"]:
        return 0
    if item["one_sent"]:
        concise = "\n" not in clean and _count_sentences(clean) == 1 and len(clean.split()) <= 32
        return 2 if concise else 1
    concise = "\n" not in clean and len(clean.split()) <= 25
    return 2 if concise else 1


def build_generative_suite():
    factual = [
        ("What is the chemical formula of water?", ["h2o"]),
        ("Which planet is closest to the Sun?", ["mercury"]),
        ("What is the largest planet in our solar system?", ["jupiter"]),
        ("What gas do plants absorb during photosynthesis?", ["carbon dioxide", "co2"]),
        ("What gas do humans need to breathe to survive?", ["oxygen"]),
        ("What is the boiling point of water in degrees Celsius?", ["100"]),
        ("What is the freezing point of water in degrees Celsius?", ["0", "zero"]),
        ("How many continents are there on Earth?", ["7", "seven"]),
        ("What is the capital city of France?", ["paris"]),
        ("What is the capital city of Japan?", ["tokyo"]),
        ("What is the capital city of India?", ["new delhi", "delhi"]),
        ("What is the capital city of the United Kingdom?", ["london"]),
        ("What is the chemical symbol for gold?", ["au"]),
        ("What is the chemical symbol for iron?", ["fe"]),
        ("What is the chemical symbol for sodium?", ["na"]),
        ("Which organ pumps blood throughout the human body?", ["heart"]),
        ("What is the powerhouse of the cell?", ["mitochondria", "mitochondrion"]),
        ("How many bones are in an adult human body?", ["206"]),
        ("What force keeps planets in orbit around the Sun?", ["gravity"]),
        ("What is the speed of light in kilometers per second?", ["300000"]),
        ("Which ocean is the largest on Earth?", ["pacific", "pacific ocean"]),
        ("What is the longest river in the world?", ["nile", "amazon", "nile river", "amazon river"]),
        ("What is the tallest mountain above sea level?", ["everest", "mount everest"]),
        ("Who wrote Romeo and Juliet?", ["shakespeare", "william shakespeare"]),
        ("Who developed the theory of general relativity?", ["einstein", "albert einstein"]),
        ("In what year did World War II end?", ["1945"]),
        ("In what year did humans first land on the Moon?", ["1969"]),
        ("What language is primarily spoken in Brazil?", ["portuguese"]),
        ("How many degrees are in a right angle?", ["90"]),
        ("How many sides does a hexagon have?", ["6", "six"]),
        ("What is the hardest natural substance on Earth?", ["diamond"]),
        ("Which planet is known as the Red Planet?", ["mars"]),
        ("What is the nearest star to Earth?", ["sun"]),
        ("What is the main gas in Earth's atmosphere?", ["nitrogen"]),
        ("What currency is used in Japan?", ["yen", "japanese yen"]),
        ("What is the largest mammal on Earth?", ["blue whale", "whale"]),
        ("How many hours are in one day?", ["24"]),
        ("How many minutes are in one hour?", ["60"]),
        ("How many days are in a non-leap year?", ["365"]),
        ("What instrument measures temperature?", ["thermometer"]),
        ("What instrument measures atmospheric pressure?", ["barometer"]),
        ("Which vitamin is produced from sunlight on skin?", ["vitamin d"]),
        ("What is the center of an atom called?", ["nucleus"]),
        ("What charge does an electron carry?", ["negative"]),
        ("What charge does a proton carry?", ["positive"]),
        ("Which continent is the Sahara Desert in?", ["africa"]),
        ("What is the chemical formula for table salt?", ["nacl"]),
        ("What is the square root of 144?", ["12"]),
        ("What is the binary representation of 5?", ["101"]),
        ("What does CPU stand for?", ["central processing unit"]),
    ]

    arithmetic = []
    for i in range(15):
        a, b = 7 + i * 3, 4 + i * 2
        arithmetic.append((f"What is {a} + {b}?", str(a + b)))
    for i in range(15):
        a, b = 45 + i * 4, 12 + i * 2
        arithmetic.append((f"What is {a} - {b}?", str(a - b)))
    for i in range(10):
        a, b = 3 + i, 4 + (i % 6)
        arithmetic.append((f"What is {a} multiplied by {b}?", str(a * b)))
    for i in range(10):
        b, ans = 2 + (i % 7), 3 + i
        arithmetic.append((f"What is {b * ans} divided by {b}?", str(ans)))

    constraints = build_axis3_constraints()
    rubric = build_axis4_rubric()

    items = []
    for q, ans in factual:
        items.append({"axis": "Gen-Factual", "prompt": q, "expected": ans, "max_tok": 32})
    for q, ans in arithmetic:
        items.append({"axis": "Gen-Math", "prompt": q, "expected": ans, "max_tok": 20})
    for idx, c in enumerate(constraints):
        items.append({"axis": "IFEval-Lite", "prompt": c["prompt"], "c_idx": idx, "max_tok": 36})
    for idx, r in enumerate(rubric):
        items.append({"axis": "Rubric", "prompt": r["prompt"], "r_idx": idx, "max_tok": 44})

    return items, constraints, rubric


def compute_4gram_diversity(token_lists: list[list[int]]) -> float:
    total_4grams = 0
    unique_4grams = set()
    for toks in token_lists:
        if len(toks) >= 4:
            grams = [tuple(toks[i : i + 4]) for i in range(len(toks) - 3)]
            total_4grams += len(grams)
            unique_4grams.update(grams)
    return round(100.0 * len(unique_4grams) / max(1, total_4grams), 1)


@torch.no_grad()
def run_generative_batched(model, encode_fn, decode_fn, eos_id: int, ctx: int, mode: str, items: list[dict]):
    t0 = time.time()
    buckets = {}
    for idx, item in enumerate(items):
        q = item["prompt"]
        formatted = f"### Instruction:\n{q}\n\n### Response:\n" if mode == "sft" else q
        p_ids = encode_fn(formatted)[-(ctx - 64) :]
        if not p_ids:
            p_ids = [eos_id]
        buckets.setdefault(len(p_ids), []).append((idx, p_ids, item["max_tok"]))

    outputs = [None] * len(items)
    all_tok_lists = []

    for _, chunk_items in buckets.items():
        for start in range(0, len(chunk_items), 32):
            chunk = chunk_items[start : start + 32]
            B = len(chunk)
            max_steps = max(x[2] for x in chunk)
            seq = torch.tensor([x[1] for x in chunk], dtype=torch.long, device=device)
            gens = [[] for _ in range(B)]
            done = [False] * B
            eos_flags = [False] * B

            for step in range(max_steps):
                if all(done):
                    break
                out = model(seq[:, -ctx:])
                raw_logits = out.logits if hasattr(out, "logits") else (out[0] if isinstance(out, tuple) else out)
                logits = raw_logits[:, -1, :].float()

                for b in range(B):
                    if done[b] or step >= chunk[b][2]:
                        continue
                    g = gens[b]
                    if g:
                        seen = torch.tensor(list(set(g)), dtype=torch.long, device=device)
                        sl = logits[b, seen]
                        logits[b, seen] = torch.where(sl < 0, sl * 1.15, sl / 1.15)
                    if len(g) >= 2:
                        prefix = tuple(g[-2:])
                        banned = {g[i + 2] for i in range(len(g) - 2) if tuple(g[i : i + 2]) == prefix}
                        if banned:
                            logits[b, torch.tensor(list(banned), dtype=torch.long, device=device)] = -float("inf")

                next_toks = torch.argmax(logits, dim=-1, keepdim=True)
                for b in range(B):
                    if done[b] or step >= chunk[b][2]:
                        continue
                    tid = next_toks[b, 0].item()
                    if tid == eos_id:
                        done[b] = True
                        eos_flags[b] = True
                    else:
                        gens[b].append(tid)
                        if mode == "sft" and len(gens[b]) >= 3 and "\n###" in decode_fn(gens[b][-6:]):
                            done[b] = True
                            eos_flags[b] = True

                seq = torch.cat([seq, next_toks], dim=1)

            for b, (orig_idx, _, _) in enumerate(chunk):
                txt = decode_fn(gens[b]).split("\n### Instruction:")[0].split("\n\n###")[0].strip()
                outputs[orig_idx] = {"response": txt, "eos": eos_flags[b], "tokens": len(gens[b])}
                all_tok_lists.append(gens[b])

    elapsed = max(1e-3, time.time() - t0)
    total_toks = sum(o["tokens"] for o in outputs)
    tok_per_sec = round(total_toks / elapsed, 1)
    diversity = compute_4gram_diversity(all_tok_lists)
    return outputs, round(elapsed, 2), tok_per_sec, diversity


def main():
    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    mc_items = build_axis1_mc()
    gen_items, constraints, rubric = build_generative_suite()

    models_spec = [
        {"key": "gqa_base_18m", "name": "GQA Base (18.57M)", "params": "18.57M / 5.70M", "kv_kb": 768, "type": "modern_llm", "path": BASE_PATH, "mode": "base"},
        {"key": "gqa_sft_18m", "name": "GQA SFT (18.57M)", "params": "18.57M / 5.70M", "kv_kb": 768, "type": "modern_llm", "path": SFT_PATH, "mode": "sft"},
        {"key": "pythia_31m", "name": "Pythia-31M (30.5M)", "params": "30.50M / 4.74M", "kv_kb": 3072, "type": "hf", "hf_id": "EleutherAI/pythia-31m", "mode": "base"},
        {"key": "pythia_70m", "name": "Pythia-70M (70.4M)", "params": "70.40M / 18.92M", "kv_kb": 6144, "type": "hf", "hf_id": "EleutherAI/pythia-70m", "mode": "base"},
        {"key": "gpt2_124m", "name": "GPT-2 Base (124.4M)", "params": "124.4M / 85.80M", "kv_kb": 18432, "type": "hf", "hf_id": "gpt2", "mode": "base"},
    ]

    results = []
    for spec in models_spec:
        print(f"Evaluating {spec['name']} across 5 strict axes on {device}...", flush=True)
        if spec["type"] == "modern_llm":
            model, ctx = load_modern_llm(spec["path"])
            mc_acc = eval_axis1_mc_logprob(model, gpt2_enc.encode_ordinary, mc_items)
            outputs, elapsed, tps, div4 = run_generative_batched(
                model, gpt2_enc.encode_ordinary, gpt2_enc.decode, gpt2_enc.eot_token, ctx, spec["mode"], gen_items
            )
            del model
            clear_device_cache()
        else:
            tok = AutoTokenizer.from_pretrained(spec["hf_id"])
            model = AutoModelForCausalLM.from_pretrained(spec["hf_id"]).to(device)
            model.eval()
            ctx = min(512, getattr(model.config, "max_position_embeddings", 512))
            enc_fn = lambda s, t=tok: t.encode(s, add_special_tokens=False)
            dec_fn = lambda ids, t=tok: t.decode(ids, skip_special_tokens=True)
            mc_acc = eval_axis1_mc_logprob(model, enc_fn, mc_items)
            outputs, elapsed, tps, div4 = run_generative_batched(
                model, enc_fn, dec_fn, tok.eos_token_id, ctx, spec["mode"], gen_items
            )
            del model
            clear_device_cache()

        fact_ok, math_ok, if_ok, rub_pts = 0, 0, 0, 0
        for item, out in zip(gen_items, outputs):
            resp = out["response"]
            if item["axis"] == "Gen-Factual":
                fact_ok += int(grade_generative_factual(resp, item["expected"]))
            elif item["axis"] == "Gen-Math":
                math_ok += int(grade_generative_arithmetic(resp, item["expected"]))
            elif item["axis"] == "IFEval-Lite":
                if_ok += int(constraints[item["c_idx"]]["validator"](resp))
            elif item["axis"] == "Rubric":
                rub_pts += score_axis4_item(resp, rubric[item["r_idx"]])

        gen_qa_pct = round(100.0 * (fact_ok + math_ok) / 100, 1)
        if_pct = round(100.0 * if_ok / 25, 1)
        rub_pct = round(100.0 * rub_pts / 40, 1)
        eos_pct = round(100.0 * sum(1 for o in outputs if o["eos"]) / len(outputs), 1)
        composite = round(0.25 * mc_acc + 0.25 * gen_qa_pct + 0.25 * if_pct + 0.25 * rub_pct, 1)

        results.append({
            "name": spec["name"],
            "params": spec["params"],
            "kv_cache_kb": spec["kv_kb"],
            "axis1_mc_logprob_pct": mc_acc,
            "axis2_factual_pct": round(100.0 * fact_ok / 50, 1),
            "axis2_math_pct": round(100.0 * math_ok / 50, 1),
            "axis2_gen_qa_pct": gen_qa_pct,
            "axis3_ifeval_pct": if_pct,
            "axis4_rubric_pts": rub_pts,
            "axis4_rubric_pct": rub_pct,
            "composite_score": composite,
            "eos_rate_pct": eos_pct,
            "diversity_4gram_pct": div4,
            "tok_per_sec": tps,
            "elapsed_sec": elapsed,
        })

    lines = [
        "# Strict 5-Axis Evaluation Matrix Report (195 Unique Tasks)",
        "",
        f"- **Timestamp (UTC):** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- **Device:** `{device}` | **Generative Decoding:** `temp=0.0`, `rep_penalty=1.15`, `no_repeat_ngram=3`",
        "- **Grading Hygiene:** Zero output repair; multi-line/runaway answers rejected in Axis 2; exact list cardinality & single-token `Yes/No` enforced in Axis 3; contradictory distractors penalized with `0 pts` in Axis 4.",
        "",
        "## 1. Capability & Alignment Matrix (Axes 1–4)",
        "",
        "| Model (`Total / Backbone`) | Axis 1: MC-LogProb (50) | Axis 2: Strict Gen-QA (100) *(Fact / Math)* | Axis 3: Strict IFEval (25) | Axis 4: Strict Rubric (40 pts) | **Composite Score** |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in results:
        lines.append(
            f"| **{r['name']}** (`{r['params']}`) | "
            f"**{r['axis1_mc_logprob_pct']:.1f}%** | "
            f"**{r['axis2_gen_qa_pct']:.1f}%** *({r['axis2_factual_pct']:.0f}% / {r['axis2_math_pct']:.0f}%)* | "
            f"**{r['axis3_ifeval_pct']:.1f}%** | "
            f"**{r['axis4_rubric_pct']:.1f}%** ({r['axis4_rubric_pts']}/40) | "
            f"**{r['composite_score']:.1f}** |"
        )

    lines.extend([
        "",
        "## 2. Systems, GQA Memory Efficiency & Generation Hygiene (Axis 5)",
        "",
        "| Model | 512-Ctx KV Cache (FP16) | Throughput (`tok/s`) | EOS Stop Rate | 4-Gram Diversity | Eval Time |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in results:
        lines.append(
            f"| **{r['name']}** | **`{r['kv_cache_kb']} KB`** | "
            f"`{r['tok_per_sec']} tok/s` | {r['eos_rate_pct']:.1f}% | "
            f"{r['diversity_4gram_pct']:.1f}% | {r['elapsed_sec']:.1f}s |"
        )

    report_md = "\n".join(lines) + "\n"
    print("\n" + report_md)

    md_path = os.path.join(BENCHMARK_DIR, "eval_5axis_report.md")
    json_path = os.path.join(BENCHMARK_DIR, "eval_5axis_results.json")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "models": results}, f, indent=2)

    print(f"Saved Strict 5-Axis Evaluation Matrix to:\n  - {md_path}\n  - {json_path}")


if __name__ == "__main__":
    main()

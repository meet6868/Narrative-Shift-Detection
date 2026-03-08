"""
ideal_embedding.py
==================
Builds one SBERT prototype vector per topic from ideal articles.

Pipeline:
  Stage 1 → Sentence segmentation of every .txt article
  Stage 2 → SBERT encode all sentences (batched, GPU)
  Stage 3 → Mean-pool per topic  →  prototype vector


"""

import os
import json
import time
import logging
from pathlib import Path
from collections import defaultdict

import nltk
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ──────────────────────────────────────────────
# CONFIG  (edit only this block if needed)
# ──────────────────────────────────────────────
IDEAL_ARTICLE_DIR = Path("ideal_article")          # root folder
OUTPUT_JSON       = Path("topic_prototypes.json")  # human-readable output
OUTPUT_PT         = Path("topic_prototypes.pt")    # torch tensor output
SBERT_MODEL       = "all-mpnet-base-v2"            # high accuracy, 768-dim
BATCH_SIZE        = 32    # safe for 4 GB VRAM, no OOM risk
NUM_WORKERS       = 6     # half the logical cores for DataLoader-style work
PRECISION         = torch.float16                  # fp16 on GPU → 2× speedup
# ──────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ── Step 0: ensure punkt tokenizer is available ──────────────────────────────
def ensure_nltk_punkt() -> None:
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        log.info("Downloading NLTK punkt tokenizer …")
        nltk.download("punkt_tab", quiet=True)


# ── Step 1: collect all sentences per topic ───────────────────────────────────
def load_sentences_by_topic(root: Path) -> dict[str, list[str]]:
    """
    Walk  root/<Topic>/<Subtopic>/*.txt
    Returns  { "Climate": ["sent1", "sent2", ...], "War": [...], ... }
    """
    topic_sentences: dict[str, list[str]] = defaultdict(list)

    topic_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not topic_dirs:
        raise FileNotFoundError(f"No topic folders found under '{root}'")

    for topic_dir in topic_dirs:
        topic = topic_dir.name  # e.g. "Climate"
        txt_files = list(topic_dir.rglob("*.txt"))

        if not txt_files:
            log.warning("No .txt files found under '%s' — skipping.", topic)
            continue

        log.info("Topic %-12s │ %d article files", topic, len(txt_files))

        for txt_file in txt_files:
            try:
                raw_text = txt_file.read_text(encoding="utf-8", errors="ignore")
                sentences = nltk.sent_tokenize(raw_text)
                # strip whitespace and skip very short/empty sentences
                clean = [s.strip() for s in sentences if len(s.strip()) > 20]
                topic_sentences[topic].extend(clean)
            except Exception as exc:
                log.warning("Could not read '%s': %s", txt_file, exc)

        log.info("  → %d sentences collected for '%s'",
                 len(topic_sentences[topic]), topic)

    return dict(topic_sentences)


# ── Step 2 + 3: encode + mean-pool per topic ─────────────────────────────────
def build_prototypes(
    topic_sentences: dict[str, list[str]],
    model: SentenceTransformer,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """
    Encodes all sentences for each topic in batches and returns
    { topic: mean_vector (np.ndarray, shape [dim]) }
    """
    prototypes: dict[str, np.ndarray] = {}

    for topic, sentences in topic_sentences.items():
        log.info("Encoding %-12s │ %d sentences …", topic, len(sentences))
        t0 = time.time()

        # encode_multi_process / encode with show_progress_bar
        embeddings = model.encode(
            sentences,
            batch_size=BATCH_SIZE,
            device=str(device),
            convert_to_numpy=True,
            normalize_embeddings=False,  # raw embeddings → mean → then we can normalise
            show_progress_bar=True,
            precision="float32",         # accumulate in fp32 for numerical stability
        )

        # mean-pool → topic prototype
        prototype = embeddings.mean(axis=0)            # shape: [dim]
        # L2-normalise so cosine similarity == dot product downstream
        prototype = prototype / (np.linalg.norm(prototype) + 1e-12)

        prototypes[topic] = prototype
        elapsed = time.time() - t0
        log.info("  ✓ Prototype built │ dim=%d │ %.1fs", prototype.shape[0], elapsed)

    return prototypes


# ── Save outputs ──────────────────────────────────────────────────────────────
def save_prototypes(
    prototypes: dict[str, np.ndarray],
    json_path: Path,
    pt_path: Path,
) -> None:
    # JSON (human-readable / language-agnostic)
    json_data = {topic: vec.tolist() for topic, vec in prototypes.items()}
    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    log.info("Saved JSON  → %s", json_path)

    # PyTorch dict of tensors (fast reload in torch pipelines)
    torch_data = {topic: torch.tensor(vec, dtype=torch.float32)
                  for topic, vec in prototypes.items()}
    torch.save(torch_data, pt_path)
    log.info("Saved .pt   → %s", pt_path)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    overall_start = time.time()

    # ── device selection ──────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info("Device: GPU │ %s │ %.1f GB VRAM", gpu_name, vram_gb)
        # keep memory usage low on 4 GB card
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        log.info("Device: CPU │ AMD Ryzen 5 5600H (12 threads)")
        torch.set_num_threads(NUM_WORKERS)

    # ── load SBERT ────────────────────────────────────────────────────────
    log.info("Loading SBERT model: %s …", SBERT_MODEL)
    model = SentenceTransformer(SBERT_MODEL, device=str(device))
    # Cast model to fp16 on GPU → faster inference, less VRAM
    if device.type == "cuda":
        model = model.half()
    log.info("SBERT loaded ✓  (embedding dim = %d)", model.get_sentence_embedding_dimension())

    # ── Stage 1: sentence segmentation ───────────────────────────────────
    ensure_nltk_punkt()
    log.info("── Stage 1: Sentence Segmentation ──────────────────────")
    topic_sentences = load_sentences_by_topic(IDEAL_ARTICLE_DIR)
    total_sentences = sum(len(v) for v in topic_sentences.values())
    log.info("Total sentences across all topics: %d", total_sentences)

    # ── Stage 2 + 3: encode + prototype ──────────────────────────────────
    log.info("── Stage 2 + 3: SBERT Encoding + Prototype Construction ─")
    prototypes = build_prototypes(topic_sentences, model, device)

    # ── Save ──────────────────────────────────────────────────────────────
    log.info("── Saving Outputs ───────────────────────────────────────")
    save_prototypes(prototypes, OUTPUT_JSON, OUTPUT_PT)

    # ── Summary ───────────────────────────────────────────────────────────
    log.info("══ Done in %.1fs ══", time.time() - overall_start)
    log.info("Prototype summary:")
    for topic, vec in prototypes.items():
        log.info("  %-14s │ dim=%d │ norm=%.4f", topic, vec.shape[0], np.linalg.norm(vec))


if __name__ == "__main__":
    main()

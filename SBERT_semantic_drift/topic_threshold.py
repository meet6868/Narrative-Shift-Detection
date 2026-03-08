"""
topic_threshold.py
==================
Computes a cosine-similarity threshold for each topic using ideal articles.

Pipeline:
  1. Load topic prototype vectors from topic_prototypes.json
  2. For each topic, load all .txt articles from ideal_article/<Topic>/**/*.txt
  3. Split each article into sentences  (NLTK)
  4. Encode all sentences with SBERT    (all-mpnet-base-v2, GPU)
  5. Compute cosine similarity of every sentence vs its topic prototype
  6. Threshold = mean(similarities) - 0.5 * std(similarities)
  7. Save thresholds to topic_thresholds.json

Output example:
  {
    "Climate":    0.57,
    "Economics":  0.55,
    ...
  }
"""

import json
import logging
import time
from pathlib import Path

import nltk
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ─────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────
PROTOTYPES_FILE   = Path("topic_prototypes.json")
IDEAL_ARTICLE_DIR = Path("ideal_article")
OUTPUT_FILE       = Path("topic_thresholds.json")

SBERT_MODEL       = "all-mpnet-base-v2"   # 768-dim, matches prototypes
BATCH_SIZE        = 32                     # safe for 4 GB VRAM
THRESHOLD_ALPHA   = 0.5                    # threshold = mean - alpha * std
MIN_SENTENCE_LEN  = 20                     # ignore very short/empty sentences
# ─────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ── Utilities ────────────────────────────────────────────────────────────────

def ensure_nltk_punkt() -> None:
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        log.info("Downloading NLTK punkt tokenizer …")
        nltk.download("punkt_tab", quiet=True)


def load_prototypes(path: Path) -> dict[str, np.ndarray]:
    """Load topic prototype vectors from JSON → numpy arrays."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    prototypes = {topic: np.array(vec, dtype=np.float32) for topic, vec in raw.items()}
    log.info("Loaded %d topic prototypes from '%s'", len(prototypes), path)
    return prototypes


def load_sentences_for_topic(topic_dir: Path) -> list[str]:
    """
    Recursively find all .txt files under topic_dir,
    read each, split into sentences, return clean sentence list.
    """
    sentences = []
    txt_files = list(topic_dir.rglob("*.txt"))

    for txt_file in txt_files:
        try:
            text = txt_file.read_text(encoding="utf-8", errors="ignore")
            sents = nltk.sent_tokenize(text)
            clean = [s.strip() for s in sents if len(s.strip()) >= MIN_SENTENCE_LEN]
            sentences.extend(clean)
        except Exception as exc:
            log.warning("Could not read '%s': %s", txt_file, exc)

    return sentences


def encode_sentences(
    sentences: list[str],
    model: SentenceTransformer,
    device: torch.device,
) -> np.ndarray:
    """Encode a list of sentences → numpy array of shape (N, dim)."""
    embeddings = model.encode(
        sentences,
        batch_size=BATCH_SIZE,
        device=str(device),
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalise → cosine sim == dot product
        show_progress_bar=True,
        precision="float32",
    )
    return embeddings                # shape: (N, 768)


def compute_threshold(
    sentence_embeddings: np.ndarray,
    prototype: np.ndarray,
    alpha: float = THRESHOLD_ALPHA,
) -> tuple[float, float, float, float]:
    """
    Compute per-sentence cosine similarities vs prototype, then:
        threshold = mean - alpha * std

    Returns (threshold, mean, std, min_sim, max_sim)
    """
    # prototype: (768,) → (1, 768) for sklearn
    proto_2d   = prototype.reshape(1, -1)
    sims       = cosine_similarity(sentence_embeddings, proto_2d).flatten()  # (N,)

    mean_sim   = float(np.mean(sims))
    std_sim    = float(np.std(sims))
    threshold  = mean_sim - alpha * std_sim

    return threshold, mean_sim, std_sim, float(np.min(sims)), float(np.max(sims))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.time()

    # ── Device ───────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device   = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info("Device: GPU │ %s │ %.1f GB VRAM", gpu_name, vram_gb)
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        log.info("Device: CPU")

    # ── Load SBERT ────────────────────────────────────────────────────────
    log.info("Loading SBERT model: %s …", SBERT_MODEL)
    model = SentenceTransformer(SBERT_MODEL, device=str(device))
    if device.type == "cuda":
        model = model.half()
    log.info("SBERT loaded ✓  (dim = %d)", model.get_sentence_embedding_dimension())

    # ── Load prototypes ───────────────────────────────────────────────────
    ensure_nltk_punkt()
    prototypes = load_prototypes(PROTOTYPES_FILE)

    # ── Compute threshold per topic ───────────────────────────────────────
    thresholds: dict[str, float] = {}
    details:    dict[str, dict]  = {}

    for topic, prototype in prototypes.items():
        log.info("── Topic: %s ──────────────────────────────────────", topic)

        topic_dir = IDEAL_ARTICLE_DIR / topic
        if not topic_dir.exists():
            log.warning("Folder not found for topic '%s': %s — skipping.", topic, topic_dir)
            continue

        # Stage 1: sentence segmentation
        sentences = load_sentences_for_topic(topic_dir)
        log.info("  Sentences collected : %d", len(sentences))

        if not sentences:
            log.warning("  No sentences found for '%s' — skipping.", topic)
            continue

        # Stage 2: SBERT encoding
        log.info("  Encoding sentences …")
        embeddings = encode_sentences(sentences, model, device)

        # Stage 3: cosine similarity + threshold
        threshold, mean_sim, std_sim, min_sim, max_sim = compute_threshold(
            embeddings, prototype
        )

        thresholds[topic] = round(threshold, 6)
        details[topic] = {
            "threshold" : round(threshold, 6),
            "mean_sim"  : round(mean_sim,  6),
            "std_sim"   : round(std_sim,   6),
            "min_sim"   : round(min_sim,   6),
            "max_sim"   : round(max_sim,   6),
            "n_sentences": len(sentences),
        }

        log.info(
            "  ✓ mean=%.4f │ std=%.4f │ min=%.4f │ max=%.4f │ threshold=%.4f",
            mean_sim, std_sim, min_sim, max_sim, threshold,
        )

    # ── Save outputs ──────────────────────────────────────────────────────
    OUTPUT_FILE.write_text(
        json.dumps(thresholds, indent=2), encoding="utf-8"
    )
    log.info("Saved thresholds → %s", OUTPUT_FILE)

    # Detailed report alongside
    details_file = OUTPUT_FILE.with_name("topic_thresholds_detailed.json")
    details_file.write_text(
        json.dumps(details, indent=2), encoding="utf-8"
    )
    log.info("Saved detailed report → %s", details_file)

    # ── Summary ───────────────────────────────────────────────────────────
    log.info("══ Done in %.1fs ══", time.time() - t_start)
    log.info("Threshold Summary:")
    log.info("  %-14s │ %8s │ %8s │ %8s │ %10s", "Topic", "Mean", "Std", "Threshold", "Sentences")
    log.info("  %s", "-" * 60)
    for topic, d in details.items():
        log.info(
            "  %-14s │ %8.4f │ %8.4f │ %8.4f │ %10d",
            topic, d["mean_sim"], d["std_sim"], d["threshold"], d["n_sentences"],
        )


if __name__ == "__main__":
    main()

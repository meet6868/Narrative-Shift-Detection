"""
drift_threshold.py
==================
Computes semantic drift thresholds per topic × context-window model.

Context window models:
  w1 → single sentence  (S_i)
  w3 → 3-sentence context  (S_{i-1} + S_i + S_{i+1})
  w5 → 5-sentence context  (S_{i-2} + S_{i-1} + S_i + S_{i+1} + S_{i+2})

Pipeline (shared):
  1.  Load ALL_Combined_Data.csv
  2.  Sentence-segment every article  (NLTK)
  3.  Build w1 / w3 / w5 text representations
  4.  Encode each representation with SBERT  (all-mpnet-base-v2, GPU)
  5.  For each topic × model:
        a. Filter sentences  (cosine-sim ≥ topic_threshold)
        b. Group into 5-day windows
        c. Mean-pool each window → window embedding
        d. Drift(t) = 1 - cosine(window_t, window_{t-1})
        e. drift_threshold = mean_drift + std_drift
  6.  Save drift_thresholds.json  +  drift_thresholds_detailed.json

Output format:
  {
    "Climate": {
      "window_days": 5,
      "models": {
        "w1": {"mean": 0.20, "std": 0.11, "threshold": 0.31},
        "w3": {"mean": 0.16, "std": 0.08, "threshold": 0.24},
        "w5": {"mean": 0.14, "std": 0.07, "threshold": 0.21}
      }
    }, ...
  }
"""

import json
import logging
import time
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
CSV_FILE         = Path("ALL_Combined_Data.csv")
PROTOTYPES_FILE  = Path("topic_prototypes.json")
THRESHOLDS_FILE  = Path("topic_thresholds.json")
OUTPUT_FILE      = Path("drift_thresholds.json")
DETAILED_FILE    = Path("drift_thresholds_detailed.json")

SBERT_MODEL      = "all-mpnet-base-v2"   # must match prototype model
BATCH_SIZE       = 32                     # safe for 4 GB VRAM
WINDOW_DAYS      = 5                      # temporal window size (fixed)
MIN_SENTENCE_LEN = 20                     # ignore very short sentences
MIN_WINDOW_SENTS = 3                      # skip near-empty windows

# Context-window configs: name → half-width (sentences on each side)
CONTEXT_MODELS = {"w1": 0, "w3": 1, "w5": 2}
# ─────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
# UTILITY HELPERS
# ══════════════════════════════════════════════════════════

def ensure_nltk_punkt() -> None:
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        log.info("Downloading NLTK punkt tokenizer …")
        nltk.download("punkt_tab", quiet=True)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_device() -> torch.device:
    if torch.cuda.is_available():
        device  = torch.device("cuda")
        log.info("Device: GPU │ %s │ %.1f GB VRAM",
                 torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        log.info("Device: CPU")
    return device


# ══════════════════════════════════════════════════════════
# STAGE 1+2 : Load CSV → sentence DataFrame
# ══════════════════════════════════════════════════════════

def load_and_segment(csv_path: Path) -> pd.DataFrame:
    """
    Returns DataFrame with one row per clean sentence:
        article_idx │ sent_pos │ n_sents │ date │ sentence
    article_idx + sent_pos are needed to build context windows.
    """
    log.info("Loading CSV: %s", csv_path)
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["Article", "Date"]).reset_index(drop=True)
    log.info("  Articles : %d  |  %s → %s",
             len(df), df["Date"].min().date(), df["Date"].max().date())

    log.info("Segmenting articles into sentences …")
    rows = []
    for art_idx, row in tqdm(df.iterrows(), total=len(df), desc="Segmenting"):
        sents = nltk.sent_tokenize(row["Article"])
        clean = [s.strip() for s in sents if len(s.strip()) >= MIN_SENTENCE_LEN]
        for pos, s in enumerate(clean):
            rows.append({
                "article_idx": art_idx,
                "sent_pos"   : pos,
                "n_sents"    : len(clean),
                "date"       : row["Date"],
                "sentence"   : s,
            })

    sent_df = pd.DataFrame(rows).reset_index(drop=True)
    log.info("  Total sentences : %d", len(sent_df))
    return sent_df


# ══════════════════════════════════════════════════════════
# STAGE 3 : Build w1 / w3 / w5 context representations
# ══════════════════════════════════════════════════════════

def build_context_windows(sent_df: pd.DataFrame, half_width: int) -> list[str]:
    """
    For each sentence i, concatenate sentences in the range
        [i - half_width,  i + half_width]
    within the SAME article (boundary-safe).

        half_width=0  →  w1  (sentence only)
        half_width=1  →  w3  (prev + cur + next)
        half_width=2  →  w5  (2×prev + cur + 2×next)

    Returns a list aligned 1-to-1 with sent_df rows.
    """
    if half_width == 0:
        return sent_df["sentence"].tolist()

    # article_idx → ordered list of sentence texts
    article_sents: dict[int, list[str]] = {}
    for art_idx, grp in sent_df.groupby("article_idx", sort=False):
        article_sents[art_idx] = (
            grp.sort_values("sent_pos")["sentence"].tolist()
        )

    contexts = []
    for _, row in sent_df.iterrows():
        art_sents = article_sents[row["article_idx"]]
        pos       = int(row["sent_pos"])
        lo        = max(0, pos - half_width)
        hi        = min(len(art_sents) - 1, pos + half_width)
        contexts.append(" ".join(art_sents[lo: hi + 1]))

    return contexts


# ══════════════════════════════════════════════════════════
# STAGE 4 : SBERT encoding
# ══════════════════════════════════════════════════════════

def encode_texts(
    texts: list[str],
    model: SentenceTransformer,
    device: torch.device,
    label: str = "",
) -> np.ndarray:
    """Encode texts in batches → (N, 768) float32, L2-normalised."""
    log.info("Encoding [%s]: %d texts (batch=%d) …", label, len(texts), BATCH_SIZE)
    embs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        device=str(device),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        precision="float32",
    )
    return embs.astype(np.float32)


# ══════════════════════════════════════════════════════════
# STAGE 5a : Topic-relevance filter
# ══════════════════════════════════════════════════════════

def filter_by_topic(
    embeddings: np.ndarray,
    sent_df: pd.DataFrame,
    prototype: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """
    Keep sentences whose cosine-sim with the topic prototype ≥ threshold.
    Returns filtered sent_df with an 'embedding' column added.
    """
    sims = cosine_similarity(embeddings, prototype.reshape(1, -1)).flatten()
    mask = sims >= threshold
    out  = sent_df[mask].copy().reset_index(drop=True)
    out["embedding"] = list(embeddings[mask])
    return out


# ══════════════════════════════════════════════════════════
# STAGE 5b-d : Windowed drift computation  (reusable)
# ══════════════════════════════════════════════════════════

def compute_drift_for_topic(
    filtered_df: pd.DataFrame,
    topic: str,
    model_name: str,
) -> dict:
    """
    Groups filtered sentences into WINDOW_DAYS-day windows,
    mean-pools each window, computes drift between adjacent windows.

        drift(t) = 1 - cosine(window_t, window_{t-1})
        drift_threshold = mean_drift + std_drift

    Returns a stats dict.
    """
    if len(filtered_df) == 0:
        return {"mean": None, "std": None, "threshold": None,
                "n_windows": 0, "n_sentences": 0}

    tmp    = filtered_df.set_index("date").sort_index()
    groups = tmp.resample(f"{WINDOW_DAYS}D")

    window_vecs   = []
    window_labels = []

    for w_start, grp in groups:
        if len(grp) < MIN_WINDOW_SENTS:
            continue
        vecs     = np.vstack(grp["embedding"].values)
        mean_vec = vecs.mean(axis=0)
        mean_vec = mean_vec / (np.linalg.norm(mean_vec) + 1e-12)
        window_vecs.append(mean_vec)
        window_labels.append(str(w_start.date()))

    n_win = len(window_vecs)

    if n_win < 2:
        log.warning("  [%s/%s] Only %d window(s) — cannot compute drift.",
                    topic, model_name, n_win)
        return {"mean": None, "std": None, "threshold": None,
                "n_windows": n_win, "n_sentences": len(filtered_df)}

    drifts = [
        1.0 - float(cosine_similarity(
            window_vecs[i - 1].reshape(1, -1),
            window_vecs[i].reshape(1, -1)
        )[0, 0])
        for i in range(1, n_win)
    ]

    drift_arr  = np.array(drifts)
    mean_drift = float(drift_arr.mean())
    std_drift  = float(drift_arr.std())
    threshold  = mean_drift + std_drift

    # Console report
    pad = "─" * max(0, 48 - len(topic) - len(model_name))
    print(f"\n  ┌─ {topic} / {model_name} {pad}")
    print(f"  │  Filtered sentences : {len(filtered_df)}")
    print(f"  │  Windows            : {n_win}")
    print(f"  │  Drift values       :")
    for label, d in zip(window_labels[1:], drifts):
        print(f"  │    {label}  →  {d:.6f}")
    print(f"  │  Mean drift         : {mean_drift:.6f}")
    print(f"  │  Std  drift         : {std_drift:.6f}")
    print(f"  └► Drift threshold    : {threshold:.6f}")

    return {
        "mean"         : round(mean_drift, 6),
        "std"          : round(std_drift,  6),
        "threshold"    : round(threshold,  6),
        "n_windows"    : n_win,
        "n_sentences"  : len(filtered_df),
        "drift_values" : [round(d, 6) for d in drifts],
        "window_labels": window_labels[1:],
    }


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main() -> None:
    t_start = time.time()

    # ── Setup ──────────────────────────────────────────────────────────
    ensure_nltk_punkt()
    device = setup_device()

    log.info("Loading SBERT model: %s …", SBERT_MODEL)
    model = SentenceTransformer(SBERT_MODEL, device=str(device))
    if device.type == "cuda":
        model = model.half()
    log.info("SBERT loaded ✓  (dim = %d)", model.get_sentence_embedding_dimension())

    # ── Load calibration files ──────────────────────────────────────────
    prototypes = {t: np.array(v, dtype=np.float32)
                  for t, v in load_json(PROTOTYPES_FILE).items()}
    thresholds = {t: float(v) for t, v in load_json(THRESHOLDS_FILE).items()}
    topics     = list(prototypes.keys())
    log.info("Topics: %s", topics)

    # ── Stage 1+2 : Load & segment ─────────────────────────────────────
    sent_df = load_and_segment(CSV_FILE)

    # ── Stage 3 : Build all context representations ─────────────────────
    log.info("Building context window texts …")
    context_texts: dict[str, list[str]] = {}
    for name, half_w in CONTEXT_MODELS.items():
        log.info("  %s (half_width=%d) …", name, half_w)
        context_texts[name] = build_context_windows(sent_df, half_w)
    log.info("Context windows ready ✓")

    # ── Stage 4 : Encode each context model separately ──────────────────
    # Each model encodes its own input text so embeddings reflect context.
    embeddings: dict[str, np.ndarray] = {}
    for name, texts in context_texts.items():
        embeddings[name] = encode_texts(texts, model, device, label=name)

    # ── Stage 5 : Per-topic × per-model drift thresholds ────────────────
    results: dict[str, dict] = {}

    for topic in topics:
        log.info("══ Topic: %s ══", topic)
        proto   = prototypes[topic]
        sim_thr = thresholds[topic]
        log.info("  Prototype dim=%d | sim_threshold=%.4f", proto.shape[0], sim_thr)

        model_stats: dict[str, dict] = {}

        for model_name in CONTEXT_MODELS:
            log.info("  ── Model: %s ──", model_name)

            # 5a: filter relevant sentences
            filtered = filter_by_topic(embeddings[model_name], sent_df, proto, sim_thr)
            log.info("    Sentences after filter: %d / %d",
                     len(filtered), len(sent_df))

            # 5b-d: windowed drift
            stats = compute_drift_for_topic(filtered, topic, model_name)
            model_stats[model_name] = stats

        results[topic] = {"window_days": WINDOW_DAYS, "models": model_stats}

    # ── Save outputs ────────────────────────────────────────────────────
    # Clean (thresholds only)
    clean = {
        topic: {
            "window_days": info["window_days"],
            "models": {
                m: {"mean": s["mean"], "std": s["std"], "threshold": s["threshold"]}
                for m, s in info["models"].items()
                if s["threshold"] is not None
            },
        }
        for topic, info in results.items()
    }
    OUTPUT_FILE.write_text(json.dumps(clean, indent=2), encoding="utf-8")
    log.info("Saved clean thresholds → %s", OUTPUT_FILE)

    # Detailed (all stats + drift values)
    DETAILED_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Saved detailed report  → %s", DETAILED_FILE)

    # ── Summary table ───────────────────────────────────────────────────
    log.info("══ Done in %.1fs ══", time.time() - t_start)
    print(f"\n{'═'*72}")
    print(f"  {'Topic':<14} │ {'Model':<5} │ {'Mean':>8} │ {'Std':>8} │ "
          f"{'Threshold':>10} │ {'Windows':>7}")
    print(f"  {'─'*14}─┼─{'─'*5}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*10}─┼─{'─'*7}")
    for topic, info in results.items():
        for mname, s in info["models"].items():
            if s["threshold"] is not None:
                print(f"  {topic:<14} │ {mname:<5} │ {s['mean']:>8.4f} │ "
                      f"{s['std']:>8.4f} │ {s['threshold']:>10.4f} │ "
                      f"{s['n_windows']:>7}")
    print(f"{'═'*72}")


if __name__ == "__main__":
    main()

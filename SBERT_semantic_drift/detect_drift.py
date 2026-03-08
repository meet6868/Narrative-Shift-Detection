"""
detect_drift.py
===============
Semantic Narrative Drift Detection — interactive CLI.

The user is prompted to:
  1. Choose a topic  (Climate / Economics / Health / Technology / War)
  2. Provide one or more paths to .txt article files

Each .txt file must contain:
  - First line : the article date  (any format parseable by pandas, e.g. 2024-03-15)
  - Remaining  : the article body  (free text)

Example .txt layout
-------------------
  2024-01-15
  Central banks raised interest rates sharply amid rising inflation concerns.
  ...

  OR with a label prefix:

  Date: 2024-01-15
  Central banks raised interest rates sharply amid rising inflation concerns.
  ...

Calibration files required (must exist in the same directory):
  topic_prototypes.json   → topic centroid embeddings (768-dim, all-mpnet-base-v2)
  topic_thresholds.json   → cosine-sim relevance threshold per topic
  drift_thresholds.json   → drift threshold per topic  model (w1 / w3 / w5)

Output
------
  drift_results.json   — full structured results
  Console              — a per-window summary table printed to stdout
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import timedelta
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  (edit these only if your calibration was done with different settings)
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
PROTOTYPES_FILE  = BASE_DIR / "topic_prototypes.json"
THRESHOLDS_FILE  = BASE_DIR / "topic_thresholds.json"
DRIFT_THR_FILE   = BASE_DIR / "drift_thresholds.json"
OUTPUT_FILE      = BASE_DIR / "drift_results.json"

SBERT_MODEL      = "all-mpnet-base-v2"
BATCH_SIZE       = 32
WINDOW_DAYS      = 5
MIN_SENTENCE_LEN = 20
TOP_N_SENTENCES  = 5
CONTEXT_MODELS   = {"w1": 0, "w3": 1, "w5": 2}   # model name → half-width

VALID_TOPICS = {"Climate", "Economics", "Health", "Technology", "War"}
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 0 — User input helpers
# ══════════════════════════════════════════════════════════════════════════════

def prompt_topic() -> str:
    """Ask the user to choose a topic from the valid list."""
    topic_list = sorted(VALID_TOPICS)
    print("\n" + "═" * 55)
    print("  Narrative Drift Detector")
    print("═" * 55)
    print("  Available topics:")
    for i, t in enumerate(topic_list, 1):
        print(f"    [{i}] {t}")
    print("─" * 55)

    while True:
        raw = input("  Enter topic name or number: ").strip()
        # Accept number shortcut
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(topic_list):
                topic = topic_list[idx]
                print(f"  → Topic selected: {topic}")
                return topic
            print(f"  ✗  Please enter a number between 1 and {len(topic_list)}.")
            continue
        # Accept name (case-insensitive)
        matched = next((t for t in VALID_TOPICS if t.lower() == raw.lower()), None)
        if matched:
            print(f"  → Topic selected: {matched}")
            return matched
        print(f"  ✗  '{raw}' is not a valid topic.  Try again.")


def prompt_article_paths() -> list[Path]:
    """
    Ask the user to enter one or more paths to .txt article files.
    Accepts:
      - Absolute paths
      - Relative paths (resolved from the script's directory)
      - A space-separated list on one line
      - Multiple lines (end input with a blank line)
    Returns a list of validated Path objects.
    """
    print("\n─" * 55)
    print("  Enter paths to .txt article files.")
    print("  • You can paste multiple paths separated by spaces,")
    print("    or press Enter after each path.")
    print("  • Press Enter on a blank line when done.")
    print("─" * 55)

    raw_paths: list[str] = []
    while True:
        line = input("  Path(s): ").strip()
        if line == "":
            if raw_paths:
                break          # done
            print("  ✗  No paths entered yet — please provide at least 5 articles.")
            continue
        # Split on whitespace to allow space-separated list
        raw_paths.extend(line.split())

    # Resolve and validate each path
    valid_paths: list[Path] = []
    errors: list[str] = []
    for rp in raw_paths:
        p = Path(rp)
        if not p.is_absolute():
            p = (BASE_DIR / p).resolve()
        if not p.exists():
            errors.append(f"  ✗  Not found:   {p}")
        elif p.suffix.lower() != ".txt":
            errors.append(f"  ✗  Not a .txt:  {p}")
        else:
            valid_paths.append(p)

    if errors:
        print("\n  Warnings:")
        for e in errors:
            print(e)

    if len(valid_paths) < 5:
        raise ValueError(
            f"\n  At least 5 valid .txt files are required.  "
            f"Only {len(valid_paths)} valid path(s) were provided."
        )

    print(f"\n  ✓  {len(valid_paths)} article file(s) accepted.")
    return valid_paths


def load_articles_from_paths(paths: list[Path]) -> list[dict]:
    """
    Load each .txt file as one article.

    Expected format of each file:
        Line 1:  date string  — bare ("2024-03-15") or prefixed ("Date: 2024-03-15")
        Lines 2+ : article body text

    Returns list of {"date": Timestamp, "text": str}.
    """
    articles: list[dict] = []
    skipped = 0
    for p in paths:
        try:
            raw = p.read_text(encoding="utf-8").strip()
            lines = raw.splitlines()
            if len(lines) < 2:
                log.warning("Skipping %s — too short (need date + body).", p.name)
                skipped += 1
                continue
            # Accept both "2024-01-03" and "Date: 2024-01-03" on the first line
            date_str = lines[0].strip()
            if ":" in date_str:
                date_str = date_str.split(":", 1)[1].strip()
            body     = "\n".join(lines[1:]).strip()
            date     = pd.to_datetime(date_str)
            if not body:
                log.warning("Skipping %s — empty body.", p.name)
                skipped += 1
                continue
            articles.append({"date": date, "text": body, "source": str(p)})
        except Exception as exc:
            log.warning("Skipping %s — %s", p.name, exc)
            skipped += 1

    log.info("Loaded %d article(s)  (%d skipped).", len(articles), skipped)
    if len(articles) < 5:
        raise ValueError(
            f"At least 5 loadable articles are required; only {len(articles)} could be read."
        )
    # Sort chronologically
    articles.sort(key=lambda a: a["date"])
    return articles


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Sentence segmentation
# ══════════════════════════════════════════════════════════════════════════════

def segment_sentences(articles: list[dict]) -> pd.DataFrame:
    """
    Split each article into individual sentences.

    Returns DataFrame with columns:
        article_idx | sent_pos | date | sentence | source
    """
    _ensure_punkt()
    rows = []
    for art_idx, art in enumerate(articles):
        sents = nltk.sent_tokenize(art["text"])
        clean = [s.strip() for s in sents if len(s.strip()) >= MIN_SENTENCE_LEN]
        for pos, s in enumerate(clean):
            rows.append({
                "article_idx": art_idx,
                "sent_pos"   : pos,
                "date"       : art["date"],
                "sentence"   : s,
                "source"     : art.get("source", ""),
            })
    df = pd.DataFrame(rows).reset_index(drop=True)

    # Drop exact duplicate sentences on the same date (e.g. identical articles
    # submitted under different filenames).
    before = len(df)
    df = df.drop_duplicates(subset=["date", "sentence"]).reset_index(drop=True)
    removed = before - len(df)
    if removed:
        log.warning("Removed %d duplicate sentence(s) (same date + text).", removed)

    log.info("Segmentation → %d sentences from %d articles.", len(df), len(articles))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Context window construction
# ══════════════════════════════════════════════════════════════════════════════

def build_context_windows(sent_df: pd.DataFrame, half_width: int) -> list[str]:
    """
    Concatenate neighbouring sentences (within the same article) around each
    sentence to form its context string.

        half_width=0 → w1  : S_i  alone
        half_width=1 → w3  : S_{i-1} + S_i + S_{i+1}
        half_width=2 → w5  : S_{i-2} … S_i … S_{i+2}

    Boundary-safe — clips at article start / end.
    Returns a list aligned 1-to-1 with sent_df rows.
    """
    if half_width == 0:
        return sent_df["sentence"].tolist()

    art_sents: dict[int, list[str]] = {
        idx: grp.sort_values("sent_pos")["sentence"].tolist()
        for idx, grp in sent_df.groupby("article_idx", sort=False)
    }
    contexts = []
    for _, row in sent_df.iterrows():
        sents = art_sents[row["article_idx"]]
        pos   = int(row["sent_pos"])
        lo    = max(0, pos - half_width)
        hi    = min(len(sents) - 1, pos + half_width)
        contexts.append(" ".join(sents[lo : hi + 1]))
    return contexts


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — SBERT encoding
# ══════════════════════════════════════════════════════════════════════════════

def encode_sentences(
    texts  : list[str],
    model  : SentenceTransformer,
    device : torch.device,
    label  : str = "",
) -> np.ndarray:
    """Return (N, 768) float32 L2-normalised embeddings."""
    log.info("Encoding [%s] — %d sentences …", label, len(texts))
    embs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        device=str(device),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        precision="float32",
    )
    return embs.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — Topic-relevance filter
# ══════════════════════════════════════════════════════════════════════════════

def filter_topic_sentences(
    embeddings    : np.ndarray,
    sent_df       : pd.DataFrame,
    prototype     : np.ndarray,
    sim_threshold : float,
) -> pd.DataFrame:
    """
    Keep only sentences with cosine-sim ≥ sim_threshold against the topic prototype.
    Attaches the embedding as a new column.
    """
    sims = cosine_similarity(embeddings, prototype.reshape(1, -1)).flatten()
    mask = sims >= sim_threshold
    out  = sent_df[mask].copy().reset_index(drop=True)
    out["embedding"] = list(embeddings[mask])
    log.info(
        "  Topic filter → %d / %d sentences kept  (threshold=%.4f).",
        len(out), len(sent_df), sim_threshold,
    )
    return out


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — Temporal window grouping
# ══════════════════════════════════════════════════════════════════════════════

def group_temporal_windows(filtered_df: pd.DataFrame) -> list[dict]:
    """
    Partition sentences into fixed WINDOW_DAYS-wide time bins.

    Returns list of non-empty window dicts:
        { "label", "start", "sentences", "embeddings" }
    """
    if filtered_df.empty:
        return []

    df        = filtered_df.sort_values("date").reset_index(drop=True)
    w_start   = df["date"].min().normalize()
    end_date  = df["date"].max()
    windows   = []

    while w_start <= end_date:
        w_end = w_start + timedelta(days=WINDOW_DAYS)
        mask  = (df["date"] >= w_start) & (df["date"] < w_end)
        grp   = df[mask]
        if not grp.empty:
            windows.append({
                "label"     : str(w_start.date()),
                "start"     : w_start,
                "sentences" : grp["sentence"].tolist(),
                "embeddings": np.vstack(grp["embedding"].values),
            })
        w_start = w_end

    log.info("  Temporal windows (non-empty): %d", len(windows))
    return windows


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — Window-level embeddings
# ══════════════════════════════════════════════════════════════════════════════

def compute_window_embeddings(windows: list[dict]) -> list[dict]:
    """
    Add 'window_embedding' = mean-pool of sentence embeddings, then L2-normalise.
    """
    for w in windows:
        vec = w["embeddings"].mean(axis=0)
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        w["window_embedding"] = vec
    return windows


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 7 — Drift computation between adjacent windows
# ══════════════════════════════════════════════════════════════════════════════

def compute_drift(windows: list[dict]) -> list[dict]:
    """
    drift(t-1 → t)  =  1 − cosine( window_embedding_{t-1}, window_embedding_t )

    Returns one record per adjacent pair.
    """
    records = []
    for i in range(1, len(windows)):
        prev = windows[i - 1]
        curr = windows[i]
        cos  = float(cosine_similarity(
            prev["window_embedding"].reshape(1, -1),
            curr["window_embedding"].reshape(1, -1),
        )[0, 0])
        records.append({
            "window_previous": prev["label"],
            "window_current" : curr["label"],
            "drift"          : round(1.0 - cos, 6),
            "emb_prev"       : prev["window_embedding"],
            "emb_curr"       : curr["window_embedding"],
            "sents_prev"     : prev["sentences"],
            "sents_curr"     : curr["sentences"],
            "embs_prev"      : prev["embeddings"],
            "embs_curr"      : curr["embeddings"],
        })
    return records


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 8 — Identify sentences responsible for drift
# ══════════════════════════════════════════════════════════════════════════════

def find_drift_sentences(
    record: dict,
    top_n : int = TOP_N_SENTENCES,
) -> tuple[list[dict], list[dict]]:
    """
    For each sentence in the CURRENT window compute:
        impact = cos(sent, window_curr) / (cos(sent, window_prev) + ε)

    High impact → strongly aligned with NEW narrative, distant from old one.

    Also returns the top_n sentences from the PREVIOUS window most similar to
    the old window embedding (the "anchors" that are being left behind).

    Returns:
        current_drift_sents   : [{"sentence", "impact_score"}, ...]
        previous_anchor_sents : [{"sentence", "sim_score"},    ...]
    """
    emb_prev  = record["emb_prev"]
    emb_curr  = record["emb_curr"]
    embs_curr = record["embs_curr"]
    embs_prev = record["embs_prev"]

    sim_to_curr = cosine_similarity(embs_curr, emb_curr.reshape(1, -1)).flatten()
    sim_to_prev = cosine_similarity(embs_curr, emb_prev.reshape(1, -1)).flatten()
    impact      = sim_to_curr / (sim_to_prev + 1e-9)

    top_curr_idx = np.argsort(impact)[::-1][:top_n]
    current_drift_sents = [
        {
            "sentence"    : record["sents_curr"][i],
            "impact_score": round(float(impact[i]), 4),
        }
        for i in top_curr_idx
    ]

    sim_prev     = cosine_similarity(embs_prev, emb_prev.reshape(1, -1)).flatten()
    top_prev_idx = np.argsort(sim_prev)[::-1][:top_n]
    previous_anchor_sents = [
        {
            "sentence" : record["sents_prev"][i],
            "sim_score": round(float(sim_prev[i]), 4),
        }
        for i in top_prev_idx
    ]

    return current_drift_sents, previous_anchor_sents


def _top_similar(
    embeddings: np.ndarray,
    ref_vec   : np.ndarray,
    sentences : list[str],
    top_n     : int,
) -> list[dict]:
    """Return top_n sentences most similar to ref_vec (used when no drift)."""
    sims    = cosine_similarity(embeddings, ref_vec.reshape(1, -1)).flatten()
    top_idx = np.argsort(sims)[::-1][:top_n]
    return [
        {"sentence": sentences[i], "sim_score": round(float(sims[i]), 4)}
        for i in top_idx
    ]


# ══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_detection(
    topic    : str,
    articles : list[dict],
    model    : SentenceTransformer,
    device   : torch.device,
    prototypes     : dict[str, np.ndarray],
    sim_thresholds : dict[str, float],
    drift_thresholds_cfg: dict,
    top_n    : int = TOP_N_SENTENCES,
) -> dict:
    """
    Run the full 8-stage pipeline for all 3 context models (w1 / w3 / w5).

    Returns a dict with keys: topic, window_days, elapsed_s, w1, w3, w5.
    """
    t0 = time.perf_counter()

    prototype = prototypes[topic]
    sim_thr   = sim_thresholds[topic]
    drift_cfg = drift_thresholds_cfg[topic]

    log.info("═══ Detecting drift │ topic=%s │ articles=%d ═══",
             topic, len(articles))

    # Stage 1 — segment
    sent_df = segment_sentences(articles)
    if sent_df.empty:
        raise RuntimeError("No sentences could be extracted from the provided articles.")

    # Stage 2 — build context texts for all 3 models upfront
    context_texts: dict[str, list[str]] = {}
    for name, hw in CONTEXT_MODELS.items():
        context_texts[name] = build_context_windows(sent_df, hw)

    # Stage 3 — encode all 3 model variants
    embeddings: dict[str, np.ndarray] = {}
    for name, texts in context_texts.items():
        embeddings[name] = encode_sentences(texts, model, device, label=name)

    # Stages 4-8 — per context model
    output: dict = {
        "topic"      : topic,
        "n_articles" : len(articles),
        "window_days": WINDOW_DAYS,
    }

    for model_name, hw in CONTEXT_MODELS.items():
        log.info("── Model: %s ──", model_name)
        drift_threshold = drift_cfg["models"][model_name]["threshold"]

        # Stage 4
        filtered = filter_topic_sentences(
            embeddings[model_name], sent_df, prototype, sim_thr
        )
        if filtered.empty:
            log.warning("  No sentences passed topic filter for %s / %s.", topic, model_name)
            output[model_name] = {
                "drift_threshold": round(drift_threshold, 6),
                "n_windows"      : 0,
                "n_drifts"       : 0,
                "results"        : [],
                "warning"        : "No topic-relevant sentences found.",
            }
            continue

        # Stage 5
        windows = group_temporal_windows(filtered)
        if len(windows) < 2:
            log.warning("  Need ≥ 2 windows; got %d for %s / %s.", len(windows), topic, model_name)
            output[model_name] = {
                "drift_threshold": round(drift_threshold, 6),
                "n_windows"      : len(windows),
                "n_drifts"       : 0,
                "results"        : [],
                "warning"        : f"Only {len(windows)} window(s) — need ≥ 2.",
            }
            continue

        # Stage 6
        windows = compute_window_embeddings(windows)

        # Stage 7
        drift_records = compute_drift(windows)

        # Stage 8 — build result list
        results = []
        for rec in drift_records:
            detected = rec["drift"] > drift_threshold
            if detected:
                curr_sents, prev_sents = find_drift_sentences(rec, top_n)
            else:
                curr_sents = _top_similar(rec["embs_curr"], rec["emb_curr"],
                                          rec["sents_curr"], top_n)
                prev_sents = _top_similar(rec["embs_prev"], rec["emb_prev"],
                                          rec["sents_prev"], top_n)

            results.append({
                "window_previous"               : rec["window_previous"],
                "window_current"                : rec["window_current"],
                "drift"                         : rec["drift"],
                "drift_threshold"               : round(drift_threshold, 6),
                "drift_detected"                : detected,
                "previous_window_sentences"     : prev_sents,
                "current_window_drift_sentences": curr_sents,
            })

        n_drifts = sum(1 for r in results if r["drift_detected"])
        output[model_name] = {
            "drift_threshold": round(drift_threshold, 6),
            "n_windows"      : len(windows),
            "n_drifts"       : n_drifts,
            "results"        : results,
        }

        _print_model_summary(topic, model_name, results, drift_threshold)

    output["elapsed_s"] = round(time.perf_counter() - t0, 2)
    return output


# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def _print_model_summary(
    topic          : str,
    model_name     : str,
    results        : list[dict],
    drift_threshold: float,
) -> None:
    W = 70
    print(f"\n{'═' * W}")
    print(f"  Topic: {topic}   │   Model: {model_name}   │   "
          f"Drift threshold: {drift_threshold:.4f}")
    print(f"{'─' * W}")
    for r in results:
        flag = "🔴 DRIFT DETECTED" if r["drift_detected"] else "🟢 stable"
        print(f"  {r['window_previous']} → {r['window_current']}"
              f"   drift={r['drift']:.4f}   {flag}")
        if r["drift_detected"]:
            print("    ↳ Top drift-driving sentences (current window):")
            for s in r["current_window_drift_sentences"][:3]:
                score   = s.get("impact_score", s.get("sim_score", 0))
                preview = s["sentence"][:90].replace("\n", " ")
                print(f"      [{score:.3f}] {preview}…")
    print("═" * W)


# ══════════════════════════════════════════════════════════════════════════════
# SERIALISATION HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _serialisable(obj):
    """Recursively remove numpy arrays so the result is JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: _serialisable(v) for k, v in obj.items()
                if not isinstance(v, np.ndarray)}
    if isinstance(obj, list):
        return [_serialisable(i) for i in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


# ══════════════════════════════════════════════════════════════════════════════
# PRIVATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_punkt() -> None:
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        log.info("Downloading NLTK punkt_tab tokenizer …")
        nltk.download("punkt_tab", quiet=True)


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Required calibration file not found: {path}\n"
            "Run the calibration scripts first."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── 1. User selects topic ─────────────────────────────────────────────
    topic = prompt_topic()

    # ── 2. User provides article .txt paths ──────────────────────────────
    paths    = prompt_article_paths()
    articles = load_articles_from_paths(paths)

    # ── 3. Load calibration files ─────────────────────────────────────────
    log.info("Loading calibration files …")
    raw_proto = _load_json(PROTOTYPES_FILE)
    raw_thr   = _load_json(THRESHOLDS_FILE)
    raw_drift = _load_json(DRIFT_THR_FILE)

    prototypes        = {t: np.array(v, dtype=np.float32) for t, v in raw_proto.items()}
    sim_thresholds    = {t: float(v) for t, v in raw_thr.items()}
    drift_thresholds  = raw_drift

    # ── 4. Load SBERT model ────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info("Device: GPU │ %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        log.info("Device: CPU")

    log.info("Loading SBERT: %s …", SBERT_MODEL)
    sbert = SentenceTransformer(SBERT_MODEL, device=str(device))
    if device.type == "cuda":
        sbert = sbert.half()
    log.info("SBERT ready ✓  (dim=%d)", sbert.get_sentence_embedding_dimension())

    # ── 5. Run detection ───────────────────────────────────────────────────
    results = run_detection(
        topic                = topic,
        articles             = articles,
        model                = sbert,
        device               = device,
        prototypes           = prototypes,
        sim_thresholds       = sim_thresholds,
        drift_thresholds_cfg = drift_thresholds,
    )

    # ── 6. Save results ────────────────────────────────────────────────────
    OUTPUT_FILE.write_text(
        json.dumps(_serialisable(results), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n✅  Results saved to  {OUTPUT_FILE}")
    print(f"    Total time: {results['elapsed_s']} s")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n\nAborted.")
        sys.exit(0)
    except Exception as exc:
        log.error("Fatal: %s", exc)
        sys.exit(1)

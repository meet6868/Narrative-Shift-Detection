#!/usr/bin/env python3
"""Evaluate KMeans Drift narrative shift outputs against ground truth files.

This script:
1. Parses 15 KMeans text outputs into the expected prediction structure.
2. Reuses the existing evaluation logic from approach1_eval.py.
3. Computes metrics for hard/medium/low.
4. Prints a one-row summary table.
5. Saves metrics to kmeans_eval.json.
"""

from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

DIFFICULTIES: Tuple[str, str, str] = ("hard", "medium", "low")
NO_NARRATIVE = "NO NARRATIVE"
MODEL_NAME = "KMeans_Drift"

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_ROOT = PROJECT_ROOT / "newinput"
PREDICTION_ROOT = Path(
    "/home/prateek-tiwari/external/nlp_project/Narrative-Shift-Detection/"
    "K_Means_Drift/kmeans_output/All_15_Narrative_Shifts_TXT"
)
OUTPUT_JSON_PATH = PROJECT_ROOT / "kmeans_eval.json"


def normalize_text(text: str) -> str:
    """Normalize sentence text for robust matching."""
    if not text:
        return ""
    lowered = text.lower().strip()
    no_punct = lowered.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", no_punct).strip()


def load_ground_truth(path: str) -> str:
    """Load and return raw ground truth text."""
    gt_path = Path(path)
    with gt_path.open("r", encoding="utf-8") as handle:
        return handle.read()


def parse_ground_truth(raw_text: str) -> Dict[str, Dict[str, Any]]:
    """Parse ground truth file into topic-level narrative groups.

    Returns:
        {
            "Topic": {
                "label": "HIGH"|"LOW"|"NO NARRATIVE"|"UNKNOWN",
                "has_narrative": bool,
                "group_a": set[str],  # normalized
                "group_b": set[str],  # normalized
            }
        }
    """
    lines = raw_text.splitlines()

    label_map: Dict[str, str] = {}
    sections_map: Dict[str, Dict[str, set[str]]] = {}

    label_pattern = re.compile(r"^\s*([^:]+?)\s*:\s*(HIGH|LOW|NO NARRATIVE)\s*$", re.IGNORECASE)
    section_pattern = re.compile(r"^\s*(.+?)\s+sentences:\s*$", re.IGNORECASE)

    for line in lines:
        match = label_pattern.match(line)
        if match:
            topic = match.group(1).strip()
            label = match.group(2).strip().upper()
            label_map[topic] = label

    current_topic: str | None = None
    current_group: str | None = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        section_match = section_pattern.match(line)
        if section_match:
            current_topic = section_match.group(1).strip()
            sections_map.setdefault(current_topic, {"A": set(), "B": set()})
            current_group = None
            continue

        if line.lower() == "group a (early):":
            current_group = "A"
            continue

        if line.lower() == "group b (later):":
            current_group = "B"
            continue

        if current_topic is None or current_group is None:
            continue

        # Keep only sentence-like lines and strip wrapping quotes.
        sentence = line.strip().strip('"').strip("'").strip()
        sentence_norm = normalize_text(sentence)
        if sentence_norm:
            sections_map[current_topic][current_group].add(sentence_norm)

    topics = set(label_map.keys()) | set(sections_map.keys())
    parsed: Dict[str, Dict[str, Any]] = {}

    for topic in topics:
        label = label_map.get(topic, "UNKNOWN")
        group_a = sections_map.get(topic, {}).get("A", set())
        group_b = sections_map.get(topic, {}).get("B", set())

        has_narrative = label != NO_NARRATIVE
        if label == "UNKNOWN":
            # If label is missing, infer from available grouped narrative sentences.
            has_narrative = bool(group_a and group_b)

        parsed[topic] = {
            "label": label,
            "has_narrative": has_narrative,
            "group_a": group_a,
            "group_b": group_b,
        }

    return parsed


def _clean_kmeans_sentence(line: str) -> str:
    """Clean one KMeans bullet sentence line."""
    cleaned = line.strip()
    cleaned = re.sub(r"^\s*-\s*", "", cleaned)
    cleaned = re.sub(r"^\s*\[\d+\]\s*", "", cleaned)
    cleaned = cleaned.strip().strip('"').strip("'").strip()
    return cleaned


def _parse_shift_block(lines: List[str], start_index: int) -> Tuple[Dict[str, str], int]:
    """Parse one shift block starting after a 'Shift #n' line."""
    before_sentences: List[str] = []
    after_sentences: List[str] = []
    section: str | None = None
    idx = start_index

    while idx < len(lines):
        raw_line = lines[idx]
        line = raw_line.strip()

        if line.startswith("Topic:") or re.match(r"^Shift\s*#\d+", line):
            break

        if line.startswith("Before Sentences:"):
            section = "before"
            idx += 1
            continue

        if line.startswith("After Sentences:"):
            section = "after"
            idx += 1
            continue

        if section in {"before", "after"} and re.match(r"^\s*-\s*", raw_line):
            cleaned = _clean_kmeans_sentence(raw_line)
            if cleaned:
                if section == "before":
                    before_sentences.append(cleaned)
                else:
                    after_sentences.append(cleaned)

        idx += 1

    return {
        "context_1": "\n".join(before_sentences),
        "context_2": "\n".join(after_sentences),
    }, idx


def _parse_kmeans_prediction_text(raw_text: str) -> Dict[str, Any]:
    """Convert one KMeans .txt output into expected evaluation prediction structure."""
    lines = raw_text.splitlines()
    results_by_topic: Dict[str, Dict[str, List[Dict[str, str]]]] = {}

    current_topic: str | None = None
    current_status: str = ""
    current_shifts: List[Dict[str, str]] = []

    def flush_topic() -> None:
        nonlocal current_topic, current_status, current_shifts
        if current_topic is None:
            return

        status_norm = current_status.lower()
        if "no narrative found" in status_norm:
            shifts_out: List[Dict[str, str]] = []
        else:
            shifts_out = current_shifts

        results_by_topic[current_topic] = {
            "sentence_level_narrative_shifts": shifts_out
        }

    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()

        if line.startswith("Topic:"):
            flush_topic()
            current_topic = line.split(":", 1)[1].strip()
            current_status = ""
            current_shifts = []
            idx += 1
            continue

        if current_topic is None:
            idx += 1
            continue

        if line.startswith("Status:"):
            current_status = line.split(":", 1)[1].strip()
            idx += 1
            continue

        if re.match(r"^Shift\s*#\d+", line):
            shift_obj, next_idx = _parse_shift_block(lines, idx + 1)
            current_shifts.append(shift_obj)
            idx = next_idx
            continue

        idx += 1

    flush_topic()

    return {"results_by_topic": results_by_topic}


def load_predictions(path: str) -> Dict[str, Any]:
    """Load and convert KMeans text prediction file into evaluator JSON structure."""
    pred_path = Path(path)
    with pred_path.open("r", encoding="utf-8") as handle:
        raw_text = handle.read()
    return _parse_kmeans_prediction_text(raw_text)


def extract_sentences_from_context(context: str) -> List[str]:
    """Extract sentence strings from model context fields.

    Handles lines like:
      >>> [article_1_s5] sentence text
          [article_1_s6] sentence text
    """
    if not isinstance(context, str) or not context.strip():
        return []

    raw_lines = [line.strip() for line in context.splitlines() if line.strip()]
    cleaned: List[str] = []

    for line in raw_lines:
        line = re.sub(r"^\s*>>>\s*", "", line)
        line = re.sub(r"^\s*\[[^\]]+\]\s*", "", line)
        line = line.strip().strip('"').strip("'").strip()
        if line:
            cleaned.append(line)

    if cleaned:
        return cleaned

    fallback = context.strip().strip('"').strip("'").strip()
    return [fallback] if fallback else []


def _sentence_matches_group(normalized_context_sentence: str, normalized_gt_group: Iterable[str]) -> bool:
    """Return True if context sentence matches any GT sentence by contain-or-exact rule."""
    if not normalized_context_sentence:
        return False
    for gt_sentence in normalized_gt_group:
        if not gt_sentence:
            continue
        if (
            normalized_context_sentence == gt_sentence
            or gt_sentence in normalized_context_sentence
            or normalized_context_sentence in gt_sentence
        ):
            return True
    return False


def match_prediction(
    prediction: Dict[str, Any],
    gt_group_a: Sequence[str],
    gt_group_b: Sequence[str],
) -> bool:
    """Return True if one predicted shift satisfies the TP rule."""
    context_1 = prediction.get("context_1", "")
    context_2 = prediction.get("context_2", "")

    context_1_sentences = [normalize_text(s) for s in extract_sentences_from_context(context_1)]
    context_2_sentences = [normalize_text(s) for s in extract_sentences_from_context(context_2)]

    if not context_1_sentences or not context_2_sentences:
        return False

    has_group_a_match = any(
        _sentence_matches_group(sent, gt_group_a)
        for sent in context_1_sentences
    )

    if not has_group_a_match:
        return False

    has_group_b_match = any(
        _sentence_matches_group(sent, gt_group_b)
        for sent in context_2_sentences
    )

    return has_group_b_match


def compute_metrics_for_file(
    ground_truth_path: str,
    prediction_path: str,
    debug: bool = False,
    log_per_topic: bool = False,
) -> Dict[str, Any]:
    """Compute TP/FP/FN for one GT file and one prediction file."""
    gt_data = parse_ground_truth(load_ground_truth(ground_truth_path))

    prediction_data = load_predictions(prediction_path)
    results_by_topic = prediction_data.get("results_by_topic", {})
    if not isinstance(results_by_topic, dict):
        results_by_topic = {}

    topics = set(gt_data.keys()) | set(results_by_topic.keys())
    file_tp = 0
    file_fp = 0
    file_fn = 0

    topic_breakdown: Dict[str, Dict[str, int]] = {}

    for topic in sorted(topics):
        gt_topic = gt_data.get(
            topic,
            {
                "label": NO_NARRATIVE,
                "has_narrative": False,
                "group_a": set(),
                "group_b": set(),
            },
        )

        prediction_topic = results_by_topic.get(topic, {})
        shifts = []
        if isinstance(prediction_topic, dict):
            candidate = prediction_topic.get("sentence_level_narrative_shifts", [])
            if isinstance(candidate, list):
                shifts = [s for s in candidate if isinstance(s, dict)]

        topic_tp = 0
        topic_fp = 0
        topic_fn = 0

        if gt_topic["has_narrative"]:
            has_true_positive = False
            gt_group_a = list(gt_topic["group_a"])
            gt_group_b = list(gt_topic["group_b"])

            for idx, shift in enumerate(shifts):
                matched = match_prediction(shift, gt_group_a, gt_group_b)
                if matched:
                    topic_tp += 1
                    has_true_positive = True
                else:
                    topic_fp += 1

                if debug:
                    outcome = "TP" if matched else "FP"
                    print(
                        f"[DEBUG] topic={topic} shift_index={idx} outcome={outcome}"
                    )

            if not has_true_positive:
                topic_fn = 1
        else:
            # No narrative in GT: every predicted shift is a false positive.
            topic_fp = len(shifts)
            if debug and shifts:
                print(
                    f"[DEBUG] topic={topic} no_narrative_gt predicted_shifts={len(shifts)} -> all FP"
                )

        file_tp += topic_tp
        file_fp += topic_fp
        file_fn += topic_fn

        topic_breakdown[topic] = {"tp": topic_tp, "fp": topic_fp, "fn": topic_fn}

        if log_per_topic:
            print(
                f"[TOPIC] {topic}: TP={topic_tp} FP={topic_fp} FN={topic_fn}"
            )

    return {
        "tp": file_tp,
        "fp": file_fp,
        "fn": file_fn,
        "topics": topic_breakdown,
        "ground_truth_path": ground_truth_path,
        "prediction_path": prediction_path,
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _metrics_from_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def aggregate_metrics(
    ground_truth_paths: Dict[str, List[str]],
    prediction_paths: Dict[str, List[str]],
    debug: bool = False,
    log_per_topic: bool = False,
) -> Dict[str, Any]:
    """Aggregate metrics by difficulty across multiple files."""
    aggregated: Dict[str, Any] = {}

    for difficulty in DIFFICULTIES:
        gt_files = ground_truth_paths.get(difficulty, [])
        pred_files = prediction_paths.get(difficulty, [])

        if len(gt_files) != len(pred_files):
            raise ValueError(
                f"Mismatch for {difficulty}: "
                f"ground_truth_paths={len(gt_files)} prediction_paths={len(pred_files)}"
            )

        difficulty_tp = 0
        difficulty_fp = 0
        difficulty_fn = 0
        file_results: List[Dict[str, Any]] = []

        for gt_file, pred_file in zip(gt_files, pred_files):
            file_result = compute_metrics_for_file(
                ground_truth_path=gt_file,
                prediction_path=pred_file,
                debug=debug,
                log_per_topic=log_per_topic,
            )
            file_results.append(file_result)

            difficulty_tp += file_result["tp"]
            difficulty_fp += file_result["fp"]
            difficulty_fn += file_result["fn"]

        metrics = _metrics_from_counts(difficulty_tp, difficulty_fp, difficulty_fn)
        aggregated[difficulty] = {
            "tp": difficulty_tp,
            "fp": difficulty_fp,
            "fn": difficulty_fn,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "files": file_results,
        }

    return aggregated


def _build_ground_truth_paths() -> Dict[str, List[str]]:
    gt_paths: Dict[str, List[str]] = {difficulty: [] for difficulty in DIFFICULTIES}

    for difficulty in DIFFICULTIES:
        for ner_id in range(1, 6):
            gt_file = INPUT_ROOT / f"Ner{ner_id}_{difficulty}" / "ground_truth.txt"
            if not gt_file.exists():
                raise FileNotFoundError(f"Missing ground truth file: {gt_file}")
            gt_paths[difficulty].append(str(gt_file.resolve()))

    return gt_paths


def _build_prediction_paths() -> Dict[str, List[str]]:
    pred_paths: Dict[str, List[str]] = {difficulty: [] for difficulty in DIFFICULTIES}

    for difficulty in DIFFICULTIES:
        for ner_id in range(1, 6):
            pred_file = PREDICTION_ROOT / f"Ner{ner_id}_{difficulty}_combined.txt"
            if not pred_file.exists():
                raise FileNotFoundError(f"Missing prediction file: {pred_file}")
            pred_paths[difficulty].append(str(pred_file.resolve()))

    return pred_paths


def _print_final_table(metrics: Dict[str, Any]) -> None:
    headers = [
        "Model",
        "Hard_F1",
        "Hard_P",
        "Hard_R",
        "Medium_F1",
        "Medium_P",
        "Medium_R",
        "Low_F1",
        "Low_P",
        "Low_R",
        "Overall (P/R/F1)",
    ]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))

    hard = metrics["hard"]
    medium = metrics["medium"]
    low = metrics["low"]
    tp_total = hard["tp"] + medium["tp"] + low["tp"]
    fp_total = hard["fp"] + medium["fp"] + low["fp"]
    fn_total = hard["fn"] + medium["fn"] + low["fn"]
    overall = _metrics_from_counts(tp_total, fp_total, fn_total)

    row = [
        MODEL_NAME,
        f"{hard['f1']:.4f}",
        f"{hard['precision']:.4f}",
        f"{hard['recall']:.4f}",
        f"{medium['f1']:.4f}",
        f"{medium['precision']:.4f}",
        f"{medium['recall']:.4f}",
        f"{low['f1']:.4f}",
        f"{low['precision']:.4f}",
        f"{low['recall']:.4f}",
        f"P={overall['precision']:.4f} / R={overall['recall']:.4f} / F1={overall['f1']:.4f}",
    ]
    print(" | ".join(row))


def _build_compact_metrics(metrics: Dict[str, Any]) -> Dict[str, Dict[str, float | int]]:
    compact: Dict[str, Dict[str, float | int]] = {}
    for difficulty in DIFFICULTIES:
        d = metrics[difficulty]
        compact[difficulty] = {
            "tp": d["tp"],
            "fp": d["fp"],
            "fn": d["fn"],
            "precision": d["precision"],
            "recall": d["recall"],
            "f1": d["f1"],
        }
    return compact


def main() -> None:
    gt_paths = _build_ground_truth_paths()
    pred_paths = _build_prediction_paths()

    metrics = aggregate_metrics(
        ground_truth_paths=gt_paths,
        prediction_paths=pred_paths,
        debug=False,
        log_per_topic=False,
    )

    _print_final_table(metrics)

    payload = {
        "model": MODEL_NAME,
        "metrics": _build_compact_metrics(metrics),
    }
    with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)

    print(f"\nSaved JSON results to: {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()

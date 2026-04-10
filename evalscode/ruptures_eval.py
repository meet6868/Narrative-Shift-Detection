#!/usr/bin/env python3
"""Evaluate rupture-based narrative shift outputs against ground truth files.

This script:
1. Discovers rupture config outputs.
2. Filters only complete configs (all Ner1..5 across hard/medium/low).
3. Evaluates each complete config using existing evaluation logic.
4. Computes overall metrics and ranks by overall F1.
5. Prints top 5 configs and saves JSON results.
"""

from __future__ import annotations

import json
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Sequence, Tuple

DIFFICULTIES: Tuple[str, str, str] = ("hard", "medium", "low")
NO_NARRATIVE = "NO NARRATIVE"
INPUT_ROOT_DIRNAME = "newinput"
OUTPUT_JSON_FILENAME = "ruptures_eval.json"
RUPTURES_OUTPUT_DIR = "TCL/Approach_lambda_output"
RUPTURES_FILE_PATTERN = re.compile(
    r"^(?P<config>.+)__Ner(?P<ner>[1-5])_(?P<difficulty>hard|medium|low)\.json$",
    re.IGNORECASE,
)


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


def load_predictions(path: str) -> Dict[str, Any]:
    """Load and return model predictions JSON."""
    pred_path = Path(path)
    with pred_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def _build_ground_truth_paths(input_root: Path) -> Dict[str, List[str]]:
    """Build ordered GT path map from newinput naming: Ner{1..5}_{difficulty}."""
    gt_paths: Dict[str, List[str]] = {difficulty: [] for difficulty in DIFFICULTIES}

    for difficulty in DIFFICULTIES:
        for ner_id in range(1, 6):
            gt_file = input_root / f"Ner{ner_id}_{difficulty}" / "ground_truth.txt"
            if not gt_file.exists():
                raise FileNotFoundError(
                    f"Missing ground truth file for Ner{ner_id}_{difficulty}: {gt_file}"
                )
            gt_paths[difficulty].append(str(gt_file.resolve()))

    return gt_paths


def _discover_complete_ruptures_configs(output_root: Path) -> Tuple[List[Dict[str, Any]], int, int, int]:
    """Discover rupture configs and return only complete ones."""
    if not output_root.exists() or not output_root.is_dir():
        raise FileNotFoundError(f"Output directory not found: {output_root}")

    grouped: DefaultDict[str, Dict[str, Dict[int, str]]] = defaultdict(
        lambda: {"hard": {}, "medium": {}, "low": {}}
    )

    for pred_file in sorted(output_root.glob("approch_ruptures_*__Ner*.json")):
        match = RUPTURES_FILE_PATTERN.match(pred_file.name)
        if not match:
            continue

        config = match.group("config")
        ner_id = int(match.group("ner"))
        difficulty = match.group("difficulty").lower()
        grouped[config][difficulty][ner_id] = str(pred_file.resolve())

    expected_ners = {1, 2, 3, 4, 5}
    total_configs = len(grouped)

    complete_configs: List[Dict[str, Any]] = []
    for config, coverage in sorted(grouped.items()):
        is_complete = all(set(coverage[difficulty].keys()) == expected_ners for difficulty in DIFFICULTIES)
        if not is_complete:
            continue

        prediction_paths: Dict[str, List[str]] = {difficulty: [] for difficulty in DIFFICULTIES}
        for difficulty in DIFFICULTIES:
            for ner_id in sorted(coverage[difficulty].keys()):
                prediction_paths[difficulty].append(coverage[difficulty][ner_id])

        complete_configs.append(
            {
                "config": config,
                "prediction_paths": prediction_paths,
            }
        )

    complete_count = len(complete_configs)
    skipped_incomplete = total_configs - complete_count
    return complete_configs, total_configs, complete_count, skipped_incomplete


def _compute_overall(metrics: Dict[str, Any]) -> Dict[str, float | int]:
    tp_total = int(metrics["hard"]["tp"]) + int(metrics["medium"]["tp"]) + int(metrics["low"]["tp"])
    fp_total = int(metrics["hard"]["fp"]) + int(metrics["medium"]["fp"]) + int(metrics["low"]["fp"])
    fn_total = int(metrics["hard"]["fn"]) + int(metrics["medium"]["fn"]) + int(metrics["low"]["fn"])

    overall = _metrics_from_counts(tp_total, fp_total, fn_total)
    return {
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "precision": overall["precision"],
        "recall": overall["recall"],
        "f1": overall["f1"],
    }


def _compact_metric_block(metric: Dict[str, Any]) -> Dict[str, float | int]:
    return {
        "tp": int(metric["tp"]),
        "fp": int(metric["fp"]),
        "fn": int(metric["fn"]),
        "precision": float(metric["precision"]),
        "recall": float(metric["recall"]),
        "f1": float(metric["f1"]),
    }


def _print_top5_table(rows: List[Dict[str, Any]]) -> None:
    headers = [
        "Rank",
        "Config",
        "Hard_F1",
        "Hard_P",
        "Hard_R",
        "Medium_F1",
        "Medium_P",
        "Medium_R",
        "Low_F1",
        "Low_P",
        "Low_R",
        "Overall_F1",
        "Overall_P",
        "Overall_R",
    ]

    table_rows: List[List[str]] = []
    for row in rows:
        hard = row["metrics"]["hard"]
        medium = row["metrics"]["medium"]
        low = row["metrics"]["low"]
        overall = row["metrics"]["overall"]
        table_rows.append(
            [
                str(row["rank"]),
                row["config"],
                f"{hard['f1']:.4f}",
                f"{hard['precision']:.4f}",
                f"{hard['recall']:.4f}",
                f"{medium['f1']:.4f}",
                f"{medium['precision']:.4f}",
                f"{medium['recall']:.4f}",
                f"{low['f1']:.4f}",
                f"{low['precision']:.4f}",
                f"{low['recall']:.4f}",
                f"{overall['f1']:.4f}",
                f"{overall['precision']:.4f}",
                f"{overall['recall']:.4f}",
            ]
        )

    widths = [len(h) for h in headers]
    for row in table_rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def format_row(values: List[str]) -> str:
        return " | ".join(values[i].ljust(widths[i]) for i in range(len(values)))

    print(format_row(headers))
    print("+".join("-" * w for w in widths))
    for row in table_rows:
        print(format_row(row))


def main() -> None:
    project_root = Path(__file__).resolve().parent
    input_root = project_root / INPUT_ROOT_DIRNAME
    output_root = project_root / RUPTURES_OUTPUT_DIR
    output_json_path = project_root / OUTPUT_JSON_FILENAME

    ground_truth_paths = _build_ground_truth_paths(input_root=input_root)

    complete_configs, total_configs, complete_count, skipped_count = _discover_complete_ruptures_configs(
        output_root=output_root
    )

    print(f"Total configs found: {total_configs}")
    print(f"Complete configs: {complete_count}")
    print(f"Skipped incomplete: {skipped_count}")

    if not complete_configs:
        raise ValueError("No complete rupture configs found to evaluate.")

    all_results: List[Dict[str, Any]] = []
    for entry in complete_configs:
        metrics = aggregate_metrics(
            ground_truth_paths=ground_truth_paths,
            prediction_paths=entry["prediction_paths"],
            debug=False,
            log_per_topic=False,
        )

        result = {
            "config": entry["config"],
            "metrics": {
                "hard": _compact_metric_block(metrics["hard"]),
                "medium": _compact_metric_block(metrics["medium"]),
                "low": _compact_metric_block(metrics["low"]),
                "overall": _compute_overall(metrics),
            },
        }
        all_results.append(result)

    sorted_results = sorted(
        all_results,
        key=lambda item: float(item["metrics"]["overall"]["f1"]),
        reverse=True,
    )

    for idx, result in enumerate(sorted_results, start=1):
        result["rank"] = idx

    top_5 = sorted_results[:5]

    print()
    _print_top5_table(top_5)

    payload = {
        "num_configs": len(sorted_results),
        "top_5": top_5,
        "all_results": sorted_results,
    }
    with output_json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)

    print(f"\nSaved JSON results to: {output_json_path}")


if __name__ == "__main__":
    main()

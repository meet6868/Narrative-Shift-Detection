#!/usr/bin/env python3
"""Evaluate Approach1 narrative shift predictions against ground truth files.

This script evaluates at shift-level with the following core logic:
1. A predicted shift is TP only if context_1 matches GT Group A and context_2 matches GT Group B.
2. Non-matching predicted shifts are FP.
3. If GT has narrative but no TP exists for that topic, FN=1 for that topic.
4. If GT says NO NARRATIVE, every predicted shift for that topic is FP.

This script is strict Approach1 mode:
- Ground truth root is hardcoded to: newinput
- Prediction root is hardcoded to: Approach1_output
- It auto-discovers all Approach1 configs from filenames and evaluates each config.
- It saves full evaluation output to: approach1_eval.json
"""

from __future__ import annotations

from collections import defaultdict
import json
import re
import string
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Sequence, Tuple

DIFFICULTIES: Tuple[str, str, str] = ("hard", "medium", "low")
NO_NARRATIVE = "NO NARRATIVE"
INPUT_ROOT_DIRNAME = "newinput"
OUTPUT_ROOT_DIRNAME = "Approach1_output"
OUTPUT_JSON_FILENAME = "approach1_eval.json"
APPROACH1_PATTERN = re.compile(
    r"^(?P<prefix>.+?)_w(?P<w>\d+)_s(?P<s>\d+)_t(?P<t>\d+p\d+)_Ner(?P<ner>\d+)_(?P<difficulty>hard|medium|low)_results\.json$",
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


def _format_triplet(precision: float, recall: float, f1: float) -> str:
    return f"P={precision:.4f} / R={recall:.4f} / F1={f1:.4f}"


def _temperature_token_to_float_string(token: str) -> str:
    return token.replace("p", ".")


def _build_ground_truth_paths_from_input_root(input_root: Path) -> Dict[str, List[str]]:
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


def _discover_approach1_configs(
    output_dir: Path,
    input_root: Path,
    allow_incomplete_configs: bool = False,
) -> List[Dict[str, Any]]:
    """Discover Approach1-style configs from output filenames and build approach entries."""
    if not output_dir.exists() or not output_dir.is_dir():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    gt_paths = _build_ground_truth_paths_from_input_root(input_root=input_root)

    grouped: DefaultDict[
        Tuple[str, str, str, str],
        Dict[str, Dict[int, str]],
    ] = defaultdict(lambda: {"hard": {}, "medium": {}, "low": {}})

    matched_file_count = 0
    for pred_file in sorted(output_dir.glob("*_results.json")):
        match = APPROACH1_PATTERN.match(pred_file.name)
        if not match:
            continue
        matched_file_count += 1
        prefix = match.group("prefix")
        window = match.group("w")
        stride = match.group("s")
        temperature_token = match.group("t")
        ner_id = int(match.group("ner"))
        difficulty = match.group("difficulty").lower()

        grouped[(prefix, window, stride, temperature_token)][difficulty][ner_id] = str(
            pred_file.resolve()
        )

    if matched_file_count == 0:
        raise ValueError(
            f"No prediction files matched Approach1 naming convention in {output_dir}"
        )

    discovered: List[Dict[str, Any]] = []
    expected_ners = {1, 2, 3, 4, 5}

    for (prefix, window, stride, temperature_token), coverage in sorted(grouped.items()):
        is_complete = all(set(coverage[difficulty].keys()) == expected_ners for difficulty in DIFFICULTIES)
        if not is_complete and not allow_incomplete_configs:
            continue

        prediction_paths: Dict[str, List[str]] = {difficulty: [] for difficulty in DIFFICULTIES}
        for difficulty in DIFFICULTIES:
            ner_ids = sorted(coverage[difficulty].keys())
            for ner_id in ner_ids:
                prediction_paths[difficulty].append(coverage[difficulty][ner_id])

        temperature = _temperature_token_to_float_string(temperature_token)
        label = f"{prefix}_w{window}_s{stride}_t{temperature}"

        discovered.append(
            {
                "name": label,
                "hyperparameters": {
                    "window_size": int(window),
                    "stride": int(stride),
                    "temperature": float(temperature),
                    "source_prefix": prefix,
                },
                "ground_truth_paths": gt_paths,
                "prediction_paths": prediction_paths,
            }
        )

    if not discovered:
        raise ValueError(
            "No complete configs found in output directory. "
            "Use --allow-incomplete-configs if you want to evaluate partial runs."
        )

    return discovered


def _infer_row_label(approach_config: Dict[str, Any]) -> str:
    for key in ("name", "hyperparameters", "label"):
        value = approach_config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if key == "hyperparameters" and isinstance(value, dict):
            w = value.get("window_size")
            s = value.get("stride")
            t = value.get("temperature")
            if w is not None and s is not None and t is not None:
                return f"w{w}_s{s}_t{t}"

    # Try to infer from first prediction file stem if no label was provided.
    prediction_paths = approach_config.get("prediction_paths", {})
    if isinstance(prediction_paths, dict):
        for difficulty in DIFFICULTIES:
            files = prediction_paths.get(difficulty, [])
            if files:
                return Path(files[0]).stem

    return "Unknown"


def _validate_approach_config(approach_config: Dict[str, Any]) -> None:
    required = ("ground_truth_paths", "prediction_paths")
    for key in required:
        if key not in approach_config:
            raise ValueError(f"Approach entry missing required key: {key}")
        if not isinstance(approach_config[key], dict):
            raise ValueError(f"Approach key {key} must be an object with hard/medium/low lists")


def _print_table(rows: List[Dict[str, Any]]) -> None:
    print("| Config | Hard (P/R/F1) | Medium (P/R/F1) | Low (P/R/F1) | Overall (P/R/F1) |")
    print("| ------ | ------------- | --------------- | ------------ | ---------------- |")
    ranked_rows: List[Tuple[float, Dict[str, Any]]] = []
    for row in rows:
        hard = row["metrics"]["hard"]
        medium = row["metrics"]["medium"]
        low = row["metrics"]["low"]
        tp_total = hard["tp"] + medium["tp"] + low["tp"]
        fp_total = hard["fp"] + medium["fp"] + low["fp"]
        fn_total = hard["fn"] + medium["fn"] + low["fn"]
        overall = _metrics_from_counts(tp_total, fp_total, fn_total)
        ranked_rows.append((overall["f1"], row))

    top_rows = [item[1] for item in sorted(ranked_rows, key=lambda x: x[0], reverse=True)[:5]]

    for row in top_rows:
        hard = row["metrics"]["hard"]
        medium = row["metrics"]["medium"]
        low = row["metrics"]["low"]
        tp_total = hard["tp"] + medium["tp"] + low["tp"]
        fp_total = hard["fp"] + medium["fp"] + low["fp"]
        fn_total = hard["fn"] + medium["fn"] + low["fn"]
        overall = _metrics_from_counts(tp_total, fp_total, fn_total)
        print(
            "| "
            f"{row['label']} | "
            f"{_format_triplet(hard['precision'], hard['recall'], hard['f1'])} | "
            f"{_format_triplet(medium['precision'], medium['recall'], medium['f1'])} | "
            f"{_format_triplet(low['precision'], low['recall'], low['f1'])} | "
            f"{_format_triplet(overall['precision'], overall['recall'], overall['f1'])} |"
        )


def _print_pretty_metrics(label: str, metrics: Dict[str, Any]) -> None:
    print(f"\n{label}")
    for difficulty in DIFFICULTIES:
        d = metrics[difficulty]
        print(
            f"{difficulty.capitalize():<7} -> "
            f"Precision={d['precision']:.4f}, Recall={d['recall']:.4f}, F1={d['f1']:.4f} "
            f"(TP={d['tp']}, FP={d['fp']}, FN={d['fn']})"
        )


def main() -> None:
    project_root = Path(__file__).resolve().parent
    input_root = project_root / INPUT_ROOT_DIRNAME
    output_dir = project_root / OUTPUT_ROOT_DIRNAME
    output_json_path = project_root / OUTPUT_JSON_FILENAME

    approaches = _discover_approach1_configs(
        output_dir=output_dir,
        input_root=input_root,
        allow_incomplete_configs=False,
    )

    print(f"Using input root: {input_root}")
    print(f"Using output root: {output_dir}")
    print(f"Discovered configs: {len(approaches)}")

    rows: List[Dict[str, Any]] = []

    for approach in approaches:
        _validate_approach_config(approach)
        gt_paths = approach["ground_truth_paths"]
        pred_paths = approach["prediction_paths"]

        metrics = aggregate_metrics(
            ground_truth_paths=gt_paths,
            prediction_paths=pred_paths,
            debug=False,
            log_per_topic=False,
        )

        label = _infer_row_label(approach)
        rows.append(
            {
                "label": label,
                "hyperparameters": approach.get("hyperparameters", {}),
                "ground_truth_paths": gt_paths,
                "prediction_paths": pred_paths,
                "metrics": metrics,
            }
        )

    print()
    _print_table(rows)

    output_payload = {
        "input_root": str(input_root),
        "output_root": str(output_dir),
        "num_configs": len(rows),
        "results": rows,
    }
    with output_json_path.open("w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2, ensure_ascii=True)
    print(f"\nSaved JSON results to: {output_json_path}")


if __name__ == "__main__":
    main()

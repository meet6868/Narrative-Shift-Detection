from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch


def _cfg_get(config, key, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _extract_topics(config):
    topics = _cfg_get(config, "topics", None)
    if topics is None:
        topics = _cfg_get(config, "TOPICS", [])
    return list(topics)


def _iter_topic_windows(topic_window_data, topic_name, per_topic_limit):
    payload = topic_window_data.get(topic_name, [])
    if isinstance(payload, dict):
        windows = payload.get("windows", [])
    else:
        windows = payload

    if isinstance(windows, np.ndarray):
        return list(windows[:per_topic_limit])
    return list(windows)[:per_topic_limit]


def _window_to_tensor_np(window):
    if isinstance(window, dict):
        value = window.get("tensor")
        if value is None:
            return None
        return np.asarray(value, dtype=np.float32)
    return np.asarray(window, dtype=np.float32)


def evaluate_model_quality(model, topic_window_data, config, device, per_topic_limit=200, inter_pairs=100, seed=42):
    """Shared evaluation for AP1/AP2/AP4/AP5 with robust window payload handling."""
    model.eval()
    rng = np.random.default_rng(seed)

    topics = _extract_topics(config)
    topic_embeddings = {topic_name: [] for topic_name in topics}

    with torch.no_grad():
        for topic_name in topics:
            windows = _iter_topic_windows(topic_window_data, topic_name, per_topic_limit)
            for window in windows:
                tensor_np = _window_to_tensor_np(window)
                if tensor_np is None:
                    continue
                tensor = torch.from_numpy(tensor_np).unsqueeze(0).to(device)
                encoded = model(tensor).cpu().numpy()[0].astype(np.float32)
                topic_embeddings[topic_name].append(encoded)
            topic_embeddings[topic_name] = np.asarray(topic_embeddings[topic_name], dtype=np.float32)

    intra_scores = {}
    intra_pair_sims = {}
    for topic_name, embeddings in topic_embeddings.items():
        if len(embeddings) < 2:
            continue
        sims = [float(np.dot(embeddings[i], embeddings[i + 1])) for i in range(len(embeddings) - 1)]
        intra_pair_sims[topic_name] = sims
        intra_scores[topic_name] = float(np.mean(sims)) if sims else 0.0

    inter_scores = {}
    inter_pair_sims = {}
    topic_names = list(topic_embeddings.keys())
    for i in range(len(topic_names)):
        for j in range(i + 1, len(topic_names)):
            left_name, right_name = topic_names[i], topic_names[j]
            left = topic_embeddings[left_name]
            right = topic_embeddings[right_name]
            if len(left) == 0 or len(right) == 0:
                continue

            pairs = min(int(inter_pairs), len(left) * len(right))
            if pairs <= 0:
                continue
            left_idx = rng.integers(0, len(left), size=pairs)
            right_idx = rng.integers(0, len(right), size=pairs)
            sims = [float(np.dot(left[li], right[ri])) for li, ri in zip(left_idx, right_idx)]

            pair_key = f"{left_name}-{right_name}"
            inter_pair_sims[pair_key] = sims
            inter_scores[pair_key] = float(np.mean(sims)) if sims else 0.0

    mean_intra = float(np.mean(list(intra_scores.values()))) if intra_scores else 0.0
    mean_inter = float(np.mean(list(inter_scores.values()))) if inter_scores else 0.0
    separation_score = mean_intra / (mean_inter + 1e-8) if mean_inter != 0 else 0.0

    intra_all = [v for values in intra_pair_sims.values() for v in values]
    inter_all = [v for values in inter_pair_sims.values() for v in values]

    similarity_extremes = {
        "intra_min": float(np.min(intra_all)) if intra_all else None,
        "intra_max": float(np.max(intra_all)) if intra_all else None,
        "inter_min": float(np.min(inter_all)) if inter_all else None,
        "inter_max": float(np.max(inter_all)) if inter_all else None,
    }

    return {
        "intra_scores": intra_scores,
        "inter_scores": inter_scores,
        "separation_score": separation_score,
        "similarity_extremes": similarity_extremes,
    }


def plot_evaluation_heatmaps(evaluation_metrics, config, intra_path, inter_path):
    topics = _extract_topics(config)
    intra_scores = evaluation_metrics.get("intra_scores", {})
    inter_scores = evaluation_metrics.get("inter_scores", {})

    intra_values = [float(intra_scores.get(topic, np.nan)) for topic in topics]
    intra_matrix = np.asarray(intra_values, dtype=np.float32).reshape(1, -1)

    inter_matrix = np.full((len(topics), len(topics)), np.nan, dtype=np.float32)
    for i, left in enumerate(topics):
        inter_matrix[i, i] = 1.0
        for j, right in enumerate(topics):
            if i >= j:
                continue
            key = f"{left}-{right}"
            if key not in inter_scores:
                key = f"{right}-{left}"
            if key in inter_scores:
                inter_matrix[i, j] = float(inter_scores[key])
                inter_matrix[j, i] = float(inter_scores[key])

    plt.figure(figsize=(10, 2.6))
    sns.heatmap(
        intra_matrix,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        xticklabels=topics,
        yticklabels=["Intra Mean"],
        cbar=True,
    )
    plt.title("Intra-topic Similarity Heatmap")
    plt.tight_layout()
    plt.savefig(intra_path, dpi=150)
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        inter_matrix,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        xticklabels=topics,
        yticklabels=topics,
        cbar=True,
    )
    plt.title("Inter-topic Similarity Heatmap")
    plt.tight_layout()
    plt.savefig(inter_path, dpi=150)
    plt.show()
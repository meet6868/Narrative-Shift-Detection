import os
import json
import ast
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm
from config import load_approach_2_config
from dataloader import TemporalWindowDataset

from load_csv import (
    load_topic_csv as shared_load_topic_csv,
)
from inference import (
    compute_topic_drift as compute_topic_drift_inference,
    detect_shifts as detect_shifts_inference,
    extract_sentence_level_narrative_shifts as extract_sentence_level_narrative_shifts_inference,
    print_batch_inference_outputs,
    run_batch_inference_approach2,
    run_user_inference_approach2,
    run_user_level_inference_approach2 as run_user_level_inference_inference,
)
from grouping import create_grouped_vectors_from_daily_ap2
from losses import EnhancedNTXentLossA12
from models import TCLTemporalEncoderA12
from evaluation import evaluate_model_quality, plot_evaluation_heatmaps
from plotting import plot_training_loss
from temporal_feature import aggregate_daily_embeddings, build_temporal_feature_records
from training import train_tcl_model_a12
from windowing import build_window_embeddings

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
config = load_approach_2_config()


# Stage A: Load topic-wise sentence data for AP2.
topic_sentence_data = {}
for topic_name in config["topics"]:
    topic_sentence_data[topic_name] = shared_load_topic_csv(topic_name, config, mode="ap2")
    print(f"Loaded {topic_name}: {len(topic_sentence_data[topic_name])} rows")


topic_daily_data = {}
topic_group_data = {}
topic_window_data = {}
all_window_embeddings = []

# Stage B: Daily aggregation -> AP2 grouping -> temporal windows.
for topic_name in config["topics"]:
    daily_dataframe = aggregate_daily_embeddings(
        dataframe=topic_sentence_data[topic_name],
        topics=[topic_name],
        min_sentences_per_day=config["min_sentences_per_day"],
        embedding_column="embedding",
        weight_column_map={topic_name: [topic_name]},
        topic_embeddings_column="topic_embeddings",
        fallback_topic_embeddings_map=None,
        normalize_date=False,
        require_weight_column=False,
        entity_signature_column=None,
        output_embedding_column="daily_vectors",
        topic_column_name="topic_name",
        include_topic_id=True,
        include_avg_weight=False,
    )
    grouped_dataframe = create_grouped_vectors_from_daily_ap2(daily_dataframe, config)

    topic_daily_data[topic_name] = daily_dataframe
    topic_group_data[topic_name] = grouped_dataframe

    enhanced_records = build_temporal_feature_records(
        grouped_dataframe,
        include_tau=True,
        tau_scale=5.0,
        include_end_date=True,
        include_num_sentences=True,
        include_num_days=True,
    )
    topic_id = config["topics"].index(topic_name)
    window_embeddings = build_window_embeddings(enhanced_records, topic_name, topic_id, config)

    topic_window_data[topic_name] = window_embeddings
    all_window_embeddings.extend(window_embeddings)

    print(
        f"{topic_name}: days={len(daily_dataframe)} | groups={len(grouped_dataframe)} | windows={len(window_embeddings)}"
    )

print(f"Total window_embeddings: {len(all_window_embeddings)}")


# Stage C: Build dataset/model/loss/optimizer.
train_dataset = TemporalWindowDataset(all_window_embeddings, config["topics"])
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True)

model = TCLTemporalEncoderA12(config).to(device)
loss_fn = EnhancedNTXentLossA12(config["temperature"])
optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

print(f"Dataset windows: {len(train_dataset)}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Stage D: Train AP2 temporal encoder.
model, training_history = train_tcl_model_a12(model, train_dataset, train_loader, optimizer, loss_fn, config, device)
plot_training_loss(training_history, config["train_loss_plot_path"])
print(f"Saved best model: {config['model_best_path']}")
print(f"Saved last model: {config['model_last_path']}")
print(f"Saved train loss plot: {config['train_loss_plot_path']}")


# Stage E: Evaluate learned representations and persist artifacts.
evaluation_metrics = evaluate_model_quality(model, topic_window_data, config, device)
print("Evaluation Metrics:")
print(json.dumps(evaluation_metrics, indent=2))

extremes = evaluation_metrics.get("similarity_extremes", {})
print("\nSimilarity Extremes:")
print(
    f"Intra -> min: {extremes.get('intra_min')} | max: {extremes.get('intra_max')} | "
    f"Inter -> min: {extremes.get('inter_min')} | max: {extremes.get('inter_max')}"
)

plot_evaluation_heatmaps(
    evaluation_metrics,
    config,
    intra_path=config["eval_heatmap_intra_path"],
    inter_path=config["eval_heatmap_inter_path"],
)

with open(config["eval_metrics_path"], "w", encoding="utf-8") as file:
    json.dump(evaluation_metrics, file, indent=2)

# Store evaluated model immediately after evaluation.
evaluated_checkpoint = {
    "model_state_dict": model.state_dict(),
    "evaluation_metrics": evaluation_metrics,
    "config": config,
}
torch.save(evaluated_checkpoint, config["model_evaluated_path"])

print(f"Saved evaluation metrics: {config['eval_metrics_path']}")
print(f"Saved intra heatmap: {config['eval_heatmap_intra_path']}")
print(f"Saved inter heatmap: {config['eval_heatmap_inter_path']}")
print(f"Saved evaluated model: {config['model_evaluated_path']}")

def compute_topic_drift(model, topic_windows, config, device):
    return compute_topic_drift_inference(model, topic_windows, config, device)


def detect_shifts(drift_rows, config):
    return detect_shifts_inference(drift_rows, config)


def split_articles_into_sentences(input_dataframe):
    import re

    sentence_rows = []
    required_cols = {"date", "article"}
    missing = required_cols - set(input_dataframe.columns)
    if missing:
        raise ValueError(f"Input CSV must contain columns: {required_cols}. Missing: {missing}")

    for article_idx, row in input_dataframe.reset_index(drop=True).iterrows():
        article_id = row.get("article_id", f"article_{article_idx}")
        date_value = pd.to_datetime(row["date"], errors="coerce")
        if pd.isna(date_value):
            continue

        text = str(row["article"]).strip()
        if not text:
            continue

        sentence_list = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", text)
            if s and s.strip()
        ]

        for sentence_order, sentence_text in enumerate(sentence_list):
            sentence_id = f"{article_id}_s{sentence_order}"
            sentence_rows.append({
                "date": date_value,
                "article_id": str(article_id),
                "sentence_id": sentence_id,
                "sentence_text": sentence_text,
                "sentence_order": int(sentence_order)
            })

    sentence_dataframe = pd.DataFrame(sentence_rows)
    if sentence_dataframe.empty:
        return pd.DataFrame(columns=["date", "article_id", "sentence_id", "sentence_text", "sentence_order"])
    return sentence_dataframe[["date", "article_id", "sentence_id", "sentence_text", "sentence_order"]]


def build_context_texts(sentence_dataframe, context_window):
    if context_window not in (3, 5):
        raise ValueError("context_window must be 3 or 5")

    radius = context_window // 2
    sentence_dataframe = sentence_dataframe.sort_values(["article_id", "sentence_order"]).reset_index(drop=True).copy()
    context_texts = [""] * len(sentence_dataframe)

    for _, group in sentence_dataframe.groupby("article_id", sort=False):
        indices = group.index.tolist()
        sentences = group["sentence_text"].tolist()

        for local_idx, global_idx in enumerate(indices):
            left = max(0, local_idx - radius)
            right = min(len(sentences), local_idx + radius + 1)
            context_texts[global_idx] = " ".join(sentences[left:right])

    sentence_dataframe["context_text"] = context_texts
    return sentence_dataframe


def load_topic_embedding_prototypes(topic_embeddings_json_path, config):
    with open(topic_embeddings_json_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    topic_embeddings = {}
    for topic_name in config["topics"]:
        if topic_name not in payload:
            raise KeyError(f"Topic '{topic_name}' missing in {topic_embeddings_json_path}")
        vector = np.asarray(payload[topic_name], dtype=np.float32)
        if vector.shape[0] != config["embedding_dim"]:
            raise ValueError(
                f"Topic embedding dim mismatch for {topic_name}: {vector.shape[0]} vs {config['embedding_dim']}"
            )
        norm = np.linalg.norm(vector)
        topic_embeddings[topic_name] = (vector / (norm + 1e-8)).astype(np.float32)

    return topic_embeddings


def soft_topic_label_sentences(sentence_dataframe, topic_embeddings, config):
    # Returns one row per sentence (training-like), including all topic columns.
    rows = []
    topic_names = config["topics"]
    topic_matrix = np.stack([topic_embeddings[name] for name in topic_names]).astype(np.float32)

    if sentence_dataframe.empty:
        base_cols = [
            "date", "article_id", "sentence_id", "sentence_text", "sentence_order",
            "sentence_embeddings", "topic_embeddings", "topic_probabilities"
        ] + topic_names
        return pd.DataFrame(columns=base_cols)

    for row in sentence_dataframe.itertuples(index=False):
        emb = np.asarray(row.sentence_embeddings, dtype=np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-8)

        similarities = np.dot(topic_matrix, emb).astype(np.float32)
        exp_sim = np.exp(similarities - np.max(similarities)).astype(np.float32)
        topic_probs = (exp_sim / (exp_sim.sum() + 1e-8)).astype(np.float32)

        record = {
            "date": row.date,
            "article_id": row.article_id,
            "sentence_id": row.sentence_id,
            "sentence_text": row.sentence_text,
            "sentence_order": int(row.sentence_order),
            "sentence_embeddings": emb.astype(np.float32),
            "topic_embeddings": topic_probs.astype(np.float32),
            # Keep both aliases to match training CSV style where w3/w5 may exist.
            "w3_embedding": emb.astype(np.float32),
            "w5_embedding": emb.astype(np.float32),
            "topic_probabilities": topic_probs.astype(np.float32)
        }

        for topic_idx, topic_name in enumerate(topic_names):
            record[topic_name] = np.float32(topic_probs[topic_idx])

        rows.append(record)

    return pd.DataFrame(rows)


def build_topic_score_rows(labeled_sentence_dataframe, config):
    # Long-format explainability table: sentence_id | topic | similarity_score(weight)
    rows = []
    for row in labeled_sentence_dataframe.itertuples(index=False):
        for topic_name in config["topics"]:
            rows.append({
                "sentence_id": row.sentence_id,
                "topic": topic_name,
                "similarity_score": float(getattr(row, topic_name))
            })
    return rows


def filter_user_topic_sentences(labeled_sentence_dataframe, user_topic, config):
    if user_topic not in config["topics"]:
        raise ValueError(f"Unknown topic '{user_topic}'. Expected one of {config['topics']}")

    filtered = labeled_sentence_dataframe[
        labeled_sentence_dataframe[user_topic] >= float(config["topic_threshold"])
    ].copy()
    filtered["selected_topic"] = user_topic
    filtered["similarity_score"] = filtered[user_topic].astype(np.float32)

    return filtered.sort_values(["date", "article_id", "sentence_order"]).reset_index(drop=True)


def validate_inference_alignment(config, filtered_sentence_dataframe):
    required_cols = {"date", "sentence_embeddings", "topic_probabilities", "sentence_text", "sentence_id"}
    missing = required_cols - set(filtered_sentence_dataframe.columns)
    if missing:
        raise ValueError(f"Filtered inference data missing required columns: {missing}")

    emb_dim_ok = filtered_sentence_dataframe["sentence_embeddings"].apply(lambda x: len(x) == config["embedding_dim"]).all()
    topic_dim_ok = filtered_sentence_dataframe["topic_probabilities"].apply(lambda x: len(x) == config["topic_dim"]).all()

    if not emb_dim_ok:
        raise ValueError("Inference sentence embedding dimension mismatch detected")
    if not topic_dim_ok:
        raise ValueError("Inference topic embedding dimension mismatch detected")

    return True


def save_training_artifacts(config, training_history, evaluation_metrics):
    payload = {
        "config": config,
        "training_history": training_history,
        "evaluation_metrics": evaluation_metrics,
        "artifacts": {
            "model_best_path": config.get("model_best_path"),
            "model_last_path": config.get("model_last_path"),
            "model_evaluated_path": config.get("model_evaluated_path"),
            "train_loss_plot_path": config.get("train_loss_plot_path"),
            "eval_heatmap_intra_path": config.get("eval_heatmap_intra_path"),
            "eval_heatmap_inter_path": config.get("eval_heatmap_inter_path"),
            "eval_metrics_path": config.get("eval_metrics_path"),
        },
    }
    save_path = config.get("run_summary_path", os.path.join(config["output_path"], "run_summary_new_2.json"))
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)
    return save_path


summary_path = save_training_artifacts(config, training_history, evaluation_metrics)
print(f"Saved summary: {summary_path}")

# CPU-safe inference embedding generation.
def generate_contextual_sbert_embeddings(sentence_dataframe, config, sbert_model_name="all-mpnet-base-v2"):
    from sentence_transformers import SentenceTransformer

    if sentence_dataframe.empty:
        sentence_dataframe["sentence_embeddings"] = []
        return sentence_dataframe

    if int(config["embedding_dim"]) != 768:
        raise ValueError("Inference requires config['embedding_dim'] == 768 to match trained Approach-2 pipeline")

    model_sbert = SentenceTransformer(sbert_model_name, device="cpu")
    encoded = model_sbert.encode(
        sentence_dataframe["context_text"].tolist(),
        batch_size=int(config["inference_batch_size"]),
        show_progress_bar=False,
        convert_to_numpy=True
    )
    encoded = np.asarray(encoded, dtype=np.float32)

    if encoded.shape[1] != config["embedding_dim"]:
        raise ValueError(
            f"SBERT output dim {encoded.shape[1]} does not match config['embedding_dim']={config['embedding_dim']}"
        )

    sentence_dataframe = sentence_dataframe.copy()
    sentence_dataframe["sentence_embeddings"] = [vec.astype(np.float32) for vec in encoded]
    return sentence_dataframe

# Enhanced user inference: sentence-level narrative shift is the final goal.
def get_user_inference_call_order():
    return [
        "1. split_articles_into_sentences",
        "2. build_context_texts",
        "3. generate_contextual_sbert_embeddings",
        "4. soft_topic_label_sentences",
        "5. filter_user_topic_sentences",
        "6. aggregate_daily_vectors -> add_temporal_features -> build_window_embeddings",
        "7. compute_topic_drift + detect_shifts (day level trigger)",
        "8. sentence-level shift detection with context (final output)"
    ]


def _normalize_topic_label(text):
    text = str(text).strip().lower()
    return "".join(ch for ch in text if ch.isalnum())


def resolve_topic_name(user_topic, config_topics):
    # Accept aliases such as "war topic" or "health-news" for canonical config topics.
    if user_topic in config_topics:
        return user_topic

    target = _normalize_topic_label(user_topic)
    scored = []
    for topic in config_topics:
        norm_topic = _normalize_topic_label(topic)
        if not norm_topic:
            continue
        if target == norm_topic:
            return topic
        if target in norm_topic or norm_topic in target:
            scored.append((len(norm_topic), topic))

    if scored:
        scored.sort(reverse=True)
        return scored[0][1]

    raise ValueError(
        f"Unsupported topic '{user_topic}'. Choose one of: {config_topics}"
    )


def _to_int_article_id(article_id_value):
    text = str(article_id_value)
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else -1


def _build_sentence_context_string(row, source_df, context_window=2):
    article_rows = source_df[source_df["article_id"].astype(str) == str(row["article_id"])].copy()
    if article_rows.empty:
        return f">>> [{row['sentence_id']}] {row['sentence_text']}"

    article_rows = article_rows.sort_values("sentence_order").reset_index(drop=True)
    positions = article_rows.index[article_rows["sentence_id"].astype(str) == str(row["sentence_id"])].tolist()
    if not positions:
        return f">>> [{row['sentence_id']}] {row['sentence_text']}"

    pos = positions[0]
    start = max(0, pos - int(context_window))
    end = min(len(article_rows), pos + int(context_window) + 1)
    subset = article_rows.iloc[start:end]

    lines = []
    current_order = int(row["sentence_order"])
    for _, r in subset.iterrows():
        prefix = ">>> " if int(r["sentence_order"]) == current_order else "    "
        lines.append(f"{prefix}[{r['sentence_id']}] {r['sentence_text']}")
    return "\n".join(lines)


def extract_sentence_level_narrative_shifts(
    filtered_sentence_dataframe,
    drift_rows,
    config,
    top_k_shifts=5,
    per_date_sent_limit=40,
    context_window=2
):
    return extract_sentence_level_narrative_shifts_inference(
        filtered_sentence_dataframe=filtered_sentence_dataframe,
        drift_rows=drift_rows,
        config=config,
        top_k_shifts=top_k_shifts,
        per_date_sent_limit=per_date_sent_limit,
        context_window=context_window,
    )


# Pipeline 2 override: keep same inference style, but build temporal windows from grouped vectors.
def run_user_level_inference(
    user_csv_path,
    model,
    config,
    topic_name,
    topic_embeddings_json_path,
    sbert_model_name="all-mpnet-base-v2"
):
    # Inference path is centralized in inference.py for AP2 parity.
    return run_user_level_inference_inference(
        user_csv_path=user_csv_path,
        model=model,
        config=config,
        topic_name=topic_name,
        topic_embeddings_json_path=topic_embeddings_json_path,
        sbert_model_name=sbert_model_name,
    )


def run_user_level_inference_approach2_compatible(
    user_csv_path,
    model,
    config,
    topic_name,
    topic_embeddings_json_path,
    sbert_model_name="all-mpnet-base-v2"
):
    return run_user_inference_approach2(
        user_csv_path=user_csv_path,
        topic_embeddings_json_path=topic_embeddings_json_path,
        topic_name=topic_name,
        config=config,
        model_variant="best",
        sbert_model_name=sbert_model_name
    )


# Backward alias kept for legacy notebook/script references.
def run_user_level_inference_approach1_compatible(
    user_csv_path,
    model,
    config,
    topic_name,
    topic_embeddings_json_path,
    sbert_model_name="all-mpnet-base-v2"
):
    return run_user_level_inference_approach2_compatible(
        user_csv_path=user_csv_path,
        model=model,
        config=config,
        topic_name=topic_name,
        topic_embeddings_json_path=topic_embeddings_json_path,
        sbert_model_name=sbert_model_name
    )

def main():
    # Batch AP2 inference entrypoint (delegates to inference.py).
    input_directory = "/home/hp/SEM2/INLP/Naretve_Shift/Output/Model_Testing/Aprroch_2"
    output_directory = "/home/hp/SEM2/INLP/Naretve_Shift/Output/Model_Testing/Aprroch_2/outputs"
    topic_embeddings_json_path = "/home/hp/SEM2/INLP/Naretve_Shift/Processed_Data/topic_embeddings.json"
    selected_topics = config["topics"]

    all_batch_inference_outputs = run_batch_inference_approach2(
        config=config,
        input_directory=input_directory,
        output_directory=output_directory,
        topic_embeddings_json_path=topic_embeddings_json_path,
        selected_topics=selected_topics,
        inference_overrides={
            "topic_threshold": 0.2,
            "zscore_threshold": 0.5,
            "percentile_threshold": 10,
            "drift_smoothing_window": 1,
        },
        model_variant="best",
        sbert_model_name="all-mpnet-base-v2",
    )

    print_batch_inference_outputs(all_batch_inference_outputs)
    return all_batch_inference_outputs


if __name__ == "__main__":
    main()

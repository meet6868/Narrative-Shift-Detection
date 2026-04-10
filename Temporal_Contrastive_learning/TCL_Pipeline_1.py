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
from config import load_approach_1_config
from dataloader import TemporalWindowDataset

from load_csv import (
    load_topic_csv as shared_load_topic_csv,
)
from inference import print_batch_inference_outputs, run_batch_inference_approach1, run_user_inference_approach1
from losses import EnhancedNTXentLossA12
from models import TCLTemporalEncoderA12
from evaluation import evaluate_model_quality, plot_evaluation_heatmaps
from plotting import plot_training_loss
from temporal_feature import aggregate_daily_embeddings, build_temporal_feature_records
from training import train_tcl_model_a12
from utils import (
    build_context_texts,
    build_topic_score_rows,
    extract_sentence_level_narrative_shifts,
    filter_user_topic_sentences,
    get_user_inference_call_order,
    load_topic_embedding_prototypes,
    resolve_topic_name,
    soft_topic_label_sentences,
    split_articles_into_sentences,
    validate_inference_alignment,
)
from windowing import build_window_embeddings

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
config = load_approach_1_config()


# Stage A: Load topic-wise sentence data prepared by shared preprocessing.
topic_sentence_data = {}
for topic_name in config["topics"]:
    topic_sentence_data[topic_name] = shared_load_topic_csv(topic_name, config, mode="ap1")
    print(f"Loaded {topic_name}: {len(topic_sentence_data[topic_name])} rows")

# Stage B: Build day-level representations, then temporal windows per topic.
topic_daily_data = {}
topic_window_data = {}
all_window_embeddings = []

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
    topic_daily_data[topic_name] = daily_dataframe

    enhanced_records = build_temporal_feature_records(
        daily_dataframe,
        include_tau=True,
        tau_scale=5.0,
        include_end_date=False,
        include_num_sentences=True,
        include_num_days=False,
    )
    topic_id = config["topics"].index(topic_name)
    window_embeddings = build_window_embeddings(enhanced_records, topic_name, topic_id, config)

    topic_window_data[topic_name] = window_embeddings
    all_window_embeddings.extend(window_embeddings)

print(f"Total window_embeddings: {len(all_window_embeddings)}")


# Stage C: Build training dataset + model + loss + optimizer.
train_dataset = TemporalWindowDataset(all_window_embeddings, config["topics"])
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True)

model = TCLTemporalEncoderA12(config).to(device)
loss_fn = EnhancedNTXentLossA12(config["temperature"])
optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

print(f"Dataset windows: {len(train_dataset)}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Stage D: Train temporal encoder with contrastive objective.
model, training_history = train_tcl_model_a12(model, train_dataset, train_loader, optimizer, loss_fn, config, device)
plot_training_loss(training_history, config["train_loss_plot_path"])
print(f"Saved best model: {config['model_best_path']}")
print(f"Saved last model: {config['model_last_path']}")
print(f"Saved train loss plot: {config['train_loss_plot_path']}")


# Stage E: Evaluate representation quality and persist artifacts.
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
    model.eval()
    window_embeddings = []

    with torch.no_grad():
        for window in topic_windows:
            tensor = torch.from_numpy(window["tensor"]).unsqueeze(0).to(device)
            encoded = model(tensor).cpu().numpy()[0].astype(np.float32)
            window_embeddings.append(encoded)

    window_embeddings = np.asarray(window_embeddings, dtype=np.float32)
    if len(window_embeddings) < 2:
        return [], window_embeddings

    raw_scores = []
    for i in range(1, len(window_embeddings)):
        raw_scores.append(np.float32(1.0 - float(np.dot(window_embeddings[i], window_embeddings[i - 1]))))

    score_series = pd.Series(raw_scores, dtype=np.float32)
    smooth_scores = score_series.rolling(
        window=config["drift_smoothing_window"], center=True, min_periods=1
    ).mean().astype(np.float32).values

    mean_score = np.float32(np.mean(smooth_scores))
    std_score = np.float32(np.std(smooth_scores)) + np.float32(1e-8)
    z_scores = ((smooth_scores - mean_score) / std_score).astype(np.float32)

    drift_rows = []
    for i, (raw_score, smooth_score, z_score) in enumerate(zip(raw_scores, smooth_scores, z_scores)):
        drift_rows.append({
            "window_idx": int(i + 1),
            "date": topic_windows[i + 1]["start_date"],
            "raw_drift": float(raw_score),
            "drift_score": float(smooth_score),
            "z_score": float(z_score)
        })

    return drift_rows, window_embeddings


def detect_shifts(drift_rows, config):
    if not drift_rows:
        return []
    z_scores = np.asarray([row["z_score"] for row in drift_rows], dtype=np.float32)
    percentile_cutoff = np.percentile(z_scores, config["percentile_threshold"])

    shifts = []
    for row in drift_rows:
        if row["z_score"] > config["zscore_threshold"] or row["z_score"] > percentile_cutoff:
            shifts.append(row)
    return shifts


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
    save_path = config.get("run_summary_path", os.path.join(config["output_path"], "run_summary_new_1.json"))
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)
    return save_path


summary_path = save_training_artifacts(config, training_history, evaluation_metrics)
print(f"Saved summary: {summary_path}")

# Inference helper: CPU-safe SBERT embedding generation.
def generate_contextual_sbert_embeddings(sentence_dataframe, config, sbert_model_name="all-mpnet-base-v2"):
    from sentence_transformers import SentenceTransformer

    if sentence_dataframe.empty:
        sentence_dataframe["sentence_embeddings"] = []
        return sentence_dataframe

    if int(config["embedding_dim"]) != 768:
        raise ValueError("Inference requires config['embedding_dim'] == 768 to match trained Approach-1 pipeline")

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

def run_user_level_inference(
    user_csv_path,
    model,
    config,
    topic_name,
    topic_embeddings_json_path,
    sbert_model_name="all-mpnet-base-v2"
):
    # Follow Approach-1 inference sequence defined in TCL/docs/approach_1.md.
    # 1) split -> 2) context -> 3) SBERT -> 4) soft topic labeling ->
    # 5) topic filtering -> 6) daily temporal features -> 7) drift -> 8) sentence-level shifts.
    call_order = get_user_inference_call_order()
    resolved_topic = resolve_topic_name(topic_name, config["topics"])

    input_dataframe = pd.read_csv(user_csv_path)
    sentence_dataframe = split_articles_into_sentences(input_dataframe)
    if sentence_dataframe.empty:
        return {
            "call_order": call_order,
            "resolved_topic": resolved_topic,
            "sentence_level_narrative_shifts": [],
            "top_topic_sentences": [],
            "topic_score_rows": [],
            "training_like_rows": []
        }

    sentence_dataframe = build_context_texts(sentence_dataframe, int(config["context_window"]))
    sentence_dataframe = generate_contextual_sbert_embeddings(
        sentence_dataframe, config, sbert_model_name=sbert_model_name
    )

    topic_embeddings = load_topic_embedding_prototypes(topic_embeddings_json_path, config)
    labeled_sentence_dataframe = soft_topic_label_sentences(sentence_dataframe, topic_embeddings, config)
    topic_score_rows = build_topic_score_rows(labeled_sentence_dataframe, config)

    try:
        filtered_sentence_dataframe = filter_user_topic_sentences(labeled_sentence_dataframe, resolved_topic, config)
    except KeyError:
        # Fallback if helper was edited and raises on topic lookup.
        if resolved_topic not in labeled_sentence_dataframe.columns:
            raise
        threshold = float(config.get("topic_threshold", 0.0))
        filtered_sentence_dataframe = labeled_sentence_dataframe[
            labeled_sentence_dataframe[resolved_topic].astype(np.float32) >= threshold
        ].copy()
        filtered_sentence_dataframe["selected_topic"] = resolved_topic
        filtered_sentence_dataframe["similarity_score"] = filtered_sentence_dataframe[resolved_topic].astype(np.float32)

    if filtered_sentence_dataframe.empty:
        return {
            "call_order": call_order,
            "resolved_topic": resolved_topic,
            "sentence_level_narrative_shifts": [],
            "top_topic_sentences": [],
            "topic_score_rows": topic_score_rows,
            "training_like_rows": labeled_sentence_dataframe.to_dict(orient="records")
        }

    validate_inference_alignment(config, filtered_sentence_dataframe)

    base_cols = [
        "date", "sentence_embeddings", "topic_probabilities", "sentence_text", "sentence_id", "similarity_score"
    ] + config["topics"]
    training_aligned_input = filtered_sentence_dataframe[base_cols].rename(columns={
        # aggregate_daily_embeddings expects `embedding` column in training/inference parity.
        "sentence_embeddings": "embedding",
        "sentence_text": "main_sentence",
        "topic_probabilities": "topic_embeddings",
        "similarity_score": "weight"
    })

    user_daily_df = aggregate_daily_embeddings(
        dataframe=training_aligned_input,
        topics=[resolved_topic],
        min_sentences_per_day=config["min_sentences_per_day"],
        embedding_column="embedding",
        weight_column_map={resolved_topic: [resolved_topic]},
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
    if user_daily_df.empty:
        return {
            "call_order": call_order,
            "resolved_topic": resolved_topic,
            "sentence_level_narrative_shifts": [],
            "top_topic_sentences": [],
            "topic_score_rows": topic_score_rows,
            "training_like_rows": labeled_sentence_dataframe.to_dict(orient="records")
        }

    user_records = build_temporal_feature_records(
        user_daily_df,
        include_tau=True,
        tau_scale=5.0,
        include_end_date=False,
        include_num_sentences=True,
        include_num_days=False,
    )
    user_windows = build_window_embeddings(
        user_records, resolved_topic, config["topics"].index(resolved_topic), config
    )

    if len(user_windows) < 2:
        return {
            "call_order": call_order,
            "resolved_topic": resolved_topic,
            "sentence_level_narrative_shifts": [],
            "top_topic_sentences": filtered_sentence_dataframe
                .sort_values("similarity_score", ascending=False)
                .head(20)[["date", "sentence_id", "sentence_text", "similarity_score"]]
                .to_dict(orient="records"),
            "topic_score_rows": topic_score_rows,
            "training_like_rows": labeled_sentence_dataframe.to_dict(orient="records")
        }

    # Use model device directly to avoid accidental CPU/GPU mismatch at inference time.
    model_device = next(model.parameters()).device
    drift_rows, _ = compute_topic_drift(model, user_windows, config, model_device)

    # Final target output: sentence-level narrative shift details.
    sentence_level_shifts = extract_sentence_level_narrative_shifts(
        filtered_sentence_dataframe=filtered_sentence_dataframe,
        drift_rows=drift_rows,
        config=config,
        detect_shifts_fn=detect_shifts,
        top_k_shifts=20,
        per_date_sent_limit=40,
        context_window=2
    )

    top_topic_sentences = (
        filtered_sentence_dataframe
        .sort_values("similarity_score", ascending=False)
        .head(20)[["date", "sentence_id", "sentence_text", "similarity_score"]]
        .to_dict(orient="records")
    )

    return {
        "call_order": call_order,
        "resolved_topic": resolved_topic,
        "sentence_level_narrative_shifts": sentence_level_shifts,
        "top_topic_sentences": top_topic_sentences,
        "topic_score_rows": topic_score_rows,
        "training_like_rows": labeled_sentence_dataframe.to_dict(orient="records")
    }


def run_user_level_inference_approach1_compatible(
    user_csv_path,
    model,
    config,
    topic_name,
    topic_embeddings_json_path,
    sbert_model_name="all-mpnet-base-v2"
):
    return run_user_inference_approach1(
        user_csv_path=user_csv_path,
        topic_embeddings_json_path=topic_embeddings_json_path,
        topic_name=topic_name,
        config=config,
        model_variant="best",
        sbert_model_name=sbert_model_name
    )

def main():
    # Batch user inference entrypoint (delegates to centralized inference.py wrapper).
    input_directory = "/home/hp/SEM2/INLP/Naretve_Shift/Output/Model_Testing/Aprroch_1"
    output_directory = "/home/hp/SEM2/INLP/Naretve_Shift/Output/Model_Testing/Aprroch_1/outputs"
    topic_embeddings_json_path = "/home/hp/SEM2/INLP/Naretve_Shift/Processed_Data/topic_embeddings.json"
    selected_topics = config["topics"]

    all_batch_inference_outputs = run_batch_inference_approach1(
        config=config,
        input_directory=input_directory,
        output_directory=output_directory,
        topic_embeddings_json_path=topic_embeddings_json_path,
        selected_topics=selected_topics,
        inference_overrides={
            "topic_threshold": 0.2,
            "zscore_threshold": 0.2,
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

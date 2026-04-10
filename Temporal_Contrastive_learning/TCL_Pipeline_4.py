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
from config import build_artifact_paths, load_approach_4_config, load_checkpoint_compat
from dataloader import TemporalWindowDataset

from load_csv import (
    load_topic_csv as shared_load_topic_csv,
)
from grouping import create_grouped_vectors_from_daily_ap4
from losses import EnhancedNTXentLossA4
from models import TCLTemporalEncoderA4
from evaluation import evaluate_model_quality, plot_evaluation_heatmaps
from plotting import plot_training_loss
from temporal_feature import aggregate_daily_embeddings, build_temporal_feature_records
from training import train_tcl_model_a4
from windowing import build_window_embeddings
from inference import (
    compute_topic_drift_a4 as compute_topic_drift_inference,
    detect_shifts_a4 as detect_shifts_inference,
    extract_sentence_level_narrative_shifts_a4 as extract_sentence_level_narrative_shifts_inference,
    print_batch_inference_outputs,
    run_batch_inference_approach4,
    run_user_inference_approach4,
    run_user_level_inference_approach4 as run_user_level_inference_inference,
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
config = load_approach_4_config()


# Stage A: Load topic-wise sentence data for AP4.
topic_sentence_data = {}
for topic_name in config["topics"]:
    topic_sentence_data[topic_name] = shared_load_topic_csv(topic_name, config, mode="ap4")
    print(f"Loaded {topic_name}: {len(topic_sentence_data[topic_name])} rows")


topic_daily_data = {}
topic_group_data = {}
topic_window_data = {}
all_window_embeddings = []

# Stage B: Daily aggregation -> AP4 feature concat -> ruptures grouping -> windows.
for topic_name in config["topics"]:
    topic_index = config["topics"].index(topic_name)
    topic_embedding = np.asarray(config["topic_embedding_table"][topic_index], dtype=np.float32)

    daily_dataframe = aggregate_daily_embeddings(
        dataframe=topic_sentence_data[topic_name],
        topics=[topic_name],
        min_sentences_per_day=config["min_sentences_per_day"],
        embedding_column="embedding",
        weight_column_map={topic_name: ["topic_weight", topic_name]},
        topic_embeddings_column="topic_embeddings",
        fallback_topic_embeddings_map={topic_name: topic_embedding},
        normalize_date=True,
        require_weight_column=False,
        entity_signature_column=None,
        output_embedding_column="daily_vectors",
        topic_column_name="topic_name",
        include_topic_id=True,
        include_avg_weight=False,
    )

    if daily_dataframe.empty:
        daily_dataframe["feature"] = []
        daily_dataframe = daily_dataframe.reindex(columns=[
            "date", "daily_vectors", "topic_embeddings", "feature", "topic_name", "topic_id", "num_sentences"
        ])
    else:
        # AP4 process parity: daily feature used for ruptures is [daily_vectors + topic_embeddings].
        features = []
        for _, row in daily_dataframe.iterrows():
            feature = np.concatenate([
                np.asarray(row["daily_vectors"], dtype=np.float32),
                np.asarray(row["topic_embeddings"], dtype=np.float32),
            ], axis=0).astype(np.float32)
            feature = feature / (np.linalg.norm(feature) + 1e-8)
            features.append(feature)

        daily_dataframe = daily_dataframe.copy()
        daily_dataframe["feature"] = features
        daily_dataframe = daily_dataframe[["date", "daily_vectors", "topic_embeddings", "feature", "topic_name", "topic_id", "num_sentences"]]

    grouped_dataframe = create_grouped_vectors_from_daily_ap4(daily_dataframe, config)

    topic_daily_data[topic_name] = daily_dataframe
    topic_group_data[topic_name] = grouped_dataframe

    enhanced_records = build_temporal_feature_records(
        grouped_dataframe,
        include_tau=False,
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

class BalancedTopicBatchSampler:
    def __init__(self, dataset, batch_size, topics):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.topics = list(topics)

        self.topic_to_indices = {
            topic: list(range(len(dataset.window_embeddings))) for topic in []
        }
        self.topic_to_indices = {
            topic: [i for i, row in enumerate(dataset.window_embeddings) if row["topic_name"] == topic]
            for topic in self.topics
        }

        min_samples = min((len(v) for v in self.topic_to_indices.values()), default=0)
        if min_samples <= 0:
            raise ValueError("At least one topic has zero windows; cannot build balanced batches")

        ideal_per_topic = max(1, self.batch_size // max(len(self.topics), 1))
        self.samples_per_topic = min(ideal_per_topic, min_samples)
        self.actual_batch_size = self.samples_per_topic * len(self.topics)
        self.num_batches = max(1, min_samples // self.samples_per_topic)

    def __iter__(self):
        shuffled = {}
        for topic, idxs in self.topic_to_indices.items():
            perm = list(idxs)
            np.random.shuffle(perm)
            shuffled[topic] = perm

        pointers = {topic: 0 for topic in self.topics}

        for _ in range(self.num_batches):
            batch = []
            for topic in self.topics:
                arr = shuffled[topic]
                for _ in range(self.samples_per_topic):
                    if pointers[topic] >= len(arr):
                        np.random.shuffle(arr)
                        pointers[topic] = 0
                    batch.append(arr[pointers[topic]])
                    pointers[topic] += 1
            yield batch

    def __len__(self):
        return self.num_batches




# Stage C: Build balanced topic batches + model + AP4 multi-loss.
train_dataset = TemporalWindowDataset(all_window_embeddings, config["topics"])

batch_sampler = BalancedTopicBatchSampler(
    train_dataset,
    batch_size=config["batch_size"],
    topics=config["topics"],
)

train_loader = DataLoader(
    train_dataset,
    batch_sampler=batch_sampler,
    num_workers=0,
    pin_memory=True if device.type == "cuda" else False,
)

model = TCLTemporalEncoderA4(config).to(device)
loss_fn = EnhancedNTXentLossA4(
    temperature=config["temperature"],
    lambda_temporal=config["lambda_temporal"],
    lambda_topic_sep=config["lambda_topic_sep"],
    lambda_hard_neg=config["lambda_hard_neg"],
    topic_sep_margin=config["topic_sep_margin"],
    hard_neg_margin=config["hard_neg_margin"],
)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"],
)

topic_window_counts = {topic: len(train_dataset.topic_groups.get(topic, [])) for topic in config["topics"]}
print(f"Dataset windows: {len(train_dataset)}")
print(f"Window counts by topic: {topic_window_counts}")
print(f"Train loader batches/epoch: {len(train_loader)}")
print(f"Balanced batch size used: {getattr(batch_sampler, 'actual_batch_size', config['batch_size'])}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(
    f"Loss weights -> temporal={config['lambda_temporal']}, "
    f"topic_sep={config['lambda_topic_sep']}, hard_neg={config['lambda_hard_neg']}"
)

# Stage D: Train AP4 temporal encoder.
model, training_history = train_tcl_model_a4(model, train_loader, optimizer, loss_fn, config, device)
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
    save_path = config.get("run_summary_path", os.path.join(config["output_path"], "run_summary_new_4.json"))
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
        raise ValueError("Inference requires config['embedding_dim'] == 768 to match trained Approach-4 pipeline")

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
    rows = []
    topic_names = config["topics"]
    topic_matrix = np.stack([topic_embeddings[name] for name in topic_names]).astype(np.float32)
    topic_table_64 = np.asarray(config["topic_embedding_table"], dtype=np.float32)
    expected_shape = (len(topic_names), int(config["topic_embedding_dim"]))
    if topic_table_64.shape != expected_shape:
        raise ValueError(
            f"topic_embedding_table shape mismatch: {topic_table_64.shape} vs {expected_shape}"
        )

    if sentence_dataframe.empty:
        base_cols = [
            "date",
            "article_id",
            "sentence_id",
            "sentence_text",
            "sentence_order",
            "sentence_embeddings",
            "topic_embeddings",
            "topic_probabilities",
        ] + topic_names
        return pd.DataFrame(columns=base_cols)

    for row in sentence_dataframe.itertuples(index=False):
        emb = np.asarray(row.sentence_embeddings, dtype=np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-8)

        similarities = np.dot(topic_matrix, emb).astype(np.float32)
        exp_sim = np.exp(similarities - np.max(similarities)).astype(np.float32)
        topic_probs = (exp_sim / (exp_sim.sum() + 1e-8)).astype(np.float32)

        topic_vec_64 = np.dot(topic_probs, topic_table_64).astype(np.float32)
        topic_vec_64 = topic_vec_64 / (np.linalg.norm(topic_vec_64) + 1e-8)

        record = {
            "date": row.date,
            "article_id": row.article_id,
            "sentence_id": row.sentence_id,
            "sentence_text": row.sentence_text,
            "sentence_order": int(row.sentence_order),
            "sentence_embeddings": emb.astype(np.float32),
            "topic_embeddings": topic_vec_64.astype(np.float32),
            "w3_embedding": emb.astype(np.float32),
            "w5_embedding": emb.astype(np.float32),
            "topic_probabilities": topic_probs.astype(np.float32),
        }

        for topic_idx, topic_name in enumerate(topic_names):
            record[topic_name] = np.float32(topic_probs[topic_idx])

        rows.append(record)

    return pd.DataFrame(rows)


def build_topic_score_rows(labeled_sentence_dataframe, config):
    rows = []
    for row in labeled_sentence_dataframe.itertuples(index=False):
        for topic_name in config["topics"]:
            rows.append(
                {
                    "sentence_id": row.sentence_id,
                    "topic": topic_name,
                    "similarity_score": float(getattr(row, topic_name)),
                }
            )
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


def _get_expected_topic_prob_dim(config):
    return int(config.get("topic_prob_dim", len(config.get("topics", []))))


def _get_expected_topic_emb_dim(config):
    return int(config.get("topic_dim", config.get("topic_embedding_dim", 0)))


def validate_inference_alignment(config, filtered_sentence_dataframe):
    required_cols = {"date", "sentence_embeddings", "topic_embeddings", "topic_probabilities", "sentence_text", "sentence_id"}
    missing = required_cols - set(filtered_sentence_dataframe.columns)
    if missing:
        raise ValueError(f"Filtered inference data missing required columns: {missing}")

    emb_dim_ok = filtered_sentence_dataframe["sentence_embeddings"].apply(
        lambda x: len(x) == int(config["embedding_dim"])
    ).all()

    expected_topic_emb_dim = _get_expected_topic_emb_dim(config)
    topic_emb_dim_ok = filtered_sentence_dataframe["topic_embeddings"].apply(
        lambda x: len(x) == expected_topic_emb_dim
    ).all()

    expected_topic_prob_dim = _get_expected_topic_prob_dim(config)
    topic_prob_dim_ok = filtered_sentence_dataframe["topic_probabilities"].apply(
        lambda x: len(x) == expected_topic_prob_dim
    ).all()

    if not emb_dim_ok:
        raise ValueError("Inference sentence embedding dimension mismatch detected")
    if not topic_emb_dim_ok:
        raise ValueError(
            f"Inference topic embedding dimension mismatch detected: expected {expected_topic_emb_dim}"
        )
    if not topic_prob_dim_ok:
        raise ValueError(
            f"Inference topic probability dimension mismatch detected: expected {expected_topic_prob_dim}"
        )

    return True

def get_user_inference_call_order():
    return [
        "1. split_articles_into_sentences",
        "2. build_context_texts",
        "3. generate_contextual_sbert_embeddings",
        "4. compute topic weights using ideal 768-d embeddings",
        "5. filter by topic threshold",
        "6. daily weighted pooling + add 64-d TCL topic embedding",
        "7. build temporal windows (adaptive if days < window_size)",
        "8. compute_topic_drift + detect_shifts",
        "9. sentence-level shift detection with context",
    ]


def _normalize_topic_label(text):
    text = str(text).strip().lower()
    return "".join(ch for ch in text if ch.isalnum())


def resolve_topic_name(user_topic, config_topics):
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

    raise ValueError(f"Unsupported topic '{user_topic}'. Choose one of: {config_topics}")


def _to_int_article_id(article_id_value):
    text = str(article_id_value)
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else -1


def load_ideal_topic_embeddings_for_inference(topic_embeddings_json_path, config):
    topic_embeddings = load_topic_embedding_prototypes(topic_embeddings_json_path, config)
    for topic_name in config["topics"]:
        vec = np.asarray(topic_embeddings[topic_name], dtype=np.float32)
        if vec.shape[0] != int(config["embedding_dim"]):
            raise ValueError(
                f"Ideal topic embedding dim mismatch for {topic_name}: {vec.shape[0]} vs {config['embedding_dim']}"
            )
    return topic_embeddings


def load_tcl_topic_embeddings_for_inference(tcl_topic_embeddings_json_path, config):
    expected_dim = int(config["topic_embedding_dim"])

    if tcl_topic_embeddings_json_path and os.path.exists(tcl_topic_embeddings_json_path):
        with open(tcl_topic_embeddings_json_path, "r", encoding="utf-8") as file:
            payload = json.load(file)

        topic_embeddings = {}
        for topic_name in config["topics"]:
            if topic_name not in payload:
                raise KeyError(f"Topic '{topic_name}' missing in {tcl_topic_embeddings_json_path}")
            vector = np.asarray(payload[topic_name], dtype=np.float32)
            if vector.shape[0] != expected_dim:
                raise ValueError(
                    f"TCL topic embedding dim mismatch for {topic_name}: {vector.shape[0]} vs {expected_dim}"
                )
            topic_embeddings[topic_name] = vector
        return topic_embeddings

    # Fallback to notebook config table when a saved TCL embedding file is not provided.
    topic_table = np.asarray(config["topic_embedding_table"], dtype=np.float32)
    expected_shape = (len(config["topics"]), expected_dim)
    if topic_table.shape != expected_shape:
        raise ValueError(f"config['topic_embedding_table'] shape mismatch: {topic_table.shape} vs {expected_shape}")

    return {topic_name: topic_table[idx].copy() for idx, topic_name in enumerate(config["topics"])}


def compute_topic_similarity_with_embeddings(sentence_embeddings, topic_embedding):
    sentence_embeddings = np.asarray(sentence_embeddings, dtype=np.float32)
    topic_embedding = np.asarray(topic_embedding, dtype=np.float32)

    sentence_embeddings = sentence_embeddings / (np.linalg.norm(sentence_embeddings, axis=1, keepdims=True) + 1e-8)
    topic_embedding = topic_embedding / (np.linalg.norm(topic_embedding) + 1e-8)

    similarities = sentence_embeddings @ topic_embedding
    similarities = (similarities + 1.0) / 2.0
    similarities = np.maximum(similarities, 0.3).astype(np.float32)
    return similarities


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
    for _, record in subset.iterrows():
        prefix = ">>> " if int(record["sentence_order"]) == current_order else "    "
        lines.append(f"{prefix}[{record['sentence_id']}] {record['sentence_text']}")
    return "\n".join(lines)

def extract_sentence_level_narrative_shifts(
    filtered_sentence_dataframe,
    drift_rows,
    config,
    top_k_shifts=5,
    per_date_sent_limit=40,
    context_window=2,
):
    return extract_sentence_level_narrative_shifts_inference(
        filtered_sentence_dataframe=filtered_sentence_dataframe,
        drift_rows=drift_rows,
        config=config,
        top_k_shifts=top_k_shifts,
        per_date_sent_limit=per_date_sent_limit,
        context_window=context_window,
    )

# AP4-aligned user inference: ideal 768-d topic weighting + TCL 64-d topic feature concat.
def run_user_level_inference(
    user_csv_path,
    model,
    config,
    topic_name,
    ideal_topic_embeddings_json_path=None,
    topic_embeddings_json_path=None,
    tcl_topic_embeddings_json_path=None,
    sbert_model_name="all-mpnet-base-v2"
):
    # Inference path is centralized in inference.py for AP4 parity.
    return run_user_level_inference_inference(
        user_csv_path=user_csv_path,
        model=model,
        config=config,
        topic_name=topic_name,
        ideal_topic_embeddings_json_path=ideal_topic_embeddings_json_path,
        topic_embeddings_json_path=topic_embeddings_json_path,
        tcl_topic_embeddings_json_path=tcl_topic_embeddings_json_path,
        sbert_model_name=sbert_model_name,
    )


def run_user_level_inference_approach4_compatible(
    user_csv_path,
    model,
    config,
    topic_name,
    ideal_topic_embeddings_json_path=None,
    topic_embeddings_json_path=None,
    tcl_topic_embeddings_json_path=None,
    sbert_model_name="all-mpnet-base-v2"
):
    selected_topic_embeddings_path = ideal_topic_embeddings_json_path or topic_embeddings_json_path
    return run_user_inference_approach4(
        user_csv_path=user_csv_path,
        ideal_topic_embeddings_json_path=selected_topic_embeddings_path,
        topic_name=topic_name,
        config=config,
        tcl_topic_embeddings_json_path=tcl_topic_embeddings_json_path,
        model_variant="best",
        sbert_model_name=sbert_model_name,
    )

def main():
    # Batch AP4 inference entrypoint (delegates to inference.py).
    input_directory = "/home/hp/SEM2/INLP/Naretve_Shift/Output/Model_Testing/Aprroch_4"
    output_directory = "/home/hp/SEM2/INLP/Naretve_Shift/Output/Model_Testing/Aprroch_4/outputs"
    ideal_topic_embeddings_json_path = "/home/hp/SEM2/INLP/Naretve_Shift/Processed_Data/topic_embeddings.json"

    default_tcl_topic_embeddings_path = os.path.join(config["output_path"], "topic_embeddings.json")
    tcl_topic_embeddings_json_path = default_tcl_topic_embeddings_path if os.path.exists(default_tcl_topic_embeddings_path) else None

    selected_topics = config["topics"]

    all_batch_inference_outputs = run_batch_inference_approach4(
        config=config,
        input_directory=input_directory,
        output_directory=output_directory,
        ideal_topic_embeddings_json_path=ideal_topic_embeddings_json_path,
        tcl_topic_embeddings_json_path=tcl_topic_embeddings_json_path,
        selected_topics=selected_topics,
        inference_overrides={
            "topic_threshold": 0.60,
            "manual_shift_threshold": config.get("manual_shift_threshold", 0.1),
        },
        model_variant="best",
        sbert_model_name="all-mpnet-base-v2",
    )

    print_batch_inference_outputs(all_batch_inference_outputs)
    return all_batch_inference_outputs


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Interactive HF loader for TCL models.

Run with no args:
    python testing.py
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

from Temporal_Contrastive_learning.models import (
    TCLTemporalEncoderA12,
    TCLTemporalEncoderA4,
    TCLTemporalEncoderA5,
)


def load_dotenv_exports(dotenv_path: Path) -> None:
    """Load lines like `export KEY=value` from .env if present."""
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_token(token_env: str = "HF_TOKEN_READ") -> str:
    token = (os.getenv(token_env) or "").strip()
    if token:
        return token
    raise ValueError(
        f"Missing Hugging Face token. Set {token_env} in environment/.env."
    )


def choose_model_class(approach_id: str):
    mapping = {
        "1": TCLTemporalEncoderA12,
        "2": TCLTemporalEncoderA12,
        "4": TCLTemporalEncoderA4,
        "5": TCLTemporalEncoderA5,
    }
    if approach_id not in mapping:
        raise ValueError("Approach id must be one of: 1, 2, 4, 5")
    return mapping[approach_id]


def load_checkpoint_compat(path: str, map_location: torch.device):
    try:
        return torch.load(path, map_location=map_location)
    except Exception as exc:
        if "Weights only load failed" in str(exc):
            return torch.load(path, map_location=map_location, weights_only=False)
        raise


def download_config_and_checkpoint(
    repo_id: str,
    revision: str,
    approach_id: str,
    token: str,
) -> Tuple[Dict, str, str, str]:
    base_hf_name = f"base_approch_{approach_id}"
    config_filename = f"{base_hf_name}_config.json"
    checkpoint_filename = f"{base_hf_name}_best.pt"

    default_subfolder = f"approach_{approach_id}"
    env_subfolder = (os.getenv("HF_SUBFOLDER") or "").strip("/")
    candidate_subfolders: List[str] = []
    if env_subfolder:
        candidate_subfolders.append(env_subfolder)
    candidate_subfolders.append(default_subfolder)
    # Keep root fallback for older uploads that were not subfoldered.
    candidate_subfolders.append("")

    # Keep order while removing duplicates.
    deduped_subfolders: List[str] = []
    for item in candidate_subfolders:
        if item not in deduped_subfolders:
            deduped_subfolders.append(item)

    attempted_paths: List[str] = []
    local_config_path = None
    local_checkpoint_path = None
    config_file_in_repo = None
    checkpoint_file_in_repo = None

    for subfolder in deduped_subfolders:
        cfg_path = f"{subfolder}/{config_filename}" if subfolder else config_filename
        ckpt_path = f"{subfolder}/{checkpoint_filename}" if subfolder else checkpoint_filename
        attempted_paths.extend([cfg_path, ckpt_path])

        try:
            local_config_path = hf_hub_download(
                repo_id=repo_id,
                filename=cfg_path,
                revision=revision,
                token=token,
                force_download=True,
                local_files_only=False,
            )
            local_checkpoint_path = hf_hub_download(
                repo_id=repo_id,
                filename=ckpt_path,
                revision=revision,
                token=token,
                force_download=True,
                local_files_only=False,
            )
            config_file_in_repo = cfg_path
            checkpoint_file_in_repo = ckpt_path
            break
        except Exception:
            continue

    if not local_config_path or not local_checkpoint_path:
        raise FileNotFoundError(
            "Could not find HF artifacts. Tried: " + ", ".join(attempted_paths)
        )

    with open(local_config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    return config, local_config_path, local_checkpoint_path, checkpoint_file_in_repo


def build_and_load_model(config: Dict, checkpoint_path: str, approach_id: str, device: torch.device):
    model_class = choose_model_class(approach_id)
    model = model_class(config).to(device)

    checkpoint = load_checkpoint_compat(checkpoint_path, map_location=device)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def ask_approach_id() -> str:
    print("Select model to run (number only):")
    print("1. Approach 1")
    print("2. Approach 2")
    print("3. Approach 4")

    raw = input("Enter option: ").strip()
    option_to_approach = {
        "1": "1",
        "2": "2",
        "3": "4",
    }
    if raw not in option_to_approach:
        raise ValueError("Invalid option. Enter 1, 2, or 3.")
    return option_to_approach[raw]


def resolve_repo_id() -> str:
    repo_id = (os.getenv("HF_REPO_ID") or "meet5568/tcl-approach").strip()
    if not repo_id:
        raise ValueError("Missing HF_REPO_ID in environment/.env")
    return repo_id


# -----------------------------
# CSV -> sentence rows -> embedding -> soft labels
# -----------------------------
def load_user_csv(user_csv_path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(user_csv_path)
    required_cols = {"date", "article"}
    missing = required_cols - set(dataframe.columns)
    if missing:
        raise ValueError(f"Input CSV must contain columns: {required_cols}. Missing: {missing}")
    return dataframe


def split_articles_into_sentences(input_dataframe: pd.DataFrame) -> pd.DataFrame:
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
            sentence_rows.append(
                {
                    "date": date_value,
                    "article_id": str(article_id),
                    "sentence_id": sentence_id,
                    "sentence_text": sentence_text,
                    "sentence_order": int(sentence_order),
                }
            )

    sentence_dataframe = pd.DataFrame(sentence_rows)
    if sentence_dataframe.empty:
        return pd.DataFrame(
            columns=["date", "article_id", "sentence_id", "sentence_text", "sentence_order"]
        )
    return sentence_dataframe[["date", "article_id", "sentence_id", "sentence_text", "sentence_order"]]


def add_sbert_embedding_column(
    sentence_dataframe: pd.DataFrame,
    embedding_dim: int,
    sbert_model_name: str = "all-mpnet-base-v2",
) -> pd.DataFrame:
    from sentence_transformers import SentenceTransformer

    if sentence_dataframe.empty:
        sentence_dataframe = sentence_dataframe.copy()
        sentence_dataframe["embedding"] = []
        return sentence_dataframe

    model_sbert = SentenceTransformer(sbert_model_name, device="cpu")
    encoded = model_sbert.encode(
        sentence_dataframe["sentence_text"].tolist(),
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    encoded = np.asarray(encoded, dtype=np.float32)

    if encoded.shape[1] != int(embedding_dim):
        raise ValueError(
            f"SBERT output dim {encoded.shape[1]} does not match expected embedding_dim={embedding_dim}"
        )

    sentence_dataframe = sentence_dataframe.copy()
    vectors = [vec.astype(np.float32) for vec in encoded]
    # Keep both names so this frame is easy to inspect and compatible with notebook naming.
    sentence_dataframe["embedding"] = vectors
    return sentence_dataframe


def load_topic_embedding_prototypes(topic_embeddings_json_path: Path, config: Dict) -> Dict[str, np.ndarray]:
    with open(topic_embeddings_json_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    topic_embeddings: Dict[str, np.ndarray] = {}
    for topic_name in config["topics"]:
        if topic_name not in payload:
            raise KeyError(f"Topic '{topic_name}' missing in {topic_embeddings_json_path}")
        vector = np.asarray(payload[topic_name], dtype=np.float32)
        if vector.shape[0] != int(config["embedding_dim"]):
            raise ValueError(
                f"Topic embedding dim mismatch for {topic_name}: {vector.shape[0]} vs {config['embedding_dim']}"
            )
        topic_embeddings[topic_name] = vector / (np.linalg.norm(vector) + 1e-8)

    return topic_embeddings


def soft_label_sentences(sentence_dataframe: pd.DataFrame, topic_embeddings: Dict[str, np.ndarray], config: Dict) -> pd.DataFrame:
    topic_names = config["topics"]
    topic_matrix = np.stack([topic_embeddings[name] for name in topic_names]).astype(np.float32)

    if sentence_dataframe.empty:
        base_cols = [
            "date",
            "article_id",
            "sentence_id",
            "sentence_text",
            "sentence_order",
            "embedding",
            "topic_probabilities",
        ] + topic_names
        return pd.DataFrame(columns=base_cols)

    rows = []
    for row in sentence_dataframe.itertuples(index=False):
        emb = np.asarray(row.embedding, dtype=np.float32)
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
            "embedding": emb,
            "topic_probabilities": topic_probs,
        }

        for topic_idx, topic_name in enumerate(topic_names):
            record[topic_name] = np.float32(topic_probs[topic_idx])

        rows.append(record)

    return pd.DataFrame(rows)


def build_labeled_sentence_dataframe(
    user_csv_path: Path,
    topic_embeddings_json_path: Path,
    config: Dict,
    approach_id: str,
    sbert_model_name: str = "all-mpnet-base-v2",
) -> pd.DataFrame:
    # Approach id is passed explicitly for traceability/debug parity with model selection.
    if str(approach_id) not in {"1", "2", "4", "5"}:
        raise ValueError("approach_id must be one of: 1, 2, 4, 5")

    user_df = load_user_csv(user_csv_path)
    sentence_df = split_articles_into_sentences(user_df)
    sentence_df = add_sbert_embedding_column(
        sentence_dataframe=sentence_df,
        embedding_dim=int(config["embedding_dim"]),
        sbert_model_name=sbert_model_name,
    )
    topic_embeddings = load_topic_embedding_prototypes(topic_embeddings_json_path, config)
    labeled_df = soft_label_sentences(sentence_df, topic_embeddings, config)
    return labeled_df


# -----------------------------
# Topic-wise filtering -> daily aggregation from labeled dataframe
# -----------------------------
def aggregate_daily_vectors_for_topic(topic_dataframe: pd.DataFrame, topic_name: str, config: Dict) -> pd.DataFrame:
    temp_df = topic_dataframe.copy()
    temp_df["date_only"] = pd.to_datetime(temp_df["date"], errors="coerce").dt.date
    temp_df = temp_df.dropna(subset=["date_only"]).reset_index(drop=True)

    daily_vectors = []
    for date_only, group in temp_df.groupby("date_only"):
    
        sentence_embeddings = np.stack(group["embedding"].values).astype(np.float32)

        # Weight by the selected topic score when available, otherwise use uniform mean.
        if topic_name in group.columns:
            raw_weights = group[topic_name].astype(np.float32).values
        else:
            raw_weights = None

        if raw_weights is not None:
            raw_weights = np.clip(raw_weights, a_min=0.0, a_max=None)
            if raw_weights.sum() > 0:
                weights = raw_weights / raw_weights.sum()
            else:
                weights = np.ones(len(group), dtype=np.float32) / max(len(group), 1)
        else:
            weights = np.ones(len(group), dtype=np.float32) / max(len(group), 1)

        daily_embedding = (sentence_embeddings.T @ weights).astype(np.float32)
        norm = np.linalg.norm(daily_embedding)
        if norm > 1e-8:
            daily_embedding = daily_embedding / norm

        daily_vectors.append(
            {
                "date": pd.Timestamp(date_only),
                "daily_vectors": daily_embedding,
                "topic_name": topic_name,
                "topic_id": int(config["topics"].index(topic_name)),
                "num_sentences": int(len(group)),
            }
        )

    if not daily_vectors:
        return pd.DataFrame(columns=["date", "daily_vectors", "topic_name", "topic_id", "num_sentences"])

    return pd.DataFrame(daily_vectors).sort_values("date").reset_index(drop=True)


def extract_entities_batch(df: pd.DataFrame, batch_size: int = 256) -> pd.DataFrame:
    """Extract named entities from sentence text using spaCy and attach entity metadata columns."""
    try:
        import spacy
    except ImportError as exc:
        raise ImportError("spaCy is required for approach 5 NER. Install with: pip install spacy") from exc

    text_col = "main_sentence" if "main_sentence" in df.columns else "sentence_text"
    if text_col not in df.columns:
        raise ValueError("Expected 'main_sentence' or 'sentence_text' column for NER extraction")

    try:
        nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer", "textcat"])
    except Exception as exc:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is missing. Run: python -m spacy download en_core_web_sm"
        ) from exc

    print(f"Extracting entities (batch_size={batch_size})...")
    sentences = df[text_col].astype(str).tolist()
    all_entities: List[List[str]] = []
    entity_signatures: List[str] = []

    for i in tqdm(range(0, len(sentences), batch_size), desc="NER"):
        batch_sentences = sentences[i : i + batch_size]
        docs = list(nlp.pipe(batch_sentences, batch_size=batch_size))

        for doc in docs:
            entities = [ent.text.strip() for ent in doc.ents if ent.text and ent.text.strip()]
            all_entities.append(entities)
            if entities:
                normalized = sorted({e.lower() for e in entities})
                entity_signatures.append(" | ".join(normalized[:5]))
            else:
                entity_signatures.append("__NO_ENTITY__")

    out = df.copy()
    out["entities"] = all_entities
    out["entity_signature"] = entity_signatures
    return out


def compute_entity_embeddings(
    df: pd.DataFrame,
    embedding_dim: int,
    sbert_model_name: str = "all-mpnet-base-v2",
) -> pd.DataFrame:
    """Compute normalized entity embeddings by encoding joined entities per sentence."""
    from sentence_transformers import SentenceTransformer

    print("Computing entity embeddings...")
    sbert_model = SentenceTransformer(sbert_model_name, device="cpu")
    entity_embeddings: List[np.ndarray] = []

    for entities in tqdm(df["entities"], desc="Entity embeddings"):
        if len(entities) == 0:
            entity_embedding = np.zeros(int(embedding_dim), dtype=np.float32)
        else:
            entity_text = " ".join(entities)
            entity_embedding = sbert_model.encode(entity_text, convert_to_numpy=True).astype(np.float32)

        norm = np.linalg.norm(entity_embedding)
        if norm > 1e-6:
            entity_embedding = entity_embedding / norm

        entity_embeddings.append(entity_embedding.astype(np.float32))

    out = df.copy()
    out["entity_embedding"] = entity_embeddings
    return out


def aggregate_daily_vectors_for_topic_approach5(topic_dataframe: pd.DataFrame, topic_name: str, config: Dict) -> pd.DataFrame:
    """Approach 5 daily aggregation using weighted final embeddings and entity context metadata."""
    df = topic_dataframe.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    weight_col = topic_name

    day_records = []
    for date, group in tqdm(df.groupby("date"), desc="Day aggregation"):
        if len(group) < int(config["min_sentences_per_day"]):
            continue

        embeddings = np.stack(group["final_embedding"].values).astype(np.float32)
        entity_embs = np.stack(group["entity_embedding"].values).astype(np.float32)
        weights = group[weight_col].astype(np.float32).values

        weighted_sum = (embeddings.T * weights).T.sum(axis=0)
        entity_weighted_sum = (entity_embs.T * weights).T.sum(axis=0)
        weight_sum = weights.sum()

        if weight_sum > 0:
            day_embedding = weighted_sum / weight_sum
            day_entity_embedding = entity_weighted_sum / weight_sum
        else:
            day_embedding = embeddings.mean(axis=0)
            day_entity_embedding = entity_embs.mean(axis=0)

        norm = np.linalg.norm(day_embedding)
        if norm > 1e-6:
            day_embedding = day_embedding / norm

        ent_norm = np.linalg.norm(day_entity_embedding)
        if ent_norm > 1e-6:
            day_entity_embedding = day_entity_embedding / ent_norm

        entity_set = sorted(
            {str(x) for x in group.get("entity_signature", pd.Series([], dtype=str)).tolist() if str(x)}
        )
        entity_context = " ; ".join(entity_set[:10]) if entity_set else "__NO_ENTITY__"

        day_records.append(
            {
                "date": date,
                "daily_vectors": day_embedding.astype(np.float32),
                "entity_embedding": day_entity_embedding.astype(np.float32),
                "topic_name": topic_name,
                "topic_id": int(config["topics"].index(topic_name)),
                "num_sentences": int(len(group)),
                "avg_weight": float(weights.mean()) if len(weights) > 0 else 0.0,
                "entity_context": entity_context,
            }
        )

    if not day_records:
        return pd.DataFrame(
            columns=[
                "date",
                "daily_vectors",
                "entity_embedding",
                "topic_name",
                "topic_id",
                "num_sentences",
                "avg_weight",
                "entity_context",
            ]
        )

    return pd.DataFrame(day_records).sort_values("date").reset_index(drop=True)


def build_topic_embedding_table(cfg: Dict) -> np.ndarray:
    # AP4-style topic table initialization (64-d default unless config overrides).
    topic_embedding_dim = int(cfg.get("topic_embedding_dim", 64))
    rng_local = np.random.default_rng(int(cfg.get("seed", 42)))
    table = rng_local.standard_normal((len(cfg["topics"]), topic_embedding_dim)).astype(np.float32)
    table /= np.linalg.norm(table, axis=1, keepdims=True) + 1e-8
    return table


def build_topic_embedding_table_approach5(cfg: Dict) -> np.ndarray:
    """Approach 5 topic embedding initialization using nn.Embedding + Xavier."""
    topic_dim = int(cfg.get("topic_embedding_dim", cfg.get("TOPIC_EMB_DIM", 64)))
    emb_layer = nn.Embedding(len(cfg["topics"]), topic_dim)
    torch.manual_seed(int(cfg.get("seed", 42)))
    nn.init.xavier_uniform_(emb_layer.weight)
    return emb_layer.weight.detach().cpu().numpy().astype(np.float32)


def _parse_topic_table_payload(payload, config: Dict) -> np.ndarray:
    """Parse topic embedding table payload from JSON-compatible objects."""
    topics = list(config["topics"])

    if isinstance(payload, dict):
        if "topic_embedding_table" in payload:
            table = np.asarray(payload["topic_embedding_table"], dtype=np.float32)
        elif all(topic in payload for topic in topics):
            table = np.stack([np.asarray(payload[topic], dtype=np.float32) for topic in topics]).astype(np.float32)
        else:
            raise ValueError("JSON payload does not contain topic_embedding_table or topic-wise vectors")
    else:
        table = np.asarray(payload, dtype=np.float32)

    return table


def load_topic_embedding_table_approach5_from_hf(
    repo_id: str,
    revision: str,
    token: str,
    config: Dict,
    approach_id: str = "5",
) -> np.ndarray:
    """Load approach-5 topic embedding table from HF artifacts uploaded previously."""
    topic_dim = int(config.get("topic_embedding_dim", config.get("TOPIC_EMB_DIM", 64)))
    expected_shape = (len(config["topics"]), topic_dim)

    default_subfolder = f"approach_{approach_id}"
    env_subfolder = (os.getenv("HF_SUBFOLDER") or "").strip("/")
    candidate_subfolders: List[str] = []
    if env_subfolder:
        candidate_subfolders.append(env_subfolder)
    candidate_subfolders.append(default_subfolder)
    # Keep root fallback for older uploads that were not subfoldered.
    candidate_subfolders.append("")

    deduped_subfolders: List[str] = []
    for subfolder in candidate_subfolders:
        if subfolder not in deduped_subfolders:
            deduped_subfolders.append(subfolder)

    candidate_files = [
        f"base_approch_{approach_id}_topic_embedding_table.npy",
        f"base_approch_{approach_id}_topic_embedding_table.json",
        f"approch_{approach_id}_topic_embedding_table.npy",
        f"approch_{approach_id}_topic_embedding_table.json",
        "topic_embedding_table.npy",
        "topic_embedding_table.json",
    ]

    attempted_paths: List[str] = []
    last_error = None

    for subfolder in deduped_subfolders:
        for file_name in candidate_files:
            remote_path = f"{subfolder}/{file_name}" if subfolder else file_name
            attempted_paths.append(remote_path)
            try:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=remote_path,
                    revision=revision,
                    token=token,
                    force_download=True,
                    local_files_only=False,
                )

                if file_name.endswith(".npy"):
                    table = np.load(local_path).astype(np.float32)
                else:
                    with open(local_path, "r", encoding="utf-8") as file:
                        payload = json.load(file)
                    table = _parse_topic_table_payload(payload, config)

                if table.shape != expected_shape:
                    raise ValueError(
                        f"Approach-5 topic table shape mismatch: {table.shape} vs {expected_shape}"
                    )

                return table.astype(np.float32)
            except Exception as exc:
                last_error = exc
                continue

    raise FileNotFoundError(
        "Could not load approach-5 topic embedding table from HF. "
        f"Tried: {', '.join(attempted_paths)}"
    ) from last_error


def add_temporal_features(daily_dataframe: pd.DataFrame, config: Dict, approach_id: str) -> pd.DataFrame:
    rows = daily_dataframe.sort_values("date").to_dict(orient="records")
    records = []

    if str(approach_id) in {"1", "2"}:
        # Approach 1/2 use one-hot topic embedding in final_vector.
        for row in rows:
            topic_index = int(row["topic_id"])
            one_hot = np.eye(len(config["topics"]), dtype=np.float32)[topic_index]
            row["topic_embeddings"] = one_hot.copy()
        
        for index, row in enumerate(rows):
            if index == 0:
                tau = 0.0
            else:
                day_gap = (row["date"] - rows[index - 1]["date"]).days
                tau = np.log1p(day_gap) / 5.0

            final_vector = np.concatenate([row["daily_vectors"],np.array([tau], dtype=np.float32),row["topic_embeddings"],]).astype(np.float32)

            records.append(
                {
                    "date": row["date"],
                    "final_vector": final_vector,
                    "topic_name": row["topic_name"],
                    "topic_id": int(row["topic_id"]),
                    "num_sentences": int(row["num_sentences"]),
                }
            )

            
    elif str(approach_id) == "4":
        # Approach 4 uses topic embedding table and does not use tau.
        if "topic_embedding_table" not in config:
            raise ValueError(
                "Missing config['topic_embedding_table'] for approach 4. "
                "Provide it in config before calling add_temporal_features."
            )
        table = np.asarray(config["topic_embedding_table"], dtype=np.float32)
        for row in rows:
            topic_index = int(row["topic_id"])
            topic_vector = table[topic_index]
            row["topic_embeddings"] = topic_vector.copy()
            final_vector = np.concatenate(
                [
                    row["daily_vectors"],
                    row["topic_embeddings"],
                ]
            ).astype(np.float32)

            records.append(
                {
                    "date": row["date"],
                    "final_vector": final_vector,
                    "topic_name": row["topic_name"],
                    "topic_id": int(row["topic_id"]),
                    "num_sentences": int(row["num_sentences"]),
                }
            )

    elif str(approach_id) == "5":
        # Approach 5: final vector combines daily vector, entity embedding, and topic embedding.
        if "approach5_topic_embedding_table" not in config:
            raise ValueError(
                "Missing approach5_topic_embedding_table in config. "
                "Load it from HF before calling add_temporal_features for approach 5."
            )
        table = np.asarray(config["approach5_topic_embedding_table"], dtype=np.float32)
        for row in rows:
            topic_index = int(row["topic_id"])
            topic_vector = table[topic_index]
            row["topic_embeddings"] = topic_vector.copy()

            entity_embedding = np.asarray(
                row.get("entity_embedding", np.zeros(int(config.get("embedding_dim", 768)), dtype=np.float32)),
                dtype=np.float32,
            )
            final_vector = np.concatenate(
                [
                    row["daily_vectors"],
                    entity_embedding,
                    row["topic_embeddings"],
                ]
            ).astype(np.float32)

            records.append(
                {
                    "date": row["date"],
                    "final_vector": final_vector,
                    "topic_name": row["topic_name"],
                    "topic_id": int(row["topic_id"]),
                    "num_sentences": int(row["num_sentences"]),
                }
            )
    else:
        raise ValueError("add_temporal_features supports approach_id 1, 2, 4, or 5")

    

    if not records:
        return pd.DataFrame(columns=["date", "final_vector", "topic_name", "topic_id", "num_sentences"])

    return pd.DataFrame(records)


def build_window_embeddings(enhanced_records, topic_name: str, topic_id: int, config: Dict):
    # Ensure temporal consistency before creating windows.
    enhanced_records = sorted(enhanced_records, key=lambda x: x["date"])

    window_embeddings = []
    total_days = len(enhanced_records)

    for start in range(0, total_days - int(config["window_size"]) + 1, int(config["stride"])):
        chunk = enhanced_records[start : start + int(config["window_size"])]
        window_matrix = np.stack([item["final_vector"] for item in chunk]).astype(np.float32)
        window_embeddings.append(
            {
                "tensor": window_matrix,
                "topic_id": int(topic_id),
                "topic_name": topic_name,
                "start_date": chunk[0]["date"],
                "end_date": chunk[-1]["date"],
                "window_idx": int(start),
            }
        )

    return window_embeddings


def compute_topic_drift(model, topic_windows, config: Dict, device: torch.device):
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

    smooth_window = int(config.get("drift_smoothing_window", 1))
    score_series = pd.Series(raw_scores, dtype=np.float32)
    smooth_scores = score_series.rolling(
        window=smooth_window, center=True, min_periods=1
    ).mean().astype(np.float32).values

    mean_score = np.float32(np.mean(smooth_scores))
    std_score = np.float32(np.std(smooth_scores)) + np.float32(1e-8)
    z_scores = ((smooth_scores - mean_score) / std_score).astype(np.float32)

    drift_rows = []
    for i, (raw_score, smooth_score, z_score) in enumerate(zip(raw_scores, smooth_scores, z_scores)):
        drift_rows.append(
            {
                "window_idx": int(i + 1),
                "date": topic_windows[i + 1]["start_date"],
                "raw_drift": float(raw_score),
                "drift_score": float(smooth_score),
                "z_score": float(z_score),
            }
        )

    return drift_rows, window_embeddings


def detect_shifts(drift_rows: List[Dict], config: Dict) -> List[Dict]:
    if not drift_rows:
        return []

    z_scores = np.asarray([row["z_score"] for row in drift_rows], dtype=np.float32)
    percentile_threshold = float(config.get("percentile_threshold", 50))
    zscore_threshold = float(config.get("zscore_threshold", 1.0))
    percentile_cutoff = float(np.percentile(z_scores, percentile_threshold))

    shifts = []
    for row in drift_rows:
        if float(row["z_score"]) > zscore_threshold or float(row["z_score"]) > percentile_cutoff:
            shifts.append(row)
    return shifts


def _to_int_article_id(article_id_value):
    text = str(article_id_value)
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else -1


def _build_sentence_context_string(row, source_df: pd.DataFrame, context_window: int = 2) -> str:
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
    for _, current in subset.iterrows():
        prefix = ">>> " if int(current["sentence_order"]) == current_order else "    "
        lines.append(f"{prefix}[{current['sentence_id']}] {current['sentence_text']}")
    return "\n".join(lines)


def extract_sentence_level_narrative_shifts(
    filtered_sentence_dataframe: pd.DataFrame,
    drift_rows: List[Dict],
    config: Dict,
    top_k_shifts: int = 5,
    per_date_sent_limit: int = 40,
    context_window: int = 2,
) -> List[Dict]:
    if filtered_sentence_dataframe.empty or not drift_rows:
        return []

    detected_shifts = detect_shifts(drift_rows, config)
    if not detected_shifts:
        return []

    filtered = filtered_sentence_dataframe.copy()
    filtered["date"] = pd.to_datetime(filtered["date"]).dt.normalize()

    unique_dates = sorted(filtered["date"].unique())
    if len(unique_dates) < 2:
        return []

    sentence_level_shifts = []
    ranked_shifts = sorted(
        detected_shifts,
        key=lambda item: item.get("z_score", 0.0),
        reverse=True,
    )[: int(top_k_shifts)]

    for shift in ranked_shifts:
        date_2 = pd.Timestamp(shift["date"]).normalize()
        previous_dates = [date for date in unique_dates if date < date_2]
        if not previous_dates:
            continue

        date_1 = previous_dates[-1]

        sents_1 = (
            filtered[filtered["date"] == date_1]
            .sort_values("similarity_score", ascending=False)
            .head(int(per_date_sent_limit))
            .reset_index(drop=True)
        )
        sents_2 = (
            filtered[filtered["date"] == date_2]
            .sort_values("similarity_score", ascending=False)
            .head(int(per_date_sent_limit))
            .reset_index(drop=True)
        )

        if sents_1.empty or sents_2.empty:
            continue

        embs_1 = np.stack(sents_1["sentence_embeddings"].values).astype(np.float32)
        embs_2 = np.stack(sents_2["sentence_embeddings"].values).astype(np.float32)

        norm_1 = embs_1 / (np.linalg.norm(embs_1, axis=1, keepdims=True) + 1e-8)
        norm_2 = embs_2 / (np.linalg.norm(embs_2, axis=1, keepdims=True) + 1e-8)
        sims = np.dot(norm_1, norm_2.T).astype(np.float32)

        min_idx = np.unravel_index(np.argmin(sims), sims.shape)
        min_similarity = float(sims[min_idx])

        sent1 = sents_1.iloc[min_idx[0]]
        sent2 = sents_2.iloc[min_idx[1]]

        sentence_level_shifts.append(
            {
                "date_1": str(pd.Timestamp(date_1).date()),
                "date_2": str(pd.Timestamp(date_2).date()),
                "sentence_id_1": str(sent1["sentence_id"]),
                "article_id_1": _to_int_article_id(sent1["article_id"]),
                "sentence_num_1": int(sent1["sentence_order"]),
                "sentence_1": str(sent1["sentence_text"]),
                "topic_weight_1": float(sent1["similarity_score"]),
                "sentence_id_2": str(sent2["sentence_id"]),
                "article_id_2": _to_int_article_id(sent2["article_id"]),
                "sentence_num_2": int(sent2["sentence_order"]),
                "sentence_2": str(sent2["sentence_text"]),
                "topic_weight_2": float(sent2["similarity_score"]),
                "context_1": _build_sentence_context_string(sent1, filtered, context_window=context_window),
                "context_2": _build_sentence_context_string(sent2, filtered, context_window=context_window),
                "similarity": min_similarity,
                "shift_score": float(1.0 - min_similarity),
                "day_level_shift_score": float(shift.get("drift_score", 0.0)),
                "day_level_z_score": float(shift.get("z_score", 0.0)),
            }
        )

    return sentence_level_shifts


def main() -> int:
    dotenv_path = Path(".env")
    revision = "main"
    device = torch.device("cpu")
    script_dir = Path(__file__).resolve().parent

    google_drive_folder_url = (
        os.getenv("GOOGLE_DRIVE_FOLDER_URL")
        or "https://drive.google.com/drive/folders/1wzDvbzYwMF9zgFpmQ_IvZpi-3_H0B-YA?usp=sharing"
    ).strip()
    topic_embeddings_json_path = None

    try:
        load_dotenv_exports(dotenv_path)
        token = resolve_token("HF_TOKEN_READ")
        repo_id = resolve_repo_id()
        approach_id = ask_approach_id()
        hf_subfolder = (os.getenv("HF_SUBFOLDER") or f"approach_{approach_id}").strip("/")
        os.environ["HF_SUBFOLDER"] = hf_subfolder

        config, local_config_path, local_checkpoint_path, checkpoint_file_in_repo = (
            download_config_and_checkpoint(
                repo_id=repo_id,
                revision=revision,
                approach_id=approach_id,
                token=token,
            )
        )

        model = build_and_load_model(config, local_checkpoint_path, approach_id, device)

        if str(approach_id) == "5":
            config["approach5_topic_embedding_table"] = load_topic_embedding_table_approach5_from_hf(
                repo_id=repo_id,
                revision=revision,
                token=token,
                config=config,
                approach_id=approach_id,
            )
            print(
                "Loaded approach-5 topic embedding table from HF: "
                f"{config['approach5_topic_embedding_table'].shape}"
            )

        print("HF load test passed")
        print(f"Repo: {repo_id}@{revision}")
        print(f"HF subfolder: {hf_subfolder}")
        print(f"Config (remote/local): {Path(local_config_path).name} | {local_config_path}")
        print(f"Checkpoint (remote/local): {checkpoint_file_in_repo} | {local_checkpoint_path}")
        print(f"Model class loaded for approach: {approach_id}")
        print(f"Device: {device}")
        print(f"Config keys: {len(config.keys())}")
        print("Input source: Google Drive folder only")
        print("Topic embeddings source: Google Drive input directory")

        with tempfile.TemporaryDirectory(prefix="tcl_drive_") as temp_dir:
            working_dir = Path(temp_dir)
            print(f"Temporary working directory: {working_dir}")

            if not google_drive_folder_url:
                raise ValueError("GOOGLE_DRIVE_FOLDER_URL is required. No fallback input source is allowed.")

            try:
                import gdown
            except ImportError as exc:
                raise ImportError(
                    "gdown is required for Drive folder download. Install with: pip install gdown"
                ) from exc

            print(f"Downloading Drive folder into temporary directory: {working_dir}")
            gdown.download_folder(
                url=google_drive_folder_url,
                output=str(working_dir),
                quiet=False,
                use_cookies=False,
                remaining_ok=True,
            )

            # Topic embeddings are expected in the same folder tree as downloaded Drive files.
            # Priority: explicit env path -> same-folder filename patterns.
            env_topic_json = (os.getenv("TOPIC_EMBEDDINGS_JSON") or "").strip()
            if env_topic_json:
                candidate_topic_path = Path(env_topic_json).expanduser()
                if candidate_topic_path.exists():
                    try:
                        candidate_topic_path.resolve().relative_to(working_dir.resolve())
                        topic_embeddings_json_path = candidate_topic_path
                    except Exception:
                        raise ValueError(
                            "TOPIC_EMBEDDINGS_JSON must point inside the active working directory tree."
                        )
                else:
                    print(f"WARNING: TOPIC_EMBEDDINGS_JSON not found: {candidate_topic_path}")

            same_folder_candidates = [
                working_dir / "topic_embeddings.json",
                working_dir / "topic_embedding.json",
                working_dir / "topic_prototypes.json",
            ]

            if not any(path.exists() for path in same_folder_candidates):
                json_candidates = sorted(working_dir.rglob("*.json"))
                preferred = [
                    path
                    for path in json_candidates
                    if "topic" in path.name.lower() and "embedding" in path.name.lower()
                ]
                if preferred:
                    same_folder_candidates.extend(preferred)
                else:
                    same_folder_candidates.extend(json_candidates)

            for candidate in same_folder_candidates:
                if candidate.exists():
                    topic_embeddings_json_path = candidate
                    break

            if topic_embeddings_json_path is None or not Path(topic_embeddings_json_path).exists():
                raise FileNotFoundError(
                    "Could not find topic embeddings JSON in working directory. "
                    "Set TOPIC_EMBEDDINGS_JSON to override."
                )

            print(f"Resolved topic embeddings json: {topic_embeddings_json_path}")

            csv_files = sorted(working_dir.rglob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in directory: {working_dir}")

            # Hardcoded max number of CSV files to process from the downloaded list.
            max_files_to_process = 1

            print(f"CSV files detected: {len(csv_files)}")
            print("CSV file list:")
            for idx, csv_path in enumerate(csv_files, 1):
                print(f"{idx:03d}. {csv_path}")

            selected_csv_files = csv_files[:max_files_to_process]
            print(
                f"Processing first {len(selected_csv_files)} file(s) only "
                f"(hardcoded max={max_files_to_process})"
            )

            for file_index, user_csv_file in enumerate(selected_csv_files, 1):
                print("\n" + "=" * 120)
                print(f"FILE {file_index}/{len(selected_csv_files)}: {user_csv_file.name}")
                print("=" * 120)

                labeled_sentences_df = build_labeled_sentence_dataframe(
                    user_csv_path=user_csv_file,
                    topic_embeddings_json_path=topic_embeddings_json_path,
                    config=config,
                    approach_id=approach_id,
                )
                print("Labeled DF columns:")
                print(list(labeled_sentences_df.columns))

                temporal_feature_frames: List[pd.DataFrame] = []
                topic_window_data: Dict[str, List[Dict]] = {}
                all_window_embeddings: List[Dict] = []
                drift_results_by_topic: Dict[str, Dict] = {}
                threshold = 0.25

                for topic_name in config["topics"]:

                    print("\n" + "=" * 60, topic_name, "=" * 60)
                    if topic_name not in labeled_sentences_df.columns:
                        continue

                    filtered_topic_df = labeled_sentences_df[
                        labeled_sentences_df[topic_name].astype(np.float32) >= threshold
                    ].copy()
                    filtered_topic_df = filtered_topic_df.sort_values(
                        ["date", "article_id", "sentence_order"]
                    ).reset_index(drop=True)

                    print(f"\nFiltered topic DF columns [{topic_name}]:", "Number of Row : ", len(filtered_topic_df))
                

                    filtered_sentence_dataframe = filtered_topic_df.copy()
                    filtered_sentence_dataframe["sentence_embeddings"] = filtered_sentence_dataframe["embedding"]
                    filtered_sentence_dataframe["similarity_score"] = filtered_sentence_dataframe[topic_name].astype(np.float32)

                    if str(approach_id) == "5":
                        approach5_df = filtered_topic_df.rename(columns={"sentence_text": "main_sentence"})
                        approach5_df = extract_entities_batch(approach5_df)
                        approach5_df = compute_entity_embeddings(
                            approach5_df,
                            embedding_dim=int(config.get("embedding_dim", 768)),
                        )
                        approach5_df["final_embedding"] = [
                            np.concatenate(
                                [
                                    np.asarray(sent_emb, dtype=np.float32),
                                    np.asarray(ent_emb, dtype=np.float32),
                                ],
                                axis=0,
                            ).astype(np.float32)
                            for sent_emb, ent_emb in zip(
                                approach5_df["embedding"],
                                approach5_df["entity_embedding"],
                            )
                        ]
                        topic_daily_df = aggregate_daily_vectors_for_topic_approach5(approach5_df, topic_name, config)
                    else:
                        topic_daily_df = aggregate_daily_vectors_for_topic(filtered_topic_df, topic_name, config)

                    print("TOpic_daily_df columns:", topic_daily_df.columns, f"rows: {len(topic_daily_df)}")
                    if not topic_daily_df.empty:
                        topic_temporal_df = add_temporal_features(
                            daily_dataframe=topic_daily_df,
                            config=config,
                            approach_id=approach_id,
                        )
                        if not topic_temporal_df.empty:
                            temporal_feature_frames.append(topic_temporal_df)
                            enhanced_records = topic_temporal_df.sort_values("date").to_dict(orient="records")
                            topic_id = int(config["topics"].index(topic_name))
                            topic_windows = build_window_embeddings(
                                enhanced_records=enhanced_records,
                                topic_name=topic_name,
                                topic_id=topic_id,
                                config=config,
                            )

                            print(f"Built {len(topic_windows)} windows for topic: {topic_name}")

                            drift_rows, _ = compute_topic_drift(model, topic_windows, config, device)
                            sentence_level_shifts = extract_sentence_level_narrative_shifts(
                                filtered_sentence_dataframe=filtered_sentence_dataframe,
                                drift_rows=drift_rows,
                                config=config,
                                top_k_shifts=20,
                                per_date_sent_limit=40,
                                context_window=2,
                            )
                            top_topic_sentences = (
                                filtered_sentence_dataframe
                                .sort_values("similarity_score", ascending=False)
                                .head(20)[["date", "sentence_id", "sentence_text", "similarity_score"]]
                                .to_dict(orient="records")
                            )

                            topic_window_data[topic_name] = topic_windows
                            all_window_embeddings.extend(topic_windows)
                            drift_results_by_topic[topic_name] = {
                                "drift_rows": drift_rows,
                                "shifts": detect_shifts(drift_rows, config),
                                "sentence_level_narrative_shifts": sentence_level_shifts,
                                "top_topic_sentences": top_topic_sentences,
                            }
                        else:
                            topic_window_data[topic_name] = []
                            drift_results_by_topic[topic_name] = {
                                "drift_rows": [],
                                "shifts": [],
                                "sentence_level_narrative_shifts": [],
                                "top_topic_sentences": [],
                            }
                    else:
                        topic_window_data[topic_name] = []
                        drift_results_by_topic[topic_name] = {
                            "drift_rows": [],
                            "shifts": [],
                            "sentence_level_narrative_shifts": [],
                            "top_topic_sentences": [],
                        }
                if temporal_feature_frames:
                    temporal_feature_df = pd.concat(temporal_feature_frames, ignore_index=True)
                    temporal_feature_df = temporal_feature_df.sort_values(["topic_name", "date"]).reset_index(drop=True)
                else:
                    temporal_feature_df = pd.DataFrame(
                        columns=["date", "final_vector", "topic_name", "topic_id", "num_sentences"]
                    )

                print(f"Labeled sentence rows: {len(labeled_sentences_df)}")
                preview_cols = ["date", "article_id", "sentence_id", "sentence_text", "War", "Health"]
                preview_cols = [col for col in preview_cols if col in labeled_sentences_df.columns]
                if preview_cols:
                    print("Labeled sentence preview:")
                    print(labeled_sentences_df[preview_cols].head(5).to_string(index=False))

                print(f"Temporal feature rows (all topics): {len(temporal_feature_df)}")
                if not temporal_feature_df.empty:
                    final_vector_dim = int(len(np.asarray(temporal_feature_df.iloc[0]["final_vector"])))
                    print(f"Final vector dimension: {final_vector_dim}")
                print(f"Total window embeddings (all topics): {len(all_window_embeddings)}")
                for topic_name in config["topics"]:
                    print(f"Window count [{topic_name}]: {len(topic_window_data.get(topic_name, []))}")

                # Console-only topic-wise report in requested format.
                output_file_path = "console_only"
                inference_results_by_topic = drift_results_by_topic

                topic_map = inference_results_by_topic
                print("")
                print("\n" + "=" * 100)
                print(f"FILE {file_index}: {os.path.basename(str(user_csv_file))}")
                print(f"Input:  {str(user_csv_file)}")
                print(f"Output: {str(output_file_path)}")
                print("=" * 100)

                if not topic_map:
                    print("No topic results found for this file.")
                else:
                    print(f"Topics in result: {list(topic_map.keys())}")

                    for topic_name, result in topic_map.items():
                        shifts = result.get("sentence_level_narrative_shifts", [])
                        print("\n" + "#" * 100)
                        print(f"TOPIC: {topic_name} | Total sentence-level shifts: {len(shifts)}")
                        print("#" * 100)

                        if not shifts:
                            print("No sentence-level shifts detected. Try lowering topic/zscore thresholds.")
                            continue

                        for i, shift in enumerate(shifts, 1):
                            date_1 = shift.get("date_1")
                            date_2 = shift.get("date_2")
                            similarity = float(shift.get("similarity", 0.0))
                            shift_score = float(shift.get("shift_score", 0.0))
                            day_z = float(shift.get("day_level_z_score", 0.0))

                            sentence_id_1 = shift.get("sentence_id_1")
                            article_id_1 = shift.get("article_id_1")
                            sentence_num_1 = shift.get("sentence_num_1")
                            topic_weight_1 = float(shift.get("topic_weight_1", 0.0))
                            context_1 = shift.get("context_1", "")

                            sentence_id_2 = shift.get("sentence_id_2")
                            article_id_2 = shift.get("article_id_2")
                            sentence_num_2 = shift.get("sentence_num_2")
                            topic_weight_2 = float(shift.get("topic_weight_2", 0.0))
                            context_2 = shift.get("context_2", "")

                            print(f"\nShift #{i}: {date_1} -> {date_2}")
                            print(
                                f"similarity={similarity:.4f} | shift_score={shift_score:.4f} | day_z={day_z:.4f}"
                            )
                            print(
                                f"\nDay 1 - {sentence_id_1} (Article {article_id_1}, Sentence {sentence_num_1})"
                            )
                            print(f"topic_weight={topic_weight_1:.3f}")
                            print(context_1)
                            print(
                                f"\nDay 2 - {sentence_id_2} (Article {article_id_2}, Sentence {sentence_num_2})"
                            )
                            print(f"topic_weight={topic_weight_2:.3f}")
                            print(context_2)
        print("Temporary working directory deleted.")
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

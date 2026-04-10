from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np
import pandas as pd

from temporal_feature import add_topic_embeddings_for_topic


def _cfg_get(config, key: str, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def parse_embedding(embedding_value):
    if isinstance(embedding_value, np.ndarray):
        return embedding_value.astype(np.float32)
    if isinstance(embedding_value, list):
        return np.asarray(embedding_value, dtype=np.float32)
    if isinstance(embedding_value, str):
        cleaned = embedding_value.strip("[]\"'").replace("\n", " ").replace("\r", " ")
        if "," in cleaned:
            cleaned = cleaned.replace(",", " ")
        parsed = np.fromstring(cleaned, sep=" ", dtype=np.float32)
        if parsed.size > 0:
            return parsed
        return np.asarray(ast.literal_eval(embedding_value), dtype=np.float32)
    return np.asarray(embedding_value, dtype=np.float32)


def apply_with_optional_progress(series, func, desc=None):
    try:
        tqdm = __import__("tqdm.auto", fromlist=["tqdm"]).tqdm
        tqdm.pandas(desc=desc)
        return series.progress_apply(func)
    except Exception:
        return series.apply(func)


def _resolve_mode(config, mode: str | None) -> str:
    if mode is not None:
        return str(mode).lower()
    approach_id = str(_cfg_get(config, "approach_id", _cfg_get(config, "APPROACH_ID", "1")))
    return {"1": "ap1", "2": "ap2", "4": "ap4", "5": "ap5"}.get(approach_id, "ap1")


def load_topic_csv(topic_name: str, config, data_dir=None, mode: str | None = None, embedding_column: str | None = None) -> pd.DataFrame:
    mode = _resolve_mode(config, mode)

    topics = _cfg_get(config, "topics", _cfg_get(config, "TOPICS", []))
    topics = list(topics)
    if topic_name not in topics:
        raise ValueError(f"Unknown topic '{topic_name}' not present in config topics: {topics}")

    topic_files = _cfg_get(config, "topic_files", _cfg_get(config, "TOPIC_FILES", {}))
    topic_file = topic_files.get(topic_name, f"{topic_name}.csv")

    base_dir = data_dir if data_dir is not None else _cfg_get(config, "data_path", _cfg_get(config, "DATA_DIR", ""))
    file_path = (Path(base_dir) / topic_file) if isinstance(base_dir, Path) else os.path.join(str(base_dir), topic_file)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")

    dataframe = pd.read_csv(file_path)
    dataframe["date"] = pd.to_datetime(dataframe["date"], format="mixed", errors="coerce")
    dataframe = dataframe.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    src_embedding_col = embedding_column or _cfg_get(
        config,
        "embedding_column",
        _cfg_get(config, "EMBEDDING_COLUMN", "w5_embedding"),
    )
    if src_embedding_col not in dataframe.columns:
        fallback = next((c for c in ["embedding", "w5_embedding", "w3_embedding"] if c in dataframe.columns), None)
        if fallback is None:
            raise ValueError(f"No embedding column found in {file_path}. Expected: {src_embedding_col}")
        src_embedding_col = fallback

    dataframe["embedding"] = apply_with_optional_progress(
        dataframe[src_embedding_col], parse_embedding, desc=f"Parsing {topic_name} embeddings"
    )

    embedding_dim = int(_cfg_get(config, "embedding_dim", _cfg_get(config, "EMBEDDING_DIM", 768)))
    dataframe = dataframe[dataframe["embedding"].apply(len) == embedding_dim].reset_index(drop=True)

    required_topic_cols = [topic for topic in topics if topic in dataframe.columns]
    if len(required_topic_cols) != len(topics):
        missing = [topic for topic in topics if topic not in dataframe.columns]
        raise ValueError(f"Missing topic columns in {file_path}: {missing}")

    for topic in topics:
        dataframe[topic] = pd.to_numeric(dataframe[topic], errors="coerce").fillna(0.0).astype(np.float32)

    topic_values = dataframe[topics].astype(np.float32).values
    denom = topic_values.sum(axis=1, keepdims=True)
    denom = np.where(denom > 1e-8, denom, 1.0)
    dataframe[topics] = (topic_values / denom).astype(np.float32)

    threshold = float(
        _cfg_get(config, "topic_threshold", _cfg_get(config, "topic_weight_threshold", _cfg_get(config, "TOPIC_WEIGHT_THRESHOLD", 0.3)))
    )
    if threshold > 0.0:
        dataframe = dataframe[dataframe[topic_name] >= threshold].copy().reset_index(drop=True)

    dataframe = add_topic_embeddings_for_topic(dataframe, topic_name, config, mode=mode)
    dataframe["topic_weight"] = dataframe[topic_name].astype(np.float32)

    if "main_sentence" not in dataframe.columns:
        for fallback in ["sentence", "article", "content", "window_text"]:
            if fallback in dataframe.columns:
                dataframe["main_sentence"] = dataframe[fallback].astype(str)
                break
        if "main_sentence" not in dataframe.columns:
            dataframe["main_sentence"] = ""

    if "sentence_id" not in dataframe.columns:
        dataframe["sentence_id"] = [f"{topic_name}_s{i}" for i in range(len(dataframe))]

    base_cols = [
        "date",
        "embedding",
        "topic_embeddings",
        "topic_weight",
        "main_sentence",
        "sentence_id",
    ]
    return dataframe[base_cols + topics]


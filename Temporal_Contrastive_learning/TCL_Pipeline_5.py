
import ast
import json
import os
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from config import load_approach_5_config
from dataloader import TemporalWindowDataset
from grouping import detect_ruptures
from load_csv import load_topic_csv
from temporal_feature import (
    add_topic_embeddings_for_topic,
    aggregate_daily_embeddings,
    approach5_add_entity_embeddings,
    approach5_extract_entities_batch,
    build_temporal_feature_records,
    compute_entity_invariant_embeddings,
    load_entity_invariant_cache_csv,
    save_entity_invariant_cache_csv,
)
from windowing import build_window_embeddings, create_windows_from_dataframe
from inference import (
    compute_topic_drift_a5 as compute_topic_drift_inference,
    detect_shifts_a5 as detect_shifts_inference,
    extract_sentence_level_narrative_shifts_a5 as extract_sentence_level_narrative_shifts_inference,
    print_multitopic_inference_outputs_approach5,
    run_multitopic_inference_approach5_minimal,
    run_user_inference_approach5,
    run_user_level_inference_approach5 as run_user_level_inference_inference,
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)

config = load_approach_5_config()
topic_loaded_data = {}


# Stage A: Load topic data early and attach topic embeddings for AP5.
print("\n" + "=" * 80)
print("LOAD ALL TOPIC FILES (EARLY LOAD)")
print("=" * 80)
for topic_name in config.TOPICS:
    try:
        topic_df = load_topic_csv(topic_name, config.DATA_DIR)
        topic_df = add_topic_embeddings_for_topic(topic_df, topic_name, config, mode="ap5")
        topic_loaded_data[topic_name] = topic_df
        print(f"  {topic_name}: {len(topic_loaded_data[topic_name]):,} rows")
    except Exception as exc:
        print(f"  {topic_name}: failed -> {exc}")
print("=" * 80)
print(f"Loaded topics now: {list(topic_loaded_data.keys())}")

# Load spaCy model
print("Loading spaCy NER model...")
nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'lemmatizer'])
if 'sentencizer' not in nlp.pipe_names and 'senter' not in nlp.pipe_names and 'parser' not in nlp.pipe_names:
    nlp.add_pipe('sentencizer')
print(f"✅ spaCy model loaded | pipeline: {nlp.pipe_names}")

def extract_entities_batch(df, batch_size=256):
    print(f"Extracting entities (batch_size={batch_size})...")
    out_df = approach5_extract_entities_batch(df, nlp, batch_size=int(batch_size))
    all_entities = out_df["entities"].tolist()
    num_with_entities = sum(1 for e in all_entities if len(e) > 0)
    total_entities = sum(len(e) for e in all_entities)
    print("Entity extraction complete")
    print(f"  Sentences with entities: {num_with_entities}/{len(out_df)} ({num_with_entities / max(len(out_df), 1) * 100:.1f}%)")
    print(f"  Total entities: {total_entities:,}")
    print(f"  Avg entities/sentence: {total_entities / max(len(out_df), 1):.2f}")
    return out_df


print("✅ NER function defined")

# Stage B: NER pass on loaded topics.
if 'topic_loaded_data' not in globals() or not topic_loaded_data:
    raise RuntimeError("topic_loaded_data is empty. Run the CSV loading cell first.")

print("\n" + "=" * 80)
print("RUN NER ON LOADED TOPIC FILES (ONE BY ONE)")
print("=" * 80)
for topic_name in config.TOPICS:
    if topic_name not in topic_loaded_data:
        print(f"  {topic_name}: skipped (not loaded)")
        continue

    topic_df = topic_loaded_data[topic_name].copy()

    # Keep embedding column in consistent ndarray float32 format.
    if 'embedding' in topic_df.columns:
        topic_df['embedding'] = topic_df['embedding'].apply(lambda x: np.asarray(x, dtype=np.float32))

    topic_df = extract_entities_batch(topic_df, batch_size=config.NER_BATCH_SIZE)
    topic_loaded_data[topic_name] = topic_df

    print(f"  {topic_name}: updated with NER | rows={len(topic_df):,}")

print("=" * 80)
print("NER update complete for loaded topics")

def compute_entity_embeddings(df, sbert_model):
    print("Computing entity embeddings...")
    out_df = approach5_add_entity_embeddings(df, sbert_model, embedding_dim=config.EMBEDDING_DIM)
    print("Entity embeddings computed")
    return out_df


print("✅ Entity embedding function defined")

# Run entity embeddings immediately on loaded topic data (one topic at a time).
if 'topic_loaded_data' not in globals() or not topic_loaded_data:
    raise RuntimeError("topic_loaded_data is empty. Run CSV loading + NER cells first.")

# Section 2.4 can run before later stage cells, so initialize SBERT here if missing.
if 'sbert_model' not in globals():
    print(f"Loading SBERT model for Section 2.4: {config.SBERT_MODEL}")
    preferred_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sbert_device = preferred_device
    try:
        sbert_model = SentenceTransformer(config.SBERT_MODEL, device=preferred_device)
    except RuntimeError as exc:
        err = str(exc).lower()
        if preferred_device == 'cuda' and ('outofmemory' in err or 'cuda out of memory' in err):
            print("CUDA OOM for SBERT in Section 2.4. Falling back to CPU.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            sbert_device = 'cpu'
            sbert_model = SentenceTransformer(config.SBERT_MODEL, device='cpu')
        else:
            raise
else:
    sbert_device = 'cuda' if config.DEVICE.type == 'cuda' else 'cpu'

print(f"SBERT device for Section 2.4: {sbert_device}")

print("\n" + "=" * 80)
print("RUN ENTITY EMBEDDINGS ON LOADED TOPIC FILES (ONE BY ONE)")
print("=" * 80)
for topic_name in config.TOPICS:
    if topic_name not in topic_loaded_data:
        print(f"  {topic_name}: skipped (not loaded)")
        continue

    topic_df = topic_loaded_data[topic_name].copy()

    if 'entities' not in topic_df.columns:
        print(f"  {topic_name}: skipped (entities column missing, run NER first)")
        continue

    topic_df = compute_entity_embeddings(topic_df, sbert_model)
    topic_loaded_data[topic_name] = topic_df

    print(f"  {topic_name}: entity embeddings updated | rows={len(topic_df):,}")

print("=" * 80)
print("Entity embedding update complete for loaded topics")

print("✅ Entity-aware dual representation function defined")

# Cache setup for section 3.5
BASE_PROCESSED_DIR = Path("Processed_Data")
BASE_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ENTITY_INVARIANT_CACHE_DIR = BASE_PROCESSED_DIR / "Stage_3_5_Entity_Invariant"
ENTITY_INVARIANT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_NEW_35_COLUMNS = {"semantic_clean_embedding", "entity_small_embedding", "final_embedding"}


def _entity_cache_path_35(topic_name):
    safe_name = str(topic_name).strip().replace("/", "_").replace(" ", "_")
    return ENTITY_INVARIANT_CACHE_DIR / f"{safe_name}_entity_invariant.csv"


def _is_cache_35_compatible(current_df, cached_df):
    if cached_df is None or len(cached_df) == 0:
        return False, "cached df empty"

    missing_new_cols = [c for c in ALLOWED_NEW_35_COLUMNS if c not in cached_df.columns]
    if missing_new_cols:
        return False, f"cached df missing 3.5 cols: {missing_new_cols}"

    current_base_cols = [c for c in current_df.columns if c not in ALLOWED_NEW_35_COLUMNS]
    cached_base_cols = [c for c in cached_df.columns if c not in ALLOWED_NEW_35_COLUMNS]

    if current_base_cols != cached_base_cols:
        return False, "base columns mismatch"

    return True, "ok"


# Run Section 3.5 on loaded topic data, but load cache and skip compute when compatible.
if 'topic_loaded_data' not in globals() or not topic_loaded_data:
    raise RuntimeError("topic_loaded_data is empty. Run CSV loading + NER + 2.4 cells first.")

# Section 3.5 can run before later stage cells, so initialize entity_proj_layer if missing.
if 'entity_proj_layer' not in globals():
    print("Initializing entity projection layer for Section 2.5")
    entity_proj_layer = nn.Linear(config.EMBEDDING_DIM, config.ENTITY_PROJ_DIM).to(config.DEVICE)
    nn.init.xavier_uniform_(entity_proj_layer.weight)
    nn.init.zeros_(entity_proj_layer.bias)

# Probe projection device; fallback to CPU if CUDA projection fails.
entity_proj_device = config.DEVICE
if str(entity_proj_device).startswith('cuda'):
    try:
        with torch.no_grad():
            _probe = torch.zeros((1, config.EMBEDDING_DIM), dtype=torch.float32, device=entity_proj_device)
            _ = entity_proj_layer(_probe)
    except RuntimeError as exc:
        err = str(exc).lower()
        if 'cublas' in err or 'cuda' in err or 'out of memory' in err:
            print("CUDA projection failed in Section 2.5. Falling back to CPU for entity projection.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            entity_proj_layer = entity_proj_layer.to('cpu')
            entity_proj_device = torch.device('cpu')
        else:
            raise
else:
    entity_proj_device = torch.device('cpu')

print(f"Entity projection device for Section 2.5: {entity_proj_device}")
print(f"3.5 cache dir: {ENTITY_INVARIANT_CACHE_DIR.resolve()}")

print("\n" + "=" * 80)
print("RUN ENTITY-INVARIANT CLEANING ON LOADED TOPIC FILES (ONE BY ONE)")
print("=" * 80)
for topic_name in config.TOPICS:
    if topic_name not in topic_loaded_data:
        print(f"  {topic_name}: skipped (not loaded)")
        continue

    topic_df = topic_loaded_data[topic_name].copy()
    cache_file = _entity_cache_path_35(topic_name)

    # 1) Try cache first; if schema matches (only 3.5 cols are new), load and skip compute.
    if cache_file.exists():
        try:
            cached_df = load_entity_invariant_cache_csv(cache_file)
            ok, reason = _is_cache_35_compatible(topic_df, cached_df)
            if ok:
                topic_loaded_data[topic_name] = cached_df
                print(f"  {topic_name}: loaded cache and skipped 3.5 compute | rows={len(cached_df):,}")
                continue
            print(f"  {topic_name}: cache exists but incompatible ({reason}); running compute")
        except Exception as exc:
            print(f"  {topic_name}: cache read failed ({exc}); running compute")

    # 2) Otherwise proceed with current 3.5 computation.
    if 'entity_embedding' not in topic_df.columns:
        print(f"  {topic_name}: skipped (entity_embedding missing, run 2.4 first)")
        continue

    topic_df = compute_entity_invariant_embeddings(
        topic_df,
        entity_proj_layer=entity_proj_layer,
        device=entity_proj_device,
        lambda_=config.ENTITY_LAMBDA,
    )
    topic_loaded_data[topic_name] = topic_df

    # Save for next runs.
    try:
        save_entity_invariant_cache_csv(topic_df, cache_file)
        print(f"  {topic_name}: final_embedding updated + cached | rows={len(topic_df):,}")
    except Exception as exc:
        print(f"  {topic_name}: final_embedding updated (cache save failed: {exc}) | rows={len(topic_df):,}")

print("=" * 80)
print("Section 2.5 update complete for loaded topics")

# 3.5 cache manager: topic-wise save/load with schema guard.
# Default behavior: if cache exists, load it and do NOT overwrite.
# Optional update mode validates that only 3.5 columns are new.

BASE_PROCESSED_DIR = Path("Processed_Data")
BASE_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ENTITY_INVARIANT_CACHE_DIR = BASE_PROCESSED_DIR / "Stage_3_5_Entity_Invariant"
ENTITY_INVARIANT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_NEW_35_COLUMNS = {
    "semantic_clean_embedding",
    "entity_small_embedding",
    "final_embedding",
}

UPDATE_EXISTING_ENTITY_CACHE = False  # keep False unless you intentionally want overwrite


def _entity_cache_path(topic_name):
    safe_name = str(topic_name).strip().replace("/", "_").replace(" ", "_")
    return ENTITY_INVARIANT_CACHE_DIR / f"{safe_name}_entity_invariant.csv"


def _base_columns(df):
    return [col for col in df.columns if col not in ALLOWED_NEW_35_COLUMNS]


print("\n" + "=" * 80)
print("3.5 CACHE: topic-wise save/load with column consistency checks")
print("=" * 80)
print(f"Processed_Data dir: {BASE_PROCESSED_DIR.resolve()}")
print(f"Cache dir: {ENTITY_INVARIANT_CACHE_DIR.resolve()}")
print(f"Update existing cache: {UPDATE_EXISTING_ENTITY_CACHE}")

for topic_name in config.TOPICS:
    if topic_name not in topic_loaded_data:
        print(f"  {topic_name}: skipped (not in topic_loaded_data)")
        continue

    current_df = topic_loaded_data[topic_name].copy()
    cache_file = _entity_cache_path(topic_name)

    if cache_file.exists() and not UPDATE_EXISTING_ENTITY_CACHE:
        cached_df = load_entity_invariant_cache_csv(cache_file)
        topic_loaded_data[topic_name] = cached_df
        print(f"  {topic_name}: loaded cached 3.5 df | rows={len(cached_df):,}")
        continue

    if cache_file.exists() and UPDATE_EXISTING_ENTITY_CACHE:
        cached_df = load_entity_invariant_cache_csv(cache_file)

        current_base = _base_columns(current_df)
        cached_base = _base_columns(cached_df)

        # Guard: all non-3.5 columns should remain identical in name/order.
        if current_base != cached_base:
            print(f"  {topic_name}: cache NOT updated (base columns changed)")
            print(f"    current_base_cols={len(current_base)} | cached_base_cols={len(cached_base)}")
            continue

        current_allowed_present = [c for c in ALLOWED_NEW_35_COLUMNS if c in current_df.columns]
        if len(current_allowed_present) < 3:
            print(f"  {topic_name}: cache NOT updated (missing one or more 3.5 columns)")
            continue

        save_entity_invariant_cache_csv(current_df, cache_file)
        print(f"  {topic_name}: cache updated after schema check | rows={len(current_df):,}")
        continue

    # Cache does not exist: first-time save.
    save_entity_invariant_cache_csv(current_df, cache_file)
    print(f"  {topic_name}: cache created | rows={len(current_df):,}")

print("=" * 80)
print("3.5 cache step complete")

# Run Section 2.6 immediately on loaded topic data (one topic at a time).
if 'topic_loaded_data' not in globals() or not topic_loaded_data:
    raise RuntimeError("topic_loaded_data is empty. Run CSV loading + 2.5 cells first.")

topic_day_data = {}

print("\n" + "=" * 80)
print("RUN DAY-LEVEL AGGREGATION ON LOADED TOPIC FILES (ONE BY ONE)")
print("=" * 80)
for topic_name in config.TOPICS:
    if topic_name not in topic_loaded_data:
        print(f"  {topic_name}: skipped (not loaded)")
        continue

    topic_df = topic_loaded_data[topic_name].copy()

    if 'final_embedding' not in topic_df.columns:
        print(f"  {topic_name}: skipped (final_embedding missing, run 2.5 first)")
        continue

    print(f"Aggregating to day-level for {topic_name}...")
    day_df = aggregate_daily_embeddings(
        dataframe=topic_df,
        topics=[topic_name],
        embedding_column='final_embedding',
        min_sentences_per_day=1,
        weight_column_map={topic_name: [topic_name]},
        topic_embeddings_column='topic_embeddings',
        fallback_topic_embeddings_map=None,
        normalize_date=False,
        require_weight_column=False,
        entity_signature_column='entity_signature',
        output_embedding_column='embedding',
        topic_column_name=None,
        include_topic_id=False,
        include_avg_weight=True,
    )
    print(f"Aggregated to {len(day_df)} days")
    if not day_df.empty:
        print(f"  Date range: {day_df['date'].min()} to {day_df['date'].max()}")
        print(f"  Avg sentences/day: {day_df['num_sentences'].mean():.1f}")

    topic_day_data[topic_name] = day_df
    print(f"  {topic_name}: day-level df ready | days={len(day_df):,}")

print("=" * 80)
print("Section 2.6 update complete for loaded topics")

print("✅ Ruptures function defined")

# Run Section 3.1 immediately on day-level topic data (one topic at a time).
if 'topic_day_data' not in globals() or not topic_day_data:
    raise RuntimeError("topic_day_data is empty. Run Section 2.6 first.")

topic_group_data = {}

print("\n" + "=" * 80)
print("RUN RUPTURES GROUPING ON DAY-LEVEL TOPIC DFS (ONE BY ONE)")
print("=" * 80)
for topic_name in config.TOPICS:
    if topic_name not in topic_day_data:
        print(f"  {topic_name}: skipped (no day-level data)")
        continue

    day_df = topic_day_data[topic_name].copy()

    if 'embedding' not in day_df.columns or len(day_df) == 0:
        print(f"  {topic_name}: skipped (missing/empty day-level embeddings)")
        continue

    grouped_df = detect_ruptures(
        day_df,
        model=config.RUPTURE_MODEL,
        pen=config.RUPTURE_PEN,
        min_size=config.MIN_GROUP_SIZE,
    )
    topic_group_data[topic_name] = grouped_df
    print(f"  {topic_name}: grouped df ready | rows={len(grouped_df):,} | groups={grouped_df['group'].nunique()}")

print("=" * 80)
print("Section 3.1 update complete for topic-wise ruptures grouping")

def create_topic_mapping(topics):
    """
    Create topic to ID mapping.
    
    Args:
        topics: List of topic names
    
    Returns:
        topic_to_id: Dictionary mapping topic name to ID
        id_to_topic: Dictionary mapping ID to topic name
    """
    unique_topics = sorted(set(topics))
    topic_to_id = {topic: idx for idx, topic in enumerate(unique_topics)}
    id_to_topic = {idx: topic for topic, idx in topic_to_id.items()}
    
    return topic_to_id, id_to_topic

def add_topic_embeddings_to_groups(group_df, topic, topic_emb_layer, topic_to_id, device):
    """
    Add topic embeddings to group embeddings.
    
    This creates the concatenated embedding [group_emb | topic_emb].
    
    Args:
        group_df: DataFrame with 'embedding' column
        topic: Topic name
        topic_emb_layer: nn.Embedding layer
        topic_to_id: Mapping dict
        device: torch device
    
    Returns:
        group_df: DataFrame with 'concat_embedding' column
    """
    topic_id = topic_to_id[topic]
    topic_tensor = torch.tensor([topic_id]).to(device)
    
    with torch.no_grad():
        topic_emb = topic_emb_layer(topic_tensor).squeeze(0).cpu().numpy().astype(np.float32)
    
    concat_embeddings = []
    
    for group_emb in group_df['embedding']:
        concat_emb = np.concatenate([np.asarray(group_emb, dtype=np.float32), topic_emb], axis=0).astype(np.float32)
        concat_embeddings.append(concat_emb)
    
    group_df['concat_embedding'] = concat_embeddings
    
    print(f"  Added topic embeddings: {group_df['concat_embedding'].iloc[0].shape}")
    
    return group_df

print("✅ Topic embedding functions defined")

# Run Section 3.2 immediately on grouped topic data (one topic at a time).
if 'topic_group_data' not in globals() or not topic_group_data:
    raise RuntimeError("topic_group_data is empty. Run Section 3.1 first.")

# Create/reuse topic mapping.
topic_to_id = globals().get("topic_to_id")
if not topic_to_id:
    topic_to_id, id_to_topic = create_topic_mapping(config.TOPICS)
else:
    id_to_topic = {idx: topic for topic, idx in topic_to_id.items()}

# Create/reuse topic embedding layer.
if 'topic_emb_layer' not in globals():
    topic_emb_layer = nn.Embedding(len(topic_to_id), config.TOPIC_EMB_DIM)
    nn.init.xavier_uniform_(topic_emb_layer.weight)
    topic_emb_layer = topic_emb_layer.to(config.DEVICE)

topic_group_data_with_topic = {}

print("\n" + "=" * 80)
print("RUN TOPIC EMBEDDING CONCAT ON GROUPED TOPIC DFS (ONE BY ONE)")
print("=" * 80)
for topic_name in config.TOPICS:
    if topic_name not in topic_group_data:
        print(f"  {topic_name}: skipped (no grouped data)")
        continue

    grouped_df = topic_group_data[topic_name].copy()
    if 'embedding' not in grouped_df.columns or len(grouped_df) == 0:
        print(f"  {topic_name}: skipped (missing/empty embedding column)")
        continue

    grouped_df = add_topic_embeddings_to_groups(
        grouped_df,
        topic_name,
        topic_emb_layer,
        topic_to_id,
        config.DEVICE,
    )

    topic_group_data_with_topic[topic_name] = grouped_df
    topic_group_data[topic_name] = grouped_df
    print(f"  {topic_name}: concat_embedding ready | rows={len(grouped_df):,}")

print("=" * 80)
print("Section 3.2 update complete for topic-wise embedding concat")

def _jaccard_overlap_from_strings(left_text, right_text):
    left_set = {x.strip() for x in str(left_text).split(';') if x and x.strip()}
    right_set = {x.strip() for x in str(right_text).split(';') if x and x.strip()}
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


print("Window creation functions defined")

# Run Section 3.3 immediately on topic-group dfs (one topic at a time).
if 'topic_group_data' not in globals() or not topic_group_data:
    raise RuntimeError("topic_group_data is empty. Run Section 3.1/3.2 first.")

topic_window_data = {}

print("\n" + "=" * 80)
print("RUN WINDOW CREATION ON TOPIC GROUP DFS (ONE BY ONE)")
print("=" * 80)
for topic_name in config.TOPICS:
    if topic_name not in topic_group_data:
        print(f"  {topic_name}: skipped (no grouped data)")
        continue

    group_df = topic_group_data[topic_name].copy()
    if len(group_df) == 0:
        print(f"  {topic_name}: skipped (empty grouped df)")
        continue

    embedding_col = 'concat_embedding' if 'concat_embedding' in group_df.columns else 'embedding'
    expected_dim = config.CONCAT_DIM if embedding_col == 'concat_embedding' else None

    try:
        windows, window_metadata = create_windows_from_dataframe(
            group_df,
            embedding_col=embedding_col,
            window_size=config.WINDOW_SIZE,
            stride=config.WINDOW_STRIDE,
            expected_dim=expected_dim,
        )
    except Exception as exc:
        print(f"  {topic_name}: window creation failed -> {exc}")
        continue

    topic_window_data[topic_name] = {
        'windows': windows,
        'metadata': window_metadata,
    }
    print(f"  {topic_name}: windows ready | shape={windows.shape}")

print("=" * 80)
print("Section 3.3 update complete for topic-wise window creation")

def create_consecutive_pairs(windows, topics, metadata):
    """
    Create pairs of consecutive windows for temporal contrastive learning.

    Returns:
        paired_windows_current, paired_windows_next, paired_topics, pair_entity_overlap, pair_window_idx
    """
    pairs_current = []
    pairs_next = []
    pairs_topics = []
    pair_entity_overlap = []
    pair_window_idx = []

    for i in range(len(windows) - 1):
        if topics[i] != topics[i + 1]:
            continue

        pairs_current.append(windows[i])
        pairs_next.append(windows[i + 1])
        pairs_topics.append(topics[i])
        pair_window_idx.append(int(i))

        left_ctx = metadata[i].get('entity_context', '__NO_ENTITY__') if i < len(metadata) else '__NO_ENTITY__'
        right_ctx = metadata[i + 1].get('entity_context', '__NO_ENTITY__') if (i + 1) < len(metadata) else '__NO_ENTITY__'
        pair_entity_overlap.append(float(_jaccard_overlap_from_strings(left_ctx, right_ctx)))

    if len(pairs_current) == 0:
        print("No valid consecutive pairs found")
        return None, None, None, None, None

    paired_windows_current = np.stack(pairs_current)
    paired_windows_next = np.stack(pairs_next)

    print(f"Created {len(pairs_current)} consecutive window pairs")
    return paired_windows_current, paired_windows_next, pairs_topics, pair_entity_overlap, pair_window_idx


print("Consecutive pairs function defined")

# Run Section 3.4 immediately on topic windows (one topic at a time).
if 'topic_window_data' not in globals() or not topic_window_data:
    raise RuntimeError("topic_window_data is empty. Run Section 3.3 first.")

topic_pair_data = {}
all_pairs_current = []
all_pairs_next = []
all_pair_topics = []
all_pair_entity_overlap = []
all_pair_window_idx = []

print("\n" + "=" * 80)
print("RUN CONSECUTIVE PAIR CREATION ON TOPIC WINDOWS (ONE BY ONE)")
print("=" * 80)
for topic_name in config.TOPICS:
    if topic_name not in topic_window_data:
        print(f"  {topic_name}: skipped (no window data)")
        continue

    topic_windows = topic_window_data[topic_name].get('windows', None)
    topic_metadata = topic_window_data[topic_name].get('metadata', [])

    if topic_windows is None or len(topic_windows) < 2:
        print(f"  {topic_name}: skipped (need >=2 windows)")
        continue

    topic_labels = [topic_name] * len(topic_windows)

    w_cur, w_next, t_list, overlap, w_idx = create_consecutive_pairs(
        topic_windows,
        topic_labels,
        topic_metadata,
    )

    if w_cur is None:
        print(f"  {topic_name}: no valid consecutive pairs")
        continue

    topic_pair_data[topic_name] = {
        'windows_current': w_cur,
        'windows_next': w_next,
        'pair_topics': t_list,
        'pair_entity_overlap': overlap,
        'pair_window_idx': w_idx,
        # Backward-compatible aliases.
        'topics': t_list,
        'entity_overlap': overlap,
    }

    all_pairs_current.append(w_cur)
    all_pairs_next.append(w_next)
    all_pair_topics.extend(t_list)
    all_pair_entity_overlap.extend(overlap)
    all_pair_window_idx.extend(w_idx)

    print(f"  {topic_name}: pairs ready | pairs={len(t_list):,}")

# Backward-compatible merged outputs for downstream cells.
if len(all_pairs_current) > 0:
    windows_current = np.concatenate(all_pairs_current, axis=0)
    windows_next = np.concatenate(all_pairs_next, axis=0)
    pair_topics = all_pair_topics
    pair_entity_overlap = all_pair_entity_overlap
    pair_window_idx = np.asarray(all_pair_window_idx, dtype=np.int64)
    print(f"Merged pairs ready: {len(pair_topics):,}")
    print(f"  windows_current: {windows_current.shape}")
    print(f"  windows_next: {windows_next.shape}")
else:
    windows_current, windows_next, pair_topics, pair_entity_overlap, pair_window_idx = None, None, None, None, None
    print("No pairs generated across topics.")

print("=" * 80)
print("Section 3.4 update complete for topic-wise consecutive pair creation")

# Use precomputed consecutive pairs from Section 3.4 topic-wise processing
print("Preparing consecutive window pairs...")

pairs_ready = (
    'windows_current' in globals() and
    'windows_next' in globals() and
    'pair_topics' in globals() and
    'pair_entity_overlap' in globals() and
    'pair_window_idx' in globals() and
    windows_current is not None and
    windows_next is not None and
    pair_window_idx is not None and
    len(pair_topics) > 0
)

if not pairs_ready:
    print("Precomputed merged pairs not found. Rebuilding from topic_pair_data...")

    if 'topic_pair_data' not in globals() or not topic_pair_data:
        raise ValueError("No topic pair data available. Run Section 3.4 first.")

    all_pairs_current = []
    all_pairs_next = []
    all_pair_topics = []
    all_pair_entity_overlap = []
    all_pair_window_idx = []

    for topic_name, pair_data in topic_pair_data.items():
        w_cur = pair_data.get('windows_current')
        w_next = pair_data.get('windows_next')
        t_list = pair_data.get('pair_topics', pair_data.get('topics', []))
        overlap = pair_data.get('pair_entity_overlap', pair_data.get('entity_overlap', []))
        w_idx = pair_data.get('pair_window_idx', None)

        if w_cur is None or w_next is None or len(t_list) == 0:
            continue

        if w_idx is None:
            w_idx = list(range(len(t_list)))

        all_pairs_current.append(w_cur)
        all_pairs_next.append(w_next)
        all_pair_topics.extend(t_list)
        all_pair_entity_overlap.extend(overlap)
        all_pair_window_idx.extend(w_idx)

    if not all_pairs_current:
        raise ValueError("Failed to build any consecutive pairs from topic_pair_data")

    windows_current = np.concatenate(all_pairs_current, axis=0)
    windows_next = np.concatenate(all_pairs_next, axis=0)
    pair_topics = all_pair_topics
    pair_entity_overlap = all_pair_entity_overlap
    pair_window_idx = np.asarray(all_pair_window_idx, dtype=np.int64)

print("\nConsecutive pairs ready:")
print(f"  Current windows: {windows_current.shape}")
print(f"  Next windows: {windows_next.shape}")
print(f"  Pairs: {len(pair_topics):,}")
print(f"  Mean entity overlap: {float(np.mean(pair_entity_overlap)):.4f}")

print("Temporal pair dataset defined")

class BalancedTopicBatchSampler:
    """AP4-style balanced topic batch sampler."""

    def __init__(self, dataset, batch_size, topics):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.topics = list(topics)

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

    def __iter__(self):
        shuffled = {
            topic: np.random.permutation(indices).tolist()
            for topic, indices in self.topic_to_indices.items()
        }

        ptr = {topic: 0 for topic in self.topics}
        while True:
            batch = []
            for topic in self.topics:
                start = ptr[topic]
                end = start + self.samples_per_topic
                if end > len(shuffled[topic]):
                    return
                batch.extend(shuffled[topic][start:end])
                ptr[topic] = end

            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return min(len(v) // self.samples_per_topic for v in self.topic_to_indices.values())


print("Balanced topic batch sampler defined")

# Build train dataset/loader from true consecutive temporal pairs.
if (
    "windows_current" not in globals()
    or "windows_next" not in globals()
    or "pair_topics" not in globals()
    or windows_current is None
    or windows_next is None
    or len(pair_topics) == 0
):
    raise RuntimeError("Consecutive pair tensors are missing. Run Section 3.4 first.")

topic_id_map = globals().get("topic_to_id", {topic: idx for idx, topic in enumerate(config.TOPICS)})
pair_topic_ids = np.asarray([int(topic_id_map[t]) for t in pair_topics], dtype=np.int64)
pair_overlap_values = pair_entity_overlap if "pair_entity_overlap" in globals() else None
pair_window_indices = pair_window_idx if "pair_window_idx" in globals() and pair_window_idx is not None else None

train_dataset = TemporalWindowDataset(
    windows_current=windows_current,
    windows_next=windows_next,
    topic_ids=pair_topic_ids,
    topics=config.TOPICS,
    entity_overlap=pair_overlap_values,
    window_indices=pair_window_indices,
)

batch_sampler = BalancedTopicBatchSampler(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    topics=config.TOPICS,
)

train_loader = DataLoader(
    train_dataset,
    batch_sampler=batch_sampler,
    num_workers=0,
    pin_memory=True if config.DEVICE.type == "cuda" else False,
)

topic_pair_counts = {topic: len(train_dataset.topic_groups.get(topic, [])) for topic in config.TOPICS}

print("=" * 80)
print("STAGE 5 COMPLETE: Temporal pair Dataset and DataLoader created")
print("=" * 80)
print(f"Dataset pairs: {len(train_dataset):,}")
print(f"Pair counts by topic: {topic_pair_counts}")
print(f"Train loader batches/epoch: {len(train_loader)}")
print(f"Balanced batch size used: {getattr(batch_sampler, 'actual_batch_size', config.BATCH_SIZE)}")
print("=" * 80)

# Ensure shared components are available when this file runs outside notebook globals.
if "TCLTemporalEncoderA5" not in globals():
    from models import TCLTemporalEncoderA5
if "SharedMultiLoss" not in globals():
    from losses import MultiLoss as SharedMultiLoss
if "train_tcl_model_a5" not in globals():
    from training import train_tcl_model_a5
if "evaluate_model_quality" not in globals() or "plot_evaluation_heatmaps" not in globals():
    from evaluation import evaluate_model_quality, plot_evaluation_heatmaps
if "plot_ap5_training_dashboard" not in globals():
    from plotting import plot_ap5_training_dashboard

# Initialize model with AP4 class name
model = TCLTemporalEncoderA5(config).to(config.DEVICE)

# Model stats
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("="*80)
print("STAGE 6 COMPLETE: Model initialized")
print("="*80)
print(f"Model: TCLTemporalEncoder")
print(f"Input dim: {config.CONCAT_DIM}D")
print(f"Hidden dim: {config.HIDDEN_DIM}D")
print(f"Output dim: {config.OUTPUT_DIM}D")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Device: {config.DEVICE}")
print("="*80)

# Initialize loss function (AP4-style temporal + topic + hard-neg + entity)
criterion = SharedMultiLoss(
    lambda_temporal=config.LAMBDA_TEMPORAL,
    lambda_topic_sep=config.LAMBDA_TOPIC_SEP,
    lambda_hard_neg=config.LAMBDA_HARD_NEG,
    lambda_entity=config.LAMBDA_ENTITY,
    temperature=config.TEMPERATURE,
).to(config.DEVICE)

print("=" * 80)
print("STAGE 7 COMPLETE: Loss functions initialized")
print("=" * 80)
print(f"Temporal loss weight: {config.LAMBDA_TEMPORAL}")
print(f"Topic separation weight: {config.LAMBDA_TOPIC_SEP}")
print(f"Hard negative weight: {config.LAMBDA_HARD_NEG}")
print(f"Entity consistency weight: {config.LAMBDA_ENTITY}")
print(f"Temperature: {config.TEMPERATURE}")
print("=" * 80)

# EXECUTE: Train the model
print("\nStarting training...\n")


def _is_cuda_oom(error_obj):
    text = str(error_obj).lower()
    return "out of memory" in text and "cuda" in text


try:
    trained_model, training_history = train_tcl_model_a5(
        model=model,
        train_loader=train_loader,
        num_epochs=config.NUM_EPOCHS,
        config=config,
        criterion=criterion,
    )
except Exception as exc:
    if not (_is_cuda_oom(exc) and torch.cuda.is_available() and str(config.DEVICE).startswith("cuda")):
        raise

    print("\nCUDA OOM detected. Retrying training on CPU...")

    import gc

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    config.DEVICE = torch.device("cpu")
    model = model.to(config.DEVICE)
    criterion = criterion.to(config.DEVICE)

    trained_model, training_history = train_tcl_model_a5(
        model=model,
        train_loader=train_loader,
        num_epochs=config.NUM_EPOCHS,
        config=config,
        criterion=criterion,
    )

print("\n" + "=" * 80)
print("STAGE 8 COMPLETE: Training finished")
print("=" * 80)
print(f"Final loss: {training_history['loss'][-1]:.4f}")
print(f"Best loss: {min(training_history['loss']):.4f}")
print("=" * 80)

# Save final trained model (AP4-style centralized artifact paths)
best_model_path = config.MODEL_BEST_PATH
evaluated_model_path = config.MODEL_EVALUATED_PATH

save_payload = {
    'model_state_dict': trained_model.state_dict(),
    'config': {
        'input_dim': config.CONCAT_DIM,
        'hidden_dim': config.HIDDEN_DIM,
        'output_dim': config.OUTPUT_DIM,
        'num_heads': config.NUM_HEADS,
        'num_layers': config.NUM_LAYERS,
        'dropout': config.DROPOUT,
        'model_base_name': config.MODEL_BASE_NAME,
    },
    'history': training_history,
}

torch.save(save_payload, best_model_path)
torch.save(save_payload, evaluated_model_path)

print(f'\nBest model saved: {best_model_path}')
print(f'Evaluated model saved: {evaluated_model_path}')
print(f'Last checkpoint saved earlier: {config.MODEL_LAST_PATH}')

plot_ap5_training_dashboard(
    training_history,
    config.TRAINING_HISTORY_PLOT_PATH,
    show=True,
)

print(f'Training history plotted: {config.TRAINING_HISTORY_PLOT_PATH}')

evaluation_metrics = evaluate_model_quality(model, topic_window_data, config, config.DEVICE)
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
    intra_path=config.OUTPUT_DIR / f"{config.MODEL_BASE_NAME}_eval_intra_heatmap.png",
    inter_path=config.OUTPUT_DIR / f"{config.MODEL_BASE_NAME}_eval_inter_heatmap.png",
)

with open(config.EVAL_METRICS_PATH, "w", encoding="utf-8") as file:
    json.dump(evaluation_metrics, file, indent=2)

evaluated_checkpoint = {
    "model_state_dict": model.state_dict(),
    "evaluation_metrics": evaluation_metrics,
    "config": {
        "input_dim": config.CONCAT_DIM,
        "hidden_dim": config.HIDDEN_DIM,
        "output_dim": config.OUTPUT_DIM,
        "num_heads": config.NUM_HEADS,
        "num_layers": config.NUM_LAYERS,
        "dropout": config.DROPOUT,
    },
}
torch.save(evaluated_checkpoint, config.MODEL_EVALUATED_PATH)

print(f"Saved evaluation metrics: {config.EVAL_METRICS_PATH}")
print(f"Saved evaluated model: {config.MODEL_EVALUATED_PATH}")

def split_articles_into_sentences(input_dataframe):
    """AP4-style splitter: one row per sentence with stable sentence_id/article_id/date fields."""
    rows = []
    for article_idx, row in input_dataframe.reset_index(drop=True).iterrows():
        article_text = str(row.get("article", "")).strip()
        if not article_text:
            continue

        date_value = pd.to_datetime(row.get("date"), errors="coerce")
        if pd.isna(date_value):
            continue

        sentences = [s.text.strip() for s in nlp(article_text).sents if s.text and s.text.strip()]
        for sentence_order, sentence_text in enumerate(sentences):
            rows.append(
                {
                    "date": date_value,
                    "article_id": int(article_idx),
                    "sentence_id": f"a{int(article_idx)}_s{int(sentence_order)}",
                    "sentence_order": int(sentence_order),
                    "sentence_text": sentence_text,
                }
            )

    return pd.DataFrame(rows)


print("AP4 sentence splitter ready")

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


def generate_contextual_sbert_embeddings(sentence_dataframe, config_dict, sbert_model_name="all-mpnet-base-v2"):
    model_sbert = SentenceTransformer(sbert_model_name, device="cpu")
    encoded = model_sbert.encode(
        sentence_dataframe["context_text"].tolist(),
        batch_size=int(config_dict["inference_batch_size"]),
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    sentence_dataframe = sentence_dataframe.copy()
    sentence_dataframe["sentence_embeddings"] = [vec.astype(np.float32) for vec in np.asarray(encoded, dtype=np.float32)]
    return sentence_dataframe


def load_topic_embedding_prototypes(topic_embeddings_json_path, config_dict):
    with open(topic_embeddings_json_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    topic_embeddings = {}
    for topic_name in config_dict["topics"]:
        vector = np.asarray(payload[topic_name], dtype=np.float32)
        vector = vector / (np.linalg.norm(vector) + 1e-8)
        topic_embeddings[topic_name] = vector
    return topic_embeddings


def soft_topic_label_sentences(sentence_dataframe, topic_embeddings, config_dict):
    rows = []
    topic_names = config_dict["topics"]
    topic_matrix = np.stack([topic_embeddings[name] for name in topic_names]).astype(np.float32)
    topic_table_64 = np.asarray(config_dict["topic_embedding_table"], dtype=np.float32)

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
            "topic_probabilities": topic_probs.astype(np.float32),
        }
        for topic_idx, topic_name in enumerate(topic_names):
            record[topic_name] = np.float32(topic_probs[topic_idx])
        rows.append(record)

    return pd.DataFrame(rows)


def build_topic_score_rows(labeled_sentence_dataframe, config_dict):
    rows = []
    for row in labeled_sentence_dataframe.itertuples(index=False):
        for topic_name in config_dict["topics"]:
            rows.append({"sentence_id": row.sentence_id, "topic": topic_name, "similarity_score": float(getattr(row, topic_name))})
    return rows


print("AP4 context and soft-label helper functions ready")

def _normalize_topic_label(text):
    text = str(text).strip().lower()
    return "".join(ch for ch in text if ch.isalnum())


def resolve_topic_name(user_topic, config_topics):
    if user_topic in config_topics:
        return user_topic
    target = _normalize_topic_label(user_topic)
    for topic in config_topics:
        if target == _normalize_topic_label(topic):
            return topic
    raise ValueError(f"Unsupported topic '{user_topic}'. Choose one of: {config_topics}")


def compute_topic_similarity_with_embeddings(sentence_embeddings, topic_embedding):
    sentence_embeddings = np.asarray(sentence_embeddings, dtype=np.float32)
    topic_embedding = np.asarray(topic_embedding, dtype=np.float32)
    sentence_embeddings = sentence_embeddings / (np.linalg.norm(sentence_embeddings, axis=1, keepdims=True) + 1e-8)
    topic_embedding = topic_embedding / (np.linalg.norm(topic_embedding) + 1e-8)
    similarities = sentence_embeddings @ topic_embedding
    similarities = (similarities + 1.0) / 2.0
    return np.maximum(similarities, 0.3).astype(np.float32)


def compute_topic_drift(model, user_windows, config_dict, device):
    return compute_topic_drift_inference(model, user_windows, config_dict, device)


def detect_shifts(drift_rows, config_dict):
    return detect_shifts_inference(drift_rows, config_dict)


print("AP4 drift and shift helper functions ready")

def load_topic_embeddings_json(json_path):
    """
    Load ideal topic embeddings from JSON.
    
    Expected format:
    {
        "Health": [0.1, 0.2, ..., 0.768],
        "War": [...],
        ...
    }
    
    Returns:
        topic_embeddings: Dict of {topic_name: np.array(768,)}
    """
    print("\n" + "="*80)
    print("STEP 4: Loading topic embeddings (ideal articles)")
    print("="*80)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    topic_embeddings = {
        topic: np.array(emb) for topic, emb in data.items()
    }
    
    print(f"✅ Loaded embeddings for topics: {list(topic_embeddings.keys())}")
    for topic, emb in topic_embeddings.items():
        print(f"   {topic}: {emb.shape}")
    
    return topic_embeddings

print("✅ Topic embedding loader defined")

def compute_topic_weights_inference(sentence_df, topic_embeddings):
    """
    Compute topic weights using cosine similarity.
    
    For each sentence, compute similarity to all 5 topic embeddings.
    
    Returns:
        sentence_df: DataFrame with topic weight columns added
    """
    print("\n" + "="*80)
    print("STEP 5: Computing topic weights (cosine similarity)")
    print("="*80)
    
    topics = list(topic_embeddings.keys())
    
    for topic in topics:
        topic_emb = topic_embeddings[topic]
        
        weights = []
        for sent_emb in tqdm(sentence_df['embedding'], desc=f"Computing {topic} weights"):
            # Cosine similarity = 1 - cosine distance
            sim = 1 - cosine(sent_emb, topic_emb)
            weights.append(max(0, sim))  # Clamp to [0, 1]
        
        sentence_df[f'{topic}_weight'] = weights
    
    print(f"✅ Topic weights computed for: {topics}")
    
    return sentence_df

print("✅ Topic weight computation defined")

def filter_by_topic_weights_inference(sentence_df, threshold=0.35):
    """
    Filter sentences where at least one topic weight > threshold.
    
    Returns:
        filtered_df: Filtered DataFrame
    """
    print("\n" + "="*80)
    print(f"STEP 6: Filtering sentences (topic weight > {threshold})")
    print("="*80)
    
    weight_cols = [col for col in sentence_df.columns if col.endswith('_weight')]
    
    # Keep if ANY topic weight > threshold
    mask = (sentence_df[weight_cols] > threshold).any(axis=1)
    
    filtered_df = sentence_df[mask].copy().reset_index(drop=True)
    
    print(f"✅ Filtered: {len(sentence_df)} → {len(filtered_df)} sentences")
    print(f"   Removed: {len(sentence_df) - len(filtered_df)} sentences below threshold")
    
    return filtered_df

print("✅ Filter function defined")

def extract_entities_inference(sentence_df, sbert_model, nlp):
    """
    Extract entities and compute 768D entity embeddings.
    """
    print("\n" + "=" * 80)
    print("STEP 7a: Extracting entities (NER)")
    print("=" * 80)

    all_entities = []
    entity_signatures = []
    entity_embeddings = []

    for sent in tqdm(sentence_df['center_sentence'], desc="Extracting entities"):
        doc = nlp(sent)
        entities = [ent.text.strip() for ent in doc.ents if ent.text and ent.text.strip()]
        all_entities.append(entities)

        if entities:
            normalized = sorted({e.lower() for e in entities})
            entity_signatures.append(" | ".join(normalized[:5]))
            ent_embs = sbert_model.encode(entities, convert_to_numpy=True).astype(np.float32)
            entity_emb = ent_embs.mean(axis=0).astype(np.float32)
        else:
            entity_signatures.append("__NO_ENTITY__")
            entity_emb = np.zeros(config.EMBEDDING_DIM, dtype=np.float32)

        norm = np.linalg.norm(entity_emb)
        if norm > 1e-6:
            entity_emb = entity_emb / norm

        entity_embeddings.append(entity_emb.astype(np.float32))

    sentence_df['entities'] = all_entities
    sentence_df['entity_signature'] = entity_signatures
    sentence_df['entity_embedding'] = entity_embeddings

    print("Entity embeddings computed")
    return sentence_df


def compute_clean_embeddings_inference(sentence_df, entity_proj_layer, device, lambda_=0.5):
    """
    Build inference dual representation:
      semantic_clean (768) + entity_small (64) -> final_embedding (832)
    """
    print("\n" + "=" * 80)
    print(f"STEP 7b: Entity-aware dual representation (lambda={lambda_})")
    print("=" * 80)

    semantic_clean = []
    entity_small = []
    final_embeddings = []

    entity_proj_layer.eval()

    for _, row in sentence_df.iterrows():
        sem_emb = np.asarray(row['embedding'], dtype=np.float32)
        ent_emb = np.asarray(row['entity_embedding'], dtype=np.float32)

        sem_clean = sem_emb - float(lambda_) * ent_emb
        sem_norm = np.linalg.norm(sem_clean)
        if sem_norm > 1e-6:
            sem_clean = sem_clean / sem_norm

        with torch.no_grad():
            ent_tensor = torch.from_numpy(ent_emb).to(device=device, dtype=torch.float32).unsqueeze(0)
            ent_small = entity_proj_layer(ent_tensor).squeeze(0).cpu().numpy().astype(np.float32)

        ent_small_norm = np.linalg.norm(ent_small)
        if ent_small_norm > 1e-6:
            ent_small = ent_small / ent_small_norm

        final_emb = np.concatenate([sem_clean.astype(np.float32), ent_small], axis=0).astype(np.float32)

        semantic_clean.append(sem_clean.astype(np.float32))
        entity_small.append(ent_small.astype(np.float32))
        final_embeddings.append(final_emb)

    sentence_df['semantic_clean_embedding'] = semantic_clean
    sentence_df['entity_small_embedding'] = entity_small
    sentence_df['final_embedding'] = final_embeddings

    print(f"Dual representation ready: {config.EMBEDDING_DIM} + {config.ENTITY_PROJ_DIM} = {config.SENTENCE_FINAL_DIM}")
    return sentence_df


print("Entity-aware inference functions defined")

def add_topic_embeddings_inference(day_df, topic_emb_layer, topic_to_id, device):
    """
    Add learned topic embeddings to day embeddings.

    832D -> 896D (832 + 64)
    """
    print("\n" + "=" * 80)
    print("STEP 9: Adding learned topic embeddings (832D -> 896D)")
    print("=" * 80)

    embeddings_with_topic = []
    topic_emb_layer.eval()

    for _, row in day_df.iterrows():
        topic = row['topic']
        day_emb = np.asarray(row['embedding'], dtype=np.float32)

        topic_id = topic_to_id[topic]
        topic_emb = topic_emb_layer(torch.tensor([topic_id]).to(device)).detach().cpu().numpy()[0].astype(np.float32)

        combined_emb = np.concatenate([day_emb, topic_emb], axis=0).astype(np.float32)
        embeddings_with_topic.append(combined_emb)

    day_df['embedding_with_topic'] = embeddings_with_topic

    print("Topic embeddings added")
    print(f"  Shape: {config.SENTENCE_FINAL_DIM} + {config.TOPIC_EMB_DIM} = {config.CONCAT_DIM}")

    return day_df


print("Topic embedding addition function defined")

def create_day_windows_inference(day_df, window_size=3):
    """
    Create windows from day-level embeddings (NO ruptures grouping).

    Returns:
        windows: (N, window_size, 896)
        window_metadata: topic/date/entity context per window
    """
    print("\n" + "=" * 80)
    print(f"STEP 10: Creating day-level windows (size={window_size}, NO grouping)")
    print("=" * 80)

    topics = day_df['topic'].unique()
    all_windows = []
    all_metadata = []

    for topic in topics:
        topic_df = day_df[day_df['topic'] == topic].sort_values('date').reset_index(drop=True)
        embeddings = np.stack(topic_df['embedding_with_topic'].values).astype(np.float32)

        for i in range(len(embeddings) - window_size + 1):
            window = embeddings[i:i + window_size]

            context_values = topic_df.iloc[i:i + window_size]['entity_context'].tolist() if 'entity_context' in topic_df.columns else []
            merged_context = " ; ".join([str(x) for x in context_values if str(x)])

            all_windows.append(window)
            all_metadata.append({
                'topic': topic,
                'start_date': topic_df.iloc[i]['date'],
                'end_date': topic_df.iloc[i + window_size - 1]['date'],
                'window_idx': i,
                'entity_context': merged_context if merged_context else '__NO_ENTITY__',
            })

    windows = np.array(all_windows)

    print(f"Windows created: {windows.shape}")
    print("  Per topic:")
    for topic in topics:
        topic_count = sum(1 for m in all_metadata if m['topic'] == topic)
        print(f"  - {topic}: {topic_count} windows")

    return windows, all_metadata


print("Day-level windowing function defined")

def predict_with_model_inference(windows, model, device, batch_size=32):
    """
    Run model forward pass on windows.
    
    Returns:
        embeddings: np.array of shape (N, 256)
    """
    print("\n" + "="*80)
    print("STEP 11: Model prediction")
    print("="*80)
    
    model.eval()
    
    all_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(windows), batch_size), desc="Predicting"):
            batch = windows[i:i+batch_size]
            batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)
            
            embeddings = model(batch_tensor)
            
            all_embeddings.append(embeddings.cpu().numpy())
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    
    print(f"✅ Predictions complete: {embeddings.shape}")
    
    return embeddings

print("✅ Model prediction function defined")

def detect_day_level_shifts_inference(embeddings, window_metadata, threshold_percentile=90, entity_overlap_threshold=0.2):
    """
    Detect narrative shifts with entity-aware filtering.

    Shift score: L2 distance between consecutive windows.
    Filter: ignore shift if entity overlap is below threshold.
    """
    print("\n" + "=" * 80)
    print("STEP 12: Day-level narrative shift detection")
    print("=" * 80)

    topics = list(set(m['topic'] for m in window_metadata))

    all_shifts = []
    all_shift_scores = []
    skipped_by_entity = 0

    for topic in topics:
        topic_indices = [i for i, m in enumerate(window_metadata) if m['topic'] == topic]
        topic_embeddings = embeddings[topic_indices]
        topic_metadata = [window_metadata[i] for i in topic_indices]

        shift_scores = []
        overlap_scores = []
        for i in range(len(topic_embeddings) - 1):
            dist = np.linalg.norm(topic_embeddings[i + 1] - topic_embeddings[i])
            shift_scores.append(float(dist))

            left_ctx = topic_metadata[i].get('entity_context', '__NO_ENTITY__')
            right_ctx = topic_metadata[i + 1].get('entity_context', '__NO_ENTITY__')
            overlap_scores.append(float(_jaccard_overlap_from_strings(left_ctx, right_ctx)))

        all_shift_scores.extend(shift_scores)

        if len(shift_scores) > 0:
            threshold = float(np.percentile(shift_scores, threshold_percentile))

            for i, score in enumerate(shift_scores):
                overlap = overlap_scores[i]
                if overlap < float(entity_overlap_threshold):
                    skipped_by_entity += 1
                    continue

                if score > threshold:
                    all_shifts.append({
                        'topic': topic,
                        'window_idx': i,
                        'start_date': topic_metadata[i]['start_date'],
                        'end_date': topic_metadata[i]['end_date'],
                        'next_start_date': topic_metadata[i + 1]['start_date'],
                        'shift_score': float(score),
                        'threshold': float(threshold),
                        'entity_overlap': float(overlap),
                        'entity_context': topic_metadata[i + 1].get('entity_context', '__NO_ENTITY__'),
                        'embedding_before': topic_embeddings[i],
                        'embedding_after': topic_embeddings[i + 1],
                    })

    print(f"Day-level shifts detected: {len(all_shifts)}")
    print(f"Skipped by entity-overlap filter: {skipped_by_entity}")
    if len(all_shift_scores) > 0:
        print(f"  Threshold (P{threshold_percentile}): {np.percentile(all_shift_scores, threshold_percentile):.4f}")

    return all_shifts, all_shift_scores


print("Day-level shift detection function defined")

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
    for _, record in subset.iterrows():
        prefix = ">>> " if int(record["sentence_order"]) == current_order else "    "
        lines.append(f"{prefix}[{record['sentence_id']}] {record['sentence_text']}")
    return "\n".join(lines)


def extract_sentence_level_narrative_shifts(
    filtered_sentence_dataframe,
    drift_rows,
    config_dict,
    top_k_shifts=5,
    context_source_dataframe=None,
):
    return extract_sentence_level_narrative_shifts_inference(
        filtered_sentence_dataframe=filtered_sentence_dataframe,
        drift_rows=drift_rows,
        config_dict=config_dict,
        top_k_shifts=top_k_shifts,
        context_source_dataframe=context_source_dataframe,
    )


print("AP4 sentence-level shift helpers ready")

def _cfg_get(config_obj, key, default=None):
    if isinstance(config_obj, dict):
        return config_obj.get(key, default)
    return getattr(config_obj, key, default)


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


def load_user_articles_csv(csv_path):
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def process_articles_to_sentences(df):
    split_df = split_articles_into_sentences(df)
    split_df = build_context_texts(split_df, 5)
    split_df["center_sentence"] = split_df["sentence_text"]
    split_df["window_text"] = split_df["context_text"]
    return split_df


def compute_sentence_embeddings_inference(sentence_df, sbert_model):
    texts = sentence_df["window_text"].tolist()
    embeddings = sbert_model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    out_df = sentence_df.copy()
    out_df["embedding"] = [vec.astype(np.float32) for vec in np.asarray(embeddings, dtype=np.float32)]
    return out_df


def detect_sentence_level_shifts_inference(day_shifts, sentence_df, model, topic_emb_layer, topic_to_id, device, top_k=5):
    _ = (day_shifts, sentence_df, model, topic_emb_layer, topic_to_id, device, top_k)
    return []


def build_inference_config(config_obj):
    return {
        "topics": list(config_obj.TOPICS),
        "context_window": 5,
        "embedding_dim": int(config_obj.EMBEDDING_DIM),
        "sentence_final_dim": int(config_obj.SENTENCE_FINAL_DIM),
        "topic_embedding_dim": int(config_obj.TOPIC_EMB_DIM),
        "concat_dim": int(config_obj.CONCAT_DIM),
        "topic_embedding_table": np.asarray(topic_emb_layer.weight.detach().cpu().numpy(), dtype=np.float32),
        "window_size": int(config_obj.WINDOW_SIZE),
        "stride": int(getattr(config_obj, "WINDOW_STRIDE", 1)),
        "temperature": float(config_obj.TEMPERATURE),
        "topic_threshold": 0.60,
        "topic_weight_threshold": 0.60,
        "manual_shift_threshold": float(config_obj.SHIFT_THRESHOLD),
        "inference_batch_size": 32,
    }


def run_user_level_inference(user_csv_path, model, config_dict, topic_name, ideal_topic_embeddings_json_path, sbert_model_name="all-mpnet-base-v2"):
    # Inference path is centralized in inference.py for AP5 parity.
    return run_user_level_inference_inference(
        user_csv_path=user_csv_path,
        model=model,
        config_dict=config_dict,
        topic_name=topic_name,
        ideal_topic_embeddings_json_path=ideal_topic_embeddings_json_path,
        sbert_model_name=sbert_model_name,
    )
def run_user_level_inference_approach5_compatible(
    user_csv_path,
    model,
    config_dict,
    topic_name,
    ideal_topic_embeddings_json_path,
    sbert_model_name="all-mpnet-base-v2",
):
    return run_user_inference_approach5(
        user_csv_path=user_csv_path,
        ideal_topic_embeddings_json_path=ideal_topic_embeddings_json_path,
        topic_name=topic_name,
        config_obj=config_dict if config_dict is not None else config,
        model_variant="best",
        sbert_model_name=sbert_model_name,
    )

def main():
    # Batch AP5 multi-topic inference entrypoint (delegates to inference.py).
    user_csv_path = str(Path("/home/hp/SEM2/INLP/Naretve_Shift/Output/Model_Testing/Approch_4/Ner1_hard_combined.csv"))
    ideal_topic_embeddings_json_path = "/home/hp/SEM2/INLP/Naretve_Shift/Processed_Data/topic_embeddings.json"
    selected_topics = list(config.TOPICS)

    multi_topic_output_path = config.OUTPUT_DIR / f"{config.MODEL_BASE_NAME}_user_inference_multi_topic.json"

    inference_output_payload = run_multitopic_inference_approach5_minimal(
        user_csv_path=user_csv_path,
        ideal_topic_embeddings_json_path=ideal_topic_embeddings_json_path,
        config_obj=config,
        selected_topics=selected_topics,
        inference_overrides={
            "topic_threshold": 0.45,
            "topic_weight_threshold": 0.20,
            "manual_shift_threshold": 0.3,
        },
        model_variant="best",
        sbert_model_name="all-mpnet-base-v2",
        output_json_path=multi_topic_output_path,
    )

    print("\nSaved multi-topic inference output:")
    print(multi_topic_output_path)
    return inference_output_payload

def print_shift_results(results, max_display=100):
    """
    Print narrative shift results in a readable format.
    """
    day_shifts = results['day_shifts']
    sentence_shifts = results['sentence_shifts']
    
    print("\n" + "="*80)
    print("DAY-LEVEL NARRATIVE SHIFTS")
    print("="*80)
    
    for i, shift in enumerate(day_shifts[:max_display]):
        print(f"\n[{i+1}] Topic: {shift['topic']}")
        print(f"    Date: {shift['start_date']} → {shift['next_start_date']}")
        print(f"    Shift Score: {shift['shift_score']:.4f} (threshold: {shift['threshold']:.4f})")
    
    if len(day_shifts) > max_display:
        print(f"\n... and {len(day_shifts) - max_display} more")
    
    print("\n" + "="*80)
    print("SENTENCE-LEVEL NARRATIVE SHIFTS")
    print("="*80)
    
    for i, shift in enumerate(sentence_shifts[:max_display]):
        print(f"\n[{i+1}] Topic: {shift['topic']} | Date: {shift['date']}")
        print(f"    Shift Contribution: {shift['shift_contribution']:.4f}")
        print(f"    Similarity Before: {shift['similarity_to_before']:.4f}")
        print(f"    Similarity After: {shift['similarity_to_after']:.4f}")
        print(f"    Day Shift Score: {shift['day_shift_score']:.4f}")
        print(f"\n    Sentence: \"{shift['sentence']}\"")
        
        if shift['context_before']:
            print(f"\n    Context Before:")
            for j, sent in enumerate(shift['context_before']):
                print(f"      [{j-len(shift['context_before'])}] {sent}")
        
        if shift['context_after']:
            print(f"\n    Context After:")
            for j, sent in enumerate(shift['context_after']):
                print(f"      [+{j+1}] {sent}")
    
    if len(sentence_shifts) > max_display:
        print(f"\n... and {len(sentence_shifts) - max_display} more")

def plot_shift_timeline(results, figsize=(14, 6)):
    """
    Plot shift scores timeline.
    """
    all_scores = results['all_shift_scores']
    day_shifts = results['day_shifts']
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Timeline
    ax1 = axes[0]
    x = np.arange(len(all_scores))
    ax1.plot(x, all_scores, linewidth=1, alpha=0.7)
    
    if len(all_scores) > 0:
        threshold = np.percentile(all_scores, 90)
        ax1.axhline(threshold, color='red', linestyle='--', label=f'P90 threshold')
    
    ax1.set_xlabel('Window Index')
    ax1.set_ylabel('Shift Score (L2 Distance)')
    ax1.set_title('Narrative Shift Timeline')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Distribution
    ax2 = axes[1]
    ax2.hist(all_scores, bins=30, alpha=0.7, edgecolor='black')
    
    if len(all_scores) > 0:
        threshold = np.percentile(all_scores, 90)
        ax2.axvline(threshold, color='red', linestyle='--', label=f'P90 threshold')
    
    ax2.set_xlabel('Shift Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Shift Score Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 Summary:")
    print(f"   Total windows: {len(all_scores)}")
    print(f"   Significant shifts: {len(day_shifts)}")
    if len(all_scores) > 0:
        print(f"   Mean shift score: {np.mean(all_scores):.4f}")
        print(f"   Max shift score: {np.max(all_scores):.4f}")

print("✅ Visualization functions defined")


if __name__ == "__main__":
    inference_output_payload = main()
    if inference_output_payload:
        print_multitopic_inference_outputs_approach5(inference_output_payload)
    else:
        print("Run the user inference call cell first.")

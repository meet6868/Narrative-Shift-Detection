from __future__ import annotations

import ast
import numpy as np
import pandas as pd


def _cfg_get(config, key: str, default=None):
	if isinstance(config, dict):
		return config.get(key, default)
	return getattr(config, key, default)


def add_topic_embeddings_for_topic(dataframe: pd.DataFrame, topic_name: str, config, mode: str = "ap5") -> pd.DataFrame:
	"""Add `topic_embeddings` column to loaded CSV rows with approach-aware dimension.

	For ap4/ap5, topic embedding defaults to 64d (config-driven).
	For ap1/ap2, topic embedding defaults to one-hot topic vector.
	"""
	topics = list(_cfg_get(config, "topics", _cfg_get(config, "TOPICS", [])))
	if topic_name not in topics:
		raise ValueError(f"Unknown topic '{topic_name}' not in topics: {topics}")

	topic_idx = topics.index(topic_name)
	out = dataframe.copy()

	if mode in {"ap4", "ap5"}:
		topic_dim = int(_cfg_get(config, "topic_embedding_dim", _cfg_get(config, "TOPIC_EMB_DIM", 64)))
		topic_table = _cfg_get(config, "topic_embedding_table", _cfg_get(config, "TOPIC_EMBEDDING_TABLE", None))
		if topic_table is None:
			seed = int(_cfg_get(config, "seed", 42))
			rng = np.random.default_rng(seed)
			topic_table = rng.standard_normal((len(topics), topic_dim)).astype(np.float32)
			topic_table /= (np.linalg.norm(topic_table, axis=1, keepdims=True) + 1e-8)
		topic_vec = np.asarray(topic_table[topic_idx], dtype=np.float32)
	else:
		topic_vec = np.eye(len(topics), dtype=np.float32)[topic_idx]

	out["topic_embeddings"] = [topic_vec.copy() for _ in range(len(out))]
	return out


def approach5_extract_entities_batch(dataframe: pd.DataFrame, nlp, batch_size: int = 256) -> pd.DataFrame:
	"""Extract entities and add `entities` and `entity_signature` columns."""
	out = dataframe.copy()
	sentences = out["main_sentence"].tolist()
	all_entities = []
	entity_signatures = []

	for i in range(0, len(sentences), int(batch_size)):
		batch_sentences = sentences[i : i + int(batch_size)]
		docs = list(nlp.pipe(batch_sentences, batch_size=int(batch_size)))
		for doc in docs:
			entities = [ent.text.strip() for ent in doc.ents if ent.text and ent.text.strip()]
			all_entities.append(entities)
			if entities:
				normalized = sorted({e.lower() for e in entities})
				entity_signatures.append(" | ".join(normalized[:5]))
			else:
				entity_signatures.append("__NO_ENTITY__")

	out["entities"] = all_entities
	out["entity_signature"] = entity_signatures
	return out


def approach5_add_entity_embeddings(dataframe: pd.DataFrame, sbert_model, embedding_dim: int = 768) -> pd.DataFrame:
	"""Add `entity_embedding` column by encoding entity text using SBERT."""
	out = dataframe.copy()
	entity_embeddings = []

	for entities in out["entities"]:
		if len(entities) == 0:
			entity_embedding = np.zeros(int(embedding_dim), dtype=np.float32)
		else:
			entity_text = " ".join(entities)
			entity_embedding = sbert_model.encode(entity_text, convert_to_numpy=True).astype(np.float32)

		norm = np.linalg.norm(entity_embedding)
		if norm > 1e-6:
			entity_embedding = entity_embedding / norm
		entity_embeddings.append(entity_embedding.astype(np.float32))

	out["entity_embedding"] = entity_embeddings
	return out


def compute_entity_invariant_embeddings(
	dataframe: pd.DataFrame,
	entity_proj_layer,
	device,
	lambda_: float = 0.3,
	embedding_column: str = "embedding",
	entity_embedding_column: str = "entity_embedding",
) -> pd.DataFrame:
	"""Compute AP5 entity-invariant features and return dataframe with 3 new columns.

	Adds:
	- semantic_clean_embedding (768)
	- entity_small_embedding (64)
	- final_embedding (832)
	"""
	out = dataframe.copy()
	semantic_clean_embeddings = []
	entity_small_embeddings = []
	final_embeddings = []

	entity_proj_layer.eval()
	proj_device = next(entity_proj_layer.parameters()).device

	for _, row in out.iterrows():
		sem_emb = np.asarray(row[embedding_column], dtype=np.float32)
		ent_emb = np.asarray(row[entity_embedding_column], dtype=np.float32)

		sem_clean = sem_emb - float(lambda_) * ent_emb
		sem_norm = np.linalg.norm(sem_clean)
		if sem_norm > 1e-6:
			sem_clean = sem_clean / sem_norm

		# Uses projection layer device; caller controls CPU/GPU fallback.
		with __import__("torch").no_grad():
			ent_tensor = __import__("torch").from_numpy(ent_emb).to(device=proj_device, dtype=__import__("torch").float32).unsqueeze(0)
			ent_small = entity_proj_layer(ent_tensor).squeeze(0).cpu().numpy().astype(np.float32)

		ent_small_norm = np.linalg.norm(ent_small)
		if ent_small_norm > 1e-6:
			ent_small = ent_small / ent_small_norm

		final_emb = np.concatenate([sem_clean.astype(np.float32), ent_small], axis=0).astype(np.float32)

		semantic_clean_embeddings.append(sem_clean.astype(np.float32))
		entity_small_embeddings.append(ent_small.astype(np.float32))
		final_embeddings.append(final_emb)

	out["semantic_clean_embedding"] = semantic_clean_embeddings
	out["entity_small_embedding"] = entity_small_embeddings
	out["final_embedding"] = final_embeddings
	return out


def save_entity_invariant_cache_csv(
	dataframe: pd.DataFrame,
	csv_path,
	vector_columns=("embedding", "entity_embedding", "semantic_clean_embedding", "entity_small_embedding", "final_embedding", "topic_embeddings"),
) -> None:
	"""Save AP5 cache to CSV by serializing vector columns as list strings."""
	out = dataframe.copy()
	for col in vector_columns:
		if col in out.columns:
			out[col] = out[col].apply(lambda x: np.asarray(x, dtype=np.float32).tolist())
	out.to_csv(csv_path, index=False)


def load_entity_invariant_cache_csv(
	csv_path,
	vector_columns=("embedding", "entity_embedding", "semantic_clean_embedding", "entity_small_embedding", "final_embedding", "topic_embeddings"),
) -> pd.DataFrame:
	"""Load AP5 CSV cache and parse serialized vector columns back to ndarrays."""
	dataframe = pd.read_csv(csv_path)
	for col in vector_columns:
		if col in dataframe.columns:
			dataframe[col] = dataframe[col].apply(
				lambda x: np.asarray(ast.literal_eval(x), dtype=np.float32) if isinstance(x, str) else np.asarray(x, dtype=np.float32)
			)
	return dataframe


def aggregate_daily_embeddings(
	dataframe: pd.DataFrame,
	topics: list[str],
	embedding_column: str = "embedding",
	min_sentences_per_day: int = 1,
	weight_column_map: dict[str, list[str]] | None = None,
	topic_embeddings_column: str = "topic_embeddings",
	fallback_topic_embeddings_map: dict[str, np.ndarray] | None = None,
	normalize_date: bool = False,
	require_weight_column: bool = False,
	entity_signature_column: str | None = None,
	output_embedding_column: str = "daily_vectors",
	topic_column_name: str | None = "topic_name",
	include_topic_id: bool = True,
	include_avg_weight: bool = False,
) -> pd.DataFrame:
	"""Single shared daily aggregation function for AP1/AP2/AP4/AP5.

	It supports:
	- single-topic daily pooling (AP1/AP2/AP4/AP5 train)
	- multi-topic daily-topic pooling (AP5 inference)
	"""
	temp_df = dataframe.copy()
	if normalize_date:
		temp_df["date_only"] = pd.to_datetime(temp_df["date"]).dt.normalize()
	else:
		temp_df["date_only"] = pd.to_datetime(temp_df["date"]).dt.date

	weight_column_map = weight_column_map or {}
	fallback_topic_embeddings_map = fallback_topic_embeddings_map or {}
	topic_to_id = {topic: idx for idx, topic in enumerate(topics)}
	records = []

	for date_only, group in temp_df.groupby("date_only"):
		if len(group) < int(min_sentences_per_day):
			continue

		embeddings = np.stack(group[embedding_column].values).astype(np.float32)

		for topic_name in topics:
			candidate_weight_cols = weight_column_map.get(topic_name, [topic_name])
			resolved_weight_col = next((c for c in candidate_weight_cols if c in group.columns), None)

			if require_weight_column and resolved_weight_col is None:
				continue

			if resolved_weight_col is not None:
				raw_weights = np.clip(group[resolved_weight_col].astype(np.float32).values, a_min=0.0, a_max=None)
			else:
				raw_weights = np.ones(len(group), dtype=np.float32)

			if raw_weights.sum() > 0:
				weights = raw_weights / raw_weights.sum()
			else:
				weights = np.ones(len(group), dtype=np.float32) / max(len(group), 1)

			daily_embedding = (embeddings.T @ weights).astype(np.float32)
			norm = np.linalg.norm(daily_embedding)
			if norm > 1e-8:
				daily_embedding = daily_embedding / norm

			if topic_embeddings_column in group.columns:
				topic_embedding = np.asarray(group[topic_embeddings_column].iloc[0], dtype=np.float32)
			elif topic_name in fallback_topic_embeddings_map:
				topic_embedding = np.asarray(fallback_topic_embeddings_map[topic_name], dtype=np.float32)
			else:
				topic_embedding = np.eye(len(topics), dtype=np.float32)[int(topic_to_id[topic_name])]

			row = {
				"date": pd.Timestamp(date_only),
				output_embedding_column: daily_embedding,
				"topic_embeddings": topic_embedding,
				"num_sentences": int(len(group)),
			}

			if topic_column_name is not None:
				row[topic_column_name] = topic_name
			if include_topic_id:
				row["topic_id"] = int(topic_to_id[topic_name])
			if include_avg_weight:
				row["avg_weight"] = float(raw_weights.mean()) if len(raw_weights) > 0 else 0.0
			if entity_signature_column is not None:
				entity_set = sorted({str(x) for x in group.get(entity_signature_column, pd.Series([], dtype=str)).tolist() if str(x)})
				row["entity_context"] = " ; ".join(entity_set[:10]) if entity_set else "__NO_ENTITY__"

			records.append(row)

	columns = ["date"]
	if topic_column_name is not None:
		columns.append(topic_column_name)
	if include_topic_id:
		columns.append("topic_id")
	columns.extend([output_embedding_column, "topic_embeddings", "num_sentences"])
	if include_avg_weight:
		columns.append("avg_weight")
	if entity_signature_column is not None:
		columns.append("entity_context")

	if not records:
		return pd.DataFrame(columns=columns)

	result_df = pd.DataFrame(records)
	sort_cols = ["date"]
	if topic_column_name is not None:
		sort_cols = [topic_column_name, "date"]
	return result_df.sort_values(sort_cols).reset_index(drop=True)


def build_temporal_feature_records(
	group_dataframe: pd.DataFrame,
	include_tau: bool = False,
	tau_scale: float = 5.0,
	daily_vectors_column: str = "daily_vectors",
	topic_embeddings_column: str = "topic_embeddings",
	output_vector_column: str = "final_vector",
	include_end_date: bool = True,
	include_num_sentences: bool = True,
	include_num_days: bool = True,
) -> list[dict]:
	"""Build temporal feature records from grouped/day-level data.

	Supports both styles used across approaches:
	- AP1/AP2: [daily_vectors + tau + topic_embeddings]
	- AP4/AP5: [daily_vectors + topic_embeddings]
	"""
	rows = group_dataframe.sort_values("date").to_dict(orient="records")
	records = []

	for index, row in enumerate(rows):
		daily_vector = np.asarray(row[daily_vectors_column], dtype=np.float32)
		topic_vector = np.asarray(row[topic_embeddings_column], dtype=np.float32)

		if include_tau:
			if index == 0:
				tau = 0.0
			else:
				day_gap = (row["date"] - rows[index - 1]["date"]).days
				tau = np.log1p(day_gap) / float(tau_scale)
			final_vector = np.concatenate(
				[daily_vector, np.array([tau], dtype=np.float32), topic_vector]
			).astype(np.float32)
		else:
			final_vector = np.concatenate([daily_vector, topic_vector]).astype(np.float32)

		record = {
			"date": row["date"],
			output_vector_column: final_vector,
			"topic_name": row["topic_name"],
			"topic_id": row["topic_id"],
		}

		if include_end_date:
			record["end_date"] = row.get("end_date", row["date"])
		if include_num_sentences and "num_sentences" in row:
			record["num_sentences"] = row["num_sentences"]
		if include_num_days:
			record["num_days"] = row.get("num_days", 1)

		records.append(record)

	return records

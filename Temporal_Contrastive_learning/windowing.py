from __future__ import annotations

import numpy as np


def _cfg_get(config, key: str, default=None):
	if isinstance(config, dict):
		return config.get(key, default)
	return getattr(config, key, default)


def build_window_embeddings(
	enhanced_records,
	topic_name,
	topic_id,
	config,
	vector_column: str = "final_vector",
	start_date_column: str = "date",
	end_date_column: str = "end_date",
):
	"""Build sliding windows from temporal feature records for AP1/AP2/AP4/AP5."""
	records = sorted(enhanced_records, key=lambda row: row[start_date_column])
	window_size = int(_cfg_get(config, "window_size", _cfg_get(config, "WINDOW_SIZE", 3)))
	stride = int(_cfg_get(config, "stride", _cfg_get(config, "WINDOW_STRIDE", 1)))

	window_embeddings = []
	for start in range(0, len(records) - window_size + 1, stride):
		chunk = records[start:start + window_size]
		window_matrix = np.stack([row[vector_column] for row in chunk]).astype(np.float32)
		window_embeddings.append(
			{
				"tensor": window_matrix,
				"topic_id": topic_id,
				"topic_name": topic_name,
				"start_date": chunk[0][start_date_column],
				"end_date": chunk[-1].get(end_date_column, chunk[-1][start_date_column]),
				"window_idx": start,
			}
		)

	return window_embeddings


def create_windows(
	embeddings,
	window_size: int = 3,
	stride: int = 1,
	expected_dim: int | None = None,
):
	"""Create sliding windows from an embedding sequence.

	If the sequence is shorter than the window size, it is padded with zeros
	to preserve AP5 notebook behavior.
	"""
	embeddings = list(embeddings)
	if expected_dim is not None and len(embeddings) > 0:
		actual_dim = int(np.asarray(embeddings[0]).shape[-1])
		if actual_dim != int(expected_dim):
			raise ValueError(f"Expected {expected_dim}D embeddings, got {actual_dim}D")

	if len(embeddings) == 0:
		return np.empty((0, int(window_size), 0), dtype=np.float32), []

	if len(embeddings) < int(window_size):
		padding = [np.zeros_like(np.asarray(embeddings[0], dtype=np.float32)) for _ in range(int(window_size) - len(embeddings))]
		embeddings = embeddings + padding

	windows = []
	indices = []
	for start in range(0, len(embeddings) - int(window_size) + 1, int(stride)):
		window = np.stack(embeddings[start:start + int(window_size)]).astype(np.float32)
		windows.append(window)
		indices.append(int(start))

	return np.stack(windows), indices


def create_windows_from_dataframe(
	df,
	embedding_col: str = "embedding",
	window_size: int = 3,
	stride: int = 1,
	expected_dim: int | None = None,
	date_col: str = "date",
	entity_context_col: str = "entity_context",
):
	"""Create sliding windows and metadata from a dataframe-like object."""
	embeddings = df[embedding_col].tolist()
	windows, indices = create_windows(
		embeddings=embeddings,
		window_size=window_size,
		stride=stride,
		expected_dim=expected_dim,
	)

	metadata = []
	for idx in indices:
		date_slice = df[date_col].iloc[idx:idx + int(window_size)].tolist() if date_col in df.columns else None

		entity_contexts = []
		if entity_context_col in df.columns:
			entity_contexts = [str(value) for value in df[entity_context_col].iloc[idx:idx + int(window_size)].tolist() if str(value)]

		merged_entity_context = " ; ".join(entity_contexts[:10]) if entity_contexts else "__NO_ENTITY__"
		metadata.append(
			{
				"start_idx": int(idx),
				"end_idx": int(idx + int(window_size) - 1),
				"dates": date_slice,
				"entity_context": merged_entity_context,
			}
		)

	return windows, metadata

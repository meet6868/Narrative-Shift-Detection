import json
import re

import numpy as np
import pandas as pd


def split_articles_into_sentences(input_dataframe):
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
			"w3_embedding": emb.astype(np.float32),
			"w5_embedding": emb.astype(np.float32),
			"topic_probabilities": topic_probs.astype(np.float32)
		}

		for topic_idx, topic_name in enumerate(topic_names):
			record[topic_name] = np.float32(topic_probs[topic_idx])

		rows.append(record)

	return pd.DataFrame(rows)


def build_topic_score_rows(labeled_sentence_dataframe, config):
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
	detect_shifts_fn,
	top_k_shifts=5,
	per_date_sent_limit=40,
	context_window=2,
):
	if filtered_sentence_dataframe.empty or not drift_rows:
		return []

	detected_shifts = detect_shifts_fn(drift_rows, config)
	if not detected_shifts:
		return []

	filtered = filtered_sentence_dataframe.copy()
	filtered["date"] = pd.to_datetime(filtered["date"]).dt.normalize()

	unique_dates = sorted(filtered["date"].unique())
	if len(unique_dates) < 2:
		return []

	sentence_level_shifts = []
	ranked_shifts = sorted(detected_shifts, key=lambda x: x.get("z_score", 0.0), reverse=True)[:int(top_k_shifts)]

	for shift in ranked_shifts:
		date_2 = pd.Timestamp(shift["date"]).normalize()
		previous_dates = [d for d in unique_dates if d < date_2]
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

		sentence_level_shifts.append({
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
			"day_level_z_score": float(shift.get("z_score", 0.0))
		})

	return sentence_level_shifts

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import build_artifact_paths, load_checkpoint_compat
from models import TCLTemporalEncoderA12, TCLTemporalEncoderA4, TCLTemporalEncoderA5
from temporal_feature import (
	aggregate_daily_embeddings,
	approach5_add_entity_embeddings,
	approach5_extract_entities_batch,
	build_temporal_feature_records,
	compute_entity_invariant_embeddings,
)
from windowing import build_window_embeddings


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
		convert_to_numpy=True,
	)
	encoded = np.asarray(encoded, dtype=np.float32)

	if encoded.shape[1] != config["embedding_dim"]:
		raise ValueError(
			f"SBERT output dim {encoded.shape[1]} does not match config['embedding_dim']={config['embedding_dim']}"
		)

	sentence_dataframe = sentence_dataframe.copy()
	sentence_dataframe["sentence_embeddings"] = [vec.astype(np.float32) for vec in encoded]
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
		"8. sentence-level shift detection with context (final output)",
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
	context_window=2,
):
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
	ranked_shifts = sorted(detected_shifts, key=lambda x: x.get("z_score", 0.0), reverse=True)[: int(top_k_shifts)]

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


def _run_user_level_inference_a12(
	*,
	user_csv_path,
	model,
	config,
	topic_name,
	topic_embeddings_json_path,
	sbert_model_name,
	min_sentences_per_day,
	include_end_date,
	include_num_days,
	top_k_shifts,
):
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
			"training_like_rows": [],
		}

	sentence_dataframe = build_context_texts(sentence_dataframe, int(config["context_window"]))
	sentence_dataframe = generate_contextual_sbert_embeddings(
		sentence_dataframe,
		config,
		sbert_model_name=sbert_model_name,
	)

	topic_embeddings = load_topic_embedding_prototypes(topic_embeddings_json_path, config)
	labeled_sentence_dataframe = soft_topic_label_sentences(sentence_dataframe, topic_embeddings, config)
	topic_score_rows = build_topic_score_rows(labeled_sentence_dataframe, config)

	filtered_sentence_dataframe = filter_user_topic_sentences(labeled_sentence_dataframe, resolved_topic, config)
	if filtered_sentence_dataframe.empty:
		return {
			"call_order": call_order,
			"resolved_topic": resolved_topic,
			"sentence_level_narrative_shifts": [],
			"top_topic_sentences": [],
			"topic_score_rows": topic_score_rows,
			"training_like_rows": labeled_sentence_dataframe.to_dict(orient="records"),
		}

	validate_inference_alignment(config, filtered_sentence_dataframe)

	base_cols = [
		"date",
		"sentence_embeddings",
		"topic_probabilities",
		"sentence_text",
		"sentence_id",
		"similarity_score",
	] + config["topics"]
	training_aligned_input = filtered_sentence_dataframe[base_cols].rename(
		columns={
			"sentence_text": "main_sentence",
			"topic_probabilities": "topic_embeddings",
			"similarity_score": "weight",
		}
	)

	user_daily_df = aggregate_daily_embeddings(
		dataframe=training_aligned_input,
		topics=[resolved_topic],
		min_sentences_per_day=int(min_sentences_per_day),
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
			"training_like_rows": labeled_sentence_dataframe.to_dict(orient="records"),
		}

	user_records = build_temporal_feature_records(
		user_daily_df,
		include_tau=True,
		tau_scale=5.0,
		include_end_date=bool(include_end_date),
		include_num_sentences=True,
		include_num_days=bool(include_num_days),
	)
	user_windows = build_window_embeddings(user_records, resolved_topic, config["topics"].index(resolved_topic), config)

	if len(user_windows) < 2:
		return {
			"call_order": call_order,
			"resolved_topic": resolved_topic,
			"sentence_level_narrative_shifts": [],
			"top_topic_sentences": filtered_sentence_dataframe.sort_values("similarity_score", ascending=False)
			.head(20)[["date", "sentence_id", "sentence_text", "similarity_score"]]
			.to_dict(orient="records"),
			"topic_score_rows": topic_score_rows,
			"training_like_rows": labeled_sentence_dataframe.to_dict(orient="records"),
		}

	model_device = next(model.parameters()).device
	drift_rows, _ = compute_topic_drift(model, user_windows, config, model_device)

	sentence_level_shifts = extract_sentence_level_narrative_shifts(
		filtered_sentence_dataframe=filtered_sentence_dataframe,
		drift_rows=drift_rows,
		config=config,
		top_k_shifts=int(top_k_shifts),
		per_date_sent_limit=40,
		context_window=2,
	)

	top_topic_sentences = (
		filtered_sentence_dataframe.sort_values("similarity_score", ascending=False)
		.head(20)[["date", "sentence_id", "sentence_text", "similarity_score"]]
		.to_dict(orient="records")
	)

	return {
		"call_order": call_order,
		"resolved_topic": resolved_topic,
		"sentence_level_narrative_shifts": sentence_level_shifts,
		"top_topic_sentences": top_topic_sentences,
		"topic_score_rows": topic_score_rows,
		"training_like_rows": labeled_sentence_dataframe.to_dict(orient="records"),
	}


def run_user_level_inference_approach1(
	user_csv_path,
	model,
	config,
	topic_name,
	topic_embeddings_json_path,
	sbert_model_name="all-mpnet-base-v2",
):
	return _run_user_level_inference_a12(
		user_csv_path=user_csv_path,
		model=model,
		config=config,
		topic_name=topic_name,
		topic_embeddings_json_path=topic_embeddings_json_path,
		sbert_model_name=sbert_model_name,
		min_sentences_per_day=config["min_sentences_per_day"],
		include_end_date=False,
		include_num_days=False,
		top_k_shifts=20,
	)


def _normalize_selected_topics(selected_topics, config_topics):
	return list(dict.fromkeys(resolve_topic_name(topic, config_topics) for topic in selected_topics))


def _run_batch_inference_a12_common(
	*,
	config,
	input_directory,
	output_directory,
	topic_embeddings_json_path,
	selected_topics,
	inference_overrides,
	model_variant,
	sbert_model_name,
	legacy_best_filename,
	runner_fn,
):
	inference_config = dict(config)
	inference_config.update(inference_overrides)
	inference_config["load_variant"] = model_variant
	inference_config.update(build_artifact_paths(inference_config))

	Path(output_directory).mkdir(parents=True, exist_ok=True)

	legacy_best_path = os.path.join(inference_config["output_path"], legacy_best_filename)
	candidate_paths = [
		inference_config["model_load_path"],
		inference_config.get("model_evaluated_path"),
		legacy_best_path,
	]
	checkpoint_path = next((p for p in candidate_paths if p and os.path.exists(p)), None)
	if checkpoint_path is None:
		raise FileNotFoundError("Checkpoint not found in candidates: " + ", ".join([str(p) for p in candidate_paths if p]))

	inference_device = torch.device("cpu")
	inference_model = TCLTemporalEncoderA12(config).to(inference_device)
	checkpoint = load_checkpoint_compat(checkpoint_path, inference_device)
	state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
	inference_model.load_state_dict(state_dict)
	inference_model.eval()

	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	normalized_topics = _normalize_selected_topics(selected_topics, config["topics"])

	input_dir = Path(input_directory)
	if not input_dir.exists():
		raise FileNotFoundError(f"Input directory not found: {input_dir}")

	csv_files = sorted(input_dir.glob("*.csv"))
	if not csv_files:
		raise FileNotFoundError(f"No CSV files found in: {input_dir}")

	all_batch_inference_outputs = []

	print("Batch inference started")
	print(f"Model variant requested: {inference_config['load_variant']}")
	print(f"Checkpoint loaded: {checkpoint_path}")
	print(f"Input directory: {input_dir}")
	print(f"Output directory: {output_directory}")
	print(
		f"Config -> approach={config.get('approach_id')} | window_size={inference_config['window_size']} | "
		f"stride={inference_config['stride']} | temperature={inference_config['temperature']}"
	)
	print(
		f"Thresholds -> topic={inference_config['topic_threshold']} | zscore={inference_config['zscore_threshold']} | "
		f"percentile={inference_config['percentile_threshold']} | smooth_window={inference_config['drift_smoothing_window']}"
	)
	print(f"Topics to run: {normalized_topics}")
	print(f"CSV files detected: {len(csv_files)}")

	for file_index, user_csv_file in enumerate(csv_files, 1):
		print("\n" + "=" * 120)
		print(f"FILE {file_index}/{len(csv_files)}: {user_csv_file.name}")
		print("=" * 120)

		inference_results_by_topic = {}
		for selected_topic in normalized_topics:
			print("\n" + "#" * 120)
			print(f"TOPIC: {selected_topic}")
			print("#" * 120)

			result = runner_fn(
				user_csv_path=str(user_csv_file),
				model=inference_model,
				config=inference_config,
				topic_name=selected_topic,
				topic_embeddings_json_path=topic_embeddings_json_path,
				sbert_model_name=sbert_model_name,
			)
			inference_results_by_topic[selected_topic] = result

			print("Function call order:")
			for step in result.get("call_order", []):
				print(f"- {step}")
			print(f"Sentence-level narrative shifts: {len(result.get('sentence_level_narrative_shifts', []))}")

		output_file_path = Path(output_directory) / f"{user_csv_file.stem}.txt"
		report_lines = _build_topic_report_lines(
			user_csv_file,
			output_file_path,
			normalized_topics,
			inference_results_by_topic,
		)
		with open(output_file_path, "w", encoding="utf-8") as f:
			f.write("\n".join(report_lines) + "\n")

		topic_shift_counts = {
			topic_name: len(result.get("sentence_level_narrative_shifts", []))
			for topic_name, result in inference_results_by_topic.items()
		}
		all_batch_inference_outputs.append(
			{
				"input_file": str(user_csv_file),
				"output_file": str(output_file_path),
				"results_by_topic": inference_results_by_topic,
				"topic_shift_counts": topic_shift_counts,
			}
		)

		print("Saved output TXT file:")
		print(output_file_path)

	print("\nBatch inference completed.")
	print(f"Total processed files: {len(all_batch_inference_outputs)}")
	return all_batch_inference_outputs


def _build_topic_report_lines(user_csv_file, output_file_path, normalized_topics, inference_results_by_topic):
	topic_shift_counts = {
		topic_name: len(result.get("sentence_level_narrative_shifts", []))
		for topic_name, result in inference_results_by_topic.items()
	}

	report_lines = []
	report_lines.append("USER INFERENCE FINAL OUTPUT")
	report_lines.append("=" * 100)
	report_lines.append("")
	report_lines.append(f"FILE: {os.path.basename(str(user_csv_file))}")
	report_lines.append(f"Input:  {str(user_csv_file)}")
	report_lines.append(f"Output: {str(output_file_path)}")
	report_lines.append("=" * 100)
	report_lines.append("")
	report_lines.append("NARRATIVE SHIFT COUNT BY TOPIC")
	report_lines.append("-" * 100)
	for topic_name in normalized_topics:
		report_lines.append(f"{topic_name}: {topic_shift_counts.get(topic_name, 0)}")

	topic_map = inference_results_by_topic
	report_lines.append("")
	report_lines.append("\n" + "=" * 100)
	report_lines.append(f"FILE: {os.path.basename(str(user_csv_file))}")
	report_lines.append(f"Input:  {str(user_csv_file)}")
	report_lines.append(f"Output: {str(output_file_path)}")
	report_lines.append("=" * 100)

	if not topic_map:
		report_lines.append("No topic results found for this file.")
		return report_lines

	report_lines.append(f"Topics in result: {list(topic_map.keys())}")
	for topic_name, result in topic_map.items():
		shifts = result.get("sentence_level_narrative_shifts", [])
		report_lines.append("\n" + "#" * 100)
		report_lines.append(f"TOPIC: {topic_name} | Total sentence-level shifts: {len(shifts)}")
		report_lines.append("#" * 100)

		if not shifts:
			report_lines.append("No sentence-level shifts detected. Try lowering topic/zscore thresholds.")
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

			report_lines.append(f"\nShift #{i}: {date_1} -> {date_2}")
			report_lines.append(f"similarity={similarity:.4f} | shift_score={shift_score:.4f} | day_z={day_z:.4f}")
			report_lines.append(f"\nDay 1 - {sentence_id_1} (Article {article_id_1}, Sentence {sentence_num_1})")
			report_lines.append(f"topic_weight={topic_weight_1:.3f}")
			report_lines.append(context_1)
			report_lines.append(f"\nDay 2 - {sentence_id_2} (Article {article_id_2}, Sentence {sentence_num_2})")
			report_lines.append(f"topic_weight={topic_weight_2:.3f}")
			report_lines.append(context_2)
			report_lines.append("-" * 100)

	return report_lines


def print_batch_inference_outputs(all_batch_inference_outputs):
	print("USER INFERENCE FINAL OUTPUT")
	print("=" * 100)

	if not all_batch_inference_outputs:
		print("Run the batch user inference call first.")
		return

	print(f"Total files in batch result: {len(all_batch_inference_outputs)}")
	for file_idx, file_result in enumerate(all_batch_inference_outputs, 1):
		input_file = file_result.get("input_file", "")
		output_file = file_result.get("output_file", "")
		topic_map = file_result.get("results_by_topic", {})

		print("\n" + "=" * 100)
		print(f"FILE {file_idx}: {os.path.basename(str(input_file))}")
		print(f"Input:  {input_file}")
		print(f"Output: {output_file}")
		print("=" * 100)

		if not topic_map:
			print("No topic results found for this file.")
			continue

		print(f"Topics in result: {list(topic_map.keys())}")
		for topic_name, result in topic_map.items():
			shifts = result.get("sentence_level_narrative_shifts", [])
			print("\n" + "#" * 100)
			print(f"TOPIC: {topic_name} | Total sentence-level shifts: {len(shifts)}")
			print("#" * 100)


def run_batch_inference_approach1(
	*,
	config,
	input_directory,
	output_directory,
	topic_embeddings_json_path,
	selected_topics=None,
	inference_overrides=None,
	model_variant="best",
	sbert_model_name="all-mpnet-base-v2",
):
	return _run_batch_inference_a12_common(
		config=config,
		input_directory=input_directory,
		output_directory=output_directory,
		topic_embeddings_json_path=topic_embeddings_json_path,
		selected_topics=selected_topics or config["topics"],
		inference_overrides=inference_overrides or {},
		model_variant=model_variant,
		sbert_model_name=sbert_model_name,
		legacy_best_filename="tcl_model_best_new_1.pt",
		runner_fn=run_user_level_inference_approach1,
	)


def _load_inference_model_a12(config, model_variant, legacy_best_filename):
	inference_config = dict(config)
	inference_config["load_variant"] = model_variant
	inference_config.update(build_artifact_paths(inference_config))

	legacy_best_path = os.path.join(inference_config["output_path"], legacy_best_filename)
	candidate_paths = [
		inference_config["model_load_path"],
		inference_config.get("model_evaluated_path"),
		legacy_best_path,
	]
	checkpoint_path = next((p for p in candidate_paths if p and os.path.exists(p)), None)
	if checkpoint_path is None:
		raise FileNotFoundError("Checkpoint not found in candidates: " + ", ".join([str(p) for p in candidate_paths if p]))

	inference_device = torch.device("cpu")
	inference_model = TCLTemporalEncoderA12(config).to(inference_device)
	checkpoint = load_checkpoint_compat(checkpoint_path, inference_device)
	state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
	inference_model.load_state_dict(state_dict)
	inference_model.eval()
	return inference_model


def run_user_inference_approach1(
	user_csv_path,
	topic_embeddings_json_path,
	topic_name,
	config,
	model_variant="best",
	sbert_model_name="all-mpnet-base-v2",
):
	model = _load_inference_model_a12(config, model_variant=model_variant, legacy_best_filename="tcl_model_best_new_1.pt")
	return run_user_level_inference_approach1(
		user_csv_path=user_csv_path,
		model=model,
		config=config,
		topic_name=topic_name,
		topic_embeddings_json_path=topic_embeddings_json_path,
		sbert_model_name=sbert_model_name,
	)


def run_user_level_inference_approach2(
	user_csv_path,
	model,
	config,
	topic_name,
	topic_embeddings_json_path,
	sbert_model_name="all-mpnet-base-v2",
):
	return _run_user_level_inference_a12(
		user_csv_path=user_csv_path,
		model=model,
		config=config,
		topic_name=topic_name,
		topic_embeddings_json_path=topic_embeddings_json_path,
		sbert_model_name=sbert_model_name,
		min_sentences_per_day=1,
		include_end_date=True,
		include_num_days=True,
		top_k_shifts=5,
	)


def run_batch_inference_approach2(
	*,
	config,
	input_directory,
	output_directory,
	topic_embeddings_json_path,
	selected_topics=None,
	inference_overrides=None,
	model_variant="best",
	sbert_model_name="all-mpnet-base-v2",
):
	return _run_batch_inference_a12_common(
		config=config,
		input_directory=input_directory,
		output_directory=output_directory,
		topic_embeddings_json_path=topic_embeddings_json_path,
		selected_topics=selected_topics or config["topics"],
		inference_overrides=inference_overrides or {},
		model_variant=model_variant,
		sbert_model_name=sbert_model_name,
		legacy_best_filename="tcl_model_best_new_2.pt",
		runner_fn=run_user_level_inference_approach2,
	)


def run_user_inference_approach2(
	user_csv_path,
	topic_embeddings_json_path,
	topic_name,
	config,
	model_variant="best",
	sbert_model_name="all-mpnet-base-v2",
):
	model = _load_inference_model_a12(config, model_variant=model_variant, legacy_best_filename="tcl_model_best_new_2.pt")
	return run_user_level_inference_approach2(
		user_csv_path=user_csv_path,
		model=model,
		config=config,
		topic_name=topic_name,
		topic_embeddings_json_path=topic_embeddings_json_path,
		sbert_model_name=sbert_model_name,
	)


def compute_topic_drift_a4(model, topic_windows, config, device):
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

	shift_scores_raw = []
	for i in range(len(window_embeddings) - 1):
		sim = float(np.dot(window_embeddings[i], window_embeddings[i + 1]))
		shift_scores_raw.append(np.float32(1.0 - sim))

	shift_scores_raw = np.asarray(shift_scores_raw, dtype=np.float32)
	score_min = float(np.min(shift_scores_raw))
	score_max = float(np.max(shift_scores_raw))
	if (score_max - score_min) > 1e-8:
		shift_scores = ((shift_scores_raw - score_min) / (score_max - score_min)).astype(np.float32)
	else:
		shift_scores = np.zeros_like(shift_scores_raw, dtype=np.float32)

	threshold = float(config.get("manual_shift_threshold", 0.1))
	mean_score = np.float32(np.mean(shift_scores))
	std_score = np.float32(np.std(shift_scores)) + np.float32(1e-8)
	z_scores = ((shift_scores - mean_score) / std_score).astype(np.float32)

	drift_rows = []
	for i, (raw_score, score, z_score) in enumerate(zip(shift_scores_raw, shift_scores, z_scores)):
		drift_rows.append(
			{
				"window_idx": int(i),
				"date": topic_windows[i + 1]["start_date"],
				"raw_drift": float(raw_score),
				"drift_score": float(score),
				"z_score": float(z_score),
				"threshold": float(threshold),
			}
		)

	return drift_rows, window_embeddings


def detect_shifts_a4(drift_rows, config):
	if not drift_rows:
		return []
	threshold = float(drift_rows[0].get("threshold", 0.0))
	shifts = []
	for row in drift_rows:
		if float(row.get("drift_score", 0.0)) > threshold:
			shifts.append(row)
	return shifts


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


def extract_sentence_level_narrative_shifts_a4(
	filtered_sentence_dataframe,
	drift_rows,
	config,
	top_k_shifts=5,
	per_date_sent_limit=40,
	context_window=2,
):
	if filtered_sentence_dataframe.empty or not drift_rows:
		return []

	detected_shifts = detect_shifts_a4(drift_rows, config)
	if not detected_shifts:
		return []

	filtered = filtered_sentence_dataframe.copy()
	filtered["date"] = pd.to_datetime(filtered["date"]).dt.normalize()

	unique_dates = sorted(filtered["date"].unique())
	if len(unique_dates) < 2:
		return []

	sentence_level_shifts = []
	used_sentence_ids = set()
	used_sentence_pairs = set()
	ranked_shifts = sorted(detected_shifts, key=lambda x: x.get("z_score", 0.0), reverse=True)[: int(top_k_shifts)]

	for shift in ranked_shifts:
		date_2 = pd.Timestamp(shift["date"]).normalize()
		previous_dates = [date_value for date_value in unique_dates if date_value < date_2]
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

		candidate_flat_indices = np.argsort(sims, axis=None)
		chosen = None
		for flat_idx in candidate_flat_indices:
			idx_1, idx_2 = np.unravel_index(int(flat_idx), sims.shape)
			cand_1 = sents_1.iloc[idx_1]
			cand_2 = sents_2.iloc[idx_2]

			sentence_id_1 = str(cand_1["sentence_id"])
			sentence_id_2 = str(cand_2["sentence_id"])
			pair_key = tuple(sorted((sentence_id_1, sentence_id_2)))

			if sentence_id_1 in used_sentence_ids or sentence_id_2 in used_sentence_ids:
				continue
			if pair_key in used_sentence_pairs:
				continue

			chosen = (idx_1, idx_2, sentence_id_1, sentence_id_2, pair_key)
			break

		if chosen is None:
			continue

		idx_1, idx_2, sentence_id_1, sentence_id_2, pair_key = chosen
		min_similarity = float(sims[idx_1, idx_2])

		sent1 = sents_1.iloc[idx_1]
		sent2 = sents_2.iloc[idx_2]

		sentence_level_shifts.append(
			{
				"date_1": str(pd.Timestamp(date_1).date()),
				"date_2": str(pd.Timestamp(date_2).date()),
				"sentence_id_1": sentence_id_1,
				"article_id_1": _to_int_article_id(sent1["article_id"]),
				"sentence_num_1": int(sent1["sentence_order"]),
				"sentence_1": str(sent1["sentence_text"]),
				"topic_weight_1": float(sent1["similarity_score"]),
				"sentence_id_2": sentence_id_2,
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

		used_sentence_ids.add(sentence_id_1)
		used_sentence_ids.add(sentence_id_2)
		used_sentence_pairs.add(pair_key)

	return sentence_level_shifts


def run_user_level_inference_approach4(
	user_csv_path,
	model,
	config,
	topic_name,
	ideal_topic_embeddings_json_path=None,
	topic_embeddings_json_path=None,
	tcl_topic_embeddings_json_path=None,
	sbert_model_name="all-mpnet-base-v2",
):
	call_order = _get_user_inference_call_order_ap45()
	resolved_topic = resolve_topic_name(topic_name, config["topics"])

	ideal_path = ideal_topic_embeddings_json_path or topic_embeddings_json_path
	if not ideal_path:
		raise ValueError("Provide ideal_topic_embeddings_json_path (or topic_embeddings_json_path for backward compatibility)")

	input_dataframe = pd.read_csv(user_csv_path)
	sentence_dataframe = split_articles_into_sentences(input_dataframe)
	if sentence_dataframe.empty:
		return {
			"call_order": call_order,
			"resolved_topic": resolved_topic,
			"sentence_level_narrative_shifts": [],
			"top_topic_sentences": [],
			"topic_score_rows": [],
			"training_like_rows": [],
		}

	sentence_dataframe = build_context_texts(sentence_dataframe, int(config["context_window"]))
	sentence_dataframe = generate_contextual_sbert_embeddings(
		sentence_dataframe,
		config,
		sbert_model_name=sbert_model_name,
	)

	ideal_embeddings = load_ideal_topic_embeddings_for_inference(ideal_path, config)
	tcl_topic_embeddings = load_tcl_topic_embeddings_for_inference(tcl_topic_embeddings_json_path, config)
	labeled_sentence_dataframe = soft_topic_label_sentences(sentence_dataframe, ideal_embeddings, config)

	sentence_matrix = np.stack(labeled_sentence_dataframe["sentence_embeddings"].values).astype(np.float32)
	selected_topic_embedding = ideal_embeddings[resolved_topic]
	topic_weights = compute_topic_similarity_with_embeddings(sentence_matrix, selected_topic_embedding)

	labeled_sentence_dataframe = labeled_sentence_dataframe.copy()
	labeled_sentence_dataframe["similarity_score"] = topic_weights
	labeled_sentence_dataframe["selected_topic"] = resolved_topic
	topic_score_rows = build_topic_score_rows(labeled_sentence_dataframe, config)

	threshold = float(config.get("topic_threshold", config.get("topic_weight_threshold", 0.55)))
	filtered_sentence_dataframe = labeled_sentence_dataframe[
		labeled_sentence_dataframe["similarity_score"].astype(np.float32) >= threshold
	].copy().sort_values(["date", "article_id", "sentence_order"]).reset_index(drop=True)

	if filtered_sentence_dataframe.empty:
		return {
			"call_order": call_order,
			"resolved_topic": resolved_topic,
			"sentence_level_narrative_shifts": [],
			"top_topic_sentences": [],
			"topic_score_rows": topic_score_rows,
			"training_like_rows": labeled_sentence_dataframe.to_dict(orient="records"),
		}

	topic_vec_64 = np.asarray(tcl_topic_embeddings[resolved_topic], dtype=np.float32)
	filtered_sentence_dataframe["topic_embeddings"] = [topic_vec_64.copy() for _ in range(len(filtered_sentence_dataframe))]

	daily_rows = []
	grouped = filtered_sentence_dataframe.groupby(pd.to_datetime(filtered_sentence_dataframe["date"]).dt.normalize())
	for date_value, group in grouped:
		embeddings = np.stack(group["embedding"].values).astype(np.float32)
		raw_weights = np.clip(group["similarity_score"].astype(np.float32).values, a_min=0.0, a_max=None)
		if raw_weights.sum() > 0:
			weights = raw_weights / raw_weights.sum()
		else:
			weights = np.ones(len(group), dtype=np.float32) / max(len(group), 1)

		daily_embedding = (embeddings.T @ weights).astype(np.float32)
		daily_embedding = daily_embedding / (np.linalg.norm(daily_embedding) + 1e-8)

		daily_rows.append(
			{
				"date": pd.Timestamp(date_value),
				"daily_vectors": daily_embedding,
				"topic_embeddings": topic_vec_64.copy(),
				"topic_name": resolved_topic,
				"topic_id": int(config["topics"].index(resolved_topic)),
				"num_sentences": int(len(group)),
			}
		)

	user_daily_df = pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)
	if user_daily_df.empty:
		return {
			"call_order": call_order,
			"resolved_topic": resolved_topic,
			"sentence_level_narrative_shifts": [],
			"top_topic_sentences": [],
			"topic_score_rows": topic_score_rows,
			"training_like_rows": labeled_sentence_dataframe.to_dict(orient="records"),
		}

	user_records = build_temporal_feature_records(
		user_daily_df,
		include_tau=False,
		include_end_date=True,
		include_num_sentences=True,
		include_num_days=True,
	)

	if len(user_records) < int(config["window_size"]):
		user_windows = []
		for idx, record in enumerate(user_records):
			padded_tensor = np.stack([record["final_vector"]] * int(config["window_size"])).astype(np.float32)
			user_windows.append(
				{
					"tensor": padded_tensor,
					"topic_id": int(config["topics"].index(resolved_topic)),
					"topic_name": resolved_topic,
					"start_date": record["date"],
					"end_date": record.get("end_date", record["date"]),
					"window_idx": idx,
					"is_adaptive": True,
				}
			)
	else:
		user_windows = build_window_embeddings(
			user_records,
			resolved_topic,
			int(config["topics"].index(resolved_topic)),
			config,
		)
		for window in user_windows:
			window["is_adaptive"] = False

	if len(user_windows) < 2:
		return {
			"call_order": call_order,
			"resolved_topic": resolved_topic,
			"sentence_level_narrative_shifts": [],
			"top_topic_sentences": filtered_sentence_dataframe.sort_values("similarity_score", ascending=False)
			.head(20)[["date", "sentence_id", "sentence_text", "similarity_score"]]
			.to_dict(orient="records"),
			"topic_score_rows": topic_score_rows,
			"training_like_rows": labeled_sentence_dataframe.to_dict(orient="records"),
		}

	model_device = next(model.parameters()).device
	drift_rows, _ = compute_topic_drift_a4(model, user_windows, config, model_device)

	sentence_level_shifts = extract_sentence_level_narrative_shifts_a4(
		filtered_sentence_dataframe=filtered_sentence_dataframe,
		drift_rows=drift_rows,
		config=config,
		top_k_shifts=5,
		per_date_sent_limit=40,
		context_window=2,
	)

	top_topic_sentences = (
		filtered_sentence_dataframe.sort_values("similarity_score", ascending=False)
		.head(20)[["date", "sentence_id", "sentence_text", "similarity_score"]]
		.to_dict(orient="records")
	)

	return {
		"call_order": call_order,
		"resolved_topic": resolved_topic,
		"sentence_level_narrative_shifts": sentence_level_shifts,
		"top_topic_sentences": top_topic_sentences,
		"topic_score_rows": topic_score_rows,
		"training_like_rows": labeled_sentence_dataframe.to_dict(orient="records"),
	}


def run_batch_inference_approach4(
	*,
	config,
	input_directory,
	output_directory,
	ideal_topic_embeddings_json_path,
	tcl_topic_embeddings_json_path=None,
	selected_topics=None,
	inference_overrides=None,
	model_variant="best",
	sbert_model_name="all-mpnet-base-v2",
):
	selected_topics = selected_topics or config["topics"]
	inference_overrides = inference_overrides or {}

	inference_config = dict(config)
	inference_config.update(inference_overrides)
	inference_config["load_variant"] = model_variant
	inference_config.update(build_artifact_paths(inference_config))

	Path(output_directory).mkdir(parents=True, exist_ok=True)

	legacy_best_path = os.path.join(inference_config["output_path"], "tcl_model_best_new_2.pt")
	candidate_paths = [
		inference_config["model_load_path"],
		inference_config.get("model_evaluated_path"),
		legacy_best_path,
	]
	checkpoint_path = next((p for p in candidate_paths if p and os.path.exists(p)), None)
	if checkpoint_path is None:
		raise FileNotFoundError("Checkpoint not found in candidates: " + ", ".join([str(p) for p in candidate_paths if p]))

	inference_device = torch.device("cpu")
	inference_model = TCLTemporalEncoderA4(config).to(inference_device)
	checkpoint = load_checkpoint_compat(checkpoint_path, inference_device)
	state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

	try:
		inference_model.load_state_dict(state_dict)
	except RuntimeError as exc:
		raise RuntimeError(
			"Checkpoint architecture mismatch after AP4-parity update. "
			"Run the training cell again to produce a new compatible checkpoint, then re-run inference. "
			f"Original load error: {exc}"
		)

	inference_model.eval()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	normalized_topics = _normalize_selected_topics(selected_topics, config["topics"])

	input_dir = Path(input_directory)
	if not input_dir.exists():
		raise FileNotFoundError(f"Input directory not found: {input_dir}")

	csv_files = sorted(input_dir.glob("*.csv"))
	if not csv_files:
		raise FileNotFoundError(f"No CSV files found in: {input_dir}")

	all_batch_inference_outputs = []

	print("Batch inference started")
	print(f"Model variant requested: {inference_config['load_variant']}")
	print(f"Checkpoint loaded: {checkpoint_path}")
	print(f"Input directory: {input_dir}")
	print(f"Output directory: {output_directory}")
	print(f"Ideal topic embeddings (768-d): {ideal_topic_embeddings_json_path}")
	print(
		"TCL topic embeddings (64-d): "
		+ (tcl_topic_embeddings_json_path if tcl_topic_embeddings_json_path else "config['topic_embedding_table'] fallback")
	)
	print(
		f"Config -> approach={config.get('approach_id')} | window_size={inference_config['window_size']} | "
		f"stride={inference_config['stride']} | temperature={inference_config['temperature']}"
	)
	print(
		f"Thresholds -> topic={inference_config['topic_threshold']} | "
		f"manual_shift={inference_config['manual_shift_threshold']}"
	)
	print(f"Topics to run: {normalized_topics}")
	print(f"CSV files detected: {len(csv_files)}")

	for file_index, user_csv_file in enumerate(csv_files, 1):
		print("\n" + "=" * 120)
		print(f"FILE {file_index}/{len(csv_files)}: {user_csv_file.name}")
		print("=" * 120)

		inference_results_by_topic = {}
		for selected_topic in normalized_topics:
			print("\n" + "#" * 120)
			print(f"TOPIC: {selected_topic}")
			print("#" * 120)

			result = run_user_level_inference_approach4(
				user_csv_path=str(user_csv_file),
				model=inference_model,
				config=inference_config,
				topic_name=selected_topic,
				ideal_topic_embeddings_json_path=ideal_topic_embeddings_json_path,
				tcl_topic_embeddings_json_path=tcl_topic_embeddings_json_path,
				sbert_model_name=sbert_model_name,
			)
			inference_results_by_topic[selected_topic] = result

			print("Function call order:")
			for step in result.get("call_order", []):
				print(f"- {step}")
			print(f"Sentence-level narrative shifts: {len(result.get('sentence_level_narrative_shifts', []))}")

		output_file_path = Path(output_directory) / f"{user_csv_file.stem}.txt"
		report_lines = _build_topic_report_lines(
			user_csv_file,
			output_file_path,
			normalized_topics,
			inference_results_by_topic,
		)
		with open(output_file_path, "w", encoding="utf-8") as f:
			f.write("\n".join(report_lines) + "\n")

		topic_shift_counts = {
			topic_name: len(result.get("sentence_level_narrative_shifts", []))
			for topic_name, result in inference_results_by_topic.items()
		}
		all_batch_inference_outputs.append(
			{
				"input_file": str(user_csv_file),
				"output_file": str(output_file_path),
				"results_by_topic": inference_results_by_topic,
				"topic_shift_counts": topic_shift_counts,
			}
		)

		print("Saved output TXT file:")
		print(output_file_path)

	print("\nBatch inference completed.")
	print(f"Total processed files: {len(all_batch_inference_outputs)}")
	return all_batch_inference_outputs


def run_user_inference_approach4(
	user_csv_path,
	ideal_topic_embeddings_json_path,
	topic_name,
	config,
	tcl_topic_embeddings_json_path=None,
	model_variant="best",
	sbert_model_name="all-mpnet-base-v2",
):
	inference_config = dict(config)
	inference_config["load_variant"] = model_variant
	inference_config.update(build_artifact_paths(inference_config))

	legacy_best_path = os.path.join(inference_config["output_path"], "tcl_model_best_new_2.pt")
	candidate_paths = [
		inference_config["model_load_path"],
		inference_config.get("model_evaluated_path"),
		legacy_best_path,
	]
	checkpoint_path = next((p for p in candidate_paths if p and os.path.exists(p)), None)
	if checkpoint_path is None:
		raise FileNotFoundError("Checkpoint not found in candidates: " + ", ".join([str(p) for p in candidate_paths if p]))

	inference_device = torch.device("cpu")
	inference_model = TCLTemporalEncoderA4(config).to(inference_device)
	checkpoint = load_checkpoint_compat(checkpoint_path, inference_device)
	state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
	inference_model.load_state_dict(state_dict)
	inference_model.eval()

	return run_user_level_inference_approach4(
		user_csv_path=user_csv_path,
		model=inference_model,
		config=config,
		topic_name=topic_name,
		ideal_topic_embeddings_json_path=ideal_topic_embeddings_json_path,
		tcl_topic_embeddings_json_path=tcl_topic_embeddings_json_path,
		sbert_model_name=sbert_model_name,
	)


def _cfg_get(config_obj, key, default=None):
	if isinstance(config_obj, dict):
		return config_obj.get(key, default)
	return getattr(config_obj, key, default)


def build_inference_config_approach5(config_obj, topic_embedding_table):
	return {
		"topics": list(_cfg_get(config_obj, "TOPICS", _cfg_get(config_obj, "topics", []))),
		"context_window": 5,
		"embedding_dim": int(_cfg_get(config_obj, "EMBEDDING_DIM", _cfg_get(config_obj, "embedding_dim", 768))),
		"sentence_final_dim": int(_cfg_get(config_obj, "SENTENCE_FINAL_DIM", _cfg_get(config_obj, "sentence_final_dim", 832))),
		"topic_embedding_dim": int(_cfg_get(config_obj, "TOPIC_EMB_DIM", _cfg_get(config_obj, "topic_embedding_dim", 64))),
		"concat_dim": int(_cfg_get(config_obj, "CONCAT_DIM", _cfg_get(config_obj, "concat_dim", 896))),
		"topic_embedding_table": np.asarray(topic_embedding_table, dtype=np.float32),
		"window_size": int(_cfg_get(config_obj, "WINDOW_SIZE", _cfg_get(config_obj, "window_size", 3))),
		"stride": int(_cfg_get(config_obj, "WINDOW_STRIDE", _cfg_get(config_obj, "stride", 1))),
		"temperature": float(_cfg_get(config_obj, "TEMPERATURE", _cfg_get(config_obj, "temperature", 0.07))),
		"topic_threshold": 0.60,
		"topic_weight_threshold": 0.60,
		"manual_shift_threshold": float(_cfg_get(config_obj, "SHIFT_THRESHOLD", _cfg_get(config_obj, "manual_shift_threshold", 0.1))),
		"inference_batch_size": 32,
	}


def compute_topic_drift_a5(model, user_windows, config_dict, device):
	model.eval()
	if len(user_windows) < 2:
		return [], None

	with torch.no_grad():
		embeddings = []
		for window in user_windows:
			tensor = torch.from_numpy(window["tensor"]).unsqueeze(0).to(device)
			emb = model(tensor).cpu().numpy()[0].astype(np.float32)
			embeddings.append(emb)

	drift_rows = []
	for i in range(len(embeddings) - 1):
		drift_score = float(np.linalg.norm(embeddings[i + 1] - embeddings[i]))
		drift_rows.append(
			{
				"date": str(pd.Timestamp(user_windows[i + 1]["end_date"]).date()),
				"drift_score": drift_score,
				"window_idx": i,
				"topic_name": user_windows[i + 1]["topic_name"],
			}
		)
	return drift_rows, embeddings


def detect_shifts_a5(drift_rows, config_dict):
	if not drift_rows:
		return []

	scores = np.asarray([row["drift_score"] for row in drift_rows], dtype=np.float32)
	mean_val = float(scores.mean())
	std_val = float(scores.std())
	std_val = std_val if std_val > 1e-8 else 1.0

	threshold = float(config_dict.get("manual_shift_threshold", 0.1))
	detected = []
	for row in drift_rows:
		z_score = float((row["drift_score"] - mean_val) / std_val)
		if row["drift_score"] >= threshold:
			item = dict(row)
			item["z_score"] = z_score
			detected.append(item)
	return detected


def extract_sentence_level_narrative_shifts_a5(
	filtered_sentence_dataframe,
	drift_rows,
	config_dict,
	top_k_shifts=5,
	context_source_dataframe=None,
):
	detected_shifts = detect_shifts_a5(drift_rows, config_dict)
	if filtered_sentence_dataframe.empty or not detected_shifts:
		return []

	filtered = filtered_sentence_dataframe.copy()
	filtered["date"] = pd.to_datetime(filtered["date"]).dt.normalize()

	if context_source_dataframe is None or context_source_dataframe.empty:
		context_source = filtered.copy()
	else:
		context_source = context_source_dataframe.copy()

	unique_dates = sorted(filtered["date"].unique())
	sentence_level_shifts = []
	ranked_shifts = sorted(detected_shifts, key=lambda x: x.get("z_score", 0.0), reverse=True)[: int(top_k_shifts)]

	for shift in ranked_shifts:
		date_2 = pd.Timestamp(shift["date"]).normalize()
		previous_dates = [d for d in unique_dates if d < date_2]
		if not previous_dates:
			continue
		date_1 = previous_dates[-1]

		sents_1 = filtered[filtered["date"] == date_1].sort_values("similarity_score", ascending=False).head(40)
		sents_2 = filtered[filtered["date"] == date_2].sort_values("similarity_score", ascending=False).head(40)
		if sents_1.empty or sents_2.empty:
			continue

		embs_1 = np.stack(sents_1["sentence_embeddings"].values).astype(np.float32)
		embs_2 = np.stack(sents_2["sentence_embeddings"].values).astype(np.float32)
		norm_1 = embs_1 / (np.linalg.norm(embs_1, axis=1, keepdims=True) + 1e-8)
		norm_2 = embs_2 / (np.linalg.norm(embs_2, axis=1, keepdims=True) + 1e-8)
		sims = np.dot(norm_1, norm_2.T).astype(np.float32)
		idx_1, idx_2 = np.unravel_index(int(np.argmin(sims)), sims.shape)

		sent1 = sents_1.reset_index(drop=True).iloc[idx_1]
		sent2 = sents_2.reset_index(drop=True).iloc[idx_2]
		min_similarity = float(sims[idx_1, idx_2])

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
				"context_1": _build_sentence_context_string(sent1, context_source, context_window=2),
				"context_2": _build_sentence_context_string(sent2, context_source, context_window=2),
				"similarity": min_similarity,
				"shift_score": float(1.0 - min_similarity),
				"day_level_shift_score": float(shift.get("drift_score", 0.0)),
				"day_level_z_score": float(shift.get("z_score", 0.0)),
			}
		)

	return sentence_level_shifts


def _get_user_inference_call_order_ap45():
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


def get_user_inference_call_order_approach5():
	return _get_user_inference_call_order_ap45()


def run_user_level_inference_approach5(
	user_csv_path,
	model,
	config_dict,
	topic_name,
	ideal_topic_embeddings_json_path,
	entity_proj_layer,
	nlp,
	sbert_model_name="all-mpnet-base-v2",
	sbert_model=None,
	device=None,
	entity_lambda=0.5,
):
	call_order = get_user_inference_call_order_approach5()
	resolved_topic = resolve_topic_name(topic_name, config_dict["topics"])
	input_dataframe = pd.read_csv(user_csv_path)
	sentence_dataframe = split_articles_into_sentences(input_dataframe)
	if sentence_dataframe.empty:
		return {
			"call_order": call_order,
			"resolved_topic": resolved_topic,
			"sentence_level_narrative_shifts": [],
			"top_topic_sentences": [],
			"topic_score_rows": [],
			"training_like_rows": [],
		}

	sentence_dataframe = build_context_texts(sentence_dataframe, int(config_dict["context_window"]))
	sentence_dataframe = generate_contextual_sbert_embeddings(sentence_dataframe, config_dict, sbert_model_name=sbert_model_name)

	if sbert_model is None:
		from sentence_transformers import SentenceTransformer

		sbert_model = SentenceTransformer(sbert_model_name, device="cpu")

	sentence_dataframe = sentence_dataframe.copy()
	sentence_dataframe["center_sentence"] = sentence_dataframe["sentence_text"]
	sentence_dataframe["main_sentence"] = sentence_dataframe["sentence_text"]
	sentence_dataframe["embedding"] = sentence_dataframe["sentence_embeddings"]

	sentence_dataframe = approach5_extract_entities_batch(sentence_dataframe, nlp, batch_size=int(config_dict.get("inference_batch_size", 32)))
	sentence_dataframe = approach5_add_entity_embeddings(
		sentence_dataframe,
		sbert_model,
		embedding_dim=int(config_dict.get("embedding_dim", 768)),
	)

	proj_device = device or next(entity_proj_layer.parameters()).device
	sentence_dataframe = compute_entity_invariant_embeddings(
		sentence_dataframe,
		entity_proj_layer=entity_proj_layer,
		device=proj_device,
		lambda_=float(entity_lambda),
		embedding_column="embedding",
		entity_embedding_column="entity_embedding",
	)

	ideal_embeddings = load_topic_embedding_prototypes(ideal_topic_embeddings_json_path, config_dict)
	labeled_sentence_dataframe = soft_topic_label_sentences(sentence_dataframe, ideal_embeddings, config_dict)

	if "final_embedding" not in labeled_sentence_dataframe.columns and "final_embedding" in sentence_dataframe.columns:
		final_lookup = sentence_dataframe[["sentence_id", "final_embedding"]].drop_duplicates(subset=["sentence_id"]).copy()
		labeled_sentence_dataframe = labeled_sentence_dataframe.merge(final_lookup, on="sentence_id", how="left")

	if "final_embedding" not in labeled_sentence_dataframe.columns:
		raise KeyError("final_embedding missing after soft labeling. Re-run entity-cleaning cells before inference.")

	sentence_matrix = np.stack(labeled_sentence_dataframe["sentence_embeddings"].values).astype(np.float32)
	selected_topic_embedding = ideal_embeddings[resolved_topic]
	topic_weights = compute_topic_similarity_with_embeddings(sentence_matrix, selected_topic_embedding)

	labeled_sentence_dataframe = labeled_sentence_dataframe.copy()
	labeled_sentence_dataframe["similarity_score"] = topic_weights
	labeled_sentence_dataframe["selected_topic"] = resolved_topic
	topic_score_rows = build_topic_score_rows(labeled_sentence_dataframe, config_dict)

	threshold = float(config_dict.get("topic_threshold", config_dict.get("topic_weight_threshold", 0.60)))
	topic_weight_threshold = float(config_dict.get("topic_weight_threshold", threshold))

	topic_filtered_source = labeled_sentence_dataframe.copy()
	if resolved_topic in topic_filtered_source.columns:
		topic_filtered_source = topic_filtered_source[
			pd.to_numeric(topic_filtered_source[resolved_topic], errors="coerce").fillna(0.0).astype(np.float32)
			>= topic_weight_threshold
		].copy()

	filtered_sentence_dataframe = topic_filtered_source[
		topic_filtered_source["similarity_score"].astype(np.float32) >= threshold
	].copy().sort_values(["date", "article_id", "sentence_order"]).reset_index(drop=True)

	dedupe_keys = [col for col in ["date", "article_id", "sentence_id", "sentence_order"] if col in filtered_sentence_dataframe.columns]
	if dedupe_keys:
		filtered_sentence_dataframe = filtered_sentence_dataframe.drop_duplicates(subset=dedupe_keys, keep="first").reset_index(drop=True)

	if filtered_sentence_dataframe.empty:
		return {
			"call_order": call_order,
			"resolved_topic": resolved_topic,
			"sentence_level_narrative_shifts": [],
			"top_topic_sentences": [],
			"topic_score_rows": topic_score_rows,
			"training_like_rows": labeled_sentence_dataframe.to_dict(orient="records"),
		}

	topic_id = int(config_dict["topics"].index(resolved_topic))
	topic_vec_64 = np.asarray(config_dict["topic_embedding_table"][topic_id], dtype=np.float32)

	expected_sentence_dim = int(config_dict.get("sentence_final_dim", 832))
	expected_topic_dim = int(config_dict.get("topic_embedding_dim", 64))
	expected_concat_dim = int(config_dict.get("concat_dim", expected_sentence_dim + expected_topic_dim))

	if filtered_sentence_dataframe["final_embedding"].iloc[0].shape[0] != expected_sentence_dim:
		raise ValueError(
			f"Inference sentence feature dim mismatch: got {filtered_sentence_dataframe['final_embedding'].iloc[0].shape[0]}, expected {expected_sentence_dim}."
		)

	user_daily_df = aggregate_daily_embeddings(
		dataframe=filtered_sentence_dataframe,
		topics=[resolved_topic],
		embedding_column="final_embedding",
		min_sentences_per_day=1,
		weight_column_map={resolved_topic: ["similarity_score"]},
		topic_embeddings_column="__unused_topic_embeddings__",
		fallback_topic_embeddings_map={resolved_topic: topic_vec_64.copy()},
		normalize_date=True,
		require_weight_column=True,
		entity_signature_column=None,
		output_embedding_column="daily_vectors",
		topic_column_name="topic_name",
		include_topic_id=True,
		include_avg_weight=False,
	)

	user_records = build_temporal_feature_records(
		user_daily_df,
		include_tau=False,
		include_end_date=True,
		include_num_sentences=False,
		include_num_days=False,
	)

	if user_records:
		if topic_vec_64.shape[0] != expected_topic_dim:
			raise ValueError(f"Inference topic embedding dim mismatch: got {topic_vec_64.shape[0]}, expected {expected_topic_dim}.")
		if int(user_records[0]["final_vector"].shape[0]) != expected_concat_dim:
			raise ValueError(
				f"Inference concat dim mismatch: got {user_records[0]['final_vector'].shape[0]}, expected {expected_concat_dim}."
			)

	if len(user_records) < int(config_dict["window_size"]):
		user_windows = []
		for idx, record in enumerate(user_records):
			padded_tensor = np.stack([record["final_vector"]] * int(config_dict["window_size"])).astype(np.float32)
			user_windows.append(
				{
					"tensor": padded_tensor,
					"topic_id": topic_id,
					"topic_name": resolved_topic,
					"start_date": record["date"],
					"end_date": record.get("end_date", record["date"]),
					"window_idx": idx,
				}
			)
	else:
		user_windows = build_window_embeddings(user_records, resolved_topic, topic_id, config_dict)

	if user_windows:
		inferred_dim = int(user_windows[0]["tensor"].shape[-1])
		if inferred_dim != expected_concat_dim:
			raise ValueError(f"Window tensor feature dim mismatch: got {inferred_dim}, expected {expected_concat_dim}.")

		model_input_dim = None
		if hasattr(model, "input_projection") and hasattr(model.input_projection, "in_features"):
			model_input_dim = int(model.input_projection.in_features)
		elif hasattr(model, "input_linear") and hasattr(model.input_linear, "in_features"):
			model_input_dim = int(model.input_linear.in_features)
		if model_input_dim is not None and model_input_dim != inferred_dim:
			raise ValueError(f"Model input dim mismatch: model expects {model_input_dim}, inference windows have {inferred_dim}.")

	if len(user_windows) < 2:
		top_topic_sentences = (
			filtered_sentence_dataframe.sort_values("similarity_score", ascending=False)
			.head(20)[["date", "sentence_id", "sentence_text", "similarity_score"]]
			.to_dict(orient="records")
		)
		return {
			"call_order": call_order,
			"resolved_topic": resolved_topic,
			"sentence_level_narrative_shifts": [],
			"top_topic_sentences": top_topic_sentences,
			"topic_score_rows": topic_score_rows,
			"training_like_rows": labeled_sentence_dataframe.to_dict(orient="records"),
		}

	model_device = next(model.parameters()).device
	drift_rows, _ = compute_topic_drift_a5(model, user_windows, config_dict, model_device)

	sentence_level_shifts = extract_sentence_level_narrative_shifts_a5(
		filtered_sentence_dataframe=filtered_sentence_dataframe,
		drift_rows=drift_rows,
		config_dict=config_dict,
		top_k_shifts=5,
		context_source_dataframe=sentence_dataframe,
	)

	top_topic_sentences = (
		filtered_sentence_dataframe.sort_values("similarity_score", ascending=False)
		.head(20)[["date", "sentence_id", "sentence_text", "similarity_score"]]
		.to_dict(orient="records")
	)

	return {
		"call_order": call_order,
		"resolved_topic": resolved_topic,
		"sentence_level_narrative_shifts": sentence_level_shifts,
		"top_topic_sentences": top_topic_sentences,
		"topic_score_rows": topic_score_rows,
		"training_like_rows": labeled_sentence_dataframe.to_dict(orient="records"),
	}


def run_multitopic_inference_approach5(
	*,
	config_obj,
	user_csv_path,
	ideal_topic_embeddings_json_path,
	entity_proj_layer,
	nlp,
	topic_embedding_table,
	selected_topics=None,
	inference_overrides=None,
	model_variant="best",
	sbert_model_name="all-mpnet-base-v2",
	output_json_path=None,
):
	inference_config = build_inference_config_approach5(config_obj, topic_embedding_table)
	if inference_overrides:
		inference_config.update(inference_overrides)
	inference_config["load_variant"] = model_variant

	checkpoint_map = {
		"best": _cfg_get(config_obj, "MODEL_BEST_PATH"),
		"last": _cfg_get(config_obj, "MODEL_LAST_PATH"),
		"evaluated": _cfg_get(config_obj, "MODEL_EVALUATED_PATH"),
	}
	checkpoint_path = checkpoint_map.get(model_variant, checkpoint_map["best"])
	if checkpoint_path is None:
		raise FileNotFoundError("No checkpoint path configured for Approach 5 inference.")

	inference_device = torch.device("cpu")
	inference_model = TCLTemporalEncoderA5(config_obj).to(inference_device)
	checkpoint = torch.load(checkpoint_path, map_location=inference_device, weights_only=False)
	state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
	inference_model.load_state_dict(state_dict)
	inference_model.eval()

	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	selected_topics = selected_topics or list(_cfg_get(config_obj, "TOPICS", inference_config["topics"]))
	normalized_topics = _normalize_selected_topics(selected_topics, inference_config["topics"])

	from sentence_transformers import SentenceTransformer

	inference_sbert = SentenceTransformer(sbert_model_name, device="cpu")

	inference_results_by_topic = {}
	for selected_topic in normalized_topics:
		result = run_user_level_inference_approach5(
			user_csv_path=user_csv_path,
			model=inference_model,
			config_dict=inference_config,
			topic_name=selected_topic,
			ideal_topic_embeddings_json_path=ideal_topic_embeddings_json_path,
			entity_proj_layer=entity_proj_layer,
			nlp=nlp,
			sbert_model_name=sbert_model_name,
			sbert_model=inference_sbert,
			device=inference_device,
			entity_lambda=float(_cfg_get(config_obj, "ENTITY_LAMBDA", 0.5)),
		)
		inference_results_by_topic[selected_topic] = result

	inference_output_payload = {
		"inference_metadata": {
			"load_variant": inference_config["load_variant"],
			"checkpoint_loaded": str(checkpoint_path),
			"thresholds": {
				"topic_threshold": inference_config.get("topic_threshold"),
				"manual_shift_threshold": inference_config.get("manual_shift_threshold"),
			},
		},
		"selected_topics": normalized_topics,
		"results_by_topic": inference_results_by_topic,
	}

	if output_json_path is not None:
		with open(output_json_path, "w", encoding="utf-8") as f:
			json.dump(inference_output_payload, f, indent=2, default=str)

	return inference_output_payload


def _build_topic_embedding_table_approach5(config_obj, ideal_topic_embeddings_json_path):
	topic_names = list(_cfg_get(config_obj, "TOPICS", []))
	topic_dim = int(_cfg_get(config_obj, "TOPIC_EMB_DIM", 64))
	ideal_embeddings = load_topic_embedding_prototypes(ideal_topic_embeddings_json_path, {"topics": topic_names})

	rows = []
	for topic_name in topic_names:
		vec = np.asarray(ideal_embeddings[topic_name], dtype=np.float32).reshape(-1)
		if vec.size >= topic_dim:
			rows.append(vec[:topic_dim])
		else:
			rows.append(np.pad(vec, (0, topic_dim - vec.size), mode="constant"))
	return np.asarray(rows, dtype=np.float32)


def _load_inference_resources_approach5(config_obj, ideal_topic_embeddings_json_path):
	import spacy

	device = torch.device("cpu")
	entity_proj_layer = torch.nn.Linear(int(_cfg_get(config_obj, "EMBEDDING_DIM", 768)), int(_cfg_get(config_obj, "ENTITY_PROJ_DIM", 64))).to(device)
	torch.nn.init.xavier_uniform_(entity_proj_layer.weight)
	torch.nn.init.zeros_(entity_proj_layer.bias)

	nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])
	if "sentencizer" not in nlp.pipe_names and "senter" not in nlp.pipe_names and "parser" not in nlp.pipe_names:
		nlp.add_pipe("sentencizer")

	topic_embedding_table = _build_topic_embedding_table_approach5(config_obj, ideal_topic_embeddings_json_path)
	return entity_proj_layer, nlp, topic_embedding_table


def run_user_inference_approach5(
	user_csv_path,
	ideal_topic_embeddings_json_path,
	topic_name,
	config_obj,
	inference_overrides=None,
	model_variant="best",
	sbert_model_name="all-mpnet-base-v2",
	output_json_path=None,
):
	entity_proj_layer, nlp, topic_embedding_table = _load_inference_resources_approach5(config_obj, ideal_topic_embeddings_json_path)
	payload = run_multitopic_inference_approach5(
		config_obj=config_obj,
		user_csv_path=user_csv_path,
		ideal_topic_embeddings_json_path=ideal_topic_embeddings_json_path,
		entity_proj_layer=entity_proj_layer,
		nlp=nlp,
		topic_embedding_table=topic_embedding_table,
		selected_topics=[topic_name],
		inference_overrides=inference_overrides,
		model_variant=model_variant,
		sbert_model_name=sbert_model_name,
		output_json_path=output_json_path,
	)
	return payload.get("results_by_topic", {}).get(topic_name, {})


def run_multitopic_inference_approach5_minimal(
	user_csv_path,
	ideal_topic_embeddings_json_path,
	config_obj,
	selected_topics=None,
	inference_overrides=None,
	model_variant="best",
	sbert_model_name="all-mpnet-base-v2",
	output_json_path=None,
):
	entity_proj_layer, nlp, topic_embedding_table = _load_inference_resources_approach5(config_obj, ideal_topic_embeddings_json_path)
	return run_multitopic_inference_approach5(
		config_obj=config_obj,
		user_csv_path=user_csv_path,
		ideal_topic_embeddings_json_path=ideal_topic_embeddings_json_path,
		entity_proj_layer=entity_proj_layer,
		nlp=nlp,
		topic_embedding_table=topic_embedding_table,
		selected_topics=selected_topics,
		inference_overrides=inference_overrides,
		model_variant=model_variant,
		sbert_model_name=sbert_model_name,
		output_json_path=output_json_path,
	)


def print_multitopic_inference_outputs_approach5(inference_output_payload):
	print("USER INFERENCE FINAL OUTPUT")
	print("=" * 100)

	inference_results_by_topic = inference_output_payload.get("results_by_topic", {})
	if not inference_results_by_topic:
		print("No topic results found.")
		return

	print(f"Topics in result: {list(inference_results_by_topic.keys())}")
	for topic_name, result in inference_results_by_topic.items():
		shifts = result.get("sentence_level_narrative_shifts", [])
		print("\n" + "#" * 100)
		print(f"TOPIC: {topic_name} | Total sentence-level shifts: {len(shifts)}")
		print("#" * 100)

		if len(shifts) == 0:
			print("No sentence-level shifts detected. Try lowering topic/zscore thresholds.")
			continue

		for i, shift in enumerate(shifts, 1):
			print(f"\nShift #{i}: {shift['date_1']} -> {shift['date_2']}")
			print(
				f"similarity={shift['similarity']:.4f} | shift_score={shift['shift_score']:.4f} | "
				f"day_z={shift['day_level_z_score']:.4f}"
			)
			print(f"\nDay 1 - {shift['sentence_id_1']} (Article {shift['article_id_1']}, Sentence {shift['sentence_num_1']})")
			print(f"topic_weight={shift['topic_weight_1']:.3f}")
			print(shift["context_1"])
			print(f"\nDay 2 - {shift['sentence_id_2']} (Article {shift['article_id_2']}, Sentence {shift['sentence_num_2']})")
			print(f"topic_weight={shift['topic_weight_2']:.3f}")
			print(shift["context_2"])
			print("-" * 100)


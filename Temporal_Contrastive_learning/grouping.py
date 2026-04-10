import numpy as np
import pandas as pd
import ruptures as rpt


def _normalize_vector(vector):
	arr = np.asarray(vector, dtype=np.float32)
	return arr / (np.linalg.norm(arr) + 1e-8)


def _empty_grouped_dataframe(include_feature=False):
	columns = [
		"group_id",
		"date",
		"end_date",
		"daily_vectors",
		"topic_embeddings",
		"topic_name",
		"topic_id",
		"num_days",
		"num_sentences",
	]
	if include_feature:
		columns.insert(5, "feature")
	return pd.DataFrame(columns=columns)


def create_groups_fixed_size(daily_dataframe, group_size):
	records = daily_dataframe.sort_values("date").to_dict(orient="records")
	grouped = []

	for index in range(0, len(records), int(group_size)):
		chunk = records[index:index + int(group_size)]
		if not chunk:
			continue

		grouped.append(
			{
				"group_id": len(grouped),
				"date": chunk[0]["date"],
				"end_date": chunk[-1]["date"],
				"daily_vectors": _normalize_vector(
					np.stack([row["daily_vectors"] for row in chunk]).mean(axis=0)
				),
				"topic_embeddings": _normalize_vector(
					np.stack([row["topic_embeddings"] for row in chunk]).mean(axis=0)
				),
				"topic_name": chunk[0]["topic_name"],
				"topic_id": chunk[0]["topic_id"],
				"num_days": len(chunk),
				"num_sentences": int(
					sum(int(row.get("num_sentences", 0)) for row in chunk)
				),
			}
		)

	return pd.DataFrame(grouped)


def create_groups_max_day_gap(daily_dataframe, max_day_gap):
	records = daily_dataframe.sort_values("date").to_dict(orient="records")
	if not records:
		return _empty_grouped_dataframe(include_feature=False)

	grouped_chunks = []
	current_chunk = [records[0]]

	for record in records[1:]:
		gap_days = int((record["date"] - current_chunk[0]["date"]).days)
		if gap_days <= int(max_day_gap):
			current_chunk.append(record)
		else:
			grouped_chunks.append(current_chunk)
			current_chunk = [record]

	if current_chunk:
		grouped_chunks.append(current_chunk)

	grouped = []
	for chunk in grouped_chunks:
		grouped.append(
			{
				"group_id": len(grouped),
				"date": chunk[0]["date"],
				"end_date": chunk[-1]["date"],
				"daily_vectors": _normalize_vector(
					np.stack([row["daily_vectors"] for row in chunk]).mean(axis=0)
				),
				"topic_embeddings": _normalize_vector(
					np.stack([row["topic_embeddings"] for row in chunk]).mean(axis=0)
				),
				"topic_name": chunk[0]["topic_name"],
				"topic_id": chunk[0]["topic_id"],
				"num_days": len(chunk),
				"num_sentences": int(
					sum(int(row.get("num_sentences", 0)) for row in chunk)
				),
			}
		)

	return pd.DataFrame(grouped)


def create_grouped_vectors_from_daily_ap2(daily_dataframe, config):
	if daily_dataframe.empty:
		return _empty_grouped_dataframe(include_feature=False)

	if config["use_fixed_group_size"]:
		return create_groups_fixed_size(daily_dataframe, config["fixed_group_size"])

	if config["use_max_day_gap"]:
		return create_groups_max_day_gap(daily_dataframe, config["max_day_gap"])

	raise ValueError("No grouping strategy enabled in config")


def detect_change_points_ruptures(daily_matrix, config):
	if daily_matrix.shape[0] <= max(2, int(config["ruptures_min_size"])):
		return [daily_matrix.shape[0]]

	algo = rpt.Pelt(
		model=config["ruptures_model"],
		min_size=int(config["ruptures_min_size"]),
	).fit(daily_matrix)

	change_points = algo.predict(pen=float(config["ruptures_penalty"]))
	unique_change_points = sorted(set(int(cp) for cp in change_points if int(cp) > 0))
	if not unique_change_points or unique_change_points[-1] != daily_matrix.shape[0]:
		unique_change_points.append(daily_matrix.shape[0])
	return unique_change_points


def create_groups_ruptures(daily_dataframe, config):
	records = daily_dataframe.sort_values("date").to_dict(orient="records")
	if not records:
		return _empty_grouped_dataframe(include_feature=True)

	matrix = np.stack([row["feature"] for row in records]).astype(np.float32)
	change_points = detect_change_points_ruptures(matrix, config)

	grouped = []
	start_index = 0
	min_days = int(config["ruptures_min_size"])
	for end_index in change_points:
		chunk = records[start_index:end_index]
		start_index = end_index
		if not chunk:
			continue

		if grouped and len(chunk) < min_days:
			previous_start_index = grouped[-1]["_start_idx"]
			chunk = records[previous_start_index:end_index]
			grouped.pop()

		grouped.append(
			{
				"group_id": len(grouped),
				"date": chunk[0]["date"],
				"end_date": chunk[-1]["date"],
				"daily_vectors": _normalize_vector(
					np.stack([row["daily_vectors"] for row in chunk]).mean(axis=0)
				),
				"topic_embeddings": _normalize_vector(
					np.stack([row["topic_embeddings"] for row in chunk]).mean(axis=0)
				),
				"feature": _normalize_vector(
					np.stack([row["feature"] for row in chunk]).mean(axis=0)
				),
				"topic_name": chunk[0]["topic_name"],
				"topic_id": chunk[0]["topic_id"],
				"num_days": len(chunk),
				"num_sentences": int(
					sum(int(row.get("num_sentences", 0)) for row in chunk)
				),
				"_start_idx": records.index(chunk[0]),
			}
		)

	for group in grouped:
		group.pop("_start_idx", None)

	return pd.DataFrame(grouped)


def create_grouped_vectors_from_daily_ap4(daily_dataframe, config):
	if daily_dataframe.empty:
		return _empty_grouped_dataframe(include_feature=True)
	return create_groups_ruptures(daily_dataframe, config)


def detect_ruptures(day_dataframe, model="rbf", pen=10, min_size=5, verbose=True):
	if verbose:
		print(f"Detecting ruptures (model={model}, pen={pen}, min_size={min_size})...")

	if len(day_dataframe) < min_size:
		if verbose:
			print(f"  Too few days ({len(day_dataframe)} < {min_size}), creating single group")
		result = day_dataframe.copy()
		result["group"] = 0
		return result

	embeddings = np.stack(day_dataframe["embedding"].values)
	algo = rpt.Pelt(model=model, min_size=min_size).fit(embeddings)
	change_points = algo.predict(pen=pen)

	if not change_points:
		if verbose:
			print("  No change points detected, creating single group")
		result = day_dataframe.copy()
		result["group"] = 0
		return result

	if verbose:
		print(f"   Detected {len(change_points) - 1} change points")
		print(f"   Change points at indices: {change_points[:-1]}")

	groups = np.zeros(len(day_dataframe), dtype=int)
	group_id = 0
	previous_change_point = 0
	for change_point in change_points:
		groups[previous_change_point:change_point] = group_id
		group_id += 1
		previous_change_point = change_point

	result = day_dataframe.copy()
	result["group"] = groups

	if verbose:
		group_sizes = result.groupby("group").size()
		print(f"Created {group_id} groups")
		print(
			f"   Group sizes: min={group_sizes.min()}, max={group_sizes.max()}, mean={group_sizes.mean():.1f}"
		)

	return result

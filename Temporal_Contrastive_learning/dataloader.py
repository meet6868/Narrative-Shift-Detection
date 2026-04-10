from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class TemporalWindowDataset(Dataset):
	"""Shared temporal dataset for all approaches.

	Supports two modes:
	- Single-window mode (AP1/AP2/AP4):
	  TemporalWindowDataset(window_embeddings=[...], topics=[...])
	- Pair mode (AP5):
	  TemporalWindowDataset(
		  windows_current=..., windows_next=..., topic_ids=..., topics=...,
		  entity_overlap=..., window_indices=...
	  )
	"""

	def __init__(
		self,
		window_embeddings=None,
		topics=None,
		windows_current=None,
		windows_next=None,
		topic_ids=None,
		entity_overlap=None,
		window_indices=None,
	):
		self.topics = list(topics or [])
		self.mode = "pairs" if windows_current is not None else "single"

		if self.mode == "single":
			self.window_embeddings = list(window_embeddings or [])
			self.topic_groups = {topic: [] for topic in self.topics}
			for item in self.window_embeddings:
				topic_name = item["topic_name"]
				if topic_name not in self.topic_groups:
					self.topic_groups[topic_name] = []
				self.topic_groups[topic_name].append(item)

			for topic in self.topic_groups:
				self.topic_groups[topic] = sorted(
					self.topic_groups[topic], key=lambda x: x.get("window_idx", 0)
				)
			return

		self.windows_current = np.asarray(windows_current, dtype=np.float32)
		self.windows_next = np.asarray(windows_next, dtype=np.float32)
		self.topic_ids = np.asarray(topic_ids, dtype=np.int64)

		if self.windows_current.shape != self.windows_next.shape:
			raise ValueError(
				f"Shape mismatch for temporal pairs: {self.windows_current.shape} vs {self.windows_next.shape}"
			)
		if self.windows_current.shape[0] != len(self.topic_ids):
			raise ValueError(
				f"Pair/topic mismatch: pairs={self.windows_current.shape[0]} vs topics={len(self.topic_ids)}"
			)

		if entity_overlap is None:
			self.entity_overlap = np.ones(self.windows_current.shape[0], dtype=np.float32)
		else:
			self.entity_overlap = np.asarray(entity_overlap, dtype=np.float32)
			if len(self.entity_overlap) != self.windows_current.shape[0]:
				raise ValueError(
					f"Pair/overlap mismatch: pairs={self.windows_current.shape[0]} vs overlaps={len(self.entity_overlap)}"
				)

		if window_indices is None:
			self.window_indices = np.arange(self.windows_current.shape[0], dtype=np.int64)
		else:
			self.window_indices = np.asarray(window_indices, dtype=np.int64)
			if len(self.window_indices) != self.windows_current.shape[0]:
				raise ValueError(
					f"Pair/window_idx mismatch: pairs={self.windows_current.shape[0]} vs idx={len(self.window_indices)}"
				)

		self.window_embeddings = []
		self.topic_groups = {topic: [] for topic in self.topics}
		for idx, topic_id in enumerate(self.topic_ids.tolist()):
			topic_name = self.topics[int(topic_id)] if self.topics else str(topic_id)
			row = {
				"pair_idx": int(idx),
				"topic_id": int(topic_id),
				"topic_name": topic_name,
				"entity_overlap": float(self.entity_overlap[idx]),
				"window_idx": int(self.window_indices[idx]),
			}
			self.window_embeddings.append(row)
			if topic_name not in self.topic_groups:
				self.topic_groups[topic_name] = []
			self.topic_groups[topic_name].append(row)

	def __len__(self):
		if self.mode == "single":
			return len(self.window_embeddings)
		return int(self.windows_current.shape[0])

	def __getitem__(self, index):
		if self.mode == "single":
			item = self.window_embeddings[index]
			return torch.from_numpy(item["tensor"]), int(item["topic_id"])

		window_current = torch.from_numpy(self.windows_current[index])
		window_next = torch.from_numpy(self.windows_next[index])
		topic_id = int(self.topic_ids[index])
		entity_overlap = float(self.entity_overlap[index])
		window_idx = int(self.window_indices[index])
		return window_current, window_next, topic_id, entity_overlap, window_idx

	def sample_consecutive_pairs(self, batch_size):
		if self.mode != "single":
			raise ValueError("sample_consecutive_pairs is only available in single-window mode")

		anchors, positives = [], []
		per_topic = max(1, int(batch_size) // max(len(self.topics), 1))

		for topic_name in self.topics:
			topic_windows = self.topic_groups.get(topic_name, [])
			if len(topic_windows) < 2:
				continue

			max_start = len(topic_windows) - 1
			sampled_indices = np.random.randint(0, max_start, size=per_topic)

			for idx in sampled_indices:
				anchors.append(torch.from_numpy(topic_windows[idx]["tensor"]))
				positives.append(torch.from_numpy(topic_windows[idx + 1]["tensor"]))

		if len(anchors) == 0:
			raise ValueError("No valid consecutive pairs found for current batch sampling")

		return torch.stack(anchors), torch.stack(positives)

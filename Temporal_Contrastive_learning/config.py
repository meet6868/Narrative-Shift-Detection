from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

# -----------------------------------------------------------------------------
# Common Paths
# -----------------------------------------------------------------------------
# W3-distributed topic files used by Approach 1/2/4
COMMON_DATA_PATH_W3 = "/home/hp/SEM2/INLP/Naretve_Shift/Processed_Data/Distributed_Data/BAL_TOPIC_WISE_W3"
# W5-distributed topic files used by Approach 5
COMMON_DATA_PATH_W5 = "/home/hp/SEM2/INLP/Naretve_Shift/Processed_Data/Distributed_Data/BAL_TOPIC_WISE_W5"


def _temperature_tag(value):
	return str(value).replace(".", "p")


def build_model_base_name(cfg):
	template = cfg["model_name_template"]
	kwargs = {
		"model_name_prefix": cfg.get("model_name_prefix", "approch"),
		"approach_id": cfg.get("approach_id", "1"),
		"window_size": cfg.get("window_size", 2),
		"stride": cfg.get("stride", 1),
		"temperature_tag": _temperature_tag(cfg.get("temperature", 0.07)),
	}
	if "{model_type}" in template:
		kwargs["model_type"] = cfg.get("model_type", "base")
	if "{model_group_size}" in template:
		kwargs["model_group_size"] = cfg.get("model_group_size", "1")
	return template.format(**kwargs)


def build_artifact_paths(cfg):
	base_name = build_model_base_name(cfg)
	output_path = cfg["output_path"]
	paths = {
		"model_base_name": base_name,
		"model_best_path": os.path.join(output_path, f"{base_name}_best.pt"),
		"model_last_path": os.path.join(output_path, f"{base_name}_last.pt"),
		"model_evaluated_path": os.path.join(output_path, f"{base_name}_evaluated.pt"),
		"train_loss_plot_path": os.path.join(output_path, f"{base_name}_train_loss.png"),
		"eval_heatmap_intra_path": os.path.join(output_path, f"{base_name}_intra_heatmap.png"),
		"eval_heatmap_inter_path": os.path.join(output_path, f"{base_name}_inter_heatmap.png"),
		"run_summary_path": os.path.join(output_path, f"{base_name}_run_summary.json"),
		"eval_metrics_path": os.path.join(output_path, f"{base_name}_evaluation_metrics.json"),
	}

	variant_to_path = {
		"best": paths["model_best_path"],
		"last": paths["model_last_path"],
		"evaluated": paths["model_evaluated_path"],
	}
	load_variant = str(cfg.get("load_variant", "best")).lower().strip()
	paths["model_load_path"] = variant_to_path.get(load_variant, paths["model_best_path"])
	return paths


def load_checkpoint_compat(path, map_location):
	try:
		return torch.load(path, map_location=map_location)
	except Exception as exc:
		if "Weights only load failed" in str(exc):
			return torch.load(path, map_location=map_location, weights_only=False)
		raise


def build_topic_embedding_table(cfg):
	rng_local = np.random.default_rng(cfg["seed"])
	table = rng_local.standard_normal((len(cfg["topics"]), cfg["topic_embedding_dim"]))
	table = table.astype(np.float32)
	table /= (np.linalg.norm(table, axis=1, keepdims=True) + 1e-8)
	return table


def _apply_shared_runtime(config):
	os.makedirs(config["output_path"], exist_ok=True)
	config.update(build_artifact_paths(config))
	np.random.seed(config["seed"])
	torch.manual_seed(config["seed"])
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(config["seed"])
	return config


def load_approach_1_config():
	# Grouping config (Approach 1): day-level windows, no extra day-grouping stage.
	config = {
		"data_path": COMMON_DATA_PATH_W3,
		"output_path": "./tcl_output_new_1",
		"topic_files": {
			"War": "War.csv",
			"Health": "Health.csv",
			"Economics": "Economics.csv",
			"Technology": "Technology.csv",
			"Climate": "Climate.csv",
		},
		"topics": ["War", "Health", "Economics", "Technology", "Climate"],
		"embedding_column": "w5_embedding",
		"embedding_dim": 768,
		"context_window": 5,
		"min_sentences_per_day": 3,
		"window_size": 2,
		"stride": 1,
		"topic_threshold": 0.3,
		"inference_batch_size": 32,
		"hidden_dim": 256,
		"num_heads": 8,
		"num_layers": 3,
		"feed_forward_dim": 512,
		"dropout": 0.1,
		"projection_dim": 128,
		"batch_size": 32,
		"learning_rate": 1e-4,
		"epochs": 100,
		"weight_decay": 0.01,
		"warmup_epochs": 5,
		"min_lr": 1e-6,
		"temperature": 0.07,
		"gradient_clip": 1.0,
		"use_amp": True,
		"patience": 10,
		"min_delta": 1e-3,
		"drift_smoothing_window": 3,
		"zscore_threshold": 1.0,
		"percentile_threshold": 50,
		"seed": 42,
		"approach_id": "1",
		"model_name_prefix": "approch",
		"model_name_template": "{model_name_prefix}_{approach_id}_w{window_size}_s{stride}_t{temperature_tag}",
		"load_variant": "best",
	}

	config["time_dim"] = 1
	config["topic_dim"] = len(config["topics"])
	config["final_dim"] = config["embedding_dim"] + config["time_dim"] + config["topic_dim"]
	if config["context_window"] not in (3, 5):
		raise ValueError("config['context_window'] must be either 3 or 5")
	return _apply_shared_runtime(config)


def load_approach_2_config():
	# Grouping config (Approach 2): exactly one of these strategies must be enabled.
	use_max_day_gap = False
	fixed_group_size = 2
	max_day_gap = 2
	config = {
		"data_path": COMMON_DATA_PATH_W3,
		"output_path": "./tcl_output_new_2",
		"topic_files": {
			"War": "War.csv",
			"Health": "Health.csv",
			"Economics": "Economics.csv",
			"Technology": "Technology.csv",
			"Climate": "Climate.csv",
		},
		"topics": ["War", "Health", "Economics", "Technology", "Climate"],
		"embedding_column": "w5_embedding",
		"embedding_dim": 768,
		"context_window": 5,
		"min_sentences_per_day": 3,
		"window_size": 3,
		"stride": 3,
		# Group by fixed number of consecutive days.
		"use_fixed_group_size": True,
		"fixed_group_size": fixed_group_size,
		# Group by max day-gap from group start.
		"use_max_day_gap": use_max_day_gap,
		"max_day_gap": max_day_gap,
		"topic_threshold": 0.3,
		"inference_batch_size": 32,
		"hidden_dim": 256,
		"num_heads": 8,
		"num_layers": 3,
		"feed_forward_dim": 512,
		"dropout": 0.1,
		"projection_dim": 128,
		"batch_size": 32,
		"learning_rate": 1e-4,
		"epochs": 100,
		"weight_decay": 0.01,
		"warmup_epochs": 5,
		"min_lr": 1e-6,
		"temperature": 0.07,
		"gradient_clip": 1.0,
		"use_amp": True,
		"patience": 10,
		"min_delta": 1e-3,
		"drift_smoothing_window": 3,
		"zscore_threshold": 1.0,
		"percentile_threshold": 50,
		"seed": 42,
		"approach_id": "2",
		"model_name_prefix": "approch",
		"model_type": "day_gap" if use_max_day_gap else "fixed_group",
		"model_group_size": max_day_gap if use_max_day_gap else fixed_group_size,
		"model_name_template": "{model_name_prefix}_{model_type}_{model_group_size}_{approach_id}_w{window_size}_s{stride}_t{temperature_tag}",
		"load_variant": "best",
	}

	config["time_dim"] = 1
	config["topic_dim"] = len(config["topics"])
	config["final_dim"] = config["embedding_dim"] + config["time_dim"] + config["topic_dim"]
	if config["context_window"] not in (3, 5):
		raise ValueError("config['context_window'] must be either 3 or 5")
	if bool(config["use_fixed_group_size"]) == bool(config["use_max_day_gap"]):
		raise ValueError("Enable exactly one grouping strategy: use_fixed_group_size xor use_max_day_gap")
	return _apply_shared_runtime(config)


def load_approach_4_config():
	# Grouping config (Approach 4): ruptures-only temporal segmentation.
	config = {
		"match_ap4_behavior": True,
		"topic_weight_threshold": 0.3,
		"data_path": COMMON_DATA_PATH_W3,
		"output_path": "./tcl_output_new_4",
		"topics": ["War", "Health", "Economics", "Technology", "Climate"],
		"topic_files": {
			"War": "War.csv",
			"Health": "Health.csv",
			"Economics": "Economics.csv",
			"Technology": "Technology.csv",
			"Climate": "Climate.csv",
		},
		"embedding_column": "w5_embedding",
		"embedding_dim": 768,
		"topic_embedding_dim": 64,
		"window_size": 2,
		"stride": 1,
		"context_window": 5,
		"min_sentences_per_day": 2,
		"daily_variance_alpha": 0.0,
		# Enable only ruptures grouping mode.
		"ruptures_only": True,
		"ruptures_model": "rbf",
		"ruptures_penalty": 0.1,
		"ruptures_min_size": 2,
		"topic_threshold": 0.3,
		"inference_batch_size": 32,
		"hidden_dim": 512,
		"num_heads": 8,
		"num_layers": 4,
		"feed_forward_dim": 2048,
		"dropout": 0.1,
		"projection_dim": 256,
		"batch_size": 128,
		"learning_rate": 3e-4,
		"epochs": 100,
		"weight_decay": 1e-5,
		"warmup_epochs": 5,
		"min_lr": 1e-6,
		"temperature": 0.05,
		"lambda_temporal": 1.5,
		"lambda_topic_sep": 0.5,
		"lambda_hard_neg": 0.3,
		"topic_sep_margin": 0.35,
		"hard_neg_margin": 0.25,
		"gradient_clip": 1.0,
		"use_amp": True,
		"patience": 10,
		"min_delta": 1e-3,
		"save_checkpoints": True,
		"checkpoint_freq": 5,
		"shift_threshold_multiplier": 1.5,
		"manual_shift_threshold": 0.1,
		"seed": 42,
		"approach_id": "4",
		"model_name_prefix": "approch",
		"model_type": "ruptures",
		"model_group_size": "pen0p1",
		"model_name_template": "{model_name_prefix}_{model_type}_{model_group_size}_{approach_id}_w{window_size}_s{stride}_t{temperature_tag}",
		"load_variant": "best",
	}

	config.update(
		{
			"TOPIC_WEIGHT_THRESHOLD": config["topic_weight_threshold"],
			"TOPIC_THRESHOLD": config["topic_threshold"],
			"WINDOW_SIZE": config["window_size"],
			"WINDOW_STRIDE": config["stride"],
			"TEMPERATURE": config["temperature"],
			"RUPTURES_PENALTY": config["ruptures_penalty"],
			"SHIFT_THRESHOLD_MULTIPLIER": config["shift_threshold_multiplier"],
			"MANUAL_SHIFT_THRESHOLD": config["manual_shift_threshold"],
			"INPUT_DIM": config["embedding_dim"] + config["topic_embedding_dim"],
			"MIN_SIZE": config["ruptures_min_size"],
		}
	)
	config["topic_prob_dim"] = len(config["topics"])
	config["topic_dim"] = config["topic_embedding_dim"]
	config["final_dim"] = config["embedding_dim"] + config["topic_embedding_dim"]

	if config["context_window"] not in (3, 5):
		raise ValueError("config['context_window'] must be either 3 or 5")
	if not config.get("ruptures_only", False):
		raise ValueError("Approach 4 is configured as ruptures-only grouping")

	config = _apply_shared_runtime(config)
	config["topic_embedding_table"] = build_topic_embedding_table(config)
	return config


class Approach5Config:
	ALIGNMENT_SETTINGS = {"match_ap4_structure": True}
	PATH_SETTINGS = {
		"data_dir": COMMON_DATA_PATH_W5,
		"output_dir": "./tcl_output_new_5",
	}
	TOPIC_DATA_SETTINGS = {
		"topics": ["Health", "War", "Technology", "Climate", "Economics"],
		"embedding_column": "w5_embedding",
		"topic_weight_threshold": 0.3,
	}
	FEATURE_SETTINGS = {
		"embedding_dim": 768,
		"entity_proj_dim": 64,
		"topic_emb_dim": 64,
		"window_size": 3,
		"window_stride": 1,
		"ner_batch_size": 256,
		"aggregation_method": "weighted_mean",
	}
	GROUPING_SETTINGS = {
		# Approach 5 grouping: ruptures segments with minimum group size.
		"use_ruptures": True,
		"rupture_model": "rbf",
		"rupture_pen": 1,
		"min_group_size": 5,
	}
	MODEL_SETTINGS = {
		"hidden_dim": 512,
		"num_heads": 8,
		"num_layers": 4,
		"dropout": 0.1,
		"output_dim": 256,
	}
	LOSS_SETTINGS = {
		"lambda_temporal": 1.5,
		"lambda_topic_sep": 0.2,
		"lambda_hard_neg": 0.3,
		"lambda_entity": 0.5,
		"entity_margin": 0.3,
		"temperature": 0.05,
		"entity_lambda": 0.3,
		"use_projection": False,
		"entity_overlap_threshold": 0.3,
		"shift_threshold": 0.5,
	}
	TRAINING_SETTINGS = {
		"batch_size": 16,
		"num_epochs": 100,
		"learning_rate": 1e-4,
		"weight_decay": 1e-5,
		"grad_clip": 1.0,
		"use_amp": True,
		"use_cosine_schedule": True,
		"warmup_epochs": 5,
		"save_every": 5,
	}
	ARTIFACT_SETTINGS = {
		"approach_id": "5",
		"model_name_prefix": "approch",
		"model_type": "entity_tcl",
		"model_name_template": "{model_name_prefix}_{model_type}_{model_group_size}_{approach_id}_w{window_size}_s{stride}_t{temperature_tag}",
		"load_variant": "best",
	}

	@staticmethod
	def _temperature_tag(value):
		return str(value).replace(".", "p")

	def __init__(self):
		merged = {}
		for group in [
			self.ALIGNMENT_SETTINGS,
			self.PATH_SETTINGS,
			self.TOPIC_DATA_SETTINGS,
			self.FEATURE_SETTINGS,
			self.GROUPING_SETTINGS,
			self.MODEL_SETTINGS,
			self.LOSS_SETTINGS,
			self.TRAINING_SETTINGS,
			self.ARTIFACT_SETTINGS,
		]:
			merged.update(group)

		self.DATA_DIR = Path(merged["data_dir"])
		self.OUTPUT_DIR = Path(merged["output_dir"])
		self.CHECKPOINT_DIR = self.OUTPUT_DIR

		self.TOPICS = merged["topics"]
		self.TOPIC_FILES = {topic: f"{topic}.csv" for topic in self.TOPICS}
		self.EMBEDDING_COLUMN = merged["embedding_column"]
		self.TOPIC_WEIGHT_THRESHOLD = float(merged["topic_weight_threshold"])

		self.APPROACH_ID = merged["approach_id"]
		self.MODEL_NAME_PREFIX = merged["model_name_prefix"]
		self.MODEL_TYPE = merged["model_type"]
		self.MODEL_NAME_TEMPLATE = merged["model_name_template"]
		self.LOAD_VARIANT = merged["load_variant"]

		self.ENTITY_LAMBDA = float(merged["entity_lambda"])
		self.USE_PROJECTION = bool(merged["use_projection"])
		self.ENTITY_PROJ_DIM = int(merged["entity_proj_dim"])

		self.SBERT_MODEL = "all-mpnet-base-v2"
		self.EMBEDDING_DIM = int(merged["embedding_dim"])
		self.SENTENCE_FINAL_DIM = self.EMBEDDING_DIM + self.ENTITY_PROJ_DIM
		self.TOPIC_EMB_DIM = int(merged["topic_emb_dim"])
		self.CONCAT_DIM = self.SENTENCE_FINAL_DIM + self.TOPIC_EMB_DIM

		self.NER_BATCH_SIZE = int(merged["ner_batch_size"])
		self.AGGREGATION_METHOD = merged["aggregation_method"]

		self.USE_RUPTURES = bool(merged["use_ruptures"])
		self.RUPTURE_MODEL = merged["rupture_model"]
		self.RUPTURE_PEN = merged["rupture_pen"]
		self.MIN_GROUP_SIZE = int(merged["min_group_size"])

		self.WINDOW_SIZE = int(merged["window_size"])
		self.WINDOW_STRIDE = int(merged["window_stride"])

		self.HIDDEN_DIM = int(merged["hidden_dim"])
		self.NUM_HEADS = int(merged["num_heads"])
		self.NUM_LAYERS = int(merged["num_layers"])
		self.DROPOUT = float(merged["dropout"])
		self.OUTPUT_DIM = int(merged["output_dim"])

		self.LAMBDA_TEMPORAL = float(merged["lambda_temporal"])
		self.LAMBDA_TOPIC_SEP = float(merged["lambda_topic_sep"])
		self.LAMBDA_HARD_NEG = float(merged["lambda_hard_neg"])
		self.LAMBDA_ENTITY = float(merged["lambda_entity"])
		self.ENTITY_MARGIN = float(merged["entity_margin"])
		self.TEMPERATURE = float(merged["temperature"])
		self.SHIFT_THRESHOLD = float(merged["shift_threshold"])
		self.ENTITY_OVERLAP_THRESHOLD = float(merged["entity_overlap_threshold"])

		self.BATCH_SIZE = int(merged["batch_size"])
		self.NUM_EPOCHS = int(merged["num_epochs"])
		self.LEARNING_RATE = float(merged["learning_rate"])
		self.WEIGHT_DECAY = float(merged["weight_decay"])
		self.GRAD_CLIP = float(merged["grad_clip"])
		self.USE_AMP = bool(merged["use_amp"])
		self.USE_COSINE_SCHEDULE = bool(merged["use_cosine_schedule"])
		self.WARMUP_EPOCHS = int(merged["warmup_epochs"])
		self.SAVE_EVERY = int(merged["save_every"])

		self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
		model_group_size = f"pen{str(self.RUPTURE_PEN).replace('.', 'p')}"
		self.MODEL_BASE_NAME = self.MODEL_NAME_TEMPLATE.format(
			model_name_prefix=self.MODEL_NAME_PREFIX,
			model_type=self.MODEL_TYPE,
			model_group_size=model_group_size,
			approach_id=self.APPROACH_ID,
			window_size=self.WINDOW_SIZE,
			stride=self.WINDOW_STRIDE,
			temperature_tag=self._temperature_tag(self.TEMPERATURE),
		)

		self.MODEL_BEST_PATH = self.OUTPUT_DIR / f"{self.MODEL_BASE_NAME}_best.pt"
		self.MODEL_LAST_PATH = self.OUTPUT_DIR / f"{self.MODEL_BASE_NAME}_last.pt"
		self.MODEL_EVALUATED_PATH = self.OUTPUT_DIR / f"{self.MODEL_BASE_NAME}_evaluated.pt"
		self.TRAINING_HISTORY_PLOT_PATH = self.OUTPUT_DIR / f"{self.MODEL_BASE_NAME}_train_loss.png"
		self.RUN_SUMMARY_PATH = self.OUTPUT_DIR / f"{self.MODEL_BASE_NAME}_run_summary.json"
		self.EVAL_METRICS_PATH = self.OUTPUT_DIR / f"{self.MODEL_BASE_NAME}_evaluation_metrics.json"
		self.INFERENCE_OUTPUT_DIR = self.OUTPUT_DIR
		self.INFERENCE_RESULTS_PATH = self.OUTPUT_DIR / f"{self.MODEL_BASE_NAME}_user_inference_multi_topic.json"


def load_approach_5_config():
	return Approach5Config()

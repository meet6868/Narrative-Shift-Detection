from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TCLTemporalEncoderA12(nn.Module):
	"""AP1/AP2 encoder: attention pooling over temporal axis."""

	def __init__(self, config):
		super().__init__()
		self.input_norm = nn.LayerNorm(config["final_dim"])
		self.input_projection = nn.Linear(config["final_dim"], config["hidden_dim"])
		self.dropout = nn.Dropout(config["dropout"])

		self.learned_positional = nn.Parameter(
			torch.randn(1, config["window_size"], config["hidden_dim"]) * 0.02
		)

		encoder_layer = nn.TransformerEncoderLayer(
			d_model=config["hidden_dim"],
			nhead=config["num_heads"],
			dim_feedforward=config["feed_forward_dim"],
			dropout=config["dropout"],
			activation="gelu",
			batch_first=True,
			norm_first=True,
		)
		self.transformer = nn.TransformerEncoder(
			encoder_layer,
			num_layers=config["num_layers"],
			norm=nn.LayerNorm(config["hidden_dim"]),
		)

		self.attention_score = nn.Linear(config["hidden_dim"], 1)
		self.post_mlp = nn.Sequential(
			nn.Linear(config["hidden_dim"], config["hidden_dim"]),
			nn.GELU(),
			nn.Dropout(config["dropout"]),
			nn.Linear(config["hidden_dim"], config["hidden_dim"]),
		)

		self.projection_head = nn.Sequential(
			nn.Linear(config["hidden_dim"], config["projection_dim"]),
			nn.LayerNorm(config["projection_dim"]),
			nn.GELU(),
			nn.Dropout(config["dropout"]),
			nn.Linear(config["projection_dim"], config["projection_dim"]),
		)

	def forward(self, inputs):
		hidden = self.input_norm(inputs)
		hidden = self.input_projection(hidden)
		hidden = self.dropout(hidden)
		hidden = hidden + self.learned_positional
		encoded = self.transformer(hidden)

		weights = F.softmax(self.attention_score(encoded), dim=1)
		pooled = (encoded * weights).sum(dim=1)
		pooled = pooled + self.post_mlp(pooled)

		projected = self.projection_head(pooled)
		return F.normalize(projected, p=2, dim=1)


class TCLTemporalEncoderA4(nn.Module):
	"""AP4 encoder: mean pooling over temporal axis."""

	def __init__(self, config):
		super().__init__()
		self.input_projection = nn.Sequential(
			nn.Linear(config["final_dim"], config["hidden_dim"]),
			nn.LayerNorm(config["hidden_dim"]),
			nn.Dropout(config["dropout"]),
		)

		self.learned_positional = nn.Parameter(
			torch.randn(1, config["window_size"], config["hidden_dim"]) * 0.02
		)

		encoder_layer = nn.TransformerEncoderLayer(
			d_model=config["hidden_dim"],
			nhead=config["num_heads"],
			dim_feedforward=config["feed_forward_dim"],
			dropout=config["dropout"],
			activation="gelu",
			batch_first=True,
			norm_first=True,
		)
		self.transformer = nn.TransformerEncoder(
			encoder_layer,
			num_layers=config["num_layers"],
		)

		self.projection_head = nn.Sequential(
			nn.Linear(config["hidden_dim"], config["hidden_dim"]),
			nn.GELU(),
			nn.Linear(config["hidden_dim"], config["projection_dim"]),
		)

	def forward(self, inputs):
		hidden = self.input_projection(inputs)
		hidden = hidden + self.learned_positional
		encoded = self.transformer(hidden)
		pooled = encoded.mean(dim=1)
		projected = self.projection_head(pooled)
		return F.normalize(projected, p=2, dim=1)


class PositionalEncoding(nn.Module):
	def __init__(self, d_model, max_len=10):
		super().__init__()
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		self.register_buffer("pe", pe)

	def forward(self, x):
		return x + self.pe[:x.size(1), :].unsqueeze(0)


class TCLTemporalEncoderA5(nn.Module):
	"""AP5 encoder with flexible constructor (config object or dict)."""

	def __init__(self, config_or_input_dim=896, hidden_dim=512, output_dim=256, num_heads=8, num_layers=4, dropout=0.1):
		super().__init__()

		if hasattr(config_or_input_dim, "CONCAT_DIM"):
			cfg = config_or_input_dim
			input_dim = int(getattr(cfg, "CONCAT_DIM", 896))
			hidden_dim = int(getattr(cfg, "HIDDEN_DIM", hidden_dim))
			output_dim = int(getattr(cfg, "OUTPUT_DIM", output_dim))
			num_heads = int(getattr(cfg, "NUM_HEADS", num_heads))
			num_layers = int(getattr(cfg, "NUM_LAYERS", num_layers))
			dropout = float(getattr(cfg, "DROPOUT", dropout))
		elif isinstance(config_or_input_dim, dict):
			cfg = config_or_input_dim
			input_dim = int(cfg.get("input_dim", cfg.get("concat_dim", 896)))
			hidden_dim = int(cfg.get("hidden_dim", hidden_dim))
			output_dim = int(cfg.get("projection_dim", cfg.get("output_dim", output_dim)))
			num_heads = int(cfg.get("num_heads", num_heads))
			num_layers = int(cfg.get("num_layers", num_layers))
			dropout = float(cfg.get("dropout", dropout))
		else:
			input_dim = int(config_or_input_dim)

		self.input_proj = nn.Linear(input_dim, hidden_dim)
		self.pos_encoder = PositionalEncoding(hidden_dim, max_len=10)

		encoder_layer = nn.TransformerEncoderLayer(
			d_model=hidden_dim,
			nhead=num_heads,
			dim_feedforward=hidden_dim * 4,
			dropout=dropout,
			activation="gelu",
			batch_first=True,
		)
		self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

		self.output_proj = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, output_dim),
		)
		self.layer_norm = nn.LayerNorm(output_dim)

	def forward(self, x, mask=None):
		x = self.input_proj(x)
		x = self.pos_encoder(x)
		x = self.transformer(x, src_key_padding_mask=mask)
		x = x.mean(dim=1)
		x = self.output_proj(x)
		x = self.layer_norm(x)
		return F.normalize(x, p=2, dim=-1)


TemporalTransformer = TCLTemporalEncoderA5

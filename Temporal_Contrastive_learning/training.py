from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def build_scheduler_a12(optimizer, config):
	def lr_lambda(epoch):
		if epoch < config["warmup_epochs"]:
			return float(epoch + 1) / float(config["warmup_epochs"])
		progress = (epoch - config["warmup_epochs"]) / max(1, config["epochs"] - config["warmup_epochs"])
		cosine = 0.5 * (1 + np.cos(np.pi * progress))
		return max(config["min_lr"] / config["learning_rate"], cosine)

	return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_tcl_model_a12(model, train_dataset, train_loader, optimizer, loss_fn, config, device):
	scheduler = build_scheduler_a12(optimizer, config)
	use_amp = bool(config["use_amp"] and torch.cuda.is_available())
	scaler = torch.cuda.amp.GradScaler() if use_amp else None

	history = {"epoch": [], "loss": [], "lr": [], "best_loss": float("inf"), "best_epoch": 0}
	patience_counter = 0

	for epoch in range(config["epochs"]):
		model.train()
		batch_losses = []
		for _ in train_loader:
			anchor_windows, positive_windows = train_dataset.sample_consecutive_pairs(config["batch_size"])
			anchor_windows = anchor_windows.to(device)
			positive_windows = positive_windows.to(device)
			optimizer.zero_grad(set_to_none=True)

			if use_amp:
				with torch.cuda.amp.autocast():
					anchor_repr = model(anchor_windows)
					positive_repr = model(positive_windows)
					loss = loss_fn(anchor_repr, positive_repr)
				scaler.scale(loss).backward()
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
				scaler.step(optimizer)
				scaler.update()
			else:
				anchor_repr = model(anchor_windows)
				positive_repr = model(positive_windows)
				loss = loss_fn(anchor_repr, positive_repr)
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
				optimizer.step()

			batch_losses.append(float(loss.item()))

		scheduler.step()
		epoch_loss = float(np.mean(batch_losses)) if batch_losses else float("inf")
		current_lr = optimizer.param_groups[0]["lr"]
		history["epoch"].append(epoch + 1)
		history["loss"].append(epoch_loss)
		history["lr"].append(current_lr)

		improvement = history["best_loss"] - epoch_loss
		if improvement > config["min_delta"]:
			history["best_loss"] = epoch_loss
			history["best_epoch"] = epoch + 1
			patience_counter = 0
			torch.save(
				{
					"epoch": epoch + 1,
					"model_state_dict": model.state_dict(),
					"optimizer_state_dict": optimizer.state_dict(),
					"loss": epoch_loss,
					"config": config,
				},
				config["model_best_path"],
			)
		else:
			patience_counter += 1

		print(f"Epoch {epoch + 1:03d} | loss={epoch_loss:.5f} | lr={current_lr:.2e}")
		if patience_counter >= config["patience"]:
			print("Early stopping triggered")
			break

	torch.save(
		{
			"epoch": history["epoch"][-1] if history["epoch"] else 0,
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"loss": history["loss"][-1] if history["loss"] else None,
			"config": config,
		},
		config["model_last_path"],
	)
	return model, history


def build_scheduler_a4(optimizer, config):
	return torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer,
		T_max=max(1, int(config["epochs"]) - int(config["warmup_epochs"])),
		eta_min=float(config["min_lr"]),
	)


def train_tcl_model_a4(model, train_loader, optimizer, loss_fn, config, device):
	scheduler = build_scheduler_a4(optimizer, config)
	use_amp = bool(config["use_amp"] and torch.cuda.is_available())
	scaler = torch.cuda.amp.GradScaler() if use_amp else None

	history = {
		"epoch": [],
		"loss": [],
		"temporal_loss": [],
		"topic_sep_loss": [],
		"hard_neg_loss": [],
		"lr": [],
		"best_loss": float("inf"),
		"best_epoch": 0,
	}

	if len(train_loader) == 0:
		raise ValueError("Train loader has zero batches. Reduce batch_size or increase available windows.")

	for epoch in range(int(config["epochs"])):
		model.train()
		batch_losses = []
		batch_temporal = []
		batch_topic_sep = []
		batch_hard_neg = []

		if epoch < int(config["warmup_epochs"]):
			warmup_lr = float(config["learning_rate"]) * float(epoch + 1) / float(config["warmup_epochs"])
			for param_group in optimizer.param_groups:
				param_group["lr"] = warmup_lr

		for windows, topic_ids in train_loader:
			windows = windows.to(device)
			topic_ids = topic_ids.to(device)
			optimizer.zero_grad(set_to_none=True)

			if use_amp:
				with torch.cuda.amp.autocast():
					embeddings = model(windows)
					loss, loss_dict = loss_fn(embeddings, topic_ids)
				scaler.scale(loss).backward()
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
				scaler.step(optimizer)
				scaler.update()
			else:
				embeddings = model(windows)
				loss, loss_dict = loss_fn(embeddings, topic_ids)
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
				optimizer.step()

			batch_losses.append(float(loss.item()))
			batch_temporal.append(float(loss_dict["temporal"]))
			batch_topic_sep.append(float(loss_dict["topic_separation"]))
			batch_hard_neg.append(float(loss_dict["hard_negative"]))

		if epoch >= int(config["warmup_epochs"]):
			scheduler.step()

		epoch_loss = float(np.mean(batch_losses))
		epoch_temporal = float(np.mean(batch_temporal))
		epoch_topic_sep = float(np.mean(batch_topic_sep))
		epoch_hard_neg = float(np.mean(batch_hard_neg))
		current_lr = float(optimizer.param_groups[0]["lr"])

		history["epoch"].append(epoch + 1)
		history["loss"].append(epoch_loss)
		history["temporal_loss"].append(epoch_temporal)
		history["topic_sep_loss"].append(epoch_topic_sep)
		history["hard_neg_loss"].append(epoch_hard_neg)
		history["lr"].append(current_lr)

		if epoch_loss < history["best_loss"]:
			history["best_loss"] = epoch_loss
			history["best_epoch"] = epoch + 1
			torch.save(
				{
					"epoch": epoch + 1,
					"model_state_dict": model.state_dict(),
					"optimizer_state_dict": optimizer.state_dict(),
					"loss": epoch_loss,
					"loss_components": {
						"temporal": epoch_temporal,
						"topic_separation": epoch_topic_sep,
						"hard_negative": epoch_hard_neg,
					},
					"config": config,
				},
				config["model_best_path"],
			)

		if bool(config.get("save_checkpoints", True)) and ((epoch + 1) % int(config.get("checkpoint_freq", 5)) == 0):
			checkpoint_path = os.path.join(config["output_path"], f"checkpoint_epoch_{epoch + 1}.pt")
			torch.save(
				{
					"epoch": epoch + 1,
					"model_state_dict": model.state_dict(),
					"optimizer_state_dict": optimizer.state_dict(),
					"scheduler_state_dict": scheduler.state_dict(),
					"loss": epoch_loss,
					"loss_components": {
						"temporal": epoch_temporal,
						"topic_separation": epoch_topic_sep,
						"hard_negative": epoch_hard_neg,
					},
					"config": config,
				},
				checkpoint_path,
			)

		print(
			f"Epoch {epoch + 1:03d} | total={epoch_loss:.5f} | temporal={epoch_temporal:.5f} | "
			f"topic_sep={epoch_topic_sep:.5f} | hard_neg={epoch_hard_neg:.5f} | lr={current_lr:.2e}"
		)

	torch.save(
		{
			"epoch": history["epoch"][-1] if history["epoch"] else 0,
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"loss": history["loss"][-1] if history["loss"] else None,
			"loss_components": {
				"temporal": history["temporal_loss"][-1] if history["temporal_loss"] else None,
				"topic_separation": history["topic_sep_loss"][-1] if history["topic_sep_loss"] else None,
				"hard_negative": history["hard_neg_loss"][-1] if history["hard_neg_loss"] else None,
			},
			"config": config,
		},
		config["model_last_path"],
	)
	return model, history


def train_tcl_model_a5(model, train_loader, num_epochs, config, criterion):
	device = config.DEVICE
	model = model.to(device)
	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=config.LEARNING_RATE,
		weight_decay=config.WEIGHT_DECAY,
	)

	if config.USE_COSINE_SCHEDULE:
		scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
			optimizer,
			T_0=max(1, config.NUM_EPOCHS // 2),
			T_mult=1,
			eta_min=1e-5,
		)
	else:
		scheduler = None

	use_amp = bool(config.USE_AMP and torch.cuda.is_available())
	scaler = torch.cuda.amp.GradScaler() if use_amp else None

	history = {
		"loss": [],
		"temporal_loss": [],
		"topic_sep_loss": [],
		"hard_neg_loss": [],
		"entity_loss": [],
		"lr": [],
	}

	for epoch in range(num_epochs):
		model.train()
		epoch_losses = {"total": [], "temporal": [], "topic_sep": [], "hard_neg": [], "entity": []}
		epoch_temporal_sim = []
		pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

		for window_current, window_next, topic_ids, entity_overlap, window_idx in pbar:
			window_current = window_current.to(device)
			window_next = window_next.to(device)
			topic_ids = topic_ids.to(device)
			entity_overlap = entity_overlap.to(device, dtype=torch.float32)
			window_idx = window_idx.to(device)
			optimizer.zero_grad(set_to_none=True)

			if use_amp:
				with torch.cuda.amp.autocast():
					z_current = model(window_current)
					z_next = model(window_next)
					loss, loss_dict = criterion(
						z_current,
						z_next,
						topic_ids,
						entity_overlap=entity_overlap,
						window_idx=window_idx,
					)
				scaler.scale(loss).backward()
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
				scaler.step(optimizer)
				scaler.update()
			else:
				z_current = model(window_current)
				z_next = model(window_next)
				loss, loss_dict = criterion(
					z_current,
					z_next,
					topic_ids,
					entity_overlap=entity_overlap,
					window_idx=window_idx,
				)
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
				optimizer.step()

			with torch.no_grad():
				sim_mean = (F.normalize(z_current, dim=1) * F.normalize(z_next, dim=1)).sum(dim=1).mean().item()
				epoch_temporal_sim.append(float(sim_mean))

			epoch_losses["total"].append(float(loss.item()))
			epoch_losses["temporal"].append(float(loss_dict["temporal"]))
			epoch_losses["topic_sep"].append(float(loss_dict["topic_sep"]))
			epoch_losses["hard_neg"].append(float(loss_dict["hard_neg"]))
			epoch_losses["entity"].append(float(loss_dict["entity"]))

			pbar.set_postfix(
				{
					"loss": f"{loss.item():.4f}",
					"temp": f"{loss_dict['temporal']:.4f}",
					"topic": f"{loss_dict['topic_sep']:.4f}",
					"hard": f"{loss_dict['hard_neg']:.4f}",
					"ent": f"{loss_dict['entity']:.4f}",
					"sim": f"{sim_mean:.3f}",
				}
			)

		if scheduler is not None:
			scheduler.step()

		avg_loss = float(np.mean(epoch_losses["total"]))
		avg_temp = float(np.mean(epoch_losses["temporal"]))
		avg_topic = float(np.mean(epoch_losses["topic_sep"]))
		avg_hard = float(np.mean(epoch_losses["hard_neg"]))
		avg_entity = float(np.mean(epoch_losses["entity"]))
		avg_sim = float(np.mean(epoch_temporal_sim)) if epoch_temporal_sim else float("nan")
		current_lr = float(optimizer.param_groups[0]["lr"])

		history["loss"].append(avg_loss)
		history["temporal_loss"].append(avg_temp)
		history["topic_sep_loss"].append(avg_topic)
		history["hard_neg_loss"].append(avg_hard)
		history["entity_loss"].append(avg_entity)
		history["lr"].append(current_lr)

		print(
			f"\nEpoch {epoch + 1}/{num_epochs}:\n  Loss: {avg_loss:.4f} | Temporal: {avg_temp:.4f} | "
			f"Topic: {avg_topic:.4f} | Hard: {avg_hard:.4f} | Entity: {avg_entity:.4f}"
		)
		print(f"  Temporal sim mean: {avg_sim:.4f}")
		print(f"  LR: {current_lr:.6f}")

		if (epoch + 1) % config.SAVE_EVERY == 0:
			checkpoint_path = config.OUTPUT_DIR / f"checkpoint_epoch_{epoch + 1}.pt"
			torch.save(
				{
					"epoch": epoch + 1,
					"model_state_dict": model.state_dict(),
					"optimizer_state_dict": optimizer.state_dict(),
					"loss": avg_loss,
					"history": history,
					"config": {
						"input_dim": config.CONCAT_DIM,
						"hidden_dim": config.HIDDEN_DIM,
						"output_dim": config.OUTPUT_DIM,
						"num_heads": config.NUM_HEADS,
						"num_layers": config.NUM_LAYERS,
						"dropout": config.DROPOUT,
						"model_base_name": config.MODEL_BASE_NAME,
					},
				},
				checkpoint_path,
			)

	torch.save(
		{
			"epoch": num_epochs,
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"loss": float(history["loss"][-1]) if history["loss"] else None,
			"history": history,
			"config": {
				"input_dim": config.CONCAT_DIM,
				"hidden_dim": config.HIDDEN_DIM,
				"output_dim": config.OUTPUT_DIM,
				"num_heads": config.NUM_HEADS,
				"num_layers": config.NUM_LAYERS,
				"dropout": config.DROPOUT,
				"model_base_name": config.MODEL_BASE_NAME,
			},
		},
		config.MODEL_LAST_PATH,
	)

	return model, history

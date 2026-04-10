from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedNTXentLossA12(nn.Module):
	def __init__(self, temperature):
		super().__init__()
		self.temperature = temperature

	def forward(self, anchor_embeddings, positive_embeddings):
		batch_size = anchor_embeddings.shape[0]
		pair_embeddings = torch.cat([anchor_embeddings, positive_embeddings], dim=0)
		pair_embeddings = F.normalize(pair_embeddings, p=2, dim=1)

		logits = torch.mm(pair_embeddings, pair_embeddings.t()) / self.temperature
		logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

		labels = torch.arange(2 * batch_size, device=pair_embeddings.device)
		labels = (labels + batch_size) % (2 * batch_size)

		mask = torch.eye(2 * batch_size, dtype=torch.bool, device=pair_embeddings.device)
		logits = logits.masked_fill(mask, -1e4)

		return F.cross_entropy(logits, labels)


class EnhancedNTXentLossA4(nn.Module):
	def __init__(
		self,
		temperature,
		lambda_temporal=1.5,
		lambda_topic_sep=0.5,
		lambda_hard_neg=0.3,
		topic_sep_margin=0.35,
		hard_neg_margin=0.25,
	):
		super().__init__()
		self.temperature = float(temperature)
		self.lambda_temporal = float(lambda_temporal)
		self.lambda_topic_sep = float(lambda_topic_sep)
		self.lambda_hard_neg = float(lambda_hard_neg)
		self.topic_sep_margin = float(topic_sep_margin)
		self.hard_neg_margin = float(hard_neg_margin)

	def forward(self, embeddings, topic_ids):
		embeddings = F.normalize(embeddings, p=2, dim=1)
		batch_size = embeddings.shape[0]
		device_local = embeddings.device

		sim_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature
		diag_mask = torch.eye(batch_size, device=device_local, dtype=torch.bool)
		sim_matrix = sim_matrix.masked_fill(diag_mask, -1e4)

		topic_match = topic_ids.unsqueeze(0) == topic_ids.unsqueeze(1)
		positive_mask = topic_match.float().masked_fill(diag_mask, 0.0)

		exp_sim = torch.exp(sim_matrix)
		pos_exp = (exp_sim * positive_mask).sum(dim=1) + 1e-8
		all_exp = exp_sim.sum(dim=1) + 1e-8
		has_pos = (positive_mask.sum(dim=1) > 0).float()
		temporal_vec = -torch.log(pos_exp / all_exp + 1e-8)
		temporal_loss = (temporal_vec * has_pos).sum() / (has_pos.sum() + 1e-8)

		topic_sep_loss = torch.tensor(0.0, device=device_local)
		unique_topics = torch.unique(topic_ids)
		if len(unique_topics) > 1:
			centroids = []
			for topic_id in unique_topics:
				topic_mask = topic_ids == topic_id
				if topic_mask.sum() > 0:
					centroids.append(embeddings[topic_mask].mean(dim=0))
			if len(centroids) > 1:
				centroid_tensor = torch.stack(centroids)
				centroid_sim = torch.mm(centroid_tensor, centroid_tensor.t())
				centroid_mask = torch.eye(len(centroids), device=device_local, dtype=torch.bool)
				centroid_sim = centroid_sim.masked_fill(centroid_mask, 0.0)
				topic_sep_loss = centroid_sim.abs().mean()

		hard_neg_loss = torch.tensor(0.0, device=device_local)
		negative_mask = (~topic_match).float().masked_fill(diag_mask, 0.0)
		if negative_mask.sum() > 0:
			neg_sims = sim_matrix * negative_mask
			k = max(1, int(batch_size * 0.3))
			hardest, _ = torch.topk(neg_sims, k=min(k, neg_sims.shape[1]), dim=1)
			hard_neg_loss = torch.exp(hardest).mean()

		total_loss = (
			self.lambda_temporal * temporal_loss
			+ self.lambda_topic_sep * topic_sep_loss
			+ self.lambda_hard_neg * hard_neg_loss
		)

		loss_dict = {
			"temporal": float(temporal_loss.detach().item()),
			"topic_separation": float(topic_sep_loss.detach().item()),
			"hard_negative": float(hard_neg_loss.detach().item()),
			"total": float(total_loss.detach().item()),
		}
		return total_loss, loss_dict


class TemporalPairLoss(nn.Module):
	def __init__(self, temperature=0.07):
		super().__init__()
		self.temperature = float(temperature)

	def forward(self, z1, z2, topic_ids=None, window_idx=None, pair_weights=None):
		z1 = F.normalize(z1, dim=1)
		z2 = F.normalize(z2, dim=1)
		sim = torch.mm(z1, z2.t()) / self.temperature
		log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

		if window_idx is not None:
			wi = window_idx.view(-1, 1)
			wj = window_idx.view(1, -1)
			pos_mask = wi == wj
		else:
			pos_mask = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)

		if topic_ids is not None:
			topic_mask = topic_ids.view(-1, 1) == topic_ids.view(1, -1)
			pos_mask = pos_mask & topic_mask

		diag_mask = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
		row_has_pos = pos_mask.any(dim=1, keepdim=True)
		pos_mask = torch.where(row_has_pos, pos_mask, diag_mask)

		pos_count = pos_mask.sum(dim=1).clamp(min=1)
		per_pair_loss = -(log_prob.masked_fill(~pos_mask, 0.0).sum(dim=1) / pos_count)

		if pair_weights is not None:
			weights = pair_weights.to(per_pair_loss.device, dtype=per_pair_loss.dtype)
			return (per_pair_loss * weights).sum() / (weights.sum() + 1e-8)

		return per_pair_loss.mean()


class TopicSeparationLoss(nn.Module):
	def __init__(self, margin=0.35):
		super().__init__()
		self.margin = float(margin)

	def forward(self, embeddings, topic_ids):
		topic_sep_loss = torch.tensor(0.0, device=embeddings.device)
		unique_topics = torch.unique(topic_ids)
		if len(unique_topics) > 1:
			centroids = []
			for topic_id in unique_topics:
				topic_mask = topic_ids == topic_id
				if topic_mask.sum() > 0:
					centroids.append(embeddings[topic_mask].mean(dim=0))
			if len(centroids) > 1:
				centroid_tensor = torch.stack(centroids)
				centroid_sim = torch.mm(centroid_tensor, centroid_tensor.t())
				centroid_mask = torch.eye(len(centroids), device=embeddings.device, dtype=torch.bool)
				centroid_sim = centroid_sim.masked_fill(centroid_mask, 0.0)
				topic_sep_loss = centroid_sim.abs().mean()
		return topic_sep_loss


class HardNegativeLoss(nn.Module):
	def __init__(self, temperature=0.07, hard_ratio=0.3):
		super().__init__()
		self.temperature = float(temperature)
		self.hard_ratio = float(hard_ratio)

	def forward(self, z_current, z_next, topic_ids):
		z_current = F.normalize(z_current, dim=1)
		z_next = F.normalize(z_next, dim=1)
		sim = torch.mm(z_current, z_next.t()) / self.temperature
		topic_diff = topic_ids.unsqueeze(1) != topic_ids.unsqueeze(0)
		if topic_diff.sum() <= 0:
			return torch.tensor(0.0, device=z_current.device)
		masked = sim.masked_fill(~topic_diff, -1e4)
		k = max(1, int(sim.size(1) * self.hard_ratio))
		hardest = torch.topk(masked, k=min(k, sim.size(1)), dim=1).values
		valid_rows = topic_diff.any(dim=1)
		if valid_rows.sum() <= 0:
			return torch.tensor(0.0, device=z_current.device)
		hardest = hardest[valid_rows]
		return F.softplus(hardest).mean()


class EntityConsistencyLoss(nn.Module):
	def __init__(self, temporal_threshold=2, margin=0.3, min_weight=0.05):
		super().__init__()
		self.temporal_threshold = temporal_threshold
		self.margin = margin
		self.min_weight = min_weight

	def forward(self, z_current, z_next, entity_overlap=None, window_idx=None):
		if entity_overlap is None or window_idx is None:
			return torch.tensor(0.0, device=z_current.device)
		z_current = F.normalize(z_current, dim=1)
		z_next = F.normalize(z_next, dim=1)
		sim = F.cosine_similarity(z_current, z_next, dim=1)
		overlap = entity_overlap.to(z_current.device, dtype=sim.dtype).clamp(0.0, 1.0)
		temporal_gap = torch.ones_like(overlap)
		temporal_weight = torch.exp(-temporal_gap / self.temporal_threshold)
		weight = torch.clamp(overlap * temporal_weight, min=self.min_weight)
		pos_loss = weight * (1.0 - sim)
		neg_loss = weight * F.relu(sim - self.margin)
		return (pos_loss + neg_loss).mean()


class MultiLoss(nn.Module):
	def __init__(
		self,
		lambda_temporal=1.0,
		lambda_topic_sep=0.3,
		lambda_hard_neg=0.5,
		lambda_entity=0.3,
		temperature=0.07,
		topic_sep_margin=0.35,
		hard_neg_margin=0.25,
	):
		super().__init__()
		self.lambda_temporal = float(lambda_temporal)
		self.lambda_topic_sep = float(lambda_topic_sep)
		self.lambda_hard_neg = float(lambda_hard_neg)
		self.lambda_entity = float(lambda_entity)
		self.temporal_pair_loss = TemporalPairLoss(temperature=temperature)
		self.topic_sep_loss = TopicSeparationLoss(margin=topic_sep_margin)
		self.hard_neg_loss = HardNegativeLoss(temperature=temperature)
		self.entity_loss = EntityConsistencyLoss()

	def forward(self, z_current, z_next, topic_ids, entity_overlap=None, window_idx=None, pair_weights=None):
		temporal_loss = self.temporal_pair_loss(
			z_current,
			z_next,
			topic_ids=topic_ids,
			window_idx=window_idx,
			pair_weights=pair_weights,
		)
		pair_center = F.normalize((z_current + z_next) * 0.5, dim=1)
		topic_loss = self.topic_sep_loss(pair_center, topic_ids)
		hard_neg_loss = self.hard_neg_loss(z_current, z_next, topic_ids)
		entity_loss = self.entity_loss(z_current, z_next, entity_overlap=entity_overlap, window_idx=window_idx)
		total_loss = (
			self.lambda_temporal * temporal_loss
			+ self.lambda_topic_sep * topic_loss
			+ self.lambda_hard_neg * hard_neg_loss
			+ self.lambda_entity * entity_loss
		)
		return total_loss, {
			"total": float(total_loss.item()),
			"temporal": float(temporal_loss.item()),
			"topic_sep": float(topic_loss.item()),
			"hard_neg": float(hard_neg_loss.item()),
			"entity": float(entity_loss.item()),
		}

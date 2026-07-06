"""Loss helpers for ProSys Stage 2 V2."""

from __future__ import annotations

import torch
import torch.nn.functional as F

DEFAULT_EPS = 1e-10


def _listmle_single(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = DEFAULT_EPS) -> torch.Tensor:
    random_indices = torch.randperm(y_pred.shape[0], device=y_pred.device)
    y_pred_shuffled = y_pred[random_indices]
    y_true_shuffled = y_true[random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True)
    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=0, index=indices)

    max_pred_value = preds_sorted_by_true.max()
    preds_shifted = preds_sorted_by_true - max_pred_value
    cumsums = torch.cumsum(preds_shifted.exp().flip(dims=[0]), dim=0).flip(dims=[0])
    observation_loss = torch.log(cumsums + eps) - preds_shifted

    if y_true_sorted.numel() == 0:
        return torch.zeros((), device=y_pred.device)
    return observation_loss.sum()


def weighted_listMLE(
    logits: torch.Tensor,
    relevance: torch.Tensor,
    sample_index: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    losses = []
    weights = []
    for group_id in torch.unique(sample_index):
        group_mask = sample_index == group_id
        group_targets = relevance[group_mask]
        if group_targets.numel() == 0 or torch.max(group_targets) <= 0:
            continue

        group_logits = logits[group_mask]
        group_loss = _listmle_single(group_logits, group_targets, eps=eps)
        group_weight = (
            sample_weight[group_mask].mean()
            if sample_weight is not None
            else torch.ones((), device=logits.device)
        )
        losses.append(group_loss * group_weight)
        weights.append(group_weight)

    if not losses:
        return torch.zeros((), device=logits.device)

    loss_tensor = torch.stack(losses)
    weight_tensor = torch.stack(weights).clamp_min(DEFAULT_EPS)
    return loss_tensor.sum() / weight_tensor.sum()


def binary_candidate_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    losses = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    if sample_weight is None:
        return losses.mean()

    weights = sample_weight.clamp_min(DEFAULT_EPS)
    return (losses * weights).sum() / weights.sum()


def pairwise_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    route_match: torch.Tensor,
    context_match: torch.Tensor,
    sample_index: torch.Tensor,
    *,
    margin: float = 0.5,
    max_negatives: int = 8,
) -> torch.Tensor:
    losses = []
    for group_id in torch.unique(sample_index):
        group_mask = sample_index == group_id
        group_logits = logits[group_mask]
        group_labels = labels[group_mask]
        group_route = route_match[group_mask]
        group_context = context_match[group_mask]

        pos_mask = group_labels > 0.5
        if not torch.any(pos_mask):
            continue

        pos_logits = group_logits[pos_mask]
        priority_masks = [
            (~pos_mask) & (group_route > 0.5) & (group_context < 0.5),
            (~pos_mask) & (group_route < 0.5) & (group_context > 0.5),
            (~pos_mask) & (group_route < 0.5) & (group_context < 0.5),
        ]

        selected_neg_logits = []
        for neg_mask in priority_masks:
            if not torch.any(neg_mask):
                continue
            neg_logits = group_logits[neg_mask]
            sorted_logits, _ = torch.sort(neg_logits, descending=True)
            selected_neg_logits.append(sorted_logits)

        if not selected_neg_logits:
            continue

        neg_logits = torch.cat(selected_neg_logits)[:max_negatives]
        if neg_logits.numel() == 0:
            continue

        diff = margin - pos_logits.unsqueeze(1) + neg_logits.unsqueeze(0)
        losses.append(F.relu(diff).mean())

    if not losses:
        return torch.zeros((), device=logits.device)

    return torch.stack(losses).mean()

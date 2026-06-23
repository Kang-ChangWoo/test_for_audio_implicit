"""Losses for the ray-sampled implicit model. All operate on sampled rays
(B,M) with a per-ray weight w = cos-lat area * valid-mask."""

import torch
import torch.nn.functional as F


def weighted_l1(pred, gt, w):
    e = (pred - gt).abs() * w
    return e.sum() / w.sum().clamp(min=1e-6)


def bin_targets(gt, centers):
    """(B,M) gt, (K,) centers -> (B,M) nearest-bin index."""
    d = (gt[..., None] - centers[None, None, :]).abs()
    return d.argmin(-1)


def depth_bin_ce(logits, gt, centers, w):
    """weighted cross-entropy to nearest log-depth bin."""
    B, M, K = logits.shape
    tgt = bin_targets(gt, centers).reshape(-1)
    ce = F.cross_entropy(logits.reshape(-1, K), tgt, reduction="none").reshape(B, M)
    return (ce * w).sum() / w.sum().clamp(min=1e-6)


def compute_loss(out, gt, w, cfg, centers=None):
    """Returns (total, parts dict). gt,w are (B,M)."""
    parts = {}
    total = weighted_l1(out["depth"], gt, w)
    parts["l1"] = float(total.detach())

    if cfg.use_depth_bins and out["logits"] is not None:
        ce = depth_bin_ce(out["logits"], gt, centers, w)
        total = total + ce
        parts["ce"] = float(ce.detach())

    if cfg.model == "hybrid":
        cl = weighted_l1(out["coarse"], gt, w)
        rl = (out["residual"].abs() * w).sum() / w.sum().clamp(min=1e-6)
        total = total + cfg.w_coarse * cl + cfg.w_res * rl
        parts["coarse"] = float(cl.detach()); parts["res"] = float(rl.detach())

    parts["total"] = float(total.detach())
    return total, parts

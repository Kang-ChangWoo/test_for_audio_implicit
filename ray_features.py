"""Precomputed per-ray feature bank for the ERP grid.

The ERP grid is fixed, so every ray feature (direction, Fourier PE, SH basis,
ear-axis mic PE) is identical across samples -> precompute once as (H*W, F) and
just gather rows at train time / chunk them at eval time.

Frame:  x = front, y = left(+)/right(-) ear axis, z = up.
    az in (-pi, pi],  el in (-pi/2, pi/2)   (matches sh.erp_grid / diag_lib).
    dir = (cos el cos az, cos el sin az, sin el)
Ears (self-emitting active rig, source == origin):
    p_L = (0, +head_r, 0),  p_R = (0, -head_r, 0)
    mic-PE(r) = [r . yhat, -r . yhat] = [y, -y]  (swaps under L/R mirror).
"""

import os
import sys
import math
import numpy as np
import torch

# Single source of truth for the real-SH (ACN/SN3D) convention + ERP grid.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "test_for_audio_clip"))
from sh import erp_grid, sh_basis_matrix  # noqa: E402


def _fourier_pe(dirs, bands):
    """(N,3) unit dirs -> (N, 3*2*bands) Fourier features."""
    freqs = (2.0 ** np.arange(bands)) * math.pi          # (bands,)
    ang = dirs[:, :, None] * freqs[None, None, :]        # (N,3,bands)
    ang = ang.reshape(dirs.shape[0], -1)                 # (N, 3*bands)
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1)


class RayBank:
    """Holds the fixed ERP ray grid + assembled feature matrix for a config."""

    def __init__(self, cfg, device="cpu"):
        H, W = cfg.img_h, cfg.img_w
        self.H, self.W, self.N = H, W, H * W
        el, az = erp_grid(H, W)                           # (H,W) each
        el_f, az_f = el.ravel(), az.ravel()
        dirs = np.stack([np.cos(el_f) * np.cos(az_f),
                         np.cos(el_f) * np.sin(az_f),
                         np.sin(el_f)], axis=1).astype(np.float32)   # (N,3)

        feats, parts = [], []
        if cfg.use_xyz:
            feats.append(dirs); parts.append(("xyz", 3))
        if cfg.use_fourier_pe:
            f = _fourier_pe(dirs, cfg.fourier_bands).astype(np.float32)
            feats.append(f); parts.append(("fourier", f.shape[1]))
        if cfg.use_sh_pe:
            sh = sh_basis_matrix(cfg.sh_order, el, az).astype(np.float32)  # (N,(L+1)^2)
            feats.append(sh); parts.append(("sh", sh.shape[1]))
        if cfg.use_mic_pe:
            y = dirs[:, 1:2]
            mic = np.concatenate([y, -y], axis=1).astype(np.float32)       # (N,2)
            feats.append(mic); parts.append(("mic", 2))

        feat = np.concatenate(feats, axis=1) if feats else np.zeros((self.N, 0), np.float32)
        self.feat = torch.from_numpy(feat).to(device)                     # (N, F)
        self.dirs = torch.from_numpy(dirs).to(device)                     # (N, 3)
        # cos-lat area weight per ray (solid angle on the sphere)
        w = np.cos(el_f).clip(min=1e-3).astype(np.float32)
        self.area = torch.from_numpy(w).to(device)                        # (N,)
        self.parts = parts
        self.feat_dim = feat.shape[1]

    def mirror_index(self):
        """Column permutation that mirrors the ERP left<->right (azimuth -> -az),
        i.e. y -> -y. Used by the L/R-swap test. Returns LongTensor (W,) col map
        broadcastable per row -> we return full (N,) index."""
        H, W = self.H, self.W
        cols = (W - np.arange(W)) % W                     # az -> -az on the grid
        idx = (np.arange(H)[:, None] * W + cols[None, :]).ravel()
        return torch.from_numpy(idx.astype(np.int64))

    def to(self, device):
        self.feat = self.feat.to(device); self.dirs = self.dirs.to(device)
        self.area = self.area.to(device)
        return self


def log_depth_bins(n_bins, dmin=0.02, dmax=1.0, device="cpu"):
    """Log-spaced bin centres in NORMALISED depth space [0,1]. Returns (n_bins,)."""
    edges = torch.logspace(math.log10(dmin), math.log10(dmax), n_bins, device=device)
    return edges

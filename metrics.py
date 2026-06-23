"""Full-grid ERP depth metrics in METRES. Inputs are (B,1,H,W).

Standard: MAE, RMSE, AbsRel, delta<1.25, SILog (cos-lat weighted, masked).
Layout-focused (the metrics that matter for the SH/implicit claim):
  low-pass MAE (blur sigma=3), SH-coefficient L1 error (low-order layout),
  sector MAE (front/back/left/right/upper/lower).
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "test_for_audio_clip"))
from sh import SHGrid  # noqa: E402

_SECTOR_ORDER = ["front", "right", "back", "left", "upper", "lower"]


# vendored from baseline_diag/diag_lib.py (avoid its heavy transitive imports)
def gaussian_blur_erp(x, sigma):
    """Separable Gaussian low-pass on (B,1,H,W): circular pad on width
    (azimuth wraps), reflect pad on height."""
    k = int(2 * round(3 * sigma) + 1)
    c = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    x = F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1))
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="circular"), g.view(1, 1, 1, k))
    return x


def erp_region_masks(h, w):
    """Direction slices (boolean H,W); slices overlap, not a partition."""
    ii, jj = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    az = (jj + 0.5) / w * 2 * np.pi - np.pi
    el = np.pi / 2 - (ii + 0.5) / h * np.pi
    d = np.deg2rad
    return {"front": np.abs(az) < d(45), "right": (az >= d(45)) & (az < d(135)),
            "back": np.abs(az) >= d(135), "left": (az <= -d(45)) & (az > -d(135)),
            "upper": el > d(30), "lower": el < -d(30)}


def cos_lat(h, device, dtype=torch.float32):
    v = torch.arange(h, device=device, dtype=dtype)
    return torch.cos((math.pi / 2) - (v + 0.5) / h * math.pi).clamp(min=1e-3)


def _wmae(p, g, w):
    return float((((p - g).abs() * w).sum() / w.sum().clamp(min=1e-6)).item())


class MetricBank:
    """Accumulates metrics over batches; call .add() then .result()."""

    def __init__(self, H, W, max_depth, sh_order=4, device="cpu"):
        self.H, self.W, self.md = H, W, max_depth
        self.wlat = cos_lat(H, device).view(1, 1, H, 1)
        self.shg = SHGrid(H, W, order=sh_order)
        self.regions = {k: torch.from_numpy(v.astype(np.float32)).view(1, 1, H, W).to(device)
                        for k, v in erp_region_masks(H, W).items()}
        self.acc = {}
        self.n = 0

    def _push(self, k, v, n):
        self.acc[k] = self.acc.get(k, 0.0) + v * n

    @torch.no_grad()
    def add(self, pred, gt, mask):
        """pred,gt,mask (B,1,H,W). pred/gt in METRES, mask in {0,1}."""
        B = pred.shape[0]
        w = self.wlat * mask                                  # cos-lat * valid
        self._push("MAE", _wmae(pred, gt, w), B)
        self._push("MAE_plain", _wmae(pred, gt, mask), B)     # mask-only (matches A0 0.802)
        self._push("RMSE", math.sqrt((((pred - gt) ** 2) * w).sum().item()
                                     / w.sum().clamp(min=1e-6).item()), B)
        ar = (pred - gt).abs() / gt.clamp(min=0.1)
        self._push("AbsRel", float(((ar * w).sum() / w.sum().clamp(min=1e-6)).item()), B)
        rt = torch.maximum(pred.clamp(min=0.1) / gt.clamp(min=0.1),
                           gt.clamp(min=0.1) / pred.clamp(min=0.1))
        self._push("delta1", float((((rt < 1.25).float() * w).sum()
                                    / w.sum().clamp(min=1e-6)).item()), B)
        lg = (torch.log(pred.clamp(min=0.1)) - torch.log(gt.clamp(min=0.1)))
        wsum = w.sum().clamp(min=1e-6)
        v = ((lg ** 2) * w).sum() / wsum - (((lg * w).sum() / wsum) ** 2)
        self._push("SILog", float((v.clamp(min=0).sqrt()).item()), B)

        # far-field-excluded MAE: drop the 10m-clamp ceiling pixels (gt>=9.99m)
        wff = w * (gt < 9.99 * (self.md / 10.0)).float()
        self._push("MAE_ffx", _wmae(pred, gt, wff), B)

        # low-pass (coarse layout) MAE
        pl, gl = gaussian_blur_erp(pred, 3.0), gaussian_blur_erp(gt, 3.0)
        self._push("MAE_low", _wmae(pl, gl, w), B)

        # SH-coefficient L1 error (per-sample, low-order layout)
        pn, gn = pred.cpu().numpy(), gt.cpu().numpy()
        sherr = 0.0
        for b in range(B):
            cp = self.shg.project(pn[b, 0]); cg = self.shg.project(gn[b, 0])
            sherr += float(np.abs(cp - cg).mean())
        self._push("SHcoefL1", sherr / B, B)

        # sector MAE
        for name, reg in self.regions.items():
            wr = w * reg
            self._push(f"sec_{name}", _wmae(pred, gt, wr), B)
        self.n += B

    def result(self):
        return {k: self.acc[k] / max(self.n, 1) for k in self.acc}

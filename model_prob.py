"""Probabilistic COARSE-layout head (audio -> ERP radial depth).

Motivated by the oracle error decomposition (out/oracle_decomp.json):
  * all global-transform fixes (scale/offset/LR/UD/azimuth) recover <0.04m and the
    biggest is a control artifact -> systematic-bias correction is exhausted;
  * the residual splits into (A) coarse-layout AMBIGUITY (audio underdetermines which
    coarse layout, ~0.68) and (B) fine angular detail that is essentially absent from
    binaural audio (~0.23, irreducible for any point model).

A deterministic decoder regresses to the conditional MEAN of layouts (blurry average).
This head instead emits K DIVERSE coarse hypotheses (relaxed Winner-Take-All / Multiple
Choice Learning) so best-of-K can cover the multi-modal layout, plus a per-pixel Laplace
log-scale for calibrated aleatoric uncertainty. Heads are band-limited (avg-pool to a
coarse grid then upsample) so capacity is not wasted on unobservable fine detail.

Hypothesis under test: best-of-K coarse MAE < deterministic 0.78m (i.e. the ceiling is
multi-modal ambiguity that a distribution can capture, not pure noise).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import conv_bn, Refine, AudioEncoder


class ProbCoarseNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.H, self.W, self.width = cfg.img_h, cfg.img_w, cfg.width
        self.K = cfg.prob_k
        self.coarse = getattr(cfg, "prob_coarse", True)
        self.ph, self.pw = cfg.prob_head_h, cfg.prob_head_w
        self.bb = AudioEncoder(cfg.width, audio_dim=256, dim=cfg.dim,
                               in_ch=getattr(cfg, "in_ch", 2))
        self.to_z = nn.Linear(256, cfg.embed_dim)
        self.h0, self.w0 = self.H // 8, self.W // 8
        self.fc = nn.Linear(cfg.embed_dim, cfg.width * 4 * self.h0 * self.w0)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"), conv_bn(cfg.width*4, cfg.width*2), Refine(cfg.width*2),
            nn.Upsample(scale_factor=2, mode="nearest"), conv_bn(cfg.width*2, cfg.width), Refine(cfg.width),
            nn.Upsample(scale_factor=2, mode="nearest"), conv_bn(cfg.width, cfg.width), Refine(cfg.width))
        self.head_mu = nn.Conv2d(cfg.width, self.K, 3, 1, 1)     # K hypothesis maps
        self.head_logb = nn.Conv2d(cfg.width, 1, 3, 1, 1)        # per-pixel Laplace log-scale

    def _bandlimit(self, x):
        if not self.coarse:
            return x
        x = F.adaptive_avg_pool2d(x, (self.ph, self.pw))
        return F.interpolate(x, (self.H, self.W), mode="bilinear", align_corners=False)

    def forward(self, spec, *args):
        z = self.to_z(self.bb(spec)[0])
        x = self.up(self.fc(z).view(-1, self.width * 4, self.h0, self.w0))   # (B,width,H,W)
        mu = torch.sigmoid(self._bandlimit(self.head_mu(x)))                 # (B,K,H,W) in [0,1]
        logb = self._bandlimit(self.head_logb(x))                           # (B,1,H,W)
        # "D" = ensemble mean keeps this drop-in compatible with MetricBank/eval.
        return {"mu": mu, "logb": logb, "D": mu.mean(1, keepdim=True)}

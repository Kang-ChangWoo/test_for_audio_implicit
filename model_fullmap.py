"""A0-style full-map decoder + audio correction branches (A9-A12).

A9   FullMapNet                : reproduce the strong A0 det decoder (~0.80).
A10  correction='cross'        : D = D0 + 0.1*tanh(alpha)*upsample(coarse cross-attn)
A11  correction='sh'           : D = D0 + 0.1*tanh(alpha)*SH_synth(audio->coef)  (+aux)
A12  correction='film'         : decoder feature * (1+s*gamma) + s*beta from audio

Design rule: every correction starts at ZERO strength (alpha / scale init 0) so it
can only help, never wreck the baseline. The cross/SH correction is COARSE by
construction (predicted on a 16x32 grid then upsampled / low-order SH).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import conv_bn, Refine, AudioEncoder, CrossBlock


class FullMapNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.H, self.W, self.width = cfg.img_h, cfg.img_w, cfg.width
        self.corr = getattr(cfg, "correction", "none")
        self.bb = AudioEncoder(cfg.width, audio_dim=256, dim=cfg.dim,
                               in_ch=getattr(cfg, "in_ch", 2))
        self.to_z = nn.Linear(256, cfg.embed_dim)
        self.h0, self.w0 = self.H // 8, self.W // 8
        self.fc = nn.Linear(cfg.embed_dim, cfg.width * 4 * self.h0 * self.w0)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"), conv_bn(cfg.width*4, cfg.width*2), Refine(cfg.width*2),
            nn.Upsample(scale_factor=2, mode="nearest"), conv_bn(cfg.width*2, cfg.width), Refine(cfg.width),
            nn.Upsample(scale_factor=2, mode="nearest"), conv_bn(cfg.width, cfg.width), Refine(cfg.width))
        self.head = nn.Sequential(conv_bn(cfg.width, cfg.width), nn.Conv2d(cfg.width, 1, 3, 1, 1))

        if self.corr == "cross":
            self.hc, self.wc = cfg.coarse_h, cfg.coarse_w
            self.ray_mlp = nn.Sequential(nn.Linear(cfg.coarse_feat_dim, cfg.dim), nn.GELU(),
                                         nn.Linear(cfg.dim, cfg.dim), nn.GELU())
            self.cross = nn.ModuleList([CrossBlock(cfg.dim, cfg.n_heads) for _ in range(cfg.n_cross)])
            self.res_head = nn.Linear(cfg.dim, 1)
            self.alpha = nn.Parameter(torch.zeros(1))
        elif self.corr == "sh":
            self.sh_head = nn.Linear(256, (cfg.corr_sh_order + 1) ** 2)
            self.alpha = nn.Parameter(torch.zeros(1))
        elif self.corr == "film":
            self.film = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 2 * cfg.width))
            self.film_scale = 0.05

    def decode(self, z, film=None):
        x = self.fc(z).view(-1, self.width * 4, self.h0, self.w0)
        x = self.up(x)                                   # (B,width,H,W)
        if film is not None:
            g, b = film[:, :self.width], film[:, self.width:]
            x = x * (1 + self.film_scale * g[:, :, None, None]) + self.film_scale * b[:, :, None, None]
        return torch.sigmoid(self.head(x))               # (B,1,H,W)

    def forward(self, spec, coarse_feat=None, sh_basis=None):
        want_tok = self.corr == "cross"
        z256, tok = self.bb(spec, want_tokens=want_tok)
        z = self.to_z(z256)
        film = self.film(z256) if self.corr == "film" else None
        D0 = self.decode(z, film)
        out = {"D0": D0, "extras": {}}

        if self.corr == "cross":
            B = spec.size(0)
            q = self.ray_mlp(coarse_feat[None].expand(B, -1, -1))   # (B,Mc,dim)
            for blk in self.cross:
                q = blk(q, tok)
            dc = self.res_head(q).squeeze(-1).view(B, 1, self.hc, self.wc)
            dc = F.interpolate(dc, (self.H, self.W), mode="bilinear", align_corners=False)
            D = (D0 + 0.1 * torch.tanh(self.alpha) * dc).clamp(1e-3, 1.0)
            out.update(D=D, extras={"alpha": float(self.alpha.detach()), "Dcorr": dc})
            return out
        if self.corr == "sh":
            coef = self.sh_head(z256)                               # (B,Kc)
            coarse = torch.einsum("bk,nk->bn", coef, sh_basis).view(-1, 1, self.H, self.W)
            D = (D0 + 0.1 * torch.tanh(self.alpha) * coarse).clamp(1e-3, 1.0)
            out.update(D=D, extras={"alpha": float(self.alpha.detach()), "coef": coef})
            return out
        out.update(D=D0)
        return out

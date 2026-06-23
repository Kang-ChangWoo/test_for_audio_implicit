"""Modular ray-conditioned implicit audio->depth model.

One module, config-flagged, covering the whole ablation ladder:
  rayonly    depth = f(ray)                         (A1 prior control)
  raymlp     depth = f(global_audio, ray)           (A2/A3 implicit MLP)
  cross      ray queries cross-attend audio tokens  (A4/A5)
  crossself  cross + ray self-attention             (A6)
  hybrid     SH-coarse(audio) + implicit residual   (A8)

Heads: scalar (sigmoid) or log-depth bins (A7). Audio backbone is the same conv
encoder family as the sibling experiments (global latent + pre-pool tokens).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# audio backbone (vendored conv encoder; matches test_for_audio_better.Backbone)
# --------------------------------------------------------------------------- #
def conv_bn(ci, co, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv2d(ci, co, k, s, p, bias=False),
                         nn.BatchNorm2d(co), nn.GELU())


class Refine(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(conv_bn(ch, ch),
                                  nn.Conv2d(ch, ch, 3, 1, 1, bias=False), nn.BatchNorm2d(ch))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.body(x))


class AudioEncoder(nn.Module):
    """spec (B,2,H,W) -> global latent (B,audio_dim) and tokens (B,T,dim)."""

    def __init__(self, width=48, audio_dim=256, dim=192, in_ch=2):
        super().__init__()
        self.net = nn.Sequential(
            conv_bn(in_ch, width), conv_bn(width, width, s=2), Refine(width),
            conv_bn(width, width*2), conv_bn(width*2, width*2, s=2), Refine(width*2),
            conv_bn(width*2, width*4), conv_bn(width*4, width*4, s=2), Refine(width*4))
        c = width * 4
        self.gpool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                   nn.Linear(c, audio_dim), nn.GELU())
        self.tok = nn.Linear(c, dim)
        self.tok_dim = dim

    def forward(self, spec, want_tokens=False):
        fmap = self.net(spec)                       # (B,C,h,w)
        z = self.gpool(fmap)                        # (B,audio_dim)
        if not want_tokens:
            return z, None
        B, C, h, w = fmap.shape
        tok = self.tok(fmap.flatten(2).transpose(1, 2))   # (B, h*w, dim)
        return z, tok


# --------------------------------------------------------------------------- #
# transformer blocks
# --------------------------------------------------------------------------- #
class FFN(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim*mult), nn.GELU(),
                                 nn.Linear(dim*mult, dim))

    def forward(self, x):
        return self.net(x)


class CrossBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = FFN(dim)

    def forward(self, q, kv):
        a, _ = self.attn(self.n1(q), kv, kv)
        q = q + a
        return q + self.ffn(self.n2(q))


class SelfBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = FFN(dim)

    def forward(self, x):
        a, _ = self.attn(self.n1(x), self.n1(x), self.n1(x))
        x = x + a
        return x + self.ffn(self.n2(x))


# --------------------------------------------------------------------------- #
# heads
# --------------------------------------------------------------------------- #
class DepthHead(nn.Module):
    """scalar -> sigmoid depth, or n_bins logits -> expected depth."""

    def __init__(self, dim, use_bins, n_bins, bin_centers=None):
        super().__init__()
        self.use_bins = use_bins
        if use_bins:
            self.fc = nn.Linear(dim, n_bins)
            self.register_buffer("centers", bin_centers)
        else:
            self.fc = nn.Linear(dim, 1)

    def forward(self, h):
        if self.use_bins:
            logits = self.fc(h)                              # (B,M,K)
            prob = logits.softmax(-1)
            depth = (prob * self.centers).sum(-1)            # (B,M)
            return depth, logits
        return torch.sigmoid(self.fc(h)).squeeze(-1), None   # (B,M)


# --------------------------------------------------------------------------- #
# main model
# --------------------------------------------------------------------------- #
class RaySkipMLP(nn.Module):
    """tip6: re-inject ray coordinates mid-network so PE/SH features don't vanish."""
    def __init__(self, in_dim, dim):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, dim)
        self.b1 = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.skip = nn.Linear(in_dim + dim, dim)
        self.b2 = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, rf):
        h = F.gelu(self.in_proj(rf))
        h = F.gelu(h + self.b1(h))
        h = F.gelu(self.skip(torch.cat([h, rf], -1)))
        return F.gelu(h + self.b2(h))


class RayDepthModel(nn.Module):
    def __init__(self, cfg, ray_feat_dim, bin_centers=None):
        super().__init__()
        self.kind = cfg.model
        self.dim = cfg.dim
        self.use_audio = cfg.model != "rayonly"
        self.is_attn = cfg.model in ("cross", "crossself", "hybrid")
        self.use_self = cfg.model == "crossself"
        self.ray_film = getattr(cfg, "ray_film", False)

        if self.use_audio:
            self.audio = AudioEncoder(cfg.width, cfg.audio_dim, cfg.dim)

        # ray embedding (tip6: optional skip-MLP)
        if getattr(cfg, "ray_mlp_skip", False):
            self.ray_mlp = RaySkipMLP(ray_feat_dim, cfg.dim)
        else:
            self.ray_mlp = nn.Sequential(
                nn.Linear(ray_feat_dim, cfg.dim), nn.GELU(),
                nn.Linear(cfg.dim, cfg.dim), nn.GELU())
        if self.ray_film and self.is_attn:
            self.film = nn.Linear(cfg.audio_dim, 2 * cfg.dim)   # tip5

        if self.kind == "raymlp":
            fuse_in = cfg.dim + (cfg.audio_dim if self.use_audio else 0)
            self.fuse = nn.Sequential(nn.Linear(fuse_in, cfg.dim), nn.GELU(),
                                      nn.Linear(cfg.dim, cfg.dim), nn.GELU())
        if self.is_attn:
            self.cross = nn.ModuleList([CrossBlock(cfg.dim, cfg.n_heads)
                                        for _ in range(cfg.n_cross)])
            if self.use_self:
                self.selfb = nn.ModuleList([SelfBlock(cfg.dim, cfg.n_heads)
                                            for _ in range(cfg.n_self)])
        if self.kind == "hybrid":
            self.sh_head = nn.Linear(cfg.audio_dim, (cfg.hybrid_sh_order + 1) ** 2)
            self.res_head = nn.Linear(cfg.dim, 1)

        self.head = DepthHead(cfg.dim, cfg.use_depth_bins, cfg.n_bins, bin_centers)

    def forward(self, spec, ray_feat, sh_coarse=None):
        """spec (B,2,H,W); ray_feat (B,M,F); sh_coarse (B,M,Kc) for hybrid.
        returns dict: depth (B,M), logits|None, coarse|None, residual|None."""
        q = self.ray_mlp(ray_feat)                           # (B,M,dim)
        out = {"logits": None, "coarse": None, "residual": None}

        if self.kind == "rayonly":
            depth, logits = self.head(q)
            out.update(depth=depth, logits=logits); return out

        if self.kind == "raymlp":
            z, _ = self.audio(spec, want_tokens=False)       # (B,Da)
            zc = z[:, None, :].expand(-1, q.size(1), -1)
            h = self.fuse(torch.cat([q, zc], dim=-1))
            depth, logits = self.head(h)
            out.update(depth=depth, logits=logits); return out

        # attention family
        z, tok = self.audio(spec, want_tokens=True)          # (B,Da),(B,T,dim)
        h = q
        for blk in self.cross:
            h = blk(h, tok)
        if self.ray_film:                                    # tip5: global-audio FiLM on ray tokens
            g, b = self.film(z).chunk(2, dim=-1)
            h = h * (1 + 0.1 * g[:, None, :]) + 0.1 * b[:, None, :]
        if self.use_self:
            for blk in self.selfb:
                h = blk(h)

        if self.kind == "hybrid":
            coeff = self.sh_head(z)                           # (B,Kc)
            coarse = torch.einsum("bk,bmk->bm", coeff, sh_coarse)   # (B,M)
            residual = self.res_head(h).squeeze(-1)           # (B,M)
            depth = (coarse + residual).clamp(1e-3, 1.0)
            out.update(depth=depth, coarse=coarse, residual=residual)
            return out

        depth, logits = self.head(h)
        out.update(depth=depth, logits=logits); return out

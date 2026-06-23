"""Train a ray-conditioned implicit audio->ERP-depth model.

Train time: supervise N randomly-sampled rays per sample (cheap, dense coverage
over epochs). Eval time: predict the full ERP grid by chunking all rays.

Examples:
  # Q1 gate: ray-only prior vs global-audio RayMLP (+ shuffled control)
  python train.py --model rayonly --run-name A1_rayonly --audio-mode none
  python train.py --model raymlp  --run-name A2_raymlp
  python train.py --model raymlp  --run-name A2_shuf --shuffle-audio True
"""

import os
import json
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from types import SimpleNamespace

from config import get_cfg
from data import make_loader, apply_audio_mode, shuffle_audio_batch
from ray_features import RayBank, log_depth_bins
from model import RayDepthModel
from losses import compute_loss
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "test_for_audio_clip"))
from sh import erp_grid, sh_basis_matrix  # noqa: E402

N_VAL = 1000          # quick val subset during training


def build_sh_coarse(cfg, device):
    el, az = erp_grid(cfg.img_h, cfg.img_w)
    B = sh_basis_matrix(cfg.hybrid_sh_order, el, az).astype(np.float32)
    return torch.from_numpy(B).to(device)             # (N, Kc)


def prep_audio(spec, cfg):
    spec = apply_audio_mode(spec, cfg.audio_mode)
    if cfg.shuffle_audio:
        spec = shuffle_audio_batch(spec)
    return spec


@torch.no_grad()
def predict_full(model, spec, bank, cfg, sh_full):
    """Predict the full ERP grid -> (B,1,H,W) normalised depth."""
    B = spec.shape[0]; N = bank.N
    feat = bank.feat                                   # (N,F)
    out = torch.zeros(B, N, device=spec.device)
    if model.use_self:                                 # self-attn couples rays/sample
        sc = 8                                          # sample-chunk (mem-bounded)
        for b0 in range(0, B, sc):
            sl = slice(b0, min(b0 + sc, B)); nb = sl.stop - sl.start
            rf = feat[None].expand(nb, -1, -1)         # (nb,N,F)
            shc = sh_full[None].expand(nb, -1, -1) if cfg.model == "hybrid" else None
            out[sl] = model(spec[sl], rf, shc)["depth"]
    else:
        for i in range(0, N, cfg.eval_chunk):
            sl = slice(i, min(i + cfg.eval_chunk, N))
            rf = feat[sl][None].expand(B, -1, -1)
            shc = sh_full[sl][None].expand(B, -1, -1) if cfg.model == "hybrid" else None
            out[:, sl] = model(spec, rf, shc)["depth"]
    return out.view(B, 1, bank.H, bank.W)


@torch.no_grad()
def quick_val(model, loader, bank, cfg, sh_full, device, wlat):
    """cos-lat weighted MAE [m] over a val subset (smaller for slow self-attn)."""
    cap = 256 if model.use_self else N_VAL
    model.eval(); tot = 0.0; wn = 0.0; seen = 0
    for b in loader:
        spec = prep_audio(b["spec"].to(device), cfg)
        depth = b["depth"].to(device); mask = b["mask"].to(device)
        pred = predict_full(model, spec, bank, cfg, sh_full) * cfg.max_depth
        gt = depth * cfg.max_depth
        w = wlat * mask
        tot += ((pred - gt).abs() * w).sum().item()
        wn += w.sum().item()
        seen += spec.size(0)
        if seen >= cap:
            break
    return tot / max(wn, 1e-6)


def main():
    cfg = get_cfg()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(cfg.out_dir, cfg.run_name); os.makedirs(run_dir, exist_ok=True)
    print(f"[cfg] {vars(cfg)}", flush=True)

    bank = RayBank(cfg, device=device)
    sh_full = build_sh_coarse(cfg, device) if cfg.model == "hybrid" else None
    centers = log_depth_bins(cfg.n_bins, device=device) if cfg.use_depth_bins else None
    model = RayDepthModel(cfg, bank.feat_dim, centers).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[model] {cfg.model} feat_dim={bank.feat_dim} params={nparam/1e6:.2f}M", flush=True)

    tr = make_loader(cfg, "train", shuffle=True)
    va = make_loader(cfg, "val", shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(tr); warmup = max(1, len(tr))   # ~1 epoch warmup
    def lr_lam(s):
        if s < warmup:
            return (s + 1) / warmup
        t = (s - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * t))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lam)
    wlat = bank.area.view(1, 1, bank.H, bank.W)        # area weight as (1,1,H,W)

    best = 1e9; hist = []
    total_steps = cfg.epochs * len(tr); gstep = 0
    for ep in range(cfg.epochs):
        model.train(); t0 = time.time(); run = {}
        for b in tr:
            spec = prep_audio(b["spec"].to(device, non_blocking=True), cfg)
            depth = b["depth"].to(device).view(spec.size(0), -1)   # (B,N)
            mask = b["mask"].to(device).view(spec.size(0), -1)
            B = spec.size(0)
            if getattr(cfg, "sector_sample", False):                # tip4: half uniform + half sector-balanced
                nh = cfg.n_rays // 2
                u = torch.randint(0, bank.N, (B, cfg.n_rays - nh), device=device)
                per = max(1, nh // len(bank.sector_pools)); secs = []
                for pool in bank.sector_pools:
                    secs.append(pool[torch.randint(0, len(pool), (B, per), device=device)])
                idx = torch.cat([u] + secs, dim=1)
            else:
                idx = torch.randint(0, bank.N, (B, cfg.n_rays), device=device)
            rf = bank.feat[idx]                                     # (B,M,F)
            if getattr(cfg, "prog_pe", False) and bank.fourier_slice is not None:  # tip3
                prog = gstep / total_steps
                bw = ((prog - bank.fourier_band.float() / cfg.fourier_bands) / 0.2).clamp(0, 1)
                s, e = bank.fourier_slice
                rf = rf.clone(); rf[..., s:e] = rf[..., s:e] * bw
            gstep += 1
            gt = depth.gather(1, idx)                               # (B,M)
            w = bank.area[idx] * mask.gather(1, idx)                # (B,M)
            if cfg.mask_farfield:                                   # drop 10m-clamp rays
                w = w * (gt < 0.999).float()
            shc = sh_full[idx] if cfg.model == "hybrid" else None
            out = model(spec, rf, shc)
            loss, parts = compute_loss(out, gt, w, cfg, centers)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            for k, v in parts.items():
                run[k] = run.get(k, 0.0) + v
        nb = len(tr); run = {k: v / nb for k, v in run.items()}

        vmae = quick_val(model, va, bank, cfg, sh_full, device, wlat)
        hist.append({"epoch": ep, "val_mae_m": vmae, **run})
        print(f"[ep {ep:02d}] {time.time()-t0:5.1f}s  loss={run.get('total',0):.4f}  "
              f"val_MAE={vmae:.4f} m", flush=True)
        if vmae < best:
            best = vmae
            torch.save({"state_dict": model.state_dict(), "cfg": vars(cfg),
                        "feat_dim": bank.feat_dim}, os.path.join(run_dir, "best.pth"))

    json.dump({"best_val_mae_m": best, "hist": hist, "cfg": vars(cfg)},
              open(os.path.join(run_dir, "train_done.json"), "w"), indent=2)
    print(f"[done] best val MAE = {best:.4f} m -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()

"""Train the A0-style full-map decoder + optional audio correction (A9-A12).

  A9   python train_fullmap.py --run-name A9_fullmap  --correction none
  A10  python train_fullmap.py --run-name A10_cross   --correction cross
  A11  python train_fullmap.py --run-name A11_shaux    --correction sh
  A12  python train_fullmap.py --run-name A12_film     --correction film

Whole-map prediction + masked MAE (matches the A0 det baseline recipe). The
correction branches add a ZERO-initialised coarse term on top of D0.
"""

import os
import sys
import json
import math
import time
import copy
import numpy as np
import torch
from types import SimpleNamespace

from config import get_cfg
from data import make_loader, apply_audio_mode, shuffle_audio_batch
from ray_features import RayBank
from model_fullmap import FullMapNet
from metrics import MetricBank, cos_lat, gaussian_blur_erp
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "test_for_audio_clip"))
from sh import SHGrid  # noqa: E402

N_VAL = 1500


def prep_audio(spec, cfg, norm=None):
    if spec.shape[1] > cfg.in_ch:            # channel ablation: keep first in_ch
        spec = spec[:, :cfg.in_ch]
    spec = apply_audio_mode(spec, cfg.audio_mode)
    if cfg.shuffle_audio:
        spec = shuffle_audio_batch(spec)
    if norm is not None:                     # per-channel train-set normalisation
        spec = (spec - norm[0]) / norm[1]
    return spec


def masked_mae(D, gt, mask):
    return ((D - gt).abs() * mask).sum() / mask.sum().clamp(min=1e-6)


def tv(x):
    return (x[..., :, 1:] - x[..., :, :-1]).abs().mean() + (x[..., 1:, :] - x[..., :-1, :]).abs().mean()


def warm_start(model, run, freeze, device):
    """Load decoder weights (bb/to_z/fc/up/head) from a trained run; optional freeze."""
    ck = torch.load(os.path.join("out", run, "best.pth"), map_location="cpu", weights_only=False)
    msd = model.state_dict()
    keep = {k: v for k, v in ck["state_dict"].items() if k in msd and msd[k].shape == v.shape}
    model.load_state_dict(keep, strict=False)
    print(f"[warm] loaded {len(keep)} tensors from {run}", flush=True)
    if freeze:
        dec = ("bb.", "to_z.", "fc.", "up.", "head.")
        nf = 0
        for n, p in model.named_parameters():
            if n.startswith(dec):
                p.requires_grad_(False); nf += 1
        print(f"[warm] froze {nf} decoder params", flush=True)


def chan_stats(cfg, device):
    """Per-channel mean/std over a train subset (for chan_norm)."""
    import numpy as np
    a = np.load(os.path.join(cfg.cache_dir, "train_spec.npy"), mmap_mode="r")[:2000, :cfg.in_ch]
    t = torch.from_numpy(np.ascontiguousarray(a)).float()
    mean = t.mean((0, 2, 3)).view(1, -1, 1, 1).to(device)
    std = t.std((0, 2, 3)).clamp(min=1e-4).view(1, -1, 1, 1).to(device)
    return mean, std


@torch.no_grad()
def quick_val(model, loader, cfg, device, extra, wlat, norm=None):
    model.eval(); tot = 0.0; wn = 0.0; seen = 0
    for b in loader:
        spec = prep_audio(b["spec"].to(device), cfg, norm)
        gt = b["depth"].to(device); mask = b["mask"].to(device)
        D = model(spec, extra.get("coarse_feat"), extra.get("sh_basis"))["D"] * cfg.max_depth
        w = wlat * mask
        tot += ((D - gt * cfg.max_depth).abs() * w).sum().item(); wn += w.sum().item()
        seen += spec.size(0)
        if seen >= N_VAL:
            break
    return tot / max(wn, 1e-6)


def main():
    cfg = get_cfg()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(cfg.out_dir, cfg.run_name); os.makedirs(run_dir, exist_ok=True)

    # correction-branch precompute
    extra = {}
    if cfg.correction in ("cross", "cross_sup"):
        ccfg = copy.copy(cfg); ccfg.img_h, ccfg.img_w = cfg.coarse_h, cfg.coarse_w
        cbank = RayBank(ccfg, device=device)
        cfg.coarse_feat_dim = cbank.feat_dim
        extra["coarse_feat"] = cbank.feat
    elif cfg.correction == "sh":
        shg = SHGrid(cfg.img_h, cfg.img_w, order=cfg.corr_sh_order)
        extra["sh_basis"] = torch.from_numpy(shg.B).to(device)          # (N,Kc)
        extra["sh_pinv"] = torch.from_numpy(shg.B_pinv).to(device)       # (Kc,N)
    norm = chan_stats(cfg, device) if cfg.chan_norm else None
    print(f"[cfg] correction={cfg.correction} {vars(cfg)}", flush=True)

    model = FullMapNet(cfg).to(device)
    if cfg.init_decoder:
        warm_start(model, cfg.init_decoder, cfg.freeze_decoder, device)
    print(f"[model] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    tr = make_loader(cfg, "train", shuffle=True)
    va = make_loader(cfg, "val", shuffle=False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = cfg.epochs * len(tr); warm = max(1, len(tr))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    wlat = cos_lat(cfg.img_h, device).view(1, 1, cfg.img_h, 1)

    best = 1e9; hist = []
    for ep in range(cfg.epochs):
        model.train(); t0 = time.time(); run = {}
        for b in tr:
            spec = prep_audio(b["spec"].to(device, non_blocking=True), cfg, norm)
            gt = b["depth"].to(device); mask = b["mask"].to(device)
            out = model(spec, extra.get("coarse_feat"), extra.get("sh_basis"))
            loss = masked_mae(out["D"], gt, mask)
            logs = {"mae": float(loss.detach())}
            if cfg.correction == "sh":
                gt_coef = (gt.view(gt.size(0), -1) @ extra["sh_pinv"].T)     # (B,Kc)
                aux = (out["extras"]["coef"] - gt_coef).abs().mean()
                loss = loss + cfg.w_sh_aux * aux; logs["shaux"] = float(aux.detach())
            if cfg.correction == "cross_sup":
                dc = out["extras"]["Dcorr"]                                  # (B,1,H,W)
                rtgt = gaussian_blur_erp((gt - out["D0"].detach()), 3.0).clamp(-cfg.res_scale, cfg.res_scale)
                rsup = masked_mae(dc, rtgt, mask); tvl = tv(dc)
                loss = loss + cfg.w_res_sup * rsup + cfg.w_tv * tvl
                logs["rsup"] = float(rsup.detach()); logs["tv"] = float(tvl.detach())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            for k, v in logs.items():
                run[k] = run.get(k, 0.0) + v
        run = {k: v / len(tr) for k, v in run.items()}
        vmae = quick_val(model, va, cfg, device, extra, wlat, norm)
        a = out["extras"].get("alpha")
        hist.append({"epoch": ep, "val_mae_m": vmae, "alpha": a, **run})
        print(f"[ep {ep:02d}] {time.time()-t0:5.1f}s  {run}  val_MAE={vmae:.4f}m"
              + (f"  alpha={a:+.3f}" if a is not None else ""), flush=True)
        if vmae < best:
            best = vmae
            torch.save({"state_dict": model.state_dict(), "cfg": vars(cfg),
                        "norm": (norm[0].cpu(), norm[1].cpu()) if norm is not None else None},
                       os.path.join(run_dir, "best.pth"))
    json.dump({"best_val_mae_m": best, "hist": hist, "cfg": vars(cfg)},
              open(os.path.join(run_dir, "train_done.json"), "w"), indent=2)
    print(f"[done] best val MAE = {best:.4f} m -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()

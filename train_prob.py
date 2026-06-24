"""Train the probabilistic coarse-layout head (model_prob.ProbCoarseNet).

  python train_prob.py --run-name P_k5 --prob-k 5 --epochs 25 --lr 2e-3

Loss = relaxed Winner-Take-All over K hypotheses (best-of-K coverage of the multi-modal
coarse layout) + Laplace NLL on the winner (calibrated aleatoric uncertainty).
Validation tracks best-of-K MAE in METRES (the quantity this head is meant to lower).
"""

import os
import sys
import json
import math
import time
import numpy as np
import torch
import torch.nn.functional as F

from config import get_cfg
from data import make_loader, apply_audio_mode, shuffle_audio_batch
from metrics import cos_lat
from model_prob import ProbCoarseNet

N_VAL = 1500


def prep_audio(spec, cfg):
    if spec.shape[1] > cfg.in_ch:
        spec = spec[:, :cfg.in_ch]
    spec = apply_audio_mode(spec, cfg.audio_mode)
    if cfg.shuffle_audio:
        spec = shuffle_audio_batch(spec)
    return spec


def per_head_mae(mu, gt, mask):
    """mu (B,K,H,W), gt/mask (B,1,H,W) -> per-head masked MAE (B,K)."""
    ae = (mu - gt).abs() * mask
    return ae.flatten(2).sum(2) / mask.flatten(1).sum(1, keepdim=True).clamp(min=1e-6)


def prob_loss(out, gt, mask, cfg):
    mu, logb = out["mu"], out["logb"]
    B, K = mu.shape[0], mu.shape[1]
    mae_k = per_head_mae(mu, gt, mask)                      # (B,K)
    best, win = mae_k.min(1)                                # (B,)
    wta = (1 - cfg.prob_eps) * best + cfg.prob_eps * mae_k.mean(1)   # relaxed WTA
    # Laplace NLL on the winning hypothesis (per-pixel scale b)
    mu_win = mu.gather(1, win.view(B, 1, 1, 1).expand(-1, -1, mu.shape[2], mu.shape[3]))
    b = F.softplus(logb) + 1e-3
    nll = (((mu_win - gt).abs() / b + torch.log(b)) * mask).sum() / mask.sum().clamp(min=1e-6)
    loss = wta.mean() + cfg.prob_w_nll * nll
    return loss, {"wta": float(wta.mean().detach()), "bestK": float(best.mean().detach()),
                  "nll": float(nll.detach()), "diversity": float(mu.std(1).mean().detach())}


@torch.no_grad()
def quick_val(model, loader, cfg, device, wlat):
    """best-of-K MAE in metres (cos-lat weighted, matches the headline metric)."""
    model.eval(); tot = 0.0; wn = 0.0; seen = 0
    for b in loader:
        spec = prep_audio(b["spec"].to(device), cfg)
        gt = b["depth"].to(device) * cfg.max_depth; mask = b["mask"].to(device)
        mu = model(spec)["mu"] * cfg.max_depth                       # (B,K,H,W)
        ae = (mu - gt).abs()                                         # (B,K,H,W)
        # per-sample best head by masked MAE, then cos-lat weighted error of that head
        mk = (ae * mask).flatten(2).sum(2) / mask.flatten(1).sum(1, keepdim=True).clamp(min=1e-6)
        win = mk.argmin(1)
        best = ae.gather(1, win.view(-1, 1, 1, 1).expand(-1, -1, gt.shape[2], gt.shape[3]))
        w = wlat * mask
        tot += (best * w).sum().item(); wn += w.sum().item()
        seen += spec.size(0)
        if seen >= N_VAL:
            break
    return tot / max(wn, 1e-6)


def main():
    cfg = get_cfg()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(cfg.out_dir, cfg.run_name); os.makedirs(run_dir, exist_ok=True)
    print(f"[cfg] PROB K={cfg.prob_k} eps={cfg.prob_eps} w_nll={cfg.prob_w_nll} "
          f"coarse={cfg.prob_coarse} head={cfg.prob_head_h}x{cfg.prob_head_w}", flush=True)

    model = ProbCoarseNet(cfg).to(device)
    print(f"[model] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    tr = make_loader(cfg, "train", shuffle=True)
    va = make_loader(cfg, "val", shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = cfg.epochs * len(tr); warm = max(1, len(tr))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    wlat = cos_lat(cfg.img_h, device).view(1, 1, cfg.img_h, 1)

    best = 1e9; hist = []
    for ep in range(cfg.epochs):
        model.train(); t0 = time.time(); run = {}
        for b in tr:
            spec = prep_audio(b["spec"].to(device, non_blocking=True), cfg)
            gt = b["depth"].to(device); mask = b["mask"].to(device)
            out = model(spec)
            loss, logs = prob_loss(out, gt, mask, cfg)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            for k, v in logs.items():
                run[k] = run.get(k, 0.0) + v
        run = {k: v / len(tr) for k, v in run.items()}
        vmae = quick_val(model, va, cfg, device, wlat)
        hist.append({"epoch": ep, "val_bestK_mae_m": vmae, **run})
        print(f"[ep {ep:02d}] {time.time()-t0:5.1f}s  {run}  val_bestK_MAE={vmae:.4f}m", flush=True)
        if vmae < best:
            best = vmae
            torch.save({"state_dict": model.state_dict(), "cfg": vars(cfg)},
                       os.path.join(run_dir, "best.pth"))
    json.dump({"best_val_bestK_mae_m": best, "hist": hist, "cfg": vars(cfg)},
              open(os.path.join(run_dir, "train_done.json"), "w"), indent=2)
    print(f"[done] best val best-of-K MAE = {best:.4f} m -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()

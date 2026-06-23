"""Full-grid evaluation + negative controls for a trained implicit model.

Loads out/<run>/best.pth, predicts the full ERP grid on the test split, and
reports the metric bank. With --controls it re-evaluates the SAME weights under
perturbed inputs (mono / left / right / shuffled / L-R swap) to probe whether
the model actually uses audio and binaural geometry.

Run: python eval.py --run-name A2_raymlp --controls True
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from types import SimpleNamespace

from data import make_loader, apply_audio_mode, shuffle_audio_batch
from ray_features import RayBank, log_depth_bins
from model import RayDepthModel
from metrics import MetricBank, cos_lat
from train import build_sh_coarse, predict_full


def load_model(run_dir, device):
    ck = torch.load(os.path.join(run_dir, "best.pth"), map_location="cpu", weights_only=False)
    cfg = SimpleNamespace(**ck["cfg"])
    bank = RayBank(cfg, device=device)
    centers = log_depth_bins(cfg.n_bins, device=device) if cfg.use_depth_bins else None
    m = RayDepthModel(cfg, bank.feat_dim, centers).to(device).eval()
    m.load_state_dict(ck["state_dict"])
    sh_full = build_sh_coarse(cfg, device) if cfg.model == "hybrid" else None
    return m, cfg, bank, sh_full


def mirror_w(x):
    """Mirror an ERP map (B,1,H,W) left<->right: az -> -az."""
    return torch.roll(torch.flip(x, dims=[-1]), shifts=1, dims=-1)


@torch.no_grad()
def eval_condition(model, loader, bank, cfg, sh_full, device, mode="stereo",
                   shuffle=False, swap=False, max_n=None):
    mb = MetricBank(bank.H, bank.W, cfg.max_depth, device=device)
    extra = {"swap_mirror_consistency": 0.0, "_n": 0}
    wlat = cos_lat(bank.H, device).view(1, 1, bank.H, 1)
    seen = 0
    for b in loader:
        spec0 = b["spec"].to(device); depth = b["depth"].to(device); mask = b["mask"].to(device)
        spec = apply_audio_mode(spec0, mode)
        if shuffle:
            spec = shuffle_audio_batch(spec)
        if swap:
            spec = spec.flip(1)                         # [L,R] -> [R,L]
        pred = predict_full(model, spec, bank, cfg, sh_full) * cfg.max_depth
        gt = depth * cfg.max_depth
        mb.add(pred, gt, mask)
        if swap:                                        # consistency vs mirrored stereo pred
            pred_o = predict_full(model, spec0, bank, cfg, sh_full) * cfg.max_depth
            w = wlat * mask
            d = ((pred - mirror_w(pred_o)).abs() * w).sum().item()
            extra["swap_mirror_consistency"] += d / w.sum().clamp(min=1e-6).item() * spec.size(0)
            extra["_n"] += spec.size(0)
        seen += spec0.size(0)
        if max_n and seen >= max_n:
            break
    res = mb.result()
    if extra["_n"]:
        res["swap_mirror_consistency"] = extra["swap_mirror_consistency"] / extra["_n"]
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--out-dir", default="out")
    p.add_argument("--controls", type=lambda s: s == "True", default=False)
    p.add_argument("--max-n", type=int, default=0)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(args.out_dir, args.run_name)
    model, cfg, bank, sh_full = load_model(run_dir, device)
    loader = make_loader(cfg, "test", shuffle=False)
    max_n = args.max_n or None

    report = {}
    report["test"] = eval_condition(model, loader, bank, cfg, sh_full, device,
                                    mode="stereo", max_n=max_n)
    print(f"[test] MAE={report['test']['MAE']:.4f} MAE_low={report['test']['MAE_low']:.4f} "
          f"SHcoefL1={report['test']['SHcoefL1']:.4f} delta1={report['test']['delta1']:.3f}",
          flush=True)

    if args.controls:
        for name, kw in [("mono", dict(mode="mono")), ("left", dict(mode="left")),
                         ("right", dict(mode="right")), ("shuffle", dict(shuffle=True)),
                         ("swap", dict(swap=True))]:
            r = eval_condition(model, loader, bank, cfg, sh_full, device, max_n=max_n, **kw)
            report[name] = r
            tail = (f"  swap_mirror_consistency={r['swap_mirror_consistency']:.4f}"
                    if "swap_mirror_consistency" in r else "")
            print(f"[{name}] MAE={r['MAE']:.4f} MAE_low={r['MAE_low']:.4f}{tail}", flush=True)

    json.dump(report, open(os.path.join(run_dir, "metrics_test.json"), "w"), indent=2)
    print(f"[done] -> {run_dir}/metrics_test.json", flush=True)


if __name__ == "__main__":
    main()

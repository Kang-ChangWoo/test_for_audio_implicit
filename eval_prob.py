"""Evaluate the probabilistic coarse head on full test.

Reports (plain masked MAE, metres):
  mean-of-K   : ensemble mean  (the deterministic-equivalent point estimate)
  best-of-K   : oracle pick of the closest hypothesis  <-- the key number
  per-head    : individual head MAEs (diversity check; heads must not collapse)
  oracle gain : mean-of-K - best-of-K  (how much the multi-modality buys)
Uncertainty calibration:
  corr(pred Laplace scale b, |error of mean|)  + MAE binned by predicted uncertainty.
Compares against the deterministic baseline (Aunet_s0 plain MAE 0.780).
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from types import SimpleNamespace

from data import make_loader, apply_audio_mode
from model_prob import ProbCoarseNet

BASELINE = 0.780  # Aunet_s0 deterministic plain MAE (metres)


def load(run_dir, device):
    ck = torch.load(os.path.join(run_dir, "best.pth"), map_location="cpu", weights_only=False)
    cfg = SimpleNamespace(**ck["cfg"])
    m = ProbCoarseNet(cfg).to(device).eval()
    m.load_state_dict(ck["state_dict"])
    return m, cfg


def pmae(x, gt, mask):  # (B,*,H,W) per-sample masked MAE over dims after batch
    e = ((x - gt).abs() * mask)
    return e.flatten(1).sum(1) / mask.flatten(1).sum(1).clamp(min=1e-6)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True); ap.add_argument("--out-dir", default="out")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(a.out_dir, a.run_name)
    model, cfg = load(run_dir, device)
    loader = make_loader(cfg, "test", shuffle=False)
    md = cfg.max_depth; K = cfg.prob_k

    n = 0
    s_mean = 0.0; s_best = 0.0; s_head = np.zeros(K); s_div = 0.0
    s_best_ctrl = 0.0  # CONTROL: best-of-K vs a MISMATCHED scene's GT (rules out "more heads = more lucky picks")
    # for calibration: collect per-sample (pred uncertainty, mean error)
    unc = []; err = []
    # binned calibration over pixels
    bins = np.linspace(0, 1, 11); be_num = np.zeros(10); be_den = np.zeros(10)
    for b in loader:
        spec = apply_audio_mode(b["spec"].to(device)[:, :cfg.in_ch], "stereo")
        gt = b["depth"].to(device) * md; mask = b["mask"].to(device)
        out = model(spec)
        mu = out["mu"] * md                                   # (B,K,H,W)
        bscale = (torch.nn.functional.softplus(out["logb"]) + 1e-3) * md   # (B,1,H,W)
        Dmean = mu.mean(1, keepdim=True)
        B = mu.shape[0]
        s_mean += float(pmae(Dmean, gt, mask).sum())
        mk = pmae(mu.reshape(B*K, 1, *mu.shape[2:]),
                  gt.repeat_interleave(K, 0), mask.repeat_interleave(K, 0)).view(B, K)
        s_best += float(mk.min(1).values.sum())
        s_head += mk.sum(0).cpu().numpy()
        # control: same K hypotheses, but scored against the next sample's GT/mask
        gt_c = torch.roll(gt, 1, 0); mask_c = torch.roll(mask, 1, 0)
        mk_c = pmae(mu.reshape(B*K, 1, *mu.shape[2:]),
                    gt_c.repeat_interleave(K, 0), mask_c.repeat_interleave(K, 0)).view(B, K)
        s_best_ctrl += float(mk_c.min(1).values.sum())
        s_div += float(mu.std(1).mean().item()) * B
        # calibration (use mean estimate)
        ae = (Dmean - gt).abs()
        for bi in range(B):
            m1 = mask[bi, 0] > 0.5
            unc.append(float(bscale[bi, 0][m1].mean())); err.append(float(ae[bi, 0][m1].mean()))
        # per-pixel binning: predicted scale (normalised 0..maxb) vs actual abs err
        bb = bscale[mask > 0.5].cpu().numpy(); ee = ae[mask > 0.5].cpu().numpy()
        if bb.size:
            nb = (bb - bb.min()) / (bb.max() - bb.min() + 1e-9)
            idx = np.clip((nb * 10).astype(int), 0, 9)
            for k in range(10):
                sel = idx == k; be_num[k] += ee[sel].sum(); be_den[k] += sel.sum()
        n += B

    unc = np.array(unc); err = np.array(err)
    calib = float(np.corrcoef(unc, err)[0, 1])
    res = {
        "mean_of_K": s_mean / n, "best_of_K": s_best / n,
        "best_of_K_ctrl": s_best_ctrl / n,   # mismatched-GT control
        "oracle_gain": (s_mean - s_best) / n,
        "oracle_gain_ctrl": (s_mean - s_best_ctrl) / n,   # gain explainable by free selection alone
        "real_multimodal_gain": (s_best_ctrl - s_best) / n,  # genuine sample-specific multimodality
        "per_head_mae": (s_head / n).tolist(), "diversity": s_div / n,
        "uncert_corr": calib, "baseline_det": BASELINE,
        "vs_baseline_meanK": s_mean / n - BASELINE, "vs_baseline_bestK": s_best / n - BASELINE,
        "calib_bins": (be_num / np.maximum(be_den, 1)).tolist(),
    }
    json.dump(res, open(os.path.join(run_dir, "prob_eval.json"), "w"), indent=2)
    print(f"\n=== {a.run_name}  (K={K}) ===")
    print(f"  deterministic baseline : {BASELINE:.4f} m")
    print(f"  mean-of-K              : {res['mean_of_K']:.4f} m   ({res['vs_baseline_meanK']:+.4f})")
    print(f"  best-of-K  (oracle)    : {res['best_of_K']:.4f} m   ({res['vs_baseline_bestK']:+.4f})  <-- key")
    print(f"  best-of-K  CONTROL     : {res['best_of_K_ctrl']:.4f} m   (vs mismatched GT)")
    print(f"  oracle gain (mean-best): {res['oracle_gain']:.4f} m")
    print(f"  -> free-selection gain : {res['oracle_gain_ctrl']:.4f} m   (artifact of K free picks)")
    print(f"  -> REAL multimodal gain: {res['real_multimodal_gain']:.4f} m   (ctrl - real; >0 = genuine)")
    print(f"  per-head MAE           : " + ", ".join(f"{x:.3f}" for x in res['per_head_mae']))
    print(f"  hypothesis diversity   : {res['diversity']:.4f}")
    print(f"  uncertainty corr(b,err): {res['uncert_corr']:+.3f}  (calibration; >0 = useful)")
    print(f"-> {run_dir}/prob_eval.json")


if __name__ == "__main__":
    main()

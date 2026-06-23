"""Stage-1 visualisations:
  fig_stage1_bars.png   val/test MAE: ray-only prior vs audio vs shuffled (+A0 ref)
  fig_qualitative.png   ERP depth panels GT | ray-only | RayMLP for sample scenes
  fig_controls.png      input-control sensitivity (stereo/mono/L/R/shuffle)
"""

import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import make_loader
from eval import load_model
from train import predict_full

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIG = "out/figs"; os.makedirs(FIG, exist_ok=True)
A0_DET = 0.802   # existing best full-map decoder (det_K1, test MAE, from audio_better)


def bars():
    def mean(runs, key="best_val_mae_m"):
        vs = [json.load(open(f"out/{r}/train_done.json"))[key] for r in runs]
        return np.mean(vs), np.std(vs)
    rows = [("ray-only\nprior", mean(["A1_rayonly_s0", "A1_rayonly_s1"]), "#888"),
            ("A2 RayMLP\n(global)", mean(["A2_raymlp_s0", "A2_raymlp_s1"]), "#5ad"),
            ("A2 RayMLP\nshuffled", mean(["A2_shuf_s0", "A2_shuf_s1"]), "#c54"),
            ("A4 cross\n-attn", mean(["A4_cross_s0", "A4_cross_s1"]), "#2a7"),
            ("A4 cross\nshuffled", mean(["A4_cross_shuf_s0"]), "#c54")]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(rows))
    ax.bar(x, [r[1][0] for r in rows], yerr=[r[1][1] for r in rows],
           color=[r[2] for r in rows], capsize=5, width=0.62)
    for i, r in enumerate(rows):
        ax.text(i, r[1][0] + 0.004, f"{r[1][0]:.3f}", ha="center", fontsize=10)
    ax.axhline(A0_DET, ls="--", c="k", lw=1)
    ax.text(len(rows) - 0.5, A0_DET + 0.004, f"A0 decoder {A0_DET:.2f}", ha="right", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel("val MAE [m]  (lower better)")
    ax.set_title("Ladder: prior → global-MLP → cross-attn (audio access ↑ → MAE ↓)\n"
                 "implicit uses audio (shuffled→prior); still above A0 decoder")
    ax.set_ylim(0.75, 1.20)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_stage1_bars.png", dpi=130)
    print(f"saved {FIG}/fig_stage1_bars.png")


@torch.no_grad()
def qualitative(n=5):
    # GT + every stage (A6 = current interim checkpoint)
    stages = [("A1 prior", "A1_rayonly_s0"), ("A2 RayMLP", "A2_raymlp_s0"),
              ("A4 cross", "A4_cross_s0"), ("A3 +SH", "A3_crossSH_s0"),
              ("A5 +mic", "A5_crossMic_s0"), ("A6 +self*", "A6_crossself_s0"),
              ("A8 hybrid", "A8_hybrid_s0"), ("A4 ffmask", "A4_ffmask_s0")]
    cfg0 = load_model("out/A1_rayonly_s0", DEV)[1]
    loader = make_loader(cfg0, "test", shuffle=False)
    batch = next(iter(loader))
    spec = batch["spec"][:n].to(DEV); gt = (batch["depth"][:n] * cfg0.max_depth)
    preds = []
    for _, r in stages:
        m, cfg, bank, shf = load_model(f"out/{r}", DEV)
        preds.append(predict_full(m, spec, bank, cfg, shf).cpu() * cfg.max_depth)
        del m; torch.cuda.empty_cache()
    ncol = len(stages) + 1
    fig, ax = plt.subplots(n, ncol, figsize=(2.0 * ncol, 1.9 * n))
    titles = ["GT depth"] + [s[0] for s in stages]
    for i in range(n):
        vmax = float(gt[i].max().clamp(min=1.0))
        imgs = [gt[i, 0]] + [p[i, 0] for p in preds]
        for j, img in enumerate(imgs):
            a = ax[i, j]
            im = a.imshow(img, cmap="turbo", vmin=0, vmax=vmax)
            a.set_xticks([]); a.set_yticks([])
            if i == 0:
                a.set_title(titles[j], fontsize=10)
            if j == 0:
                a.set_ylabel(batch["key"][i].split("/")[0][:8], fontsize=8)
        plt.colorbar(im, ax=ax[i, -1], fraction=0.046, pad=0.04)
    fig.suptitle("ERP radial depth — all stages (A6* = interim). cross sharpens layout; "
                 "SH/mic/hybrid look ~identical to cross", y=1.0)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_qualitative.png", dpi=125)
    print(f"saved {FIG}/fig_qualitative.png")


def controls():
    p = "out/A4_cross_s0/metrics_test.json"
    if not os.path.exists(p):
        print("no controls json; run eval.py --controls True first"); return
    j = json.load(open(p))
    order = [("stereo", "test"), ("mono", "mono"), ("left", "left"),
             ("right", "right"), ("shuffle", "shuffle"), ("swap", "swap")]
    labels = [o[0] for o in order if o[1] in j]
    mae = [j[o[1]]["MAE"] for o in order if o[1] in j]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    cols = ["#2a7", "#5ad", "#5ad", "#5ad", "#c54", "#a8a"][:len(mae)]
    ax.bar(labels, mae, color=cols)
    for i, v in enumerate(mae):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("test MAE [m]"); ax.set_ylim(min(mae) - 0.02, max(mae) + 0.02)
    ax.set_title("Input controls (same weights):\nA4 cross-attn: stereo≪mono (binaural strongly used), swap breaks it")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_controls.png", dpi=130)
    print(f"saved {FIG}/fig_controls.png")


if __name__ == "__main__":
    bars(); qualitative(); controls()

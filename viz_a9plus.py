"""Visualise the A9+ phase (A0 full-map decoder + audio corrections).
  fig_a9plus_bars.png  : plain MAE / MAE_low / alpha across A9-A12 (+A0 ref)
  fig_a9plus_qual.png  : GT | A9 | A10 | A11 | A12 ERP depth + A10 correction map
"""
import os, json
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from data import make_loader, apply_audio_mode
from eval_fullmap import load

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIG = "out/figs"; A0 = 0.802
RUNS = [("A9\nA0-repro", "A9_fullmap_s0"), ("A10\n+cross-resid", "A10_cross_s0"),
        ("A11\n+SH-aux", "A11_shaux_s0"), ("A12\n+FiLM", "A12_film_s0")]


def bars():
    mp, lo, al, names = [], [], [], []
    for n, r in RUNS:
        j = json.load(open(f"out/{r}/metrics_test.json"))
        names.append(n); mp.append(j["test"]["MAE_plain"]); lo.append(j["test"]["MAE_low"])
        al.append(j.get("alpha"))
    x = np.arange(len(names))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].bar(x - 0.2, mp, 0.4, label="MAE (plain)", color="#39c")
    ax[0].bar(x + 0.2, lo, 0.4, label="MAE_low (coarse)", color="#fa3")
    for i, (a, b) in enumerate(zip(mp, lo)):
        ax[0].text(i - 0.2, a + 0.002, f"{a:.3f}", ha="center", fontsize=8)
        ax[0].text(i + 0.2, b + 0.002, f"{b:.3f}", ha="center", fontsize=8)
    ax[0].axhline(A0, ls="--", c="k", lw=1); ax[0].text(len(x)-.5, A0+.003, "A0 0.802", ha="right", fontsize=8)
    ax[0].axhline(mp[0], ls=":", c="#555", lw=1)
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, fontsize=8.5); ax[0].set_ylim(0.74, 0.84)
    ax[0].set_ylabel("test error [m]"); ax[0].legend(fontsize=8)
    ax[0].set_title("A9+ : corrections don't beat the A0 baseline (A9)")
    # alpha (correction strength) — only A10/A11 have it
    ai = [(names[i], al[i]) for i in range(len(names)) if al[i] is not None]
    ax[1].bar([a[0] for a in ai], [abs(a[1]) for a in ai], color="#7a7")
    for i, a in enumerate(ai):
        ax[1].text(i, abs(a[1]) + 0.002, f"{a[1]:+.3f}", ha="center", fontsize=9)
    ax[1].set_ylabel("|correction gate alpha|  (0 = branch unused)")
    ax[1].set_title("learned correction strength (started at 0)")
    fig.suptitle("A9-A12: full-map decoder + zero-init audio correction", y=1.02)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_a9plus_bars.png", dpi=130, bbox_inches="tight")
    print(f"saved {FIG}/fig_a9plus_bars.png")


@torch.no_grad()
def qual(n=4):
    models = [(nm.replace("\n", " "), *load(f"out/{r}", DEV)) for nm, r in RUNS]
    cfg0 = models[0][2]
    loader = make_loader(cfg0, "test", shuffle=False)
    b = next(iter(loader))
    spec = b["spec"][:n].to(DEV); gt = b["depth"][:n, 0] * cfg0.max_depth
    cols = ["GT"] + [m[0] for m in models] + ["A10 corr."]
    fig, ax = plt.subplots(n, len(cols), figsize=(2.0 * len(cols), 1.9 * n))
    for i in range(n):
        vmax = float(gt[i].max().clamp(min=1.0)); imgs = [gt[i].cpu()]
        corr = None
        for nm, m, cfg, extra in models:
            sp = apply_audio_mode(spec, "stereo")
            if sp.shape[1] > getattr(cfg,"in_ch",2): sp = sp[:, :getattr(cfg,"in_ch",2)]
            out = m(sp, extra.get("coarse_feat"), extra.get("sh_basis"))
            imgs.append((out["D"][i, 0] * cfg.max_depth).cpu())
            if "Dcorr" in out.get("extras", {}):
                corr = out["extras"]["Dcorr"][i, 0].cpu()
        imgs.append(corr if corr is not None else torch.zeros_like(gt[i].cpu()))
        for j, img in enumerate(imgs):
            cmap = "coolwarm" if j == len(cols) - 1 else "turbo"
            vm = None if j == len(cols) - 1 else vmax
            a = ax[i, j]; a.imshow(img, cmap=cmap, vmin=0 if vm else None, vmax=vm)
            a.set_xticks([]); a.set_yticks([])
            if i == 0: a.set_title(cols[j], fontsize=9)
    fig.suptitle("A9-A12 ERP depth — corrections barely change the A0 prediction", y=1.0)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_a9plus_qual.png", dpi=120, bbox_inches="tight")
    print(f"saved {FIG}/fig_a9plus_qual.png")


if __name__ == "__main__":
    bars(); qual()

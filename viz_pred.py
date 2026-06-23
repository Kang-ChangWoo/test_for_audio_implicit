"""Predicted ERP depth for the KEY final models (mixed model types):
  GT | A9 A0-decoder | A6 self-attn(best) | A4 cross | A14 sup-resid | A2 RayMLP(worst)
"""
import os, json
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from data import make_loader, apply_audio_mode
import eval as ev_impl                # RayDepthModel loader + predict
import eval_fullmap as ev_fm          # FullMapNet loader
from train import predict_full

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIG = "out/figs"

# (label, run, type) type: 'impl'=RayDepthModel(eval.py), 'fm'=FullMapNet(eval_fullmap)
COLS = [("U-Net (skip)", "Aunet_s0", "fm"),
        ("A6 self-attn", "A6_crossself_s0", "impl"),
        ("A4 cross", "A4_cross_s0", "impl"),
        ("A9 A0-decoder", "A9_fullmap_s0", "fm"),
        ("A2 RayMLP", "A2_raymlp_s0", "impl")]


@torch.no_grad()
def main(n=5):
    # shared batch from the default (2ch) cache
    base_cfg = ev_impl.load_model("out/A4_cross_s0", DEV)[1]
    loader = make_loader(base_cfg, "test", shuffle=False)
    b = next(iter(loader))
    spec2 = b["spec"][:n].to(DEV); gt = b["depth"][:n, 0] * base_cfg.max_depth

    preds = []
    for lab, run, typ in COLS:
        if typ == "impl":
            m, cfg, bank, shf = ev_impl.load_model(f"out/{run}", DEV)
            sh = None
            P = predict_full(m, spec2, bank, cfg, sh).cpu() * cfg.max_depth
        else:
            m, cfg, extra = ev_fm.load(f"out/{run}", DEV)
            sp = spec2
            if "norm" in extra:
                sp = (sp - extra["norm"][0]) / extra["norm"][1]
            P = m(sp, extra.get("coarse_feat"), extra.get("sh_basis"))["D"].cpu() * cfg.max_depth
        preds.append(P[:, 0])
        del m; torch.cuda.empty_cache()

    titles = ["GT depth"] + [c[0] for c in COLS]
    fig, ax = plt.subplots(n, len(titles), figsize=(2.05 * len(titles), 1.9 * n))
    for i in range(n):
        vmax = float(gt[i].max().clamp(min=1.0))
        imgs = [gt[i]] + [p[i] for p in preds]
        for j, img in enumerate(imgs):
            a = ax[i, j]; im = a.imshow(img, cmap="turbo", vmin=0, vmax=vmax)
            a.set_xticks([]); a.set_yticks([])
            if i == 0:
                a.set_title(titles[j], fontsize=9.5)
            if j == 0:
                a.set_ylabel(b["key"][i].split("/")[0][:8], fontsize=8)
        plt.colorbar(im, ax=ax[i, -1], fraction=0.046, pad=0.04)
    fig.suptitle("Predicted ERP radial depth — best (A6 self-attn) vs A0 decoder vs others. "
                 "All recover room-scale; fine layout stays blobby (observability limit).", y=1.0)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_pred_depth.png", dpi=125, bbox_inches="tight")
    print(f"saved {FIG}/fig_pred_depth.png")


if __name__ == "__main__":
    main()

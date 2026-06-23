"""~100-sample GT|pred montage across diverse test scenes (default model: U-Net)."""
import os, sys, json
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from data import make_loader, apply_audio_mode
import eval_fullmap as ev_fm
import eval as ev_impl
from train import predict_full

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIG = "out/figs"
RUN = sys.argv[1] if len(sys.argv) > 1 else "Aunet_s0"
TYP = sys.argv[2] if len(sys.argv) > 2 else "fm"      # fm | impl
N = 100


@torch.no_grad()
def main():
    if TYP == "fm":
        m, cfg, extra = ev_fm.load(f"out/{RUN}", DEV)
        pred = lambda sp: m(sp, extra.get("coarse_feat"), extra.get("sh_basis"))["D"]
    else:
        m, cfg, bank, _ = ev_impl.load_model(f"out/{RUN}", DEV)
        pred = lambda sp: predict_full(m, sp, bank, cfg, None)
    ds = make_loader(cfg, "test", shuffle=False).dataset
    # span scenes: evenly strided indices across the test set
    keys = ds.keys; total = len(keys)
    sel = list(range(0, total, max(1, total // N)))[:N]
    scenes = sorted(set(k.split("/")[0] for k in keys))
    print(f"{total} test samples, {len(scenes)} scenes, showing {len(sel)} strided", flush=True)

    cols = 10              # 5 GT|pred pairs per row
    rows = (len(sel) + 4) // 5
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 0.95))
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    for n, si in enumerate(sel):
        s = ds[si]
        sp = s["spec"][None].to(DEV)
        if sp.shape[1] > getattr(cfg, "in_ch", 2):
            sp = sp[:, :getattr(cfg, "in_ch", 2)]
        if "norm" in (extra if TYP == "fm" else {}):
            sp = (sp - extra["norm"][0]) / extra["norm"][1]
        gt = (s["depth"][0] * cfg.max_depth).numpy()
        P = (pred(sp)[0, 0] * cfg.max_depth).cpu().numpy()
        vmax = max(float(gt.max()), 1.0)
        r = n // 5; c = (n % 5) * 2
        ax[r, c].imshow(gt, cmap="turbo", vmin=0, vmax=vmax)
        ax[r, c + 1].imshow(P, cmap="turbo", vmin=0, vmax=vmax)
        if r == 0:
            ax[r, c].set_title("GT", fontsize=7); ax[r, c + 1].set_title("pred", fontsize=7)
        ax[r, c].set_ylabel(s["key"].split("/")[0][:6], fontsize=5)
    for k in range(len(sel) * 2, rows * cols):
        ax.ravel()[k].axis("off")
    fig.suptitle(f"{RUN} — GT|pred across {len(sel)} strided test samples "
                 f"({len(scenes)} scenes). Recovers room-scale; fine layout blobby.", y=1.005, fontsize=11)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_pred_montage.png", dpi=110, bbox_inches="tight")
    print(f"saved {FIG}/fig_pred_montage.png")


if __name__ == "__main__":
    main()

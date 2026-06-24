"""~100-sample predicted-depth montage, GT + N models, split into multiple files
of 10 rows each for readability.
  python viz_montage.py            -> fig_pred_montage_*  (best models)
  python viz_montage.py tips       -> fig_tips_montage_*  (tip models + U-Net)
"""
import os, sys
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from data import make_loader
import eval as ev_impl
import eval_fullmap as ev_fm
from train import predict_full

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIG = "out/figs"
N = 100
PER = 10                                      # rows (samples) per file

MODE = sys.argv[1] if len(sys.argv) > 1 else "main"
if MODE == "tips":
    PREFIX = "fig_tips_montage"
    COLS = [("A4 base", "A4_cross_s0", "impl"), ("#4 sector", "T_sector", "impl"),
            ("#5 FiLM", "T_film", "impl"), ("#6 MLPskip", "T_mlpskip", "impl"),
            ("#3 progPE", "T_progpe", "impl"), ("ALL", "T_all", "impl"),
            ("U-Net(skip)", "Aunet_s0", "fm")]
else:
    PREFIX = "fig_pred_montage"
    COLS = [("U-Net", "Aunet_s0", "fm"), ("A6 self-attn", "A6_crossself_s0", "impl"),
            ("A4 cross", "A4_cross_s0", "impl"), ("A9 decoder", "A9_fullmap_s0", "fm"),
            ("A2 RayMLP", "A2_raymlp_s0", "impl")]


@torch.no_grad()
def load_all():
    models = []
    for lab, run, typ in COLS:
        if typ == "impl":
            m, cfg, bank, _ = ev_impl.load_model(f"out/{run}", DEV)
            models.append((lab, "impl", m, cfg, bank, {}))
        else:
            m, cfg, extra = ev_fm.load(f"out/{run}", DEV)
            models.append((lab, "fm", m, cfg, extra, None))
    return models


@torch.no_grad()
def predict(modelspec, spec):
    lab, typ, m, cfg, aux, _ = modelspec
    sp = spec
    if sp.shape[1] > getattr(cfg, "in_ch", 2):
        sp = sp[:, :getattr(cfg, "in_ch", 2)]
    if typ == "fm":
        if "norm" in aux:
            sp = (sp - aux["norm"][0]) / aux["norm"][1]
        return (m(sp, aux.get("coarse_feat"), aux.get("sh_basis"))["D"] * cfg.max_depth).cpu()
    return (predict_full(m, sp, aux, cfg, None) * cfg.max_depth).cpu()


@torch.no_grad()
def main():
    models = load_all()
    cfg0 = models[0][3]
    ds = make_loader(cfg0, "test", shuffle=False).dataset
    total = len(ds.keys)
    sel = list(range(0, total, max(1, total // N)))[:N]
    titles = ["GT"] + [c[0] for c in COLS]
    nfile = (len(sel) + PER - 1) // PER
    print(f"{total} samples -> {len(sel)} strided, {nfile} files", flush=True)

    for f in range(nfile):
        chunk = sel[f * PER:(f + 1) * PER]
        spec = torch.stack([ds[i]["spec"] for i in chunk]).to(DEV)
        gts = [(ds[i]["depth"][0] * cfg0.max_depth).numpy() for i in chunk]
        keys = [ds[i]["key"] for i in chunk]
        preds = [predict(ms, spec) for ms in models]       # list of (B,1,H,W)
        rows = len(chunk); ncol = len(COLS) + 1
        fig, ax = plt.subplots(rows, ncol, figsize=(ncol * 1.8, rows * 1.0))
        if rows == 1:
            ax = ax[None]
        for r in range(rows):
            vmax = max(float(gts[r].max()), 1.0)
            imgs = [gts[r]] + [preds[c][r, 0].numpy() for c in range(len(COLS))]
            for cc in range(ncol):
                a = ax[r, cc]; a.imshow(imgs[cc], cmap="turbo", vmin=0, vmax=vmax)
                a.set_xticks([]); a.set_yticks([])
                if r == 0:
                    a.set_title(titles[cc], fontsize=8)
                if cc == 0:
                    a.set_ylabel(keys[r].split("/")[0][:7], fontsize=6)
        fig.suptitle(f"{PREFIX} (file {f+1}/{nfile}, samples {f*PER}-{f*PER+rows-1})", y=1.01, fontsize=10)
        fig.tight_layout(); fig.savefig(f"{FIG}/{PREFIX}_{f:02d}.png", dpi=115, bbox_inches="tight")
        print(f"saved {PREFIX}_{f:02d}.png ({rows} rows)", flush=True)


if __name__ == "__main__":
    main()

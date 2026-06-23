"""Consolidated master summary figure for the implicit experiment (all results
so far). 2x2 panels:
  (A) full ladder val MAE A1->A8 (+A0 conv-decoder ref)
  (B) test MAE vs low-freq MAE for converged models
  (C) A4 cross-attn input controls (binaural / audio sensitivity)
  (D) SH-coefficient (coarse-layout) error
A6 self-attn uses its train-val best (in-progress) where test metrics are absent.
"""
import json, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

FIG = "out/figs"; A0 = 0.8018


def val_best(run):
    p = f"out/{run}/train_done.json"
    if os.path.exists(p):
        return json.load(open(p))["best_val_mae_m"], True
    f = f"logs/{run}.log"
    if os.path.exists(f):
        vs = [float(m.group(1)) for l in open(f)
              if (m := re.search(r"val_MAE=([0-9.]+)", l))]
        return (min(vs), False) if vs else (None, False)
    return None, False


def test_m(run, key):
    p = f"out/{run}/metrics_test.json"
    return json.load(open(p))["test"][key] if os.path.exists(p) else None


fig, ax = plt.subplots(2, 2, figsize=(15, 9))

# unified stage list used by panels A, B, D (every stage in every panel)
STAGES = [("A1\nprior", "A1_rayonly_s0", "#888"), ("A2 RayMLP\nglobal", "A2_raymlp_s0", "#5ad"),
          ("A4 cross\n-attn", "A4_cross_s0", "#2a7"), ("A3 +SH\n-PE", "A3_crossSH_s0", "#2a7"),
          ("A5 +mic\n-PE", "A5_crossMic_s0", "#2a7"), ("A6 +self\n-attn", "A6_crossself_s0", "#36b"),
          ("A8 hybrid\nSH+res", "A8_hybrid_s0", "#e9a"), ("A4 ff\nmask", "A4_ffmask_s0", "#7c7")]
labels = [s[0] for s in STAGES]; runs = [s[1] for s in STAGES]; cols = [s[2] for s in STAGES]
fin = [os.path.exists(f"out/{r}/train_done.json") for r in runs]
hatch = ["" if f else "//" for f in fin]
xs = np.arange(len(STAGES))

def annotate_interim(axis, y0):
    for i, f in enumerate(fin):
        if not f:
            axis.text(i, y0, "ep1*", ha="center", fontsize=6.5, color="#36b")

# (A) ladder (val MAE) -----------------------------------------------------
a = ax[0, 0]
for i, r in enumerate(runs):
    v, _ = val_best(r)
    a.bar(i, v, color=cols[i], hatch=hatch[i], edgecolor="white", width=0.66)
    a.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=8)
annotate_interim(a, 0.792)
a.axhline(A0, ls="--", c="k", lw=1); a.text(len(xs) - .4, A0 + .004, "A0 decoder 0.80", ha="right", fontsize=8)
a.set_xticks(xs); a.set_xticklabels(labels, fontsize=8); a.set_ylim(0.78, 1.18)
a.set_ylabel("val MAE [m]")
a.set_title("(A) Full ladder A1→A8 (val MAE) — climbs prior→global→cross, then plateaus")

# (B) test MAE vs MAE_low --------------------------------------------------
b = ax[0, 1]
mae = [test_m(r, "MAE") for r in runs]; low = [test_m(r, "MAE_low") for r in runs]
b.bar(xs - .2, mae, .4, label="MAE", color="#39c", hatch=hatch)
b.bar(xs + .2, low, .4, label="MAE_low (coarse)", color="#fa3", hatch=hatch)
for i, (m, lo) in enumerate(zip(mae, low)):
    b.text(i - .2, m + .006, f"{m:.2f}", ha="center", fontsize=6.5)
    b.text(i + .2, lo + .006, f"{lo:.2f}", ha="center", fontsize=6.5)
b.axhline(A0, ls="--", c="k", lw=1); b.text(len(xs) - .4, A0 + .006, "A0 0.80", ha="right", fontsize=8)
b.set_xticks(xs); b.set_xticklabels(labels, fontsize=8); b.set_ylim(0.75, 1.20)
b.set_ylabel("test error [m]"); b.legend(fontsize=8, loc="upper left")
b.set_title("(B) Test MAE & low-freq MAE (all stages) — cross best; SH/mic/hybrid add nothing")

# (C) controls -------------------------------------------------------------
c = ax[1, 0]; j = json.load(open("out/A4_cross_s0/metrics_test.json"))
order = [("stereo", "test"), ("mono", "mono"), ("left", "left"),
         ("right", "right"), ("shuffle", "shuffle"), ("swap", "swap")]
clab = [o[0] for o in order if o[1] in j]; cval = [j[o[1]]["MAE"] for o in order if o[1] in j]
ccol = ["#2a7", "#5ad", "#5ad", "#5ad", "#c54", "#a8a"][:len(cval)]
c.bar(clab, cval, color=ccol)
for i, v in enumerate(cval):
    c.text(i, v + .003, f"{v:.3f}", ha="center", fontsize=8)
c.set_ylim(min(cval) - .02, max(cval) + .03); c.set_ylabel("test MAE [m]")
c.set_title("(C) A4 cross controls — stereo≪mono (binaural used), shuffle/swap break it")

# (D) SH-coef --------------------------------------------------------------
d = ax[1, 1]
sh = [test_m(r, "SHcoefL1") for r in runs]
d.bar(xs, sh, .55, color="#7a7", hatch=hatch)
for i, v in enumerate(sh):
    d.text(i, v + .0006, f"{v:.3f}", ha="center", fontsize=7)
annotate_interim(d, 0.3405)
d.axhline(sh[2], ls=":", c="#2a7", lw=1)   # cross reference line
d.set_xticks(xs); d.set_xticklabels(labels, fontsize=8); d.set_ylim(0.34, 0.40)
d.set_ylabel("SH-coefficient L1 error")
d.set_title("(D) Coarse-layout (SH-coef) error (all stages) — flat vs cross → priors don't help")

fig.suptitle("Implicit ray audio→ERP-depth — verdict: cross-attention is the only win; "
             "beyond it the bottleneck is audio observability, not architecture", fontsize=12)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_master_summary.png", dpi=130)
print(f"saved {FIG}/fig_master_summary.png")

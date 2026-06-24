"""Visualise the analysis that motivated the probabilistic coarse head.
Writes standalone figures to out/figs/analysis/.

  fig01_oracle_decomp : how much MAE each PERFECT global-transform fix recovers
                        (+ azimuth-roll control proving it's an oracle artifact)
  fig02_coarse_fine   : error splits into recovered coarse layout + unobservable fine detail
  fig03_round_fail    : swap-eq / chan-norm / mic-PE all fail on MAE AND handedness
  fig04_tips          : only prog-PE hurts; its PE-anneal schedule is anti-aligned with LR decay
"""
import os, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "out/figs/analysis"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def fig_oracle():
    d = json.load(open("out/oracle_decomp.json"))["Aunet_s0"]
    raw = d["raw"]
    rows = [("optimal affine\n(scale+offset)", d["affine"]), ("L/R mirror\n(handedness)", d["mirror"]),
            ("U/D vflip\n(elevation)", d["vflip"]), ("best azimuth-roll\n(orientation)", d["bestroll"])]
    labels = [r[0] for r in rows]; gains = [raw - r[1] for r in rows]   # recovered metres (positive=good)
    ctrl = raw - d["bestroll_ctrl"]                                     # roll on MISMATCHED gt
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    colors = ["#4c72b0"]*3 + ["#c44e52"]
    y = np.arange(len(rows))
    ax.barh(y, gains, color=colors)
    ax.barh(len(rows), ctrl, color="#888", hatch="///", edgecolor="k")
    for i, g in enumerate(gains):
        ax.text(g+0.001, i, f"-{g:.3f} m", va="center", fontsize=9)
    ax.text(ctrl+0.001, len(rows), f"-{ctrl:.3f} m  (CONTROL: roll vs WRONG scene)", va="center", fontsize=9)
    ax.set_yticks(list(y)+[len(rows)])
    ax.set_yticklabels(labels+["best azimuth-roll\nCONTROL (mismatched GT)"])
    ax.invert_yaxis()
    ax.set_xlabel("MAE recovered if this factor is PERFECTLY fixed  (metres)")
    ax.set_title(f"Oracle error decomposition (Aunet_s0, raw MAE={raw:.3f} m)\n"
                 "azimuth-roll looks big but its control recovers the SAME -> oracle artifact, not real")
    ax.axvline(0, color="k", lw=0.8)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig01_oracle_decomp.png", dpi=130); plt.close(fig)
    print("wrote fig01_oracle_decomp.png")


def fig_coarse_fine():
    # measured (analyze2 on Aunet_s0)
    coarse_err, coarse_gt = 0.680, 2.194
    fine_err, fine_gt, fine_pred = 0.234, 0.231, 0.049
    raw, predtpl, gttpl = 0.780, 0.997, 1.016
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4))
    # left: coarse vs fine - GT energy, model error, model emitted energy
    x = np.arange(2); w = 0.27
    ax[0].bar(x-w, [coarse_gt, fine_gt], w, label="GT energy present", color="#55a868")
    ax[0].bar(x,   [coarse_err, fine_err], w, label="model MAE (error)", color="#c44e52")
    ax[0].bar(x+w, [coarse_gt-coarse_err, fine_pred], w, label="model recovers / emits", color="#4c72b0")
    ax[0].set_xticks(x); ax[0].set_xticklabels(["COARSE layout\n(blur σ3)", "FINE detail\n(residual)"])
    ax[0].set_ylabel("depth (metres)"); ax[0].legend(fontsize=8)
    ax[0].set_title("Coarse is ~69% recovered; fine detail\nis present in GT but model emits ~0 -> unobservable")
    ax[0].text(1, fine_err+0.03, "error ≈ all of GT's\nfine energy", ha="center", fontsize=8)
    # right: scene-specificity ladder
    bars = [("best static\nGT template", gttpl), ("model -> its\nown avg map", predtpl),
            ("model (raw)", raw)]
    ax[1].bar([b[0] for b in bars], [b[1] for b in bars],
              color=["#bbb", "#c0a0d0", "#4c72b0"])
    ax[1].axhline(raw, color="#4c72b0", ls="--", lw=0.8)
    ax[1].set_ylabel("MAE (metres)")
    ax[1].set_title("Model carries +0.217 m scene-specific signal\n(beats best static template by 0.236 m)")
    for i, b in enumerate(bars):
        ax[1].text(i, b[1]+0.01, f"{b[1]:.3f}", ha="center", fontsize=9)
    ax[1].set_ylim(0.7, 1.06)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig02_coarse_fine.png", dpi=130); plt.close(fig)
    print("wrote fig02_coarse_fine.png")


def fig_round_fail():
    # (name, plain MAE, mirror_rate, is_baseline)
    data = [("U-Net\n(base)", 0.773, 0.289, 1), ("U-Net\n+swap-eq", 0.7755, 0.295, 0),
            ("A13 ipd5\n(base)", 0.781, 0.291, 1), ("A13\n+chan-norm", 0.7919, 0.302, 0),
            ("A6 cross\n(base)", 0.7757, 0.243, 1), ("A6\n+mic-PE", 0.7834, 0.253, 0)]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    x = np.arange(len(data))
    col = ["#4c72b0" if b else "#c44e52" for *_, b in data]
    ax[0].bar(x, [d[1] for d in data], color=col)
    ax[0].axhline(0.773, color="#888", ls="--", lw=0.8, label="best (U-Net 0.773)")
    ax[0].set_xticks(x); ax[0].set_xticklabels([d[0] for d in data], fontsize=8)
    ax[0].set_ylabel("plain test MAE (m)"); ax[0].set_ylim(0.76, 0.80)
    ax[0].set_title("MAE: every intervention (red) ≥ its baseline (blue)"); ax[0].legend(fontsize=8)
    ax[1].bar(x, [d[2] for d in data], color=col)
    ax[1].set_xticks(x); ax[1].set_xticklabels([d[0] for d in data], fontsize=8)
    ax[1].set_ylabel("mirror_better_rate"); ax[1].set_ylim(0.2, 0.32)
    ax[1].set_title("Handedness: mirror_rate goes UP (worse) for every fix")
    fig.suptitle("This round (swap-equivariance / channel-norm / mic-PE): all fail both axes", y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig03_round_fail.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print("wrote fig03_round_fail.png")


def fig_tips():
    tips = [("base\nA4", 0.7790), ("#4 sector", 0.7736), ("#5 FiLM", 0.7781),
            ("#6 mlp-skip", 0.7804), ("#3 prog-PE", 0.8227), ("T_all", 0.7819)]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    x = np.arange(len(tips))
    col = ["#888", "#55a868", "#bbb", "#bbb", "#c44e52", "#dd8452"]
    ax[0].bar(x, [t[1] for t in tips], color=col)
    ax[0].axhline(0.7790, color="#888", ls="--", lw=0.8)
    ax[0].set_xticks(x); ax[0].set_xticklabels([t[0] for t in tips], fontsize=8)
    ax[0].set_ylabel("plain test MAE (m)"); ax[0].set_ylim(0.76, 0.83)
    ax[0].set_title("Only prog-PE (#3) really hurts; rest within seed noise")
    for i, t in enumerate(tips):
        ax[0].text(i, t[1]+0.001, f"{t[1]:.3f}", ha="center", fontsize=8)
    # prog-PE root cause: band-open vs LR-decay
    bands = 6; prog = np.linspace(0, 1, 200)
    ax2 = ax[1]
    for b in range(bands):
        bw = np.clip((prog - b/bands)/0.2, 0, 1)
        ax2.plot(prog, bw, lw=1.4, label=f"band {b} (freq {2**b*math.pi:.0f})")
    lr = 0.5*(1+np.cos(np.pi*prog))
    ax2.plot(prog, lr, "k--", lw=2.2, label="LR (cosine)")
    ax2.set_xlabel("training progress"); ax2.set_ylabel("weight / LR factor")
    ax2.set_title("prog-PE root cause: high-freq bands open only\nafter LR≈0 (band5 never fully opens)")
    ax2.legend(fontsize=7, loc="center right", ncol=1)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig04_tips.png", dpi=130); plt.close(fig)
    print("wrote fig04_tips.png")


def fig_prob():
    """Probabilistic coarse head result: best-of-K is an oracle-selection artifact,
    but the per-pixel uncertainty is well-calibrated."""
    runs = ["P_k1", "P_k5", "P_k10"]; Ks = [1, 5, 10]
    R = [json.load(open(f"out/{r}/prob_eval.json")) for r in runs]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    x = np.arange(len(runs)); w = 0.26
    mean = [r["mean_of_K"] for r in R]; best = [r["best_of_K"] for r in R]; ctrl = [r["best_of_K_ctrl"] for r in R]
    ax[0].bar(x-w, mean, w, label="mean-of-K (deployable point)", color="#dd8452")
    ax[0].bar(x,   best, w, label="best-of-K (oracle pick)", color="#4c72b0")
    ax[0].bar(x+w, ctrl, w, label="best-of-K CONTROL (vs WRONG scene)", color="#888", hatch="///", edgecolor="k")
    ax[0].axhline(0.78, color="k", ls="--", lw=1, label="deterministic baseline 0.78")
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"K={k}" for k in Ks])
    ax[0].set_ylabel("plain test MAE (m)")
    ax[0].set_title("best-of-K 'beats' 0.78 — but its control is IDENTICAL\n=> gain is oracle-selection artifact, not real multimodality")
    ax[0].legend(fontsize=7.5)
    for i in range(len(runs)):
        ax[0].text(x[i], best[i]-0.03, f"{best[i]:.2f}", ha="center", fontsize=8, color="w")
    # right: uncertainty calibration corr + real multimodal gain (~0)
    corr = [r["uncert_corr"] for r in R]; realg = [r["real_multimodal_gain"] for r in R]
    ax[1].bar(x-0.2, corr, 0.4, label="uncertainty corr(pred σ, error)", color="#55a868")
    ax[1].bar(x+0.2, realg, 0.4, label="REAL multimodal gain (ctrl-real)", color="#c44e52")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"K={k}" for k in Ks])
    ax[1].set_ylim(-0.1, 0.8)
    ax[1].set_title("Positive: uncertainty is well-calibrated (~+0.67)\nNegative: genuine sample-specific multimodality ≈ 0")
    ax[1].legend(fontsize=8)
    for i in range(len(runs)):
        ax[1].text(x[i]-0.2, corr[i]+0.02, f"{corr[i]:+.2f}", ha="center", fontsize=8)
        ax[1].text(x[i]+0.2, realg[i]+0.02, f"{realg[i]:+.3f}", ha="center", fontsize=8)
    fig.suptitle("Probabilistic coarse head (relaxed-WTA + Laplace NLL): the mismatched-GT control is decisive", y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig05_prob_result.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print("wrote fig05_prob_result.png")


if __name__ == "__main__":
    fig_oracle(); fig_coarse_fine(); fig_round_fail(); fig_tips(); fig_prob()
    print(f"\nAll analysis figures -> {OUT}/")

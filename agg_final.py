"""Final multi-seed ranking: plain-masked test MAE mean+-std per model + error-bar figure."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# model -> list of seed run dirs
MODELS = [
    ("A2 RayMLP (global)",      ["A2_raymlp_s0", "A2_raymlp_s1", "A2_raymlp_s2"], "#5ad"),
    ("U-Net (pix2pix skip)",    ["Aunet_s0", "Aunet_s1", "Aunet_s2"], "#b59"),
    ("A9 A0 decoder",           ["A9_fullmap_s0", "A9_fullmap_s1", "A9_fullmap_s2"], "#888"),
    ("A4 cross-attn",           ["A4_cross_s0", "A4_cross_s1", "A4_cross_s2"], "#2a7"),
    ("A6 self-attn",            ["A6_crossself_s0", "A6_crossself_s1", "A6_crossself_s2"], "#2a7"),
    ("A11 SH-aux",              ["A11_shaux_s0", "A11_shaux_s1", "A11_shaux_s2"], "#e9a"),
    ("A13 +IPD (5ch)",          ["A13_ipd5_s0", "A13_ipd5_s1", "A13_ipd5_s2"], "#c63"),
    ("A14 sup-residual",        ["A14_logmag_s0", "A14_logmag_s1"], "#fa3"),
]


def vals(runs):
    out = []
    for r in runs:
        p = f"out/{r}/metrics_test.json"
        if os.path.exists(p):
            out.append(json.load(open(p))["test"]["MAE_plain"])
    return out


def main():
    rows = []
    for name, runs, c in MODELS:
        v = vals(runs)
        if v:
            rows.append((name, np.mean(v), np.std(v), len(v), v, c))
    rows.sort(key=lambda r: r[1])

    print(f"{'model':22s} {'MAE_plain':>16s}  {'n':>2s}  seeds")
    print("-" * 64)
    for name, m, s, n, v, _ in rows:
        print(f"{name:22s}  {m:.4f} ± {s:.4f}  {n:2d}  {[round(x,3) for x in v]}")
    print("-" * 64)
    print(f"{'A0 det_K1 (orig)':22s}  0.8020          [reference]")

    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(rows))
    ax.bar(x, [r[1] for r in rows], yerr=[r[2] for r in rows], capsize=6,
           color=[r[5] for r in rows], width=0.62)
    for i, r in enumerate(rows):
        ax.text(i, r[1] + r[2] + 0.002, f"{r[1]:.3f}\n±{r[2]:.3f}", ha="center", fontsize=8)
    ax.axhline(0.802, ls="--", c="k", lw=1); ax.text(len(x) - .5, 0.804, "A0 0.802", ha="right", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], rotation=25, ha="right", fontsize=8.5)
    ax.set_ylim(0.74, 0.85); ax.set_ylabel("test MAE [m] — plain masked (mean ± std, 3 seeds)")
    ax.set_title("Final multi-seed ranking (3 seeds, 64x128, masked MAE) — self-attn & pix2pix U-Net tie best\n"
                 "(~0.775); both beat no-skip A9 decoder (0.795); RayMLP worst. Tight observability cluster")
    fig.tight_layout(); fig.savefig("out/figs/fig_final_ranking.png", dpi=130)
    print("\nsaved out/figs/fig_final_ranking.png")


if __name__ == "__main__":
    main()

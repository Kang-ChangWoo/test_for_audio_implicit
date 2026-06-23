"""Visualise WHERE the cross-attention looks.

Each ray-direction query attends over 128 audio tokens = the conv feature map
(192, 8, 16) flattened, i.e. an (8 freq x 16 time) grid of the binaural
spectrogram. Time axis ~ echo delay ~ distance.

Two views:
  fig_attn_probe.png  : for 6 probe directions (front/back/left/right/up/down),
                        the attention heatmap over the (freq x time) audio grid.
  fig_attn_erp.png    : per-ray attention-weighted MEAN time-token index, mapped
                        back to the ERP grid (= "which echo delay each direction
                        focuses on"), shown beside GT depth. If attention is
                        meaningful, attended-delay should track depth.
"""
import os
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from data import make_loader
from eval import load_model
from ray_features import RayBank

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIG = "out/figs"; os.makedirs(FIG, exist_ok=True)
TH, TW = 8, 16            # audio token grid (freq x time) after /8 downsample


@torch.no_grad()
def attn_for_sample(model, cfg, bank, spec1):
    """spec1 (1,2,H,W) -> (depth (H*W,), attn (H*W,128)) from last cross block."""
    cap = {}
    h = model.cross[-1].attn.register_forward_hook(
        lambda m, i, o: cap.__setitem__("w", o[1].detach()))   # o[1]=(1,M,128)
    feat = bank.feat[None]                                      # (1,N,F)
    out = model(spec1, feat, None)
    h.remove()
    return out["depth"][0].cpu(), cap["w"][0].cpu()             # (N,), (N,128)


def main():
    model, cfg, bank, _ = load_model("out/A4_cross_s0", DEV)
    H, W = cfg.img_h, cfg.img_w
    loader = make_loader(cfg, "test", shuffle=False)
    batch = next(iter(loader))
    spec = batch["spec"][:1].to(DEV)
    gt = (batch["depth"][0, 0] * cfg.max_depth).numpy()
    depth, attn = attn_for_sample(model, cfg, bank, spec)      # (N,),(N,128)
    attn = attn.numpy(); depth = depth.numpy().reshape(H, W) * cfg.max_depth

    # ---- (1) probe directions ----
    def idx(r, c): return r * W + c
    probes = [("front", idx(32, 63)), ("back", idx(32, 127)), ("left(+y)", idx(32, 95)),
              ("right(-y)", idx(32, 31)), ("up", idx(8, 63)), ("down", idx(56, 63))]
    fig, ax = plt.subplots(1, 6, figsize=(15, 2.7))
    for k, (name, p) in enumerate(probes):
        a = attn[p].reshape(TH, TW)                            # (freq,time)
        im = ax[k].imshow(a, cmap="magma", aspect="auto")
        ax[k].set_title(f"{name}", fontsize=10)
        ax[k].set_xlabel("time→ (echo delay)", fontsize=8)
        if k == 0:
            ax[k].set_ylabel("freq", fontsize=8)
        ax[k].set_xticks([]); ax[k].set_yticks([])
        plt.colorbar(im, ax=ax[k], fraction=0.046, pad=0.04)
    fig.suptitle("Cross-attention: where each ray direction looks in the audio "
                 "(freq × time) token grid — A4 cross", y=1.05)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_attn_probe.png", dpi=130, bbox_inches="tight")
    print(f"saved {FIG}/fig_attn_probe.png")

    # ---- (2) ERP attended-delay map ----
    tcol = np.arange(TW)                                        # time-token index 0..15
    attn_t = attn.reshape(-1, TH, TW).sum(1)                    # (N,16) marginal over freq
    attn_t = attn_t / attn_t.sum(1, keepdims=True).clip(1e-9)
    mean_t = (attn_t * tcol[None]).sum(1).reshape(H, W)         # (H,W) attended time idx
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.0))
    im0 = ax[0].imshow(gt, cmap="turbo"); ax[0].set_title("GT depth [m]")
    im1 = ax[1].imshow(depth, cmap="turbo"); ax[1].set_title("A4 cross prediction [m]")
    im2 = ax[2].imshow(mean_t, cmap="viridis")
    ax[2].set_title("attention-weighted mean echo-delay (time-token)")
    for a, im in [(ax[0], im0), (ax[1], im1), (ax[2], im2)]:
        a.set_xticks([]); a.set_yticks([]); plt.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    # correlation between attended-delay and GT depth (cos-lat weighted-ish, simple)
    from scipy.stats import pearsonr
    r = pearsonr(mean_t.ravel(), gt.ravel())[0]
    fig.suptitle(f"Does cross-attn look later (farther echo) for deeper rays?  "
                 f"corr(attended-delay, GT depth) = {r:+.3f}", y=1.04)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_attn_erp.png", dpi=130, bbox_inches="tight")
    print(f"saved {FIG}/fig_attn_erp.png  corr={r:+.3f}")


def token_colors(th, tw):
    """Assign each audio token a 2D color: time(echo delay) -> turbo (blue=early,
    red=late, NON-cyclic), freq -> brightness. Returns (th*tw,3) RGB + legend grid."""
    fr = (np.arange(th * tw) // tw) / max(th - 1, 1)        # freq 0..1
    ti = (np.arange(th * tw) % tw) / max(tw - 1, 1)         # time 0..1
    base = plt.cm.turbo(ti)[:, :3]                          # time -> turbo hue
    bright = 0.5 + 0.5 * fr                                 # freq -> brightness
    rgb = (base * bright[:, None]).clip(0, 1)
    return rgb, rgb.reshape(th, tw, 3)


@torch.no_grad()
def color_map(n=4):
    """ERP coloured by WHICH spectrogram token each ray attends to most."""
    model, cfg, bank, _ = load_model("out/A4_cross_s0", DEV)
    H, W = cfg.img_h, cfg.img_w
    loader = make_loader(cfg, "test", shuffle=False)
    batch = next(iter(loader))
    colors, legend = token_colors(TH, TW)                  # (128,3),(8,16,3)

    fig, ax = plt.subplots(n, 4, figsize=(14, 2.6 * n))
    for i in range(n):
        spec = batch["spec"][i:i+1].to(DEV)
        gt = (batch["depth"][i, 0] * cfg.max_depth).numpy()
        depth, attn = attn_for_sample(model, cfg, bank, spec)
        attn = attn.numpy(); depth = depth.numpy().reshape(H, W) * cfg.max_depth
        argc = colors[attn.argmax(1)].reshape(H, W, 3)             # hard: most-attended
        soft = (attn[:, :, None] * colors[None]).sum(1)            # soft: weighted blend
        soft = soft.reshape(H, W, 3).clip(0, 1)
        for j, (img, t, kw) in enumerate([
                (gt, "GT depth [m]", dict(cmap="turbo")),
                (depth, "cross pred [m]", dict(cmap="turbo")),
                (argc, "most-attended token (color)", {}),
                (soft, "attn-weighted token color", {})]):
            ax[i, j].imshow(img, **kw); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if i == 0:
                ax[i, j].set_title(t, fontsize=10)
            if j == 0:
                ax[i, j].set_ylabel(batch["key"][i].split("/")[0][:8], fontsize=8)
    fig.suptitle("Each ray coloured by the spectrogram token it attends to "
                 "(hue=echo-delay/time, brightness=freq). Legend below.", y=1.005)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_attn_colormap.png", dpi=130, bbox_inches="tight")
    print(f"saved {FIG}/fig_attn_colormap.png")

    # standalone legend with axis labels
    figl, axl = plt.subplots(figsize=(4.2, 2.4))
    axl.imshow(legend, aspect="auto", origin="lower")
    axl.set_xlabel("time → (echo delay = distance)"); axl.set_ylabel("frequency →")
    axl.set_xticks([0, TW-1]); axl.set_xticklabels(["early", "late"])
    axl.set_yticks([0, TH-1]); axl.set_yticklabels(["low", "high"])
    axl.set_title("token color legend (8 freq × 16 time)")
    figl.tight_layout(); figl.savefig(f"{FIG}/fig_attn_colormap_legend.png", dpi=130)
    print(f"saved {FIG}/fig_attn_colormap_legend.png")


@torch.no_grad()
def dataset_attn(n_stat=800):
    """Across the dataset: per-sample mean attended echo-delay, pick early/mid/late
    exemplars (3 each), and the global token activation map."""
    from torch.utils.data import DataLoader
    from data import CachedDataset, collate
    model, cfg, bank, _ = load_model("out/A4_cross_s0", DEV)
    H, W = cfg.img_h, cfg.img_w
    dset = CachedDataset(cfg, "test")
    loader = DataLoader(dset, batch_size=32, shuffle=False, num_workers=6, collate_fn=collate)
    colors, legend = token_colors(TH, TW)
    tcol = np.arange(TW)

    cap = {}
    hk = model.cross[-1].attn.register_forward_hook(
        lambda m, i, o: cap.__setitem__("w", o[1].detach()))
    feat = bank.feat
    per_tok = []                                    # (n,128) per-sample ray-mean attn
    gsum = np.zeros(TH * TW); seen = 0
    pr_sum = 0.0; pr_cnt = 0                         # per-ray participation ratio
    for b in loader:
        spec = b["spec"].to(DEV); B = spec.size(0)
        model(spec, feat[None].expand(B, -1, -1), None)
        w = cap["w"]                                 # (B,N,128)
        p = w / w.sum(-1, keepdim=True).clamp(min=1e-9)
        pr = 1.0 / p.pow(2).sum(-1).clamp(min=1e-9)  # (B,N): ~1 focused, ~128 uniform
        pr_sum += float(pr.sum()); pr_cnt += pr.numel()
        m = w.mean(1).cpu().numpy()                  # (B,128)
        per_tok.append(m); gsum += m.sum(0); seen += B
        if seen >= n_stat:
            break
    hk.remove()
    mean_pr = pr_sum / max(pr_cnt, 1)
    per_tok = np.concatenate(per_tok, 0)[:n_stat]   # (n,128)

    # per-sample mean attended time-token (marginalise freq)
    tprof = per_tok.reshape(-1, TH, TW).sum(1)      # (n,16)
    tprof = tprof / tprof.sum(1, keepdims=True).clip(1e-9)
    mean_t = (tprof * tcol[None]).sum(1)            # (n,) 0..15
    order = np.argsort(mean_t)
    early = order[:3]                               # smallest attended time (near/blue)
    mid = order[len(order)//2 - 1: len(order)//2 + 2]
    late = order[-3:][::-1]                         # largest attended time (far/red)
    groups = [("EARLY (남색·near echo)", early), ("MID", mid), ("LATE (빨강·far echo)", late)]

    # ---- exemplars figure: 9 rows x 3 cols ----
    fig, ax = plt.subplots(9, 3, figsize=(11, 19))
    row = 0
    for gname, idxs in groups:
        for k in idxs:
            s = dset[int(k)]
            spec = s["spec"][None].to(DEV)
            gt = (s["depth"][0] * cfg.max_depth).numpy()
            depth, attn = attn_for_sample(model, cfg, bank, spec)
            attn = attn.numpy(); depth = depth.numpy().reshape(H, W) * cfg.max_depth
            argc = colors[attn.argmax(1)].reshape(H, W, 3)
            for j, (img, kw) in enumerate([(gt, dict(cmap="turbo")),
                                           (depth, dict(cmap="turbo")), (argc, {})]):
                ax[row, j].imshow(img, **kw); ax[row, j].set_xticks([]); ax[row, j].set_yticks([])
                if row == 0:
                    ax[row, j].set_title(["GT depth", "cross pred", "most-attended token"][j], fontsize=10)
            ax[row, 0].set_ylabel(f"{gname}\n~t={mean_t[k]:.1f}/15", fontsize=7.5)
            row += 1
    fig.suptitle("Samples grouped by mean attended echo-delay "
                 "(EARLY/near → MID → LATE/far). 3rd col = token color (blue=early..red=late)", y=1.002)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_attn_samples_by_delay.png", dpi=120, bbox_inches="tight")
    print(f"saved {FIG}/fig_attn_samples_by_delay.png")

    # ---- physical unit mapping (from data.py) ----
    SR, C, MD = cfg.sample_rate, 340.0, cfg.max_depth
    win_ms = 2.0 * MD / C * 1000.0                  # analysis window = round-trip(MD)
    t_ms = (np.arange(TW) + 0.5) / TW * win_ms       # token center time [ms]
    dist_m = (np.arange(TW) + 0.5) / TW * MD         # one-way echo distance [m]
    f_khz = (np.arange(TH) + 0.5) / TH * (SR / 2) / 1000.0   # token center freq [kHz]

    # ---- global activation map (physical axes) ----
    g = (gsum / seen).reshape(TH, TW)
    gt_prof = g.sum(0); gt_prof = gt_prof / gt_prof.sum()
    fprof = g.sum(1); fprof = fprof / fprof.sum()
    fig2, ax2 = plt.subplots(1, 3, figsize=(16, 3.4))
    im = ax2[0].imshow(g, cmap="magma", aspect="auto", origin="lower",
                       extent=[0, win_ms, 0, SR/2/1000.0])
    ax2[0].set_title("global mean attention (physical)")
    ax2[0].set_xlabel("echo delay [ms]"); ax2[0].set_ylabel("frequency [kHz]")
    sec = ax2[0].secondary_xaxis("top", functions=(lambda x: x/1000*C/2, lambda d: d*2/C*1000))
    sec.set_xlabel("one-way distance [m]")
    plt.colorbar(im, ax=ax2[0], fraction=0.046)
    ax2[1].bar(t_ms, gt_prof, width=win_ms/TW*0.9, color=plt.cm.turbo(tcol/(TW-1)))
    meant = float((t_ms*gt_prof).sum()); meand = float((dist_m*gt_prof).sum())
    ax2[1].set_title(f"attention vs echo delay\nmean={meant:.1f} ms (~{meand:.1f} m)")
    ax2[1].set_xlabel("echo delay [ms]  (→ farther)"); ax2[1].set_ylabel("attention share")
    ax2[2].barh(f_khz, fprof, height=(SR/2/1000/TH)*0.85, color="#789")
    ax2[2].set_title(f"attention vs frequency\nmean={float((f_khz*fprof).sum()):.1f} kHz")
    ax2[2].set_xlabel("attention share"); ax2[2].set_ylabel("frequency [kHz]")
    fig2.suptitle(f"Dataset-wide ({seen} samples): WHERE cross-attention looks in the "
                  f"spectrogram | focus: mean per-ray participation ratio = {mean_pr:.0f}/128 "
                  f"({'broad' if mean_pr>40 else 'focused'})", y=1.06)
    fig2.tight_layout(); fig2.savefig(f"{FIG}/fig_attn_global.png", dpi=130, bbox_inches="tight")
    print(f"saved {FIG}/fig_attn_global.png")
    early = gt_prof[:TW//3].sum(); late = gt_prof[-TW//3:].sum()
    peak_t = t_ms[gt_prof.argmax()]; peak_f = f_khz[fprof.argmax()]
    print(f"peak delay={peak_t:.1f}ms (~{peak_t/1000*C/2:.2f}m)  peak freq={peak_f:.1f}kHz")
    print(f"early-third={early:.3f} late-third={late:.3f}  mean_per_ray_PR={mean_pr:.1f}/128")


@torch.no_grad()
def binaural_focus(n_stat=900):
    """(1) per-ray participation ratio (PR) by direction sector -> binaural focus.
    (2) per-sector LATE-delay activation ratio -> top-3 samples per sector."""
    from torch.utils.data import DataLoader
    from data import CachedDataset, collate
    from metrics import erp_region_masks
    model, cfg, bank, _ = load_model("out/A4_cross_s0", DEV)
    H, W = cfg.img_h, cfg.img_w
    dset = CachedDataset(cfg, "test")
    loader = DataLoader(dset, batch_size=32, shuffle=False, num_workers=6, collate_fn=collate)
    regions = erp_region_masks(H, W)
    SECT = ["front", "back", "left", "right", "upper", "lower"]
    rmask = {k: torch.from_numpy(regions[k].reshape(-1).astype(bool)).to(DEV) for k in SECT}
    late_cols = (np.arange(TW) >= (TW * 2 // 3))           # last third = far echo
    colors, _ = token_colors(TH, TW)

    cap = {}
    hk = model.cross[-1].attn.register_forward_hook(
        lambda m, i, o: cap.__setitem__("w", o[1].detach()))
    pr_pix = torch.zeros(H * W, device=DEV)
    per_late = []; n_seen = 0
    for b in loader:
        spec = b["spec"].to(DEV); B = spec.size(0)
        model(spec, bank.feat[None].expand(B, -1, -1), None)
        w = cap["w"]                                        # (B,N,128)
        p = w / w.sum(-1, keepdim=True).clamp(min=1e-9)
        pr = 1.0 / p.pow(2).sum(-1).clamp(min=1e-9)         # (B,N)
        pr_pix += pr.sum(0)
        wr = w.reshape(B, -1, TH, TW)
        late = wr[..., late_cols].sum((-2, -1)) / w.sum(-1).clamp(min=1e-9)   # (B,N)
        per_late.append(torch.stack([late[:, rmask[k]].mean(1) for k in SECT], 1).cpu().numpy())
        n_seen += B
        if n_seen >= n_stat:
            break
    hk.remove()
    pr_map = (pr_pix / n_seen).cpu().numpy().reshape(H, W)
    per_late = np.concatenate(per_late, 0)[:n_stat]        # (n, 6)
    sect_pr = {k: float(pr_map.reshape(-1)[rmask[k].cpu().numpy()].mean()) for k in SECT}

    # ---- Part 1 figure: PR map + sector PR bars ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
    im = ax[0].imshow(pr_map, cmap="viridis_r")            # darker = more focused
    ax[0].set_title("per-ray attention participation ratio (ERP)\n(lower=more focused; /128)")
    ax[0].set_xticks([]); ax[0].set_yticks([]); plt.colorbar(im, ax=ax[0], fraction=0.046)
    ks = SECT; vs = [sect_pr[k] for k in ks]
    cols_b = ["#c33" if k in ("left", "right") else "#39c" for k in ks]
    ax[1].bar(ks, vs, color=cols_b)
    for i, v in enumerate(vs): ax[1].text(i, v + 0.2, f"{v:.1f}", ha="center", fontsize=9)
    ax[1].set_ylabel("mean PR /128 (lower=focused)")
    lat = (sect_pr["left"] + sect_pr["right"]) / 2; fb = (sect_pr["front"] + sect_pr["back"]) / 2
    ax[1].set_title(f"binaural focus: lateral(L/R)={lat:.1f} vs front/back={fb:.1f}  "
                    f"(Δ={fb-lat:+.1f})")
    ax[1].set_ylim(min(vs) - 3, max(vs) + 3)
    fig.suptitle("Direction-wise attention focus — are laterally-informative rays more focused?", y=1.04)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_attn_pr_by_dir.png", dpi=130, bbox_inches="tight")
    print(f"saved fig_attn_pr_by_dir.png  sector_PR={ {k: round(v,1) for k,v in sect_pr.items()} }")

    # ---- Part 2 figure: top-3 LATE-activation samples per sector ----
    use = ["front", "back", "left", "right"]
    fig2, ax2 = plt.subplots(len(use) * 3, 3, figsize=(11, 2.2 * len(use) * 3))
    row = 0
    for si, sname in enumerate(use):
        j = SECT.index(sname)
        top = np.argsort(per_late[:, j])[::-1][:3]
        for k in top:
            s = dset[int(k)]
            spec = s["spec"][None].to(DEV)
            gt = (s["depth"][0] * cfg.max_depth).numpy()
            depth, attn = attn_for_sample(model, cfg, bank, spec)
            attn = attn.numpy(); depth = depth.numpy().reshape(H, W) * cfg.max_depth
            argc = colors[attn.argmax(1)].reshape(H, W, 3)
            for c, (img, kw) in enumerate([(gt, dict(cmap="turbo")),
                                           (depth, dict(cmap="turbo")), (argc, {})]):
                ax2[row, c].imshow(img, **kw); ax2[row, c].set_xticks([]); ax2[row, c].set_yticks([])
                if row == 0:
                    ax2[row, c].set_title(["GT depth", "cross pred", "most-attended token"][c], fontsize=10)
            ax2[row, 0].set_ylabel(f"{sname}\nlate={per_late[k,j]:.2f}", fontsize=8)
            row += 1
    fig2.suptitle("Top-3 samples by LATE-delay (far-echo) attention ratio, per sector "
                  "(3rd col: blue=early..red=late)", y=1.005)
    fig2.tight_layout(); fig2.savefig(f"{FIG}/fig_attn_late_by_sector.png", dpi=115, bbox_inches="tight")
    print(f"saved fig_attn_late_by_sector.png  mean late-ratio/sector="
          f"{ {k: round(float(per_late[:,SECT.index(k)].mean()),3) for k in use} }")


if __name__ == "__main__":
    main()
    color_map()

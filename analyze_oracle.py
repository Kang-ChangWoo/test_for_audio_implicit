"""Oracle error decomposition: for the best full-map models, measure how far
MAE_plain (metres) would drop if we PERFECTLY fixed each error factor in turn.
Tells us which lever is worth chasing next vs which is a ~noise-level red herring.

Per-sample masked-mean MAE (metres). Oracles (each is a best-case ceiling):
  raw            : the model as-is
  +offset        : subtract optimal per-sample constant bias  (radial DC error)
  +affine        : optimal per-sample a*p+b  (scale + offset ambiguity)
  +mirror(L/R)   : min(raw, horizontally-flipped)             (handedness)
  +vflip(U/D)    : min(raw, vertically-flipped)               (elevation handedness)
  +bestroll(az)  : min over all W azimuth shifts              (absolute-orientation ambiguity)
Reference floors (no model, just GT statistics):
  const=GTmean   : predict each sample's own true mean depth  (room-scale-only ceiling)
  const=globalmu : predict one global constant                (trivial baseline)
"""
import os, sys, json, argparse
import numpy as np, torch
from eval_fullmap import load
from data import make_loader

def per_sample_mae(p, g, m):                      # (B,1,H,W) -> (B,)
    e = ((p - g).abs() * m).flatten(1).sum(1) / m.flatten(1).sum(1).clamp(min=1e-6)
    return e

@torch.no_grad()
def analyze(run, device):
    model, cfg, extra = load(os.path.join("out", run), device)
    loader = make_loader(cfg, "test", shuffle=False)
    W = cfg.img_w
    acc = {}; n = 0
    gmu_num = 0.0; gmu_den = 0.0
    # first pass mean for global constant
    for b in loader:
        g = b["depth"].to(device) * cfg.max_depth; m = b["mask"].to(device)
        gmu_num += (g * m).sum().item(); gmu_den += m.sum().item()
    gmu = gmu_num / gmu_den
    for b in loader:
        spec = b["spec"].to(device)
        if spec.shape[1] > getattr(cfg, "in_ch", 2): spec = spec[:, :getattr(cfg, "in_ch", 2)]
        if "norm" in extra: spec = (spec - extra["norm"][0]) / extra["norm"][1]
        g = b["depth"].to(device) * cfg.max_depth; m = b["mask"].to(device)
        p = model(spec, extra.get("coarse_feat"), extra.get("sh_basis"))["D"] * cfg.max_depth
        B = p.shape[0]; mf = m.flatten(1); md = mf.sum(1).clamp(min=1e-6)
        def push(k, v): acc[k] = acc.get(k, 0.0) + float(v.sum())
        push("raw", per_sample_mae(p, g, m))
        # optimal offset b* (masked mean of (g-p))
        off = ((g - p) * m).flatten(1).sum(1) / md
        push("offset", per_sample_mae(p + off.view(B,1,1,1), g, m))
        # optimal affine a*p+b via masked least squares
        pf = (p*m).flatten(1); gf = (g*m).flatten(1); mm = m.flatten(1)
        sp = (pf).sum(1)/md; sg = (gf).sum(1)/md
        cov = ((pf-sp.view(B,1)*mm)*(gf-sg.view(B,1)*mm)).sum(1)
        var = (((pf-sp.view(B,1)*mm)**2)).sum(1).clamp(min=1e-6)
        a = (cov/var).clamp(0.2, 5.0); bb = sg - a*sp
        paff = a.view(B,1,1,1)*p + bb.view(B,1,1,1)
        push("affine", per_sample_mae(paff, g, m))
        # mirror L/R
        push("mirror", torch.minimum(per_sample_mae(p,g,m), per_sample_mae(torch.flip(p,[-1]),g,m)))
        # vertical flip
        push("vflip", torch.minimum(per_sample_mae(p,g,m), per_sample_mae(torch.flip(p,[-2]),g,m)))
        # best azimuth roll
        best = per_sample_mae(p,g,m)
        for k in range(1, W):
            best = torch.minimum(best, per_sample_mae(torch.roll(p, k, dims=-1), g, m))
        push("bestroll", best)
        # CONTROL: best roll of pred against a MISMATCHED gt (roll within batch by +1 sample).
        # If bestroll_ctrl ~ bestroll, the roll gain is just the oracle's free DOF, not real alignment.
        gp = torch.roll(g, 1, dims=0); mp = torch.roll(m, 1, dims=0)
        bc = per_sample_mae(p, gp, mp)
        for k in range(1, W):
            bc = torch.minimum(bc, per_sample_mae(torch.roll(p, k, dims=-1), gp, mp))
        push("bestroll_ctrl", bc)
        push("raw_ctrl", per_sample_mae(p, gp, mp))
        # floors
        cmu = (g*m).flatten(1).sum(1)/md
        push("const_GTmean", per_sample_mae(cmu.view(B,1,1,1).expand_as(g), g, m))
        push("const_globalmu", per_sample_mae(torch.full_like(g, gmu), g, m))
        n += B
    return {k: v/n for k, v in acc.items()}, gmu

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+",
        default=["Aunet_s0", "A9_fullmap_s0", "A13_ipd5_s0"]); a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    order = ["raw","offset","affine","mirror","vflip","bestroll","raw_ctrl","bestroll_ctrl","const_GTmean","const_globalmu"]
    out = {}
    for r in a.runs:
        res, gmu = analyze(r, dev); out[r] = res
        print(f"\n=== {r}  (global GT mean = {gmu:.3f} m) ===")
        raw = res["raw"]
        for k in order:
            d = res[k]-raw
            print(f"  {k:16s} {res[k]:.4f} m   Δvs_raw={d:+.4f}")
    json.dump(out, open("out/oracle_decomp.json","w"), indent=2)
    print("\n-> out/oracle_decomp.json")

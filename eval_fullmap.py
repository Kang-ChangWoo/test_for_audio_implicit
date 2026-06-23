"""Evaluate A9-A12 full-map decoder (+correction) on test, with input controls."""

import os
import sys
import json
import copy
import argparse
import numpy as np
import torch
from types import SimpleNamespace

from data import make_loader, apply_audio_mode, shuffle_audio_batch
from ray_features import RayBank
from model_fullmap import FullMapNet
from metrics import MetricBank
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "test_for_audio_clip"))
from sh import SHGrid  # noqa: E402


def build_extra(cfg, device):
    extra = {}
    if cfg.correction in ("cross", "cross_sup"):
        ccfg = copy.copy(cfg); ccfg.img_h, ccfg.img_w = cfg.coarse_h, cfg.coarse_w
        extra["coarse_feat"] = RayBank(ccfg, device=device).feat
    elif cfg.correction == "sh":
        shg = SHGrid(cfg.img_h, cfg.img_w, order=cfg.corr_sh_order)
        extra["sh_basis"] = torch.from_numpy(shg.B).to(device)
    return extra


def load(run_dir, device):
    ck = torch.load(os.path.join(run_dir, "best.pth"), map_location="cpu", weights_only=False)
    cfg = SimpleNamespace(**ck["cfg"])
    if getattr(cfg, "arch", "fullmap") == "unet":
        from model_unet import UNet
        m = UNet(cfg).to(device).eval()
    else:
        m = FullMapNet(cfg).to(device).eval()
    m.load_state_dict(ck["state_dict"])
    extra = build_extra(cfg, device)
    if ck.get("norm") is not None:
        extra["norm"] = (ck["norm"][0].to(device), ck["norm"][1].to(device))
    return m, cfg, extra


@torch.no_grad()
def evrun(model, loader, cfg, extra, device, mode="stereo", shuffle=False, swap=False, max_n=None):
    mb = MetricBank(cfg.img_h, cfg.img_w, cfg.max_depth, device=device); seen = 0
    for b in loader:
        spec = b["spec"].to(device)
        if spec.shape[1] > getattr(cfg,"in_ch",2):
            spec = spec[:, :getattr(cfg,"in_ch",2)]
        spec = apply_audio_mode(spec, mode)
        if shuffle:
            spec = shuffle_audio_batch(spec)
        if swap:
            spec = spec.flip(1)
        if "norm" in extra:
            spec = (spec - extra["norm"][0]) / extra["norm"][1]
        D = model(spec, extra.get("coarse_feat"), extra.get("sh_basis"))["D"] * cfg.max_depth
        mb.add(D, b["depth"].to(device) * cfg.max_depth, b["mask"].to(device))
        seen += spec.size(0)
        if max_n and seen >= max_n:
            break
    return mb.result()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True); p.add_argument("--out-dir", default="out")
    p.add_argument("--controls", type=lambda s: s == "True", default=False)
    p.add_argument("--max-n", type=int, default=0)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(args.out_dir, args.run_name)
    model, cfg, extra = load(run_dir, device)
    loader = make_loader(cfg, "test", shuffle=False)
    mx = args.max_n or None

    rep = {"test": evrun(model, loader, cfg, extra, device, max_n=mx)}
    a = float(model.alpha.detach()) if hasattr(model, "alpha") else None
    print(f"[test] corr={cfg.correction} MAE={rep['test']['MAE']:.4f} "
          f"MAE_low={rep['test']['MAE_low']:.4f} SHcoefL1={rep['test']['SHcoefL1']:.4f} "
          f"delta1={rep['test']['delta1']:.3f}" + (f" alpha={a:+.3f}" if a is not None else ""), flush=True)
    if args.controls:
        for nm, kw in [("mono", dict(mode="mono")), ("shuffle", dict(shuffle=True)), ("swap", dict(swap=True))]:
            rep[nm] = evrun(model, loader, cfg, extra, device, max_n=mx, **kw)
            print(f"[{nm}] MAE={rep[nm]['MAE']:.4f} MAE_low={rep[nm]['MAE_low']:.4f}", flush=True)
    if a is not None:
        rep["alpha"] = a
    json.dump(rep, open(os.path.join(run_dir, "metrics_test.json"), "w"), indent=2)
    print(f"[done] -> {run_dir}/metrics_test.json", flush=True)


if __name__ == "__main__":
    main()

"""Build a richer binaural-RIR feature cache for A13 (signal-representation lever).

The default cache is 2ch log-magnitude spectrogram -> it DISCARDS inter-aural
phase (ITD/azimuth cue). audio_wav IS the raw binaural RIR, so we recompute a
5-channel spatial feature that keeps the phase:

  [ log|L|, log|R|, ILD=log|L|-log|R|, cos(IPD), sin(IPD) ]    (5, 64, 128)
  IPD (inter-aural phase difference) = angle(L * conj(R))  -> recovers ITD.

Output cache_rir/{split}_spec.npy (5ch); depth/mask/keys are SYMLINKED from the
existing test_for_audio_better cache so ordering/targets stay identical.

  python build_rir_cache.py --splits train val test
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/root/storage/matterport3d_0303renew"
BETTER_CACHE = "../test_for_audio_better/cache"
OUT = "cache_rir"
H, W, SR, MD, C = 64, 128, 48000, 10.0, 340.0
CUT = int(2.0 * MD / C * SR)                     # 2823 samples = round-trip 10m
NFFT, HOP, WIN = 512, 160, 400
_win = torch.hann_window(WIN)


def feat_from_wave(wav):
    """wav (2, T) float -> (5, H, W) spatial feature."""
    x = torch.as_tensor(wav[:, :CUT], dtype=torch.float32)
    if x.shape[1] < CUT:
        x = F.pad(x, (0, CUT - x.shape[1]))
    st = torch.stft(x, NFFT, HOP, WIN, _win, return_complex=True)   # (2, F, T')
    L, R = st[0], st[1]
    eps = 1e-6
    lmag = torch.log1p(L.abs()); rmag = torch.log1p(R.abs())
    ild = torch.log(L.abs() + eps) - torch.log(R.abs() + eps)
    ipd = torch.angle(L * torch.conj(R))                            # (F, T')
    feat = torch.stack([lmag, rmag, ild, torch.cos(ipd), torch.sin(ipd)], 0)  # (5,F,T')
    return F.interpolate(feat.unsqueeze(0), (H, W), mode="nearest").squeeze(0).numpy()


def load_wave(scene, idx):
    p = os.path.join(ROOT, scene, "audio_npy", f"audio_{idx}.npy")
    if os.path.exists(p):
        return np.load(p).astype(np.float32)                        # (2, T)
    import soundfile as sf
    w, _ = sf.read(os.path.join(ROOT, scene, "audio_wav", f"audio_{idx}.wav"))
    return w.T.astype(np.float32)


def build(split):
    os.makedirs(OUT, exist_ok=True)
    keys = json.load(open(os.path.join(BETTER_CACHE, f"{split}_keys.json")))
    N = len(keys)
    mm = np.lib.format.open_memmap(os.path.join(OUT, f"{split}_spec.npy"),
                                   mode="w+", dtype=np.float16, shape=(N, 5, H, W))
    for i, k in enumerate(keys):
        scene, idx = k.split("/")
        try:
            mm[i] = feat_from_wave(load_wave(scene, idx)).astype(np.float16)
        except Exception as e:
            print(f"[skip {k}] {e}", flush=True); mm[i] = 0
        if i % 2000 == 0:
            print(f"  {split} {i}/{N}", flush=True)
    mm.flush()
    # symlink shared targets so CachedDataset finds everything in cache_rir/
    for suf in (f"{split}_depth.npy", f"{split}_mask.npy", f"{split}_keys.json"):
        dst = os.path.join(OUT, suf)
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(os.path.join(BETTER_CACHE, suf)), dst)
    print(f"[done] {split}: {N} -> {OUT}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--splits", nargs="+", default=["test", "val", "train"])
    for s in p.parse_args().splits:
        build(s)

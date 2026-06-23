"""Dataset wrapper over the test_for_audio_better cache.

Returns: spec (2,H,W) log-mag binaural, depth (1,H,W) radial/max_depth in [0,1],
mask (1,H,W). Ray sampling and the negative-control input transforms (mono /
left / right / none, channel-shuffle) are applied in the train/eval loops, not
here, so the same cache serves every ablation.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def _paths(cd, sp):
    return ({k: os.path.join(cd, f"{sp}_{k}.npy") for k in ("spec", "depth", "mask")},
            os.path.join(cd, f"{sp}_keys.json"))


def cache_exists(cfg, split):
    paths, kp = _paths(cfg.cache_dir, split)
    return os.path.exists(kp) and all(os.path.exists(p) for p in paths.values())


class CachedDataset(Dataset):
    def __init__(self, cfg, split):
        self.paths, kp = _paths(cfg.cache_dir, split)
        self.keys = json.load(open(kp))
        self.arr = {k: np.load(p, mmap_mode="r") for k, p in self.paths.items()}
        print(f"[{split}] {len(self.keys)} (cache:{cfg.cache_dir})", flush=True)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        d = {k: torch.from_numpy(np.ascontiguousarray(self.arr[k][i])).float()
             for k in self.arr}
        d["key"] = self.keys[i]
        return d


def collate(b):
    return {k: ([x[k] for x in b] if k == "key" else torch.stack([x[k] for x in b]))
            for k in b[0]}


def make_loader(cfg, split, shuffle):
    assert cache_exists(cfg, split), f"cache missing for {split} at {cfg.cache_dir}"
    ds = CachedDataset(cfg, split)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                      num_workers=cfg.num_workers, collate_fn=collate,
                      drop_last=shuffle, pin_memory=True)


def apply_audio_mode(spec, mode):
    """Negative-control / ablation input transforms on (B,2,H,W) log-mag spec.
    Channels are [L, R]. 'none' zeros audio (for the ray-only sanity path)."""
    if mode == "stereo":
        return spec
    if mode == "mono":
        m = spec.mean(1, keepdim=True)
        return m.expand(-1, 2, -1, -1).clone()
    if mode == "left":
        l = spec[:, 0:1]
        return l.expand(-1, 2, -1, -1).clone()
    if mode == "right":
        r = spec[:, 1:2]
        return r.expand(-1, 2, -1, -1).clone()
    if mode == "none":
        return torch.zeros_like(spec)
    raise ValueError(mode)


def shuffle_audio_batch(spec, generator=None):
    """Control B: break the audio<->scene pairing within a batch (roll by 1)."""
    return torch.roll(spec, shifts=1, dims=0)

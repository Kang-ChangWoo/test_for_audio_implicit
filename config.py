"""Config for the ray-conditioned implicit audio->ERP-depth experiment.

Hypothesis decomposition (break in order, do NOT jump to one big model):
  Q1  ray-conditioned implicit fn  >  existing global encoder-decoder?
  Q2  SH / Fourier ray-PE give real inductive bias (lower LOW-FREQ error)?
  Q3  ear-axis mic-PE helps the model exploit binaural (ILD/IPD) cues?
  Q4  ray self-attention corrects unobservable rays from neighbours?
  Q5  hybrid SH-coarse + implicit-residual reduces the mean-blob collapse?

Data reality (confirmed): the dataset is listener-centred and SELF-EMITTING
(active echolocation). There is NO per-sample p_L/p_R/p_s metadata, BUT:
  * source == listener == origin  -> a per-ray source-PE is degenerate (dropped)
  * the two ears are a FIXED known rig -> we place them at +/- y (head radius
    `head_r` m) and feature each ray by its geometry to each ear. This is the
    legitimate "mic PE". It also drives the L/R-swap mirror test.

Reuses test_for_audio_better's cache (spec/depth/mask, radial depth /max_depth in
[0,1]) so no data prep is needed.
"""

import argparse
from types import SimpleNamespace

DEFAULTS = dict(
    # --- data (reuse the better-experiment cache; no rebuild) ---
    dataset_dir="/root/storage/matterport3d_0303renew",
    cache_dir="../test_for_audio_better/cache",
    img_h=64, img_w=128, max_depth=10.0, sample_rate=48000,

    # --- model selection ---
    # rayonly | raymlp | cross | crossself | hybrid
    model="raymlp",
    width=48, embed_dim=128, dim=192, audio_dim=256,

    # --- ray feature flags (the modular ablation knobs) ---
    use_xyz=True,             # raw unit direction (3)
    use_fourier_pe=True,      # Fourier PE of xyz
    fourier_bands=6,          # -> 3*2*bands dims
    use_sh_pe=False,          # spherical-harmonic ray basis
    sh_order=4,               # -> (sh_order+1)**2 dims
    use_mic_pe=False,         # ear-axis (binaural) geometry features
    head_r=0.0875,            # head radius [m] for the +/- y ear rig

    # --- attention sizes (cross / self models) ---
    n_heads=4, n_cross=2, n_self=2,

    # --- depth head ---
    use_depth_bins=False,     # log-depth bin classification + expected value
    n_bins=64,

    # --- hybrid SH-coarse + residual (model=hybrid) ---
    hybrid_sh_order=3,        # coarse spherical geometry order
    w_coarse=0.5,             # aux loss on SH-coarse depth
    w_res=0.02,               # residual L1 magnitude penalty

    # --- ray sampling ---
    n_rays=2048,              # rays supervised per sample per step
    eval_chunk=4096,          # rays per forward at full-grid eval

    # --- training ---
    epochs=25, batch_size=64, lr=2e-3, weight_decay=1e-4,
    num_workers=10, seed=0, device="cuda",
    out_dir="out", run_name="run",

    # --- input controls (negative controls live here so they are logged) ---
    audio_mode="stereo",      # stereo | mono | left | right | none
    shuffle_audio=False,      # break audio<->scene pairing (control B)
    mask_farfield=False,      # drop >=10m clamp pixels from the TRAIN loss (ablation)
)


def get_cfg():
    p = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k.replace('_','-')}", type=lambda s: s == "True", default=v)
        else:
            p.add_argument(f"--{k.replace('_','-')}", type=type(v), default=v)
    return SimpleNamespace(**vars(p.parse_args()))

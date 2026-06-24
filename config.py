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

    # --- input channels (2 = log-mag binaural; 5 = RIR spatial feature, A13) ---
    in_ch=2,
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

    # --- implicit-field tuning tips (applied to cross/raymlp) ---
    ray_mlp_skip=False,       # tip6: coord re-injection skip-MLP for ray embedding
    ray_film=False,           # tip5: FiLM (global audio -> gamma,beta) on ray tokens after cross
    prog_pe=False,            # tip3: progressive coarse->fine Fourier PE band opening
    sector_sample=False,      # tip4: sector/near/grad-balanced ray sampling (vs uniform)

    # --- depth head ---
    use_depth_bins=False,     # log-depth bin classification + expected value
    n_bins=64,

    # --- hybrid SH-coarse + residual (model=hybrid) ---
    hybrid_sh_order=3,        # coarse spherical geometry order
    w_coarse=0.5,             # aux loss on SH-coarse depth
    w_res=0.02,               # residual L1 magnitude penalty

    # --- architecture (train_fullmap.py): fullmap(A9 no-skip) | unet(pix2pix skip) ---
    arch="fullmap",
    ngf=64, unet_downs=6,     # pix2pix U-Net width / #downsamples (64x128 -> 1x2 at 6)
    # --- A9-A12 full-map decoder + audio correction (train_fullmap.py) ---
    correction="none",        # none(A9) | cross(A10) | sh(A11) | film(A12) | cross_sup(A14)
    coarse_h=16, coarse_w=32,  # coarse grid for the cross-residual branch
    corr_sh_order=3,          # SH order for the sh-correction / aux
    w_sh_aux=1.0,             # weight on audio->SH-coef auxiliary loss (A11)
    # --- A14 supervised residual corrector ---
    init_decoder="",          # warm-start decoder weights from this run (e.g. A9_fullmap_s0)
    freeze_decoder=False,     # freeze D0 -> pure "is the A0 residual audio-predictable?" test
    res_scale=0.3,            # bound on tanh residual (normalised depth)
    w_res_sup=1.0,            # supervise Dcorr to lowpass(GT - D0)
    w_tv=0.01,                # total-variation reg on Dcorr
    chan_norm=False,          # per-channel train-set input normalisation (for 5ch A13/A14)
    w_swap_eq=0.0,            # tip8: weak swap-equivariance reg: f(swap_LR(x)) ~ mirror(f(x))

    # --- probabilistic coarse head (train_prob.py): model the coarse-layout AMBIGUITY ---
    # Finding (oracle decomp): error = coarse-layout multi-modality + unobservable fine detail.
    # K diverse coarse hypotheses (relaxed-WTA) + per-pixel Laplace scale (aleatoric uncertainty).
    prob_k=5,                 # number of hypotheses
    prob_eps=0.05,            # relaxed-WTA weight on non-winner heads (keeps them alive)
    prob_w_nll=0.2,           # weight on Laplace NLL (uncertainty calibration)
    prob_coarse=True,         # band-limit heads (avg-pool to prob_head res then upsample)
    prob_head_h=16, prob_head_w=32,   # coarse head resolution (enforces smoothness)

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

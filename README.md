# test_for_audio_implicit — ray-conditioned implicit audio→ERP-depth

Tests whether a **ray-conditioned implicit depth function** beats the existing
global-bottleneck encoder–decoder at binaural-audio → ERP radial depth, by
**decomposing the hypothesis** and breaking each sub-question in order rather
than building one big model.

```
binaural spec ──► audio encoder ──► global latent z  /  tokens
ERP ray dir r  ─► [xyz | Fourier-PE | SH basis | ear-axis mic-PE] ─► ray query q
q  (×audio)  ──► depth(r)            # implicit: predict per-ray, not a full map
```

## Hypothesis ladder (run in order)
| Q | question | runs |
|---|---|---|
| Q1 | implicit fn uses audio at all? | A1 ray-only prior vs A2 RayMLP (+shuffled control) |
| Q2 | SH/Fourier ray-PE give inductive bias? | A2 vs A3 (`--use-sh-pe`) on low-freq metrics |
| Q3 | ear-axis mic-PE helps binaural use? | A3 vs A5 (`--use-mic-pe`), L/R-swap test |
| Q4 | ray self-attn corrects unobservable rays? | A5 vs A6 (`--model crossself`) |
| Q5 | SH-coarse + residual cuts mean-blob? | A5 vs A8 (`--model hybrid`) |

## Data / reuse
- Reuses `../test_for_audio_better/cache` (spec 2×64×128 log-mag binaural,
  depth radial /max_depth∈[0,1], mask). No data prep.
- Dataset is **listener-centred & self-emitting** (active echolocation): source≈origin
  (per-ray source-PE degenerate → dropped); ears are a fixed `±y` rig (`head_r`),
  giving a legitimate mic-PE that drives the L/R-swap mirror test.

## Models (`--model`)
`rayonly` · `raymlp` · `cross` · `crossself` · `hybrid`.
Ray-feature flags: `--use-xyz --use-fourier-pe --use-sh-pe --use-mic-pe`.
Head: `--use-depth-bins`. Controls: `--audio-mode {stereo,mono,left,right,none}`,
`--shuffle-audio True`.

## Run
```bash
bash run_stage1.sh          # Q1 gate, 2 seeds
python agg_stage1.py        # verdict table
python eval.py --run-name A2_raymlp_s0 --controls True
```
Training supervises N random rays/sample; eval predicts the full grid by chunking.

## Metrics (`metrics.py`)
MAE/RMSE/AbsRel/δ<1.25/SILog (cos-lat weighted) + **layout** metrics that matter
for the SH/implicit claim: low-pass MAE, SH-coefficient L1 error, sector MAE.
Controls in `eval.py`: mono/left/right/shuffle + L/R-swap mirror consistency.

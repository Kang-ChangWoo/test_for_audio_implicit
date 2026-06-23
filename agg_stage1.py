"""Aggregate Stage-1 (Q1 gate) results across seeds and print the verdict table.

Reads out/<run>/train_done.json for best val MAE, optionally runs eval controls,
and prints: A1 ray-only prior vs A2 global-audio RayMLP vs A2-shuffled.
Q1 passes iff audio model << ray-only prior AND shuffled collapses to the prior.
"""

import json
import os
import numpy as np

GROUPS = {
    "A1_rayonly (no audio, prior)": ["A1_rayonly_s0", "A1_rayonly_s1"],
    "A2_raymlp  (global audio)":    ["A2_raymlp_s0", "A2_raymlp_s1"],
    "A2_shuffled (audio control)":  ["A2_shuf_s0", "A2_shuf_s1"],
}


def best_val(run):
    p = os.path.join("out", run, "train_done.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))["best_val_mae_m"]


def main():
    print(f"{'group':32s}  {'val MAE [m] mean±std':>22s}   seeds")
    means = {}
    for name, runs in GROUPS.items():
        vals = [v for v in (best_val(r) for r in runs) if v is not None]
        if not vals:
            print(f"{name:32s}  {'(pending)':>22s}")
            continue
        m, s = float(np.mean(vals)), float(np.std(vals))
        means[name] = m
        print(f"{name:32s}  {m:8.4f} ± {s:6.4f}        {['%.4f'%v for v in vals]}")
    if len(means) == 3:
        prior = means["A1_rayonly (no audio, prior)"]
        audio = means["A2_raymlp  (global audio)"]
        shuf = means["A2_shuffled (audio control)"]
        print("\n--- Q1 verdict ---")
        print(f"audio gain over prior : {prior - audio:+.4f} m "
              f"({100*(prior-audio)/prior:+.1f}%)")
        print(f"shuffled gain         : {prior - shuf:+.4f} m "
              f"({100*(prior-shuf)/prior:+.1f}%)  (should be ~0)")
        passed = (audio < prior - 0.01) and (shuf > audio + 0.01)
        print(f"Q1 ray-conditioned implicit USES audio: {'PASS' if passed else 'FAIL/weak'}")


if __name__ == "__main__":
    main()

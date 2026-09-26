"""Export the latent-alignment panels as JSON for the website's interactive map.

Runs exactly the pipeline behind ``figures/tsne.png`` -- the same motion selection,
the same per-panel t-SNE fit, the same compact layout, circle normalisation and
polyline smoothing as
``humanoidverse/cross_embodiment/mjlab_bfm/tools/viz_latent_panels.py`` -- and then,
instead of drawing with matplotlib, writes each trajectory out as a polyline tagged
with its motion name, behavior class and source embodiment. The website draws those
polylines as SVG paths and can therefore name a trajectory on hover.

Run with the cross_bfm environment (needs numpy/sklearn/scipy):

  ~/Documents/Research/cross_bfm/.venv/bin/python tools/export_latent_map.py \
      --repo ~/Documents/Research/cross_bfm \
      --out assets/latent_map.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from argparse import Namespace

import numpy as np

# The figure in the paper was produced with these arguments; keep them in sync or the
# exported map stops matching the printed panels.
ZPRED_DIRS = [
    ("100%", "humanoidverse/cross_encoder/infer_out/m3_zpred"),
    ("25%", "humanoidverse/cross_encoder/infer_out/m3_zpred_025"),
    ("10%", "humanoidverse/cross_encoder/infer_out/m3_zpred_010"),
]
HELD_OUT = ["walk3_subject4", "walk3_subject5", "walk4_subject1"]
ROBOT = "m3"
SEED = 6
MAX_TRACE_POINTS = 110  # after projection: enough to keep the curve, small enough to ship


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.expanduser("~/Documents/Research/cross_bfm"))
    ap.add_argument("--out", default="assets/latent_map.json")
    ap.add_argument("--max-points", type=int, default=MAX_TRACE_POINTS)
    cli = ap.parse_args()

    repo = os.path.abspath(os.path.expanduser(cli.repo))
    sys.path.insert(0, repo)

    from humanoidverse.cross_embodiment.mjlab_bfm.tools.viz_latent_2d import (
        CATEGORY_COLORS, _compact_layout, _project, _smooth, _to_circle,
    )
    from humanoidverse.cross_embodiment.mjlab_bfm.tools.viz_latent_panels import (
        MAX_PER_CAT, _gather, _select_motions,
    )

    g1_dir = os.path.join(repo, "humanoidverse/data/g1_twist2_z")
    zdirs = [(lab, os.path.join(repo, d)) for lab, d in ZPRED_DIRS]

    # Same knobs as viz_latent_panels' defaults, which is what produced the figure.
    args = Namespace(window=300, window_start=0.2, window_size=0, windows_per_motion=0,
                     max_points=300, smooth=5, cluster_shrink=1.0, cluster_spread=1.5,
                     no_circle=False, method="tsne", seed=SEED)

    motions = _select_motions(g1_dir, [d for _, d in zdirs], MAX_PER_CAT,
                              random_motions=True, seed=SEED, held=HELD_OUT)
    held_present = [m for m in HELD_OUT if m in motions]
    print(f"[export] {len(motions)} motions: {', '.join(motions)}", flush=True)
    print(f"[export] held-out: {', '.join(held_present)}", flush=True)

    panels = []
    for label, zdir in zdirs:
        print(f"[export] {label}: {zdir}", flush=True)
        X, _cats, segs = _gather(ROBOT, g1_dir, zdir, motions, args)
        Y, _sub = _project(X, args.method, args.seed)
        Y = _compact_layout(Y, segs, args.cluster_shrink, args.cluster_spread)
        Y = _to_circle(Y, 1.0)

        traces = []
        for start, end, cat, src, motion, _group in segs:
            seg = _smooth(Y[start:end], args.smooth)
            if seg.shape[0] > cli.max_points:
                keep = np.linspace(0, seg.shape[0] - 1, cli.max_points).astype(int)
                seg = seg[keep]
            traces.append({
                "m": motion,
                "cat": cat,
                "src": "g1" if src == "g1" else "target",
                "ood": motion in held_present,
                "p": [[round(float(x), 3), round(float(y), 3)] for x, y in seg],
            })
        panels.append({"label": label, "traces": traces})
        print(f"[export]   {len(traces)} traces, {sum(len(t['p']) for t in traces)} points",
              flush=True)

    out = {
        "robot": ROBOT.upper(),
        "source": "G1",
        "colors": CATEGORY_COLORS,
        "held_out": held_present,
        "panels": panels,
    }
    dest = os.path.abspath(cli.out)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"[export] wrote {dest} ({os.path.getsize(dest) / 1024:.0f} KB)", flush=True)


if __name__ == "__main__":
    main()

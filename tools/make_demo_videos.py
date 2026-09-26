"""Encode the rendered rollout frames into the web demo's clips and manifest.

The render tree is ``<group>/<motion>/<robot>/NNNN.jpeg`` -- ``gen_motions`` for
latents sampled from the flow planner, ``lafan_ghost`` for reference motions from
the dataset. Every robot in a motion is cropped and scaled *identically*, because
the demo's whole claim is that you can flip between embodiments on the same frame
and compare; different framing would read as different behavior. The groups in
RENDER_CLIPS skip the encode entirely and take the clip the render tree already
holds -- see that constant for why.

Two invariants matter and are enforced here rather than trusted:

* **Equal length.** Only frame indices that every robot has are encoded. A robot
  still rendering otherwise yields a shorter clip, which wraps early and silently
  compares the wrong frames from the first loop onward.
* **Honest placeholders.** A robot with no render at all borrows another robot's
  clip so the picker stays complete, but the manifest records whose it is and the
  page labels it. An unlabelled stand-in would claim a result that does not exist.

Writes ``assets/demo/manifest.js``, which the page reads to build its pickers, so
a newly rendered motion or robot needs no edit to the HTML -- just a re-run.

The render tree is not in the repo, so point the tool at it with CROSSBFM_RENDERS
or --src:

  CROSSBFM_RENDERS=/path/to/rendered_images python3 tools/make_demo_videos.py
  python3 tools/make_demo_videos.py --src /path/to/rendered_images
  python3 tools/make_demo_videos.py --fps 20          # override the inferred rate
  python3 tools/make_demo_videos.py --manifest-only    # reorder/relabel only
  python3 tools/make_demo_videos.py --no-placeholder  # omit un-rendered robots
  python3 tools/make_demo_videos.py --only reward_motions  # encode one new group
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

# Where the render tree lives. It sits outside the repo -- the frames are far too
# large to track -- so the location is per-machine and comes from the environment
# or --src rather than being written in here.
SRC = os.environ.get("CROSSBFM_RENDERS", "renders")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST = os.path.join(ROOT, "assets", "demo")
REL = "assets/demo"

# Picker order in the demo, and the order the clips are listed in the manifest.
# A robot found on disk but not named here still gets encoded; it just sorts to
# the end under its own uppercased name.
ROBOTS = ["n1", "m3", "t1"]
ROBOT_LABEL = {"m3": "M3", "t1": "T1", "n1": "N1"}
GROUP_LABEL = {"gen_motions": "Flow-prompted", "lafan_ghost": "LAFAN reference",
               "goal": "Goal reaching", "reward_motions": "Reward-optimized"}
# Ascending energy rather than alphabetical, so the first motion is both the
# natural one to open on and the cheapest to fetch. Groups share one order; a
# motion only one group has simply sorts among its own.
MOTION_ORDER = ["walk", "run", "dance", "reach", "rotate", "raisearms", "move"]

# The lafan directories are shorthand; these are the clips they actually are.
MOTION_LABEL = {
    ("lafan_ghost", "walk"): "walk1_subject1",
    ("lafan_ghost", "run"): "run1_subject2",
    ("lafan_ghost", "dance"): "dance1_subject2",
    ("reward_motions", "raisearms"): "Raise arms",
}

# Groups whose clips are taken from the render tree rather than encoded here.
# Their frames carry the translucent reference ghost, which the DARK mask below
# cannot see -- cropping them with it shears the ghost off the side of the frame,
# and the ghost is the whole point of those renders. ``render_jobs/encode.py``
# already wrote a ghost-aware 800x600 crop beside the frames, so those are copied
# in as they are.
RENDER_CLIPS = {"lafan_ghost", "goal"}

DENSE_FPS = 50   # playback rate for a one-frame-per-control-step render
DARK = 110       # a robot pixel is darker than this on the light backdrop
PAD = 0.18       # fraction of the subject's height added as breathing room
ASPECT = 4 / 3
OUT_W = 800


def subject_box(frames):
    """Union bounding box of the robot over every frame given."""
    box = None
    for p in frames:
        a = np.asarray(Image.open(p).convert("L"), dtype=np.int16)
        ys, xs = np.where(a < DARK)
        if not len(xs):
            continue
        b = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        box = b if box is None else (min(box[0], b[0]), min(box[1], b[1]),
                                     max(box[2], b[2]), max(box[3], b[3]))
    return box


def crop_rect(box, size):
    """Pad the subject box out to ASPECT, centred, clamped inside the frame."""
    W, H = size
    pad = int((box[3] - box[1]) * PAD)
    h = min(H, (box[3] - box[1]) + 2 * pad)
    w = min(W, int(round(h * ASPECT)))
    h = int(round(w / ASPECT))                      # width may have been clamped
    cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
    x = max(0, min(W - w, cx - w // 2))
    y = max(0, min(H - h, cy - h // 2))
    return x, y, w - w % 2, h - h % 2               # even dims for yuv420p


def longest_run(idx):
    """The longest stretch of frames at one uniform stride.

    A directory can hold two passes at once -- an old strided render left sitting
    beside a newer dense one -- and encoding the union splices them into a clip
    that jumps mid-motion. Taking the longest uniform run picks the real pass.
    """
    if len(idx) < 3:
        return idx
    best = (0, 0)
    start, step = 0, idx[1] - idx[0]
    for i in range(1, len(idx)):
        d = idx[i] - idx[i - 1]
        if d == step:
            continue
        if i - start > best[1] - best[0]:
            best = (start, i)
        start, step = i - 1, d
    if len(idx) - start > best[1] - best[0]:
        best = (start, len(idx))
    return idx[best[0]:best[1]]


def infer_fps(indices):
    """Frames are numbered by simulation step, so their stride sets the rate.

    A dense render is one frame per control step, and the controllers run well
    above 30 Hz -- played at 30 the run reads as slow motion. A strided render
    has fewer frames for the same wall-clock and would strobe at that rate.
    """
    if len(indices) < 2:
        return DENSE_FPS
    stride = min(b - a for a, b in zip(indices, indices[1:]))
    return DENSE_FPS if stride <= 1 else 12


def encode(frames, out, rect, fps):
    x, y, w, h = rect
    # ffmpeg's image2 demuxer wants a dense numeric pattern; render indices are
    # strided (0400, 0405, ...), so stage symlinks in a dense sequence instead.
    tmp = tempfile.mkdtemp()
    try:
        for i, p in enumerate(frames):
            os.symlink(os.path.abspath(p), os.path.join(tmp, f"{i:05d}.jpeg"))
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(fps), "-i", os.path.join(tmp, "%05d.jpeg"),
            "-vf", f"crop={w}:{h}:{x}:{y},scale={OUT_W}:-2:flags=lanczos",
            "-c:v", "libx264", "-profile:v", "high", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
            out,
        ], check=True)
    finally:
        shutil.rmtree(tmp)


def poster(frame, out, rect):
    x, y, w, h = rect
    Image.open(frame).crop((x, y, x + w, y + h)) \
        .resize((OUT_W, int(OUT_W / ASPECT)), Image.LANCZOS) \
        .save(out, quality=82, optimize=True)


def import_clips(mdir, motion):
    """The render tree's own web clips for one motion, as {robot: (mp4, jpg)}.

    ``render_jobs/encode.py`` writes ``<motion>_<robot>_web.mp4`` and a poster
    beside the frames; the robots are read off those names rather than off the
    frame directories, so a robot whose encode has not run yet is simply absent
    instead of arriving here as a half-copied clip.
    """
    out = {}
    for f in sorted(os.listdir(mdir)):
        m = re.fullmatch(rf"{re.escape(motion)}_(\w+)_web\.mp4", f)
        if not m:
            continue
        jpg = os.path.join(mdir, f"{motion}_{m.group(1)}.jpg")
        if os.path.exists(jpg):
            out[m.group(1)] = (os.path.join(mdir, f), jpg)
    return out


def nice(key):
    return key.replace("_", " ").replace("-", " ").strip().capitalize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=None,
                    help="playback rate; inferred from the frame stride if unset")
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--manifest-only", action="store_true",
                    help="rebuild manifest.js from the existing clips; use after "
                         "an order or label change, when no re-encode is needed")
    ap.add_argument("--no-placeholder", action="store_true",
                    help="leave a robot out entirely rather than borrowing a clip")
    ap.add_argument("--only", metavar="GROUP", action="append",
                    help="re-encode only these groups; the rest keep their existing "
                         "clips. The manifest still covers every group, so the page "
                         "never loses one -- repeatable")
    args = ap.parse_args()

    os.makedirs(DST, exist_ok=True)
    groups, written = {}, set()

    all_groups = sorted(d for d in os.listdir(args.src)
                        if os.path.isdir(os.path.join(args.src, d)))
    if args.only:
        unknown = sorted(set(args.only) - set(all_groups))
        if unknown:
            sys.exit(f"--only {', '.join(unknown)}: no such group under {args.src} "
                     f"(have {', '.join(all_groups)})")

    for group in all_groups:
        gdir = os.path.join(args.src, group)
        motions = []
        # A group left out of --only is not re-encoded, but its clips still have to
        # reach the manifest or the page would drop the group entirely.
        reuse = args.manifest_only or bool(args.only and group not in args.only)
        if reuse and not args.manifest_only:
            print(f"{group}: not in --only -- keeping its existing clips")

        for motion in sorted(d for d in os.listdir(gdir)
                             if os.path.isdir(os.path.join(gdir, d))):
            mdir = os.path.join(gdir, motion)

            if group in RENDER_CLIPS:
                clips = {}
                for robot, (mp4_src, jpg_src) in import_clips(mdir, motion).items():
                    base = f"{group}_{motion}_{robot}"
                    mp4 = os.path.join(DST, base + ".mp4")
                    jpg = os.path.join(DST, base + ".jpg")
                    if reuse:
                        if not os.path.exists(mp4):
                            print(f"  ! {base}.mp4 missing -- re-run with this group "
                                  f"encoded to create it", file=sys.stderr)
                            continue
                    else:
                        shutil.copyfile(mp4_src, mp4)
                        shutil.copyfile(jpg_src, jpg)
                        print(f"{group}/{motion}/{robot}: imported "
                              f"{os.path.basename(mp4_src)} "
                              f"({os.path.getsize(mp4) / 1024:.0f} KB)")
                    written.update({base + ".mp4", base + ".jpg"})
                    clips[robot] = {"robot": ROBOT_LABEL.get(robot, robot.upper()),
                                    "src": f"{REL}/{base}.mp4",
                                    "poster": f"{REL}/{base}.jpg"}
                if not clips:
                    print(f"  ! {group}/{motion}: no <motion>_<robot>_web.mp4 in "
                          f"{mdir} -- run render_jobs/encode.py on it first",
                          file=sys.stderr)
                    continue
                order = {r: i for i, r in enumerate(ROBOTS)}
                motions.append({
                    "key": motion,
                    "label": MOTION_LABEL.get((group, motion), nice(motion)),
                    "clips": [clips[r] for r in sorted(clips, key=lambda r: (order.get(r, 99), r))],
                })
                continue

            # index -> path, so the frame numbering on disk is never reconstructed
            # (the two groups pad it differently: 0400 against 0000).
            found = {}
            for robot in sorted(os.listdir(mdir)):
                rdir = os.path.join(mdir, robot)
                if not os.path.isdir(rdir):
                    continue
                idx = {int(m.group(1)): os.path.join(rdir, f)
                       for f in os.listdir(rdir)
                       if (m := re.fullmatch(r"(\d+)\.jpe?g", f))}
                if idx:
                    found[robot] = idx
            if not found:
                continue

            # Only the frames every rendered robot has.
            common = sorted(set.intersection(*(set(i) for i in found.values())))
            if not common:
                print(f"  ! {group}/{motion}: no frame index common to "
                      f"{sorted(found)}", file=sys.stderr)
                continue
            short = {r: len(i) - len(common) for r, i in found.items()
                     if len(i) > len(common)}
            if short:
                print(f"  {group}/{motion}: trimmed to {len(common)} common frames "
                      f"({common[0]:04d}-{common[-1]:04d}); dropped {short} still rendering")

            run = longest_run(common)
            if len(run) < len(common):
                print(f"  ! {group}/{motion}: frames are not one uniform stride "
                      f"({common[0]:04d}-{common[-1]:04d}, {len(common)} frames) -- "
                      f"encoding the longest even run {run[0]:04d}-{run[-1]:04d} "
                      f"({len(run)} frames) and ignoring {len(common) - len(run)} "
                      f"left over from another pass", file=sys.stderr)
            common = run

            seqs = {r: [found[r][i] for i in common] for r in found}
            fps = args.fps or infer_fps(common)

            # Reading every frame to find the crop is the slow part, so it is
            # skipped when only the manifest is being rebuilt.
            rect = None
            if not reuse:
                box = subject_box([f for fs in seqs.values() for f in fs])
                rect = crop_rect(box, Image.open(seqs[next(iter(seqs))][0]).size)

            clips = {}
            for robot, frames in seqs.items():
                base = f"{group}_{motion}_{robot}"
                mp4 = os.path.join(DST, base + ".mp4")
                jpg = os.path.join(DST, base + ".jpg")
                if reuse:
                    if not os.path.exists(mp4):
                        print(f"  ! {base}.mp4 missing -- re-run with this group "
                              f"encoded to create it", file=sys.stderr)
                        continue
                else:
                    encode(frames, mp4, rect, fps)
                    poster(frames[0], jpg, rect)
                written.update({base + ".mp4", base + ".jpg"})
                clips[robot] = {"robot": ROBOT_LABEL.get(robot, robot.upper()),
                                "src": f"{REL}/{base}.mp4",
                                "poster": f"{REL}/{base}.jpg"}
                if not reuse:
                    print(f"{group}/{motion}/{robot}: {len(frames)} frames @ {fps}fps "
                          f"-> {os.path.relpath(mp4, ROOT)} "
                          f"({os.path.getsize(mp4) / 1024:.0f} KB)")

            # A robot with no render borrows the first one that has it, flagged so
            # the page can say whose clip it really is.
            if not args.no_placeholder:
                donor = next((r for r in ROBOTS if r in clips), None)
                for robot in ROBOTS:
                    if robot in clips or donor is None:
                        continue
                    clips[robot] = dict(clips[donor],
                                        robot=ROBOT_LABEL.get(robot, robot.upper()),
                                        placeholder=ROBOT_LABEL.get(donor, donor.upper()))
                    print(f"{group}/{motion}/{robot}: not rendered -- showing "
                          f"{donor}'s clip, flagged as a placeholder")

            order = {r: i for i, r in enumerate(ROBOTS)}
            motions.append({
                "key": motion,
                "label": MOTION_LABEL.get((group, motion), nice(motion)),
                "clips": [clips[r] for r in sorted(clips, key=lambda r: (order.get(r, 99), r))],
            })

        if motions:
            mo = {k: i for i, k in enumerate(MOTION_ORDER)}
            motions.sort(key=lambda m: (mo.get(m["key"], 99), m["key"]))
            groups[group] = {"label": GROUP_LABEL.get(group, nice(group)),
                             "motions": motions}

    if not groups:
        sys.exit(f"no <group>/<motion>/<robot> renders under {args.src}")

    # Anything left from an earlier tree layout would otherwise sit there unused.
    for f in sorted(os.listdir(DST)):
        if f.endswith((".mp4", ".jpg")) and f not in written:
            os.remove(os.path.join(DST, f))
            print(f"removed stale {f}")

    manifest = os.path.join(DST, "manifest.js")
    with open(manifest, "w") as fh:
        fh.write("// Generated by tools/make_demo_videos.py -- do not edit.\n")
        fh.write("window.DEMO_CLIPS = " + json.dumps(groups, indent=2) + ";\n")
    summary = ", ".join(f"{g}: {len(v['motions'])} motion(s)" for g, v in groups.items())
    print(f"\nmanifest: {os.path.relpath(manifest, ROOT)} ({summary})")


if __name__ == "__main__":
    main()

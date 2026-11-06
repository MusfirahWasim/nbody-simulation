#!/usr/bin/env python3
"""Render a 3D animation from a snapshot file written by the C++ programs.

Snapshot layout: int N | repeat( int step | double x[N] | double y[N] | double z[N] )

Examples:
  python3 scripts/animate.py results/snapshots/traj.bin results/animations/disk.gif
  python3 scripts/animate.py traj.bin out.gif --ranks 4      # colour bodies by owning MPI rank
  python3 scripts/animate.py traj.bin out.mp4                # needs ffmpeg
"""
import argparse
import struct
import sys

import matplotlib

matplotlib.use("Agg")  # headless: works on WSL without a display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter


def read_snapshots(path):
    try:
        f = open(path, "rb")
    except OSError as e:
        sys.exit(f"Error: cannot open {path}: {e}")
    with f:
        N = struct.unpack("i", f.read(4))[0]
        steps, frames = [], []
        while True:
            hdr = f.read(4)
            if len(hdr) < 4:
                break
            data = np.frombuffer(f.read(24 * N), dtype=np.float64)
            if data.size != 3 * N:  # truncated final frame
                break
            steps.append(struct.unpack("i", hdr)[0])
            frames.append(data.reshape(3, N))
    if not frames:
        sys.exit("Error: no complete frames in snapshot file")
    return N, np.array(steps), np.array(frames)  # frames: (T, 3, N)


def owner_ranks(N, P):
    """Rank owning each body under the same block decomposition as the C++ code."""
    base, rem = divmod(N, P)
    counts = [base + (1 if r < rem else 0) for r in range(P)]
    return np.repeat(np.arange(P), counts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot")
    ap.add_argument("output", help=".gif or .mp4")
    ap.add_argument("--ranks", type=int, default=0, help="colour bodies by owner among P MPI ranks")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--stride", type=int, default=1, help="use every k-th frame")
    ap.add_argument("--size", type=float, default=6.0, help="marker size")
    args = ap.parse_args()

    N, steps, frames = read_snapshots(args.snapshot)
    frames, steps = frames[:: args.stride], steps[:: args.stride]

    # Robust axis limit so a few ejected bodies don't squash the view.
    lim = float(np.percentile(np.abs(frames), 99)) * 1.1 or 1.0

    if args.ranks > 0:
        c = owner_ranks(N, args.ranks)
        cmap, color_kw = plt.get_cmap("tab10"), {"c": c, "cmap": "tab10", "vmin": 0, "vmax": 9}
        legend = [plt.Line2D([], [], marker="o", ls="", color=cmap(r % 10), label=f"rank {r}")
                  for r in range(args.ranks)]
    else:
        color_kw, legend = {"c": "tab:blue"}, None

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    sc = ax.scatter(*frames[0], s=args.size, **color_kw)
    if legend:
        ax.legend(handles=legend, loc="upper right", fontsize=8)
    title = ax.set_title("")

    def update(i):
        sc._offsets3d = tuple(frames[i])
        title.set_text(f"N={N}   step {steps[i]}")
        ax.view_init(elev=20, azim=30 + 0.5 * i)  # slow rotation
        return sc, title

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    writer = FFMpegWriter(fps=args.fps) if args.output.lower().endswith(".mp4") else PillowWriter(fps=args.fps)
    anim.save(args.output, writer=writer, dpi=100)
    print(f"Saved {len(frames)} frames -> {args.output}")


if __name__ == "__main__":
    main()

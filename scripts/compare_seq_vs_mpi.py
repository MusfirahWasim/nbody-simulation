#!/usr/bin/env python3
"""Visual comparison of the sequential and MPI N-body implementations.

Produces two things:

 1. Performance dashboard (PNG) from results/csv/performance.csv
      - execution time: sequential vs MPI (P = 1,2,4,8) for every N
      - speedup, parallel efficiency
      - where the time goes: computation vs communication
 2. Side-by-side trajectory animation (GIF) from two snapshot files that were
    produced with the same parameters, one by nbody_seq and one by nbody_mpi.
    A third panel shows the per-frame position difference between them.

Examples:
  python3 scripts/compare_seq_vs_mpi.py performance
  python3 scripts/compare_seq_vs_mpi.py animate \
      --seq results/snapshots/seq_traj.bin --mpi results/snapshots/mpi_traj.bin --ranks 4
"""
import argparse
import os
import struct
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

SEQ_COLOR = "#4C72B0"
RANK_CMAP = plt.get_cmap("tab10")


# --------------------------------------------------------------------------- performance
def performance(args):
    df = pd.read_csv(args.csv)
    med = df.groupby(["mode", "N", "P"], as_index=False).median(numeric_only=True)
    seq = med[med["mode"] == "seq"].set_index("N")
    mpi = med[med["mode"] == "mpi"]
    Ns = sorted(mpi["N"].unique())
    Ps = sorted(mpi["P"].unique())
    colors = {P: RANK_CMAP(i + 1) for i, P in enumerate(Ps)}

    fig, axs = plt.subplots(2, 2, figsize=(14, 9.5))
    (ax_t, ax_s), (ax_e, ax_c) = axs
    x = np.arange(len(Ns))

    # (a) execution time grouped bars
    nb = len(Ps) + 1
    w = 0.8 / nb
    ax_t.bar(x - 0.4 + w / 2, [seq.loc[n, "t_total"] for n in Ns], w, color=SEQ_COLOR, label="Sequential")
    for i, P in enumerate(Ps):
        vals = [mpi[(mpi.N == n) & (mpi.P == P)]["t_total"].iloc[0] for n in Ns]
        ax_t.bar(x - 0.4 + w / 2 + (i + 1) * w, vals, w, color=colors[P], label=f"MPI P={P}")
    ax_t.set_yscale("log")
    ax_t.set_xticks(x, [str(n) for n in Ns])
    ax_t.set_xlabel("Number of bodies N")
    ax_t.set_ylabel("Execution time (s, log scale)")
    ax_t.set_title("(a) Execution time: sequential vs MPI")
    ax_t.legend(ncol=2, fontsize=9)

    # (b) speedup vs sequential
    for n in Ns:
        sub = mpi[mpi.N == n].sort_values("P")
        ax_s.plot(sub["P"], seq.loc[n, "t_total"] / sub["t_total"], marker="o", label=f"N={n}")
    ax_s.plot(Ps, Ps, "k--", label="ideal")
    ax_s.axhline(1, color="gray", lw=1)
    ax_s.set_xticks(Ps)
    ax_s.set_xlabel("MPI processes P")
    ax_s.set_ylabel("Speedup  T_seq / T_MPI")
    ax_s.set_title("(b) Speedup over the sequential program")
    ax_s.legend(fontsize=9)
    ax_s.grid(alpha=0.3)

    # (c) efficiency
    for n in Ns:
        sub = mpi[mpi.N == n].sort_values("P")
        ax_e.plot(sub["P"], seq.loc[n, "t_total"] / sub["t_total"] / sub["P"], marker="o", label=f"N={n}")
    ax_e.axhline(1, color="k", ls="--", label="ideal")
    ax_e.set_xticks(Ps)
    ax_e.set_ylim(0, max(1.2, float((seq['t_total'].reindex(mpi['N']).values / mpi['t_total'].values / mpi['P'].values).max()) * 1.08))
    ax_e.set_xlabel("MPI processes P")
    ax_e.set_ylabel("Parallel efficiency  S / P")
    ax_e.set_title("(c) Parallel efficiency")
    ax_e.legend(fontsize=9)
    ax_e.grid(alpha=0.3)

    # (d) computation vs communication, at the largest N shared by all
    n_show = args.n if args.n in Ns else Ns[-2]
    labels = ["Seq"] + [f"P={P}" for P in Ps]
    comp = [seq.loc[n_show, "t_comp"]] + [mpi[(mpi.N == n_show) & (mpi.P == P)]["t_comp"].iloc[0] for P in Ps]
    comm = [0.0] + [mpi[(mpi.N == n_show) & (mpi.P == P)]["t_comm"].iloc[0] for P in Ps]
    xs = np.arange(len(labels))
    ax_c.bar(xs, comp, color=["#8DA0CB"] + ["#55A868"] * len(Ps), label="computation (seq / MPI)")
    ax_c.bar(xs, comm, bottom=comp, color="#C44E52", label="communication / wait")
    for i, (a, b) in enumerate(zip(comp, comm)):
        ax_c.text(i, a + b, f"{a + b:.2f}s", ha="center", va="bottom", fontsize=9)
    ax_c.set_xticks(xs, labels)
    ax_c.set_ylim(0, max(a + b for a, b in zip(comp, comm)) * 1.1)
    ax_c.set_ylabel("Time (s)")
    ax_c.set_title(f"(d) Where the time goes (N={n_show})")
    ax_c.legend(fontsize=9)

    fig.suptitle("Sequential vs MPI N-body: measured performance", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


# --------------------------------------------------------------------------- animation
def read_snapshots(path):
    with open(path, "rb") as f:
        N = struct.unpack("i", f.read(4))[0]
        steps, frames = [], []
        while True:
            hdr = f.read(4)
            if len(hdr) < 4:
                break
            data = np.frombuffer(f.read(24 * N), dtype=np.float64)
            if data.size != 3 * N:
                break
            steps.append(struct.unpack("i", hdr)[0])
            frames.append(data.reshape(3, N))
    if not frames:
        sys.exit(f"Error: no complete frames in {path}")
    return N, np.array(steps), np.array(frames)


def owner_ranks(N, P):
    base, rem = divmod(N, P)
    return np.repeat(np.arange(P), [base + (r < rem) for r in range(P)])


def animate(args):
    Ns, steps_s, fs = read_snapshots(args.seq)
    Nm, steps_m, fm = read_snapshots(args.mpi)
    if Ns != Nm or len(steps_s) != len(steps_m):
        sys.exit("Error: snapshot files do not match (different N or frame count)")
    N = Ns
    k = max(1, args.stride)
    fs, fm, steps = fs[::k], fm[::k], steps_s[::k]
    T = len(steps)

    diff = np.linalg.norm(fs - fm, axis=1).max(axis=1)  # (T,) max position difference per frame
    lim = float(np.percentile(np.abs(fs), 99)) * 1.1 or 1.0
    cols = owner_ranks(N, args.ranks)

    fig = plt.figure(figsize=(13, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.8])
    axA = fig.add_subplot(gs[0], projection="3d")
    axB = fig.add_subplot(gs[1], projection="3d")
    axD = fig.add_subplot(gs[2])
    for ax in (axA, axB):
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    axA.set_title("Sequential (1 process)", color=SEQ_COLOR, fontweight="bold")
    axB.set_title(f"MPI ({args.ranks} processes)", fontweight="bold")
    scA = axA.scatter(*fs[0], s=args.size, c=SEQ_COLOR)
    scB = axB.scatter(*fm[0], s=args.size, c=cols, cmap="tab10", vmin=0, vmax=9)
    axB.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=RANK_CMAP(r % 10), label=f"rank {r}")
                        for r in range(args.ranks)], loc="upper left", fontsize=8)

    floor = 1e-18
    axD.set_xlim(steps[0], steps[-1])
    axD.set_yscale("log")
    axD.set_ylim(floor, 1e-3)
    axD.axhline(1e-9, color="gray", ls="--", lw=1)
    axD.text(steps[0], 1.5e-9, "verification tolerance 1e-9", fontsize=8, color="gray")
    axD.set_xlabel("time step")
    axD.set_ylabel("max |position(seq) − position(MPI)|")
    axD.set_title("Seq vs MPI difference")
    line, = axD.plot([], [], color="#C44E52", lw=2)
    txt = axD.text(0.5, 0.5, "", transform=axD.transAxes, ha="center", fontsize=11)
    sup = fig.suptitle("", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    def update(i):
        scA._offsets3d = tuple(fs[i])
        scB._offsets3d = tuple(fm[i])
        az = 30 + 0.5 * i
        axA.view_init(elev=20, azim=az); axB.view_init(elev=20, azim=az)
        line.set_data(steps[: i + 1], np.maximum(diff[: i + 1], floor))
        d = diff[i]
        txt.set_text("identical (Δ = 0)" if d == 0 else f"Δ = {d:.2e}")
        sup.set_text(f"N = {N}   step {steps[i]}")
        return scA, scB, line, txt, sup

    anim = FuncAnimation(fig, update, frames=T, blit=False)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps), dpi=90)
    print(f"Saved {T} frames -> {args.out}   (max difference over run: {diff.max():.3e})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("performance", help="sequential vs MPI performance dashboard")
    p.add_argument("--csv", default="results/csv/performance.csv")
    p.add_argument("--out", default="results/plots/seq_vs_mpi_performance.png")
    p.add_argument("--n", type=int, default=5000, help="N used for the computation/communication panel")
    p.set_defaults(fn=performance)

    a = sub.add_parser("animate", help="side-by-side trajectory animation")
    a.add_argument("--seq", required=True)
    a.add_argument("--mpi", required=True)
    a.add_argument("--ranks", type=int, required=True, help="P used for the MPI run (colours bodies by owner)")
    a.add_argument("--out", default="results/animations/seq_vs_mpi.gif")
    a.add_argument("--fps", type=int, default=15)
    a.add_argument("--stride", type=int, default=1)
    a.add_argument("--size", type=float, default=8.0)
    a.set_defaults(fn=animate)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate all performance graphs from the CSVs written by analyze.py.

Outputs (results/plots/):
  time_vs_N.png         execution time vs N, one curve per P plus sequential
  speedup_vs_P.png      speedup vs P for each N, with the ideal line
  efficiency_vs_P.png   parallel efficiency vs P for each N
  comp_vs_comm.png      computation vs communication time per P (one panel per N)
  load_balance.png      time imbalance vs N, and per-rank bodies / compute time
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save(fig, outdir, name):
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def plot_time_vs_n(s, outdir):
    fig, ax = plt.subplots(figsize=(7, 5))
    seq = s.groupby("N")["T_seq"].first()
    ax.plot(seq.index, seq.values, "k--o", label="sequential")
    for P, g in s.groupby("P"):
        g = g.sort_values("N")
        ax.plot(g["N"], g["t_total"], marker="o", label=f"MPI P={P}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of bodies N"); ax.set_ylabel("Execution time (s)")
    ax.set_title("Execution time vs problem size")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    save(fig, outdir, "time_vs_N.png")


def plot_speedup(s, outdir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for N, g in s.groupby("N"):
        g = g.sort_values("P")
        ax.plot(g["P"], g["speedup"], marker="o", label=f"N={int(N)}")
    pmax = s["P"].max()
    ax.plot([1, pmax], [1, pmax], "k--", label="ideal")
    ax.set_xlabel("Processes P"); ax.set_ylabel("Speedup  T_seq / T_P")
    ax.set_title("Speedup vs number of processes")
    ax.set_xticks(sorted(s["P"].unique())); ax.grid(True, alpha=0.3); ax.legend()
    save(fig, outdir, "speedup_vs_P.png")


def plot_efficiency(s, outdir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for N, g in s.groupby("N"):
        g = g.sort_values("P")
        ax.plot(g["P"], g["efficiency"], marker="o", label=f"N={int(N)}")
    ax.axhline(1.0, color="k", ls="--", label="ideal")
    ax.set_xlabel("Processes P"); ax.set_ylabel("Parallel efficiency  S / P")
    ax.set_title("Parallel efficiency vs number of processes")
    ax.set_xticks(sorted(s["P"].unique())); ax.grid(True, alpha=0.3); ax.legend()
    save(fig, outdir, "efficiency_vs_P.png")


def plot_comp_comm(s, outdir):
    Ns = sorted(s["N"].unique())
    fig, axes = plt.subplots(1, len(Ns), figsize=(3.6 * len(Ns), 4.2), squeeze=False)
    for ax, N in zip(axes[0], Ns):
        g = s[s["N"] == N].sort_values("P")
        x = np.arange(len(g))
        ax.bar(x, g["t_comp"], label="computation")
        ax.bar(x, g["t_comm"], bottom=g["t_comp"], label="communication")
        ax.set_xticks(x); ax.set_xticklabels([f"P={int(p)}" for p in g["P"]])
        ax.set_title(f"N={int(N)}"); ax.set_ylabel("time (s)")
    axes[0][0].legend()
    fig.suptitle("Computation vs communication time (rank-averaged)")
    save(fig, outdir, "comp_vs_comm.png")


def plot_load_balance(lb_summary, lb_raw, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # (a) time imbalance vs N, grouped by P (shared x positions; remainder annotated)
    ax = axes[0]
    Ns = sorted(lb_summary["N"].unique())
    xpos = {n: i for i, n in enumerate(Ns)}
    for P, g in lb_summary.groupby("P"):
        g = g.sort_values("N")
        xs = [xpos[n] for n in g["N"]]
        ax.plot(xs, g["time_imbalance"], marker="o", label=f"P={int(P)}")
        for x, y, r in zip(xs, g["time_imbalance"], g["N_mod_P"]):
            ax.annotate(f"rem {int(r)}", (x, y), textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=7)
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xticks(range(len(Ns))); ax.set_xticklabels([str(int(n)) for n in Ns])
    ax.set_ylabel("max / mean compute time"); ax.set_xlabel("N  (labels: N mod P)")
    ax.set_title("Compute-time imbalance"); ax.grid(True, alpha=0.3); ax.legend()

    # (b),(c) per-rank bodies and compute time for the most uneven case at the largest P
    Pmax = int(lb_summary["P"].max())
    sub = lb_summary[lb_summary["P"] == Pmax]
    N = int(sub.loc[sub["N_mod_P"].idxmax(), "N"])
    g = lb_raw[(lb_raw["N"] == N) & (lb_raw["P"] == Pmax)]
    per = g.groupby("rank").agg(bodies=("bodies", "first"), t_comp=("t_comp", "median"),
                                t_comm=("t_comm", "median"))
    axes[1].bar(per.index, per["bodies"], color="tab:green")
    axes[1].set_title(f"Bodies per rank (N={N}, P={Pmax})")
    axes[1].set_xlabel("rank"); axes[1].set_ylabel("bodies")
    axes[1].set_ylim(0, per["bodies"].max() * 1.2)  # zero-based: no exaggerated differences
    for r, b in zip(per.index, per["bodies"]):
        axes[1].text(r, b, str(int(b)), ha="center", va="bottom", fontsize=8)
    axes[2].bar(per.index - 0.2, per["t_comp"], width=0.4, label="computation")
    axes[2].bar(per.index + 0.2, per["t_comm"], width=0.4, label="communication (incl. waiting)")
    axes[2].set_title(f"Per-rank time (N={N}, P={Pmax})")
    axes[2].set_xlabel("rank"); axes[2].set_ylabel("time (s)"); axes[2].legend()
    save(fig, outdir, "load_balance.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="results/csv")
    ap.add_argument("--out", default="results/plots")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sp = os.path.join(args.csv, "summary.csv")
    if not os.path.exists(sp):
        sys.exit(f"Error: {sp} not found - run scripts/analyze.py first")
    s = pd.read_csv(sp).dropna(subset=["speedup"])
    if s.empty:
        sys.exit("Error: summary.csv has no usable rows")

    plot_time_vs_n(s, args.out)
    plot_speedup(s, args.out)
    plot_efficiency(s, args.out)
    plot_comp_comm(s, args.out)

    lsp, lrp = (os.path.join(args.csv, "load_balance_summary.csv"),
                os.path.join(args.csv, "load_balance.csv"))
    if os.path.exists(lsp) and os.path.exists(lrp):
        plot_load_balance(pd.read_csv(lsp), pd.read_csv(lrp), args.out)
    else:
        print("(load-balance files missing; skipped load_balance.png)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Turn raw experiment CSVs into the metrics required by the CCA.

Inputs  (from run_experiments.sh / run_load_balance.sh):
  results/csv/performance.csv   mode,N,P,steps,run,t_total,t_comp,t_comm,E0,E1,comp_max,comp_min,imbalance
  results/csv/load_balance.csv  N,P,run,rank,bodies,t_comp,t_comm        (optional)

Outputs:
  results/csv/summary.csv              per (N,P): medians + speedup, efficiency, comm fraction
  results/csv/summary.md               same table, Markdown (paste into the report)
  results/csv/load_balance_summary.csv per (N,P): body-count and time imbalance

Definitions (medians over repeated runs):
  speedup        S = T_seq(N) / T_P(N,P)          T_P = max over ranks of loop time
  efficiency     E = S / P
  comm fraction  = t_comm / (t_comp + t_comm)     (rank-averaged)
  body imbalance = max(bodies per rank) / (N / P)
  time imbalance = max(rank comp time) / mean(rank comp time)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd


def to_markdown(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def performance_summary(path, outdir):
    if not os.path.exists(path):
        sys.exit(f"Error: {path} not found - run scripts/run_experiments.sh first")
    df = pd.read_csv(path)
    if df.empty:
        sys.exit(f"Error: {path} has no rows")

    med = df.groupby(["mode", "N", "P"], as_index=False).agg(
        runs=("run", "count"),
        t_total=("t_total", "median"),
        t_comp=("t_comp", "median"),
        t_comm=("t_comm", "median"),
        comp_max=("comp_max", "median"),
        comp_min=("comp_min", "median"),
        imbalance=("imbalance", "median"),
        E0=("E0", "median"),
        E1=("E1", "median"),
    )
    med["energy_drift"] = (med["E1"] - med["E0"]).abs() / med["E0"].abs()

    seq = med[med["mode"] == "seq"].set_index("N")["t_total"]
    mpi = med[med["mode"] == "mpi"].copy()
    mpi["T_seq"] = mpi["N"].map(seq)
    if mpi["T_seq"].isna().any():
        print("Warning: some N have no sequential baseline; speedup is NaN for them.", file=sys.stderr)
    mpi["speedup"] = mpi["T_seq"] / mpi["t_total"]
    mpi["efficiency"] = mpi["speedup"] / mpi["P"]
    mpi["comm_frac"] = mpi["t_comm"] / (mpi["t_comp"] + mpi["t_comm"])
    mpi = mpi.sort_values(["N", "P"]).reset_index(drop=True)

    cols = ["N", "P", "runs", "T_seq", "t_total", "t_comp", "t_comm", "comm_frac",
            "speedup", "efficiency", "imbalance", "energy_drift"]
    out = mpi[cols]
    out.to_csv(os.path.join(outdir, "summary.csv"), index=False)
    with open(os.path.join(outdir, "summary.md"), "w") as f:
        f.write(to_markdown(out))
    return out


def scalability_report(summary):
    print("\n=== Scalability analysis (per problem size) ===")
    for N, g in summary.groupby("N"):
        g = g.sort_values("P")
        best = g.loc[g["speedup"].idxmax()]
        line = f"N={int(N):>6}: best speedup {best['speedup']:.2f}x at P={int(best['P'])} (E={best['efficiency']:.2f})"
        over = g[g["comm_frac"] >= 0.5]
        if len(over):
            line += f"; communication >= 50% of time from P={int(over['P'].min())}"
        else:
            line += "; communication stays < 50% for all tested P"
        low = g[g["efficiency"] < 0.5]
        if len(low):
            line += f"; efficiency < 0.5 from P={int(low['P'].min())}"
        print(line)


def load_balance_summary(path, outdir):
    if not os.path.exists(path):
        print(f"\n(no {path}; skipping load-balance analysis)")
        return None
    lb = pd.read_csv(path)
    if lb.empty:
        print("\n(load_balance.csv is empty; skipping)")
        return None

    rows = []
    for (N, P), g in lb.groupby(["N", "P"]):
        bodies = g.groupby("rank")["bodies"].first()
        per_run = g.groupby("run")["t_comp"].agg(["max", "mean", "min"])
        t_imb = (per_run["max"] / per_run["mean"]).median()
        spread = ((per_run["max"] - per_run["min"]) / per_run["mean"]).median()
        rows.append({
            "N": int(N), "P": int(P), "N_mod_P": int(N % P),
            "bodies_max": int(bodies.max()), "bodies_min": int(bodies.min()),
            "body_imbalance": bodies.max() / (N / P),
            "time_imbalance": t_imb,
            "time_spread": spread,
        })
    out = pd.DataFrame(rows).sort_values(["P", "N"]).reset_index(drop=True)
    out.to_csv(os.path.join(outdir, "load_balance_summary.csv"), index=False)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perf", default="results/csv/performance.csv")
    ap.add_argument("--lb", default="results/csv/load_balance.csv")
    ap.add_argument("--out", default="results/csv")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    summary = performance_summary(args.perf, args.out)
    pd.set_option("display.width", 200)
    print("=== Performance summary (medians) ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    scalability_report(summary)

    lb = load_balance_summary(args.lb, args.out)
    if lb is not None:
        print("\n=== Load balance (N not divisible by P) ===")
        print(lb.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    print(f"\nWrote tables to {args.out}/ (summary.csv, summary.md, load_balance_summary.csv)")


if __name__ == "__main__":
    main()

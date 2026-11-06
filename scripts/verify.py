#!/usr/bin/env python3
"""Verify that the MPI result matches the sequential reference.

Both programs are run with identical parameters and --out <file>. Each file has
one line per body:  id x y z vx vy vz

Error metrics (stated tolerance, default 1e-9):
  * max absolute error over all positions and velocities
  * max relative error  = max|a - b| / max|a|   (scale-normalised)

Usage:
  python3 scripts/verify.py ref.txt test.txt [--tol 1e-9]
Exit code 0 = PASS, 1 = FAIL, 2 = bad input.
"""
import argparse
import sys

import numpy as np


def load(path):
    try:
        a = np.loadtxt(path)
    except OSError as e:
        sys.exit(f"Error: cannot read {path}: {e}")
    if a.ndim != 2 or a.shape[1] != 7:
        sys.exit(f"Error: {path} must have 7 columns (id x y z vx vy vz)")
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference")
    ap.add_argument("test")
    ap.add_argument("--tol", type=float, default=1e-9, help="max relative error allowed")
    args = ap.parse_args()

    ref, tst = load(args.reference), load(args.test)
    if ref.shape != tst.shape:
        print(f"FAIL: shape mismatch {ref.shape} vs {tst.shape}")
        return 2
    if not np.array_equal(ref[:, 0], tst[:, 0]):
        print("FAIL: body ids differ")
        return 2

    pos_err = np.abs(ref[:, 1:4] - tst[:, 1:4]).max()
    vel_err = np.abs(ref[:, 4:7] - tst[:, 4:7]).max()
    scale = max(np.abs(ref[:, 1:7]).max(), 1e-300)
    rel = max(pos_err, vel_err) / scale

    print(f"bodies             : {ref.shape[0]}")
    print(f"max abs error (pos): {pos_err:.3e}")
    print(f"max abs error (vel): {vel_err:.3e}")
    print(f"max relative error : {rel:.3e}   (tolerance {args.tol:.1e})")
    ok = rel <= args.tol
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env bash
# Generates both trajectories (sequential + MPI, identical parameters), verifies
# they match, then builds the comparison dashboard and side-by-side animation.
# Usage: bash scripts/run_comparison.sh            (defaults: N=300, P=4, 300 steps)
#        N=500 P=4 STEPS=400 bash scripts/run_comparison.sh
set -euo pipefail
cd "$(dirname "$0")/.."
N=${N:-300}; P=${P:-4}; STEPS=${STEPS:-300}; INIT=${INIT:-1}; INTERVAL=${INTERVAL:-5}

make >/dev/null
mkdir -p results/snapshots results/plots results/animations
./bin/nbody_seq --n "$N" --steps "$STEPS" --init "$INIT" --save-trajectory --snap-interval "$INTERVAL" \
    --snap-file results/snapshots/seq_traj.bin --out results/snapshots/seq_final.txt
mpirun ${OVERSUB:+--oversubscribe} -np "$P" ./bin/nbody_mpi --n "$N" --steps "$STEPS" --init "$INIT" \
    --save-trajectory --snap-interval "$INTERVAL" \
    --snap-file results/snapshots/mpi_traj.bin --out results/snapshots/mpi_final.txt
python3 scripts/verify.py results/snapshots/seq_final.txt results/snapshots/mpi_final.txt --tol 1e-9
python3 scripts/compare_seq_vs_mpi.py performance
python3 scripts/compare_seq_vs_mpi.py animate --seq results/snapshots/seq_traj.bin \
    --mpi results/snapshots/mpi_traj.bin --ranks "$P"

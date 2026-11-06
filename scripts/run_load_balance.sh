#!/usr/bin/env bash
# run_load_balance.sh - effect of workload distribution when N is NOT evenly
# divisible by the number of processes.
#
# For each (N, P) it records the per-rank body count, computation time and
# communication time (the RANK lines printed by nbody_mpi) into
#   results/csv/load_balance.csv
#
# Defaults: N in {1000, 1001, 1003, 1007}, P in {4, 8}. With P=8 these give
# remainders 0, 1, 3 and 7, i.e. from perfectly even to the worst uneven case.
#
# Overridable: NS_LB="1000 1001 1003 1007"  PS_LB="4 8"  STEPS=100  RUNS=3
#              INIT=0 SEED=42 DT=0.01 EPS=0.05 OVERSUB=0

set -euo pipefail
cd "$(dirname "$0")/.."

NS_LB=${NS_LB:-"1000 1001 1003 1007"}
PS_LB=${PS_LB:-"4 8"}
STEPS=${STEPS:-100}
RUNS=${RUNS:-3}
INIT=${INIT:-0}
SEED=${SEED:-42}
DT=${DT:-0.01}
EPS=${EPS:-0.05}
OVERSUB=${OVERSUB:-0}

MPI=bin/nbody_mpi
[[ -x $MPI ]] || { echo "Binary missing - run 'make' first." >&2; exit 1; }

mkdir -p results/logs results/csv
CSV=results/csv/load_balance.csv
CORES=$(nproc)
MPIFLAGS=""
[[ $OVERSUB == 1 ]] && MPIFLAGS="--oversubscribe"

echo "N,P,run,rank,bodies,t_comp,t_comm" > "$CSV"

for N in $NS_LB; do
  for P in $PS_LB; do
    if (( P > CORES )) && [[ $OVERSUB != 1 ]]; then
      echo "[skip] P=$P > $CORES cores (set OVERSUB=1 to force)"; continue
    fi
    for r in $(seq 1 "$RUNS"); do
      echo "[lb] N=$N P=$P (N mod P = $((N % P))) run=$r"
      mpirun $MPIFLAGS -np "$P" $MPI --n "$N" --steps "$STEPS" --init "$INIT" \
             --seed "$SEED" --dt "$DT" --eps "$EPS" \
        | tee -a "results/logs/lb_N${N}_P${P}.log" \
        | awk -F, -v run="$r" '/^RANK/ {print $2","$3","run","$4","$5","$6","$7}' >> "$CSV"
    done
  done
done

echo "Done. Results: $CSV"

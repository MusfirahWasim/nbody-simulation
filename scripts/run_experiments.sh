#!/usr/bin/env bash
# run_experiments.sh - reproducible performance experiments.
#
# Runs the sequential baseline (P=1 reference) and the MPI version for every
# (N, P) combination, RUNS times each, and appends one row per run to
# results/csv/performance.csv. Raw program output goes to results/logs/.
#
# All parameters can be overridden through environment variables:
#   NS="100 500 1000 5000 10000"  PS="1 2 4 8"  STEPS=100  RUNS=3
#   INIT=0  SEED=42  DT=0.01  EPS=0.05  OVERSUB=0
# Example:
#   NS="100 500" PS="1 2 4" RUNS=2 bash scripts/run_experiments.sh
#
# OVERSUB=1 allows more MPI processes than CPU cores (adds --oversubscribe).
# Such runs do NOT show true scaling - report them as a limitation.

set -euo pipefail
cd "$(dirname "$0")/.."

NS=${NS:-"100 500 1000 5000 10000"}
PS=${PS:-"1 2 4 8"}
STEPS=${STEPS:-100}
RUNS=${RUNS:-3}
INIT=${INIT:-0}
SEED=${SEED:-42}
DT=${DT:-0.01}
EPS=${EPS:-0.05}
OVERSUB=${OVERSUB:-0}

SEQ=bin/nbody_seq
MPI=bin/nbody_mpi
[[ -x $SEQ && -x $MPI ]] || { echo "Binaries missing - run 'make' first." >&2; exit 1; }

mkdir -p results/logs results/csv
CSV=results/csv/performance.csv
COMMON="--steps $STEPS --init $INIT --seed $SEED --dt $DT --eps $EPS"
CORES=$(nproc)
MPIFLAGS=""
[[ $OVERSUB == 1 ]] && MPIFLAGS="--oversubscribe"

# --- record the environment (reproducibility) ---
{
  echo "date: $(date -Is)"
  echo "cores (nproc): $CORES"
  lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core|MHz" || true
  echo "compiler: $(g++ --version | head -1)"
  echo "mpi: $(mpirun --version | head -1)"
  echo "params: NS='$NS' PS='$PS' STEPS=$STEPS RUNS=$RUNS INIT=$INIT SEED=$SEED DT=$DT EPS=$EPS OVERSUB=$OVERSUB"
} > results/logs/system_info.txt

echo "mode,N,P,steps,run,t_total,t_comp,t_comm,E0,E1,comp_max,comp_min,imbalance" > "$CSV"

# Append a RESULT line as a CSV row: RESULT,mode,N,P,steps,... -> mode,N,P,steps,run,...
add_row() { # $1 = run index, stdin = program output
  awk -F, -v run="$1" '/^RESULT/ {printf "%s,%s,%s,%s,%s", $2,$3,$4,$5,run;
                        for (i=6;i<=NF;i++) printf ",%s", $i; printf "\n"}'
}

for N in $NS; do
  for r in $(seq 1 "$RUNS"); do
    echo "[seq] N=$N run=$r"
    $SEQ --n "$N" $COMMON | tee -a "results/logs/seq_N${N}.log" | add_row "$r" >> "$CSV"
  done
  for P in $PS; do
    if (( P > CORES )) && [[ $OVERSUB != 1 ]]; then
      echo "[skip] P=$P > $CORES cores (set OVERSUB=1 to force)"; continue
    fi
    if (( P > N )); then echo "[skip] P=$P > N=$N"; continue; fi
    for r in $(seq 1 "$RUNS"); do
      echo "[mpi] N=$N P=$P run=$r"
      mpirun $MPIFLAGS -np "$P" $MPI --n "$N" $COMMON \
        | tee -a "results/logs/mpi_N${N}_P${P}.log" | add_row "$r" >> "$CSV"
    done
  done
done

echo "Done. Results: $CSV   Environment: results/logs/system_info.txt"

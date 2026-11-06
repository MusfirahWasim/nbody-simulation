# Report & Presentation Outline — mapped to the CCA rubric

Fill the bracketed items with YOUR measured numbers. Every claim should point
to a file in `results/` (table or plot) so the experiments are traceable.

## A. Rubric → where it is evidenced

| Rubric criterion | What the grader looks for | Evidence in this project |
|---|---|---|
| 1. Diverse resources (CA1) | MPI, tooling, compute resources, data, visualisation, monitoring all integrated | C++/Open MPI on WSL Ubuntu, Makefile, Python (pandas/Matplotlib), animations, `system_info.txt`, optional `htop`/`psutil` screenshot |
| 2. Distributed interactions (CA2) | Correct communication, synchronisation, data exchange, workload distribution, overhead handling | Block decomposition, `Allgatherv`, barrier placement, `verify.py` PASS output, per-rank timing |
| 3. Innovation & optimisation (CA3) | Thoughtful optimisation of distribution/communication/algorithm | Packed `[x,y,z,m]` buffer → 1 collective/step, `MPI_IN_PLACE`, one shared physics kernel, remainder-balanced decomposition, [optional: ring-exchange comparison or Barnes–Hut] |
| 4. Performance & scalability (CA2) | Reproducible experiments; time, comm overhead, speedup, efficiency, load balance, scalability | `performance.csv`, `summary.md`, five plots, `load_balance_summary.csv`, scalability paragraph from `analyze.py` |
| 5. Impact & justification (CA4) | Scientific/engineering value, efficiency, resource use, scalability | Section 9 of the report |

## B. Technical report structure

1. **Introduction** — problem, N-body in astrophysics/molecular dynamics, objectives (CLO 1, CLO 2).
2. **Background** — direct O(N²) method, softening, leapfrog, MPI collectives, speedup/efficiency definitions.
3. **Environment** — CPU model, cores, RAM, WSL version, Ubuntu version, g++ and Open MPI versions (copy from `results/logs/system_info.txt`). *(Criterion 1)*
4. **Sequential design** — data layout, algorithm, parameters, energy conservation check.
5. **Distributed design** *(Criterion 2, 3)*
   - decomposition: block scheme, formula for counts/displacements, remainder rule (1 extra body to the first `N mod P` ranks)
   - communication: why `Allgatherv` (every body needs every position), packed buffer, `MPI_IN_PLACE`
   - synchronisation: collective as per-step barrier; barrier only around timing/I/O
   - unnecessary communication identified: gathering the constant mass each step is redundant (3 of 4 doubles change); velocities are never communicated; [state what you chose to keep and why]
   - complexity: computation O(N²/P) per step per rank vs communication O(N) per rank per step ⇒ comm/comp ratio grows ~P/N
6. **Correctness** — `verify.py` output for P = 1, 2, 4, 8, including non-divisible N; stated tolerance 1e-9; energy drift table.
7. **Experimental setup** — N list, P list, steps, runs (median), seed, init mode, dt, eps; how to reproduce (README §6). *(Criterion 4)*
8. **Results & analysis** *(Criterion 4)*
   - time vs N (`time_vs_N.png`), speedup (`speedup_vs_P.png`), efficiency (`efficiency_vs_P.png`)
   - computation vs communication (`comp_vs_comm.png`); the P at which communication ≥ 50% (from `analyze.py`)
   - load balance for N not divisible by P (`load_balance.png`, `load_balance_summary.csv`): body imbalance vs measured time imbalance
   - scalability: which N scales, where adding processes stops helping, why (comm ∝ N, compute ∝ N²/P)
   - sanity check: MPI at P = 1 vs sequential should be ≈ 1.0 speedup; explain any gap
9. **Impact and justification** *(Criterion 5)*
   - large-scale uses: galaxy/cluster dynamics, molecular dynamics, plasma and particle simulations, where N is far beyond one machine's reach
   - efficiency: show efficiency numbers to argue when extra processes waste energy/CPU time (e.g. low efficiency at small N means fewer cores should be used)
   - resource use: memory is replicated per rank (N positions), so memory does not scale with P; discuss implication
   - scalability limits and how Barnes–Hut or domain decomposition would reduce work/communication
10. **Limitations** — see README §9; add anything measured on your hardware (e.g. oversubscription, WSL overhead, single node).
11. **Conclusion & future work** — Barnes–Hut (O(N log N)), non-blocking overlap of communication and computation, ring exchange, cluster deployment.
12. **References** — MPI standard, Open MPI docs, N-body/Barnes–Hut paper, course material.
13. **Appendix** — code listing pointers (repo URL), full CSV, full command log.

## C. Presentation (max 12 slides)

| # | Slide | Content |
|---|---|---|
| 1 | Title | Project, course, group members |
| 2 | Problem & objectives | N-body, why distribute, CLOs |
| 3 | Sequential baseline | O(N²) algorithm, energy conservation |
| 4 | Distributed design | Block decomposition diagram incl. remainder example (e.g. N = 1003 on 8 ranks) |
| 5 | Communication & synchronisation | Per-step timeline: kick/drift → Allgatherv → force → kick |
| 6 | Optimisations | Packed buffer, `MPI_IN_PLACE`, shared kernel, [bonus] |
| 7 | Correctness | `verify.py` PASS, tolerance, energy drift |
| 8 | Experimental setup | Hardware, N, P, runs, reproducibility |
| 9 | Performance | Speedup + efficiency plots |
| 10 | Communication overhead & scalability | `comp_vs_comm.png`, crossover point |
| 11 | Load balance | `load_balance.png`, non-divisible N |
| 12 | Animation / impact / conclusion | Rank-coloured 3D animation, impact (Criterion 5), limitations, future work |

Tip: put the rank-coloured animation (`animate.py --ranks P`) on slide 4 or 12 —
it shows the data decomposition visually.

## D. Final deliverables checklist (from the brief)

- [ ] Sequential direct N-body (`src/sequential.cpp`)
- [ ] MPI version tested with P = 1, 2, 4, 8 (where hardware permits)
- [ ] Experiments for N = 100, 500, 1000, 5000, 10000
- [ ] Consistency with sequential within stated tolerance (`verify.py`)
- [ ] Computation, communication, total time, speedup, efficiency measured
- [ ] Non-divisible N investigated (`run_load_balance.sh`)
- [ ] Scalability and communication-overhead analysis
- [ ] README with install / run / parameters / reproducibility
- [ ] Logs and CSV data, tables and graphs
- [ ] Technical report
- [ ] Presentation, ≤ 12 slides

// mpi_nbody.cpp - distributed direct N-body simulation (MPI).
//
// Decomposition : 1-D block decomposition of bodies (see decomposition.hpp).
// Communication : ONE MPI_Allgatherv (MPI_IN_PLACE) of packed [x,y,z,m] per step.
// Synchronisation: the collective is the per-step synchronisation point; a
//                  barrier is used only before timing starts and around
//                  snapshot I/O so that I/O never pollutes the measurements.
//
// Per-step structure (kick-drift-kick leapfrog):
//   half_kick + drift   (local slice)          -> t_comp
//   Allgatherv positions                       -> t_comm  (includes waiting
//                                                 for slower ranks = imbalance)
//   compute_acc + half_kick (local slice)      -> t_comp
//
// Output (rank 0):
//   RESULT,mpi,N,P,steps,t_total,t_comp_avg,t_comm_avg,E0,E1,comp_max,comp_min,imbalance
//   RANK,N,P,rank,bodies,t_comp,t_comm               (one line per rank)

#include <mpi.h>

#include <cstdio>
#include <exception>
#include <iostream>
#include <vector>

#include "common.hpp"
#include "decomposition.hpp"
#include "init.hpp"
#include "physics.hpp"
#include "snapshot.hpp"
#include "timer.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, P = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    // Every rank parses the same arguments, so all ranks succeed or fail together.
    Params p;
    std::string err;
    if (!parse_args(argc, argv, p, err)) {
        if (rank == 0) {
            if (err != "help") std::cerr << "Error: " << err << "\n";
            print_usage(argv[0]);
        }
        MPI_Finalize();
        return err == "help" ? 0 : 1;
    }
    if (p.N < P) {
        if (rank == 0) std::cerr << "Error: need N >= number of processes (N=" << p.N
                                 << ", P=" << P << ")\n";
        MPI_Finalize();
        return 1;
    }

    try {
        const int N = p.N;
        const Decomposition d = block_decompose(N, P);
        const int lb = d.begin(rank), le = d.end(rank);
        const std::vector<int> counts4 = d.scaled_counts(4);
        const std::vector<int> displs4 = d.scaled_displs(4);

        // Identical initial system on every rank (same seed, no communication).
        System s(N);
        init_system(s, p);

        // Initial accelerations for the local slice; energy via Allreduce.
        compute_acc(s, lb, le, p.G, p.eps);
        double e_loc = kinetic_energy(s, lb, le) + potential_energy(s, lb, le, p.G, p.eps);
        double E0 = 0.0;
        MPI_Allreduce(&e_loc, &E0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        SnapshotWriter snap;
        if (p.save_traj && rank == 0) {
            if (!snap.open(p.snap_file, N)) {
                std::cerr << "Error: cannot open snapshot file " << p.snap_file << "\n";
                MPI_Abort(MPI_COMM_WORLD, 2);
            }
            snap.write(0, s);
        }

        // ---- timed simulation loop ----
        Stopwatch t_total, t_comp, t_comm;
        MPI_Barrier(MPI_COMM_WORLD);
        t_total.start();
        for (int step = 1; step <= p.steps; ++step) {
            t_comp.start();
            half_kick(s, lb, le, p.dt);
            drift(s, lb, le, p.dt);
            t_comp.stop();

            t_comm.start();
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, s.pos.data(),
                           counts4.data(), displs4.data(), MPI_DOUBLE, MPI_COMM_WORLD);
            t_comm.stop();

            t_comp.start();
            compute_acc(s, lb, le, p.G, p.eps);
            half_kick(s, lb, le, p.dt);
            t_comp.stop();

            if (p.save_traj && step % p.snap_interval == 0) {
                t_total.stop();                       // exclude I/O from timing
                if (rank == 0) snap.write(step, s);
                MPI_Barrier(MPI_COMM_WORLD);          // others don't charge the wait to t_comm
                t_total.start();
            }
        }
        t_total.stop();
        // -------------------------------
        snap.close();

        e_loc = kinetic_energy(s, lb, le) + potential_energy(s, lb, le, p.G, p.eps);
        double E1 = 0.0;
        MPI_Allreduce(&e_loc, &E1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // ---- reduce timings ----
        const double comp = t_comp.total(), comm = t_comm.total(), total = t_total.total();
        double comp_max = 0, comp_min = 0, comp_sum = 0, comm_sum = 0, total_max = 0;
        MPI_Reduce(&comp, &comp_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&comp, &comp_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&comp, &comp_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&comm, &comm_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&total, &total_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        // Per-rank breakdown for the load-balance analysis.
        const double mine[2] = {comp, comm};
        std::vector<double> all(rank == 0 ? 2 * static_cast<size_t>(P) : 0);
        MPI_Gather(mine, 2, MPI_DOUBLE, all.data(), 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // ---- optional final-state dump for verification ----
        if (!p.out_file.empty()) {
            const std::vector<int> counts3 = d.scaled_counts(3);
            const std::vector<int> displs3 = d.scaled_displs(3);
            if (rank == 0)
                MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, s.vel.data(), counts3.data(),
                            displs3.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            else
                MPI_Gatherv(s.vel.data() + 3 * lb, 3 * (le - lb), MPI_DOUBLE, nullptr, nullptr,
                            nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            if (rank == 0 && !dump_state(s, p.out_file)) {
                std::cerr << "Error: cannot write " << p.out_file << "\n";
                MPI_Abort(MPI_COMM_WORLD, 2);
            }
        }

        if (rank == 0) {
            const double comp_avg = comp_sum / P;
            std::printf("RESULT,mpi,%d,%d,%d,%.6f,%.6f,%.6f,%.12e,%.12e,%.6f,%.6f,%.4f\n", N, P,
                        p.steps, total_max, comp_avg, comm_sum / P, E0, E1, comp_max, comp_min,
                        comp_avg > 0 ? comp_max / comp_avg : 1.0);
            for (int r = 0; r < P; ++r)
                std::printf("RANK,%d,%d,%d,%d,%.6f,%.6f\n", N, P, r, d.counts[r], all[2 * r],
                            all[2 * r + 1]);
            std::fflush(stdout);
        }
    } catch (const std::exception& e) {
        std::cerr << "Rank " << rank << " error: " << e.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    MPI_Finalize();
    return 0;
}

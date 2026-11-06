// sequential.cpp - baseline direct O(N^2) N-body simulation.
// Serves as the correctness reference and the T_seq used for speedup.
//
// Output: one machine-readable line, parsed by scripts/analyze.py
//   RESULT,seq,N,1,steps,t_total,t_comp,t_comm,E0,E1,comp_max,comp_min,imbalance
// (same columns as the MPI version; comm = 0, imbalance = 1)

#include <cstdio>
#include <exception>
#include <iostream>

#include "common.hpp"
#include "init.hpp"
#include "physics.hpp"
#include "snapshot.hpp"
#include "timer.hpp"

int main(int argc, char** argv) {
    Params p;
    std::string err;
    if (!parse_args(argc, argv, p, err)) {
        if (err != "help") std::cerr << "Error: " << err << "\n";
        print_usage(argv[0]);
        return err == "help" ? 0 : 1;
    }

    try {
        System s(p.N);
        init_system(s, p);
        const int N = p.N;

        // Initial accelerations and energy (outside the timed region).
        compute_acc(s, 0, N, p.G, p.eps);
        const double E0 = kinetic_energy(s, 0, N) + potential_energy(s, 0, N, p.G, p.eps);

        SnapshotWriter snap;
        if (p.save_traj) {
            if (!snap.open(p.snap_file, N)) {
                std::cerr << "Error: cannot open snapshot file " << p.snap_file << "\n";
                return 2;
            }
            snap.write(0, s);
        }

        // ---- timed simulation loop ----
        Stopwatch t_total, t_comp;
        t_total.start();
        for (int step = 1; step <= p.steps; ++step) {
            t_comp.start();
            half_kick(s, 0, N, p.dt);
            drift(s, 0, N, p.dt);
            compute_acc(s, 0, N, p.G, p.eps);
            half_kick(s, 0, N, p.dt);
            t_comp.stop();

            if (p.save_traj && step % p.snap_interval == 0) {
                t_total.stop();           // exclude I/O from timing
                snap.write(step, s);
                t_total.start();
            }
        }
        t_total.stop();
        snap.close();
        // -------------------------------

        const double E1 = kinetic_energy(s, 0, N) + potential_energy(s, 0, N, p.G, p.eps);

        if (!p.out_file.empty() && !dump_state(s, p.out_file)) {
            std::cerr << "Error: cannot write " << p.out_file << "\n";
            return 2;
        }

        std::printf("RESULT,seq,%d,1,%d,%.6f,%.6f,%.6f,%.12e,%.12e,%.6f,%.6f,%.4f\n", N, p.steps,
                    t_total.total(), t_comp.total(), 0.0, E0, E1, t_comp.total(),
                    t_comp.total(), 1.0);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 3;
    }
}

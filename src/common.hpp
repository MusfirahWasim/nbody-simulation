// common.hpp - shared types, simulation parameters and CLI parsing.
// Used by both the sequential and the MPI implementations.
#pragma once

#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Simulation parameters (identical on every MPI rank)
// ---------------------------------------------------------------------------
struct Params {
    int N = 1000;                 // number of bodies
    int steps = 100;              // number of time steps
    double dt = 0.01;             // time step
    double G = 1.0;               // gravitational constant (normalised units)
    double eps = 0.05;            // softening length (avoids singularity)
    unsigned long long seed = 42; // RNG seed -> reproducible initial state
    int init_mode = 0;            // 0 = random cube, 1 = two clusters, 2 = disk

    bool save_traj = false;       // write snapshots for visualisation
    int snap_interval = 10;       // snapshot every k steps
    std::string snap_file = "results/snapshots/traj.bin";
    std::string out_file;         // optional final-state dump (for verify.py)
};

// ---------------------------------------------------------------------------
// System state.
//   pos : packed [x, y, z, m] per body  -> ONE Allgatherv per step (4 doubles
//         per body) instead of four separate collectives.
//   vel, acc : [x, y, z] per body. Full length on every rank for simple
//         global indexing; each rank only updates its own slice.
// ---------------------------------------------------------------------------
struct System {
    int N = 0;
    std::vector<double> pos;
    std::vector<double> vel;
    std::vector<double> acc;

    explicit System(int n = 0)
        : N(n), pos(4 * static_cast<size_t>(n), 0.0),
          vel(3 * static_cast<size_t>(n), 0.0),
          acc(3 * static_cast<size_t>(n), 0.0) {}
};

// ---------------------------------------------------------------------------
// Command-line parsing with validation (error handling requirement)
// ---------------------------------------------------------------------------
inline void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [options]\n"
        << "  --n <int>               number of bodies            (default 1000)\n"
        << "  --steps <int>           number of time steps        (default 100)\n"
        << "  --dt <double>           time step                   (default 0.01)\n"
        << "  --eps <double>          softening length            (default 0.05)\n"
        << "  --seed <uint>           RNG seed                    (default 42)\n"
        << "  --init <0|1|2>          0=cube 1=clusters 2=disk    (default 0)\n"
        << "  --save-trajectory       write snapshots for animation\n"
        << "  --snap-interval <int>   snapshot every k steps      (default 10)\n"
        << "  --snap-file <path>      snapshot output file\n"
        << "  --out <path>            dump final state (for verification)\n"
        << "  -h, --help              show this message\n";
}

// Returns true on success. On failure (or --help) returns false and sets err.
inline bool parse_args(int argc, char** argv, Params& p, std::string& err) {
    auto to_int = [](const std::string& opt, const std::string& v) {
        try {
            size_t pos = 0;
            int r = std::stoi(v, &pos);
            if (pos != v.size()) throw std::invalid_argument("");
            return r;
        } catch (...) {
            throw std::invalid_argument("invalid integer for " + opt + ": '" + v + "'");
        }
    };
    auto to_double = [](const std::string& opt, const std::string& v) {
        try {
            size_t pos = 0;
            double r = std::stod(v, &pos);
            if (pos != v.size()) throw std::invalid_argument("");
            return r;
        } catch (...) {
            throw std::invalid_argument("invalid number for " + opt + ": '" + v + "'");
        }
    };

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
                return argv[++i];
            };

            if (a == "--n")                     p.N = to_int(a, next());
            else if (a == "--steps")            p.steps = to_int(a, next());
            else if (a == "--dt")               p.dt = to_double(a, next());
            else if (a == "--eps")              p.eps = to_double(a, next());
            else if (a == "--seed")             p.seed = static_cast<unsigned long long>(to_int(a, next()));
            else if (a == "--init")             p.init_mode = to_int(a, next());
            else if (a == "--snap-interval")    p.snap_interval = to_int(a, next());
            else if (a == "--snap-file")        p.snap_file = next();
            else if (a == "--out")              p.out_file = next();
            else if (a == "--save-trajectory")  p.save_traj = true;
            else if (a == "-h" || a == "--help") { err = "help"; return false; }
            else throw std::invalid_argument("unknown option: " + a);
        }
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }

    // Validation
    if (p.N < 2)              { err = "--n must be >= 2"; return false; }
    if (p.steps < 1)          { err = "--steps must be >= 1"; return false; }
    if (p.dt <= 0.0)          { err = "--dt must be > 0"; return false; }
    if (p.eps <= 0.0)         { err = "--eps must be > 0"; return false; }
    if (p.init_mode < 0 || p.init_mode > 2) { err = "--init must be 0, 1 or 2"; return false; }
    if (p.snap_interval < 1)  { err = "--snap-interval must be >= 1"; return false; }
    return true;
}

// ---------------------------------------------------------------------------
// Final-state dump (text, full precision) - compared by scripts/verify.py.
// Format: one line per body: id x y z vx vy vz
// ---------------------------------------------------------------------------
inline bool dump_state(const System& s, const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return false;
    for (int i = 0; i < s.N; ++i) {
        std::fprintf(f, "%d %.17g %.17g %.17g %.17g %.17g %.17g\n", i,
                     s.pos[4 * i], s.pos[4 * i + 1], s.pos[4 * i + 2],
                     s.vel[3 * i], s.vel[3 * i + 1], s.vel[3 * i + 2]);
    }
    std::fclose(f);
    return true;
}

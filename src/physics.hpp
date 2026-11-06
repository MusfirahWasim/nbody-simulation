// physics.hpp - direct O(N^2) force computation and kick-drift-kick leapfrog.
// Every function works on a body range [begin, end), so the SAME code runs
// the whole system in the sequential version (0..N) and only the rank-local
// slice in the MPI version. This guarantees both versions share one physics
// implementation.
//
// Per-step structure (kick-drift-kick):
//   v += 0.5*dt*a      half_kick
//   x += dt*v          drift
//   [MPI: Allgatherv of positions]
//   a  = f(x)          compute_acc   (needs ALL positions)
//   v += 0.5*dt*a      half_kick
#pragma once

#include <cmath>

#include "common.hpp"

// Acceleration of bodies [begin, end) due to ALL N bodies.
// The j-loop always runs 0..N-1 in the same order, so each body's force sum
// is evaluated identically in the sequential and the MPI version.
// The i == j term contributes zero (dx = dy = dz = 0), so no branch is needed.
inline void compute_acc(System& s, int begin, int end, double G, double eps) {
    const double eps2 = eps * eps;
    const double* p = s.pos.data();
    const int N = s.N;

    for (int i = begin; i < end; ++i) {
        const double xi = p[4 * i], yi = p[4 * i + 1], zi = p[4 * i + 2];
        double ax = 0.0, ay = 0.0, az = 0.0;
        for (int j = 0; j < N; ++j) {
            const double dx = p[4 * j] - xi;
            const double dy = p[4 * j + 1] - yi;
            const double dz = p[4 * j + 2] - zi;
            const double r2 = dx * dx + dy * dy + dz * dz + eps2;
            const double inv_r = 1.0 / std::sqrt(r2);
            const double w = p[4 * j + 3] * inv_r * inv_r * inv_r; // m_j / r^3
            ax += dx * w;
            ay += dy * w;
            az += dz * w;
        }
        s.acc[3 * i] = G * ax;
        s.acc[3 * i + 1] = G * ay;
        s.acc[3 * i + 2] = G * az;
    }
}

inline void half_kick(System& s, int begin, int end, double dt) {
    const double h = 0.5 * dt;
    for (int i = begin; i < end; ++i) {
        s.vel[3 * i] += h * s.acc[3 * i];
        s.vel[3 * i + 1] += h * s.acc[3 * i + 1];
        s.vel[3 * i + 2] += h * s.acc[3 * i + 2];
    }
}

inline void drift(System& s, int begin, int end, double dt) {
    for (int i = begin; i < end; ++i) {
        s.pos[4 * i] += dt * s.vel[3 * i];
        s.pos[4 * i + 1] += dt * s.vel[3 * i + 1];
        s.pos[4 * i + 2] += dt * s.vel[3 * i + 2];
    }
}

// ---------------------------------------------------------------------------
// Energy diagnostics (partial sums over [begin, end)).
// MPI version: MPI_Allreduce(SUM) the partials to get the global value.
// Used as a physical sanity check: total energy should stay nearly constant.
// ---------------------------------------------------------------------------
inline double kinetic_energy(const System& s, int begin, int end) {
    double ke = 0.0;
    for (int i = begin; i < end; ++i) {
        const double v2 = s.vel[3 * i] * s.vel[3 * i] +
                          s.vel[3 * i + 1] * s.vel[3 * i + 1] +
                          s.vel[3 * i + 2] * s.vel[3 * i + 2];
        ke += 0.5 * s.pos[4 * i + 3] * v2;
    }
    return ke;
}

// Each pair is visited twice across all i, hence the 0.5 factor.
// The j == i term is skipped here (softened self-energy would be non-zero).
inline double potential_energy(const System& s, int begin, int end, double G, double eps) {
    const double eps2 = eps * eps;
    const double* p = s.pos.data();
    double pe = 0.0;
    for (int i = begin; i < end; ++i) {
        for (int j = 0; j < s.N; ++j) {
            if (j == i) continue;
            const double dx = p[4 * j] - p[4 * i];
            const double dy = p[4 * j + 1] - p[4 * i + 1];
            const double dz = p[4 * j + 2] - p[4 * i + 2];
            pe -= 0.5 * G * p[4 * i + 3] * p[4 * j + 3] /
                  std::sqrt(dx * dx + dy * dy + dz * dz + eps2);
        }
    }
    return pe;
}

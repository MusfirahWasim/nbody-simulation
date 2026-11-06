// init.hpp - deterministic, seeded initial conditions.
// Every MPI rank calls this with the same seed, so all ranks build an
// identical initial system without any communication.
#pragma once

#include <cmath>
#include <random>
#include <stdexcept>

#include "common.hpp"

inline void init_system(System& s, const Params& p) {
    std::mt19937_64 rng(p.seed);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    const int N = s.N;
    const double PI = 3.14159265358979323846;

    auto set = [&](int i, double x, double y, double z,
                   double vx, double vy, double vz, double m) {
        s.pos[4 * i + 0] = x;  s.pos[4 * i + 1] = y;  s.pos[4 * i + 2] = z;
        s.pos[4 * i + 3] = m;
        s.vel[3 * i + 0] = vx; s.vel[3 * i + 1] = vy; s.vel[3 * i + 2] = vz;
    };

    // Uniform random point inside a sphere of radius R (rejection sampling)
    auto in_sphere = [&](double R, double& x, double& y, double& z) {
        do { x = U(rng); y = U(rng); z = U(rng); } while (x * x + y * y + z * z > 1.0);
        x *= R; y *= R; z *= R;
    };

    const double m_each = 1.0 / N; // total mass = 1

    switch (p.init_mode) {
    case 0: { // random cube [-1,1]^3, small random velocities
        for (int i = 0; i < N; ++i) {
            double x = U(rng), y = U(rng), z = U(rng);
            set(i, x, y, z, 0.1 * U(rng), 0.1 * U(rng), 0.1 * U(rng), m_each);
        }
        break;
    }
    case 1: { // two colliding clusters
        for (int i = 0; i < N; ++i) {
            double x, y, z;
            in_sphere(0.5, x, y, z);
            const bool left = (i < N / 2);
            const double cx = left ? -1.5 : 1.5;
            const double bvx = left ? 0.3 : -0.3;
            const double bvy = left ? 0.1 : -0.1;
            set(i, cx + x, y, z,
                bvx + 0.05 * U(rng), bvy + 0.05 * U(rng), 0.05 * U(rng), m_each);
        }
        break;
    }
    case 2: { // rotating disk around a heavy central body (nice for animation)
        const double M = 1.0;
        set(0, 0, 0, 0, 0, 0, 0, M);
        const double m_disk = 0.1 / (N - 1);
        for (int i = 1; i < N; ++i) {
            const double r = 0.3 + 1.7 * (U(rng) + 1.0) / 2.0;
            const double th = PI * (U(rng) + 1.0);
            const double z = 0.02 * U(rng);
            const double v = std::sqrt(p.G * M / r); // circular orbit speed
            set(i, r * std::cos(th), r * std::sin(th), z,
                -v * std::sin(th), v * std::cos(th), 0.0, m_disk);
        }
        break;
    }
    default:
        throw std::invalid_argument("unknown init mode");
    }
}

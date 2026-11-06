// timer.hpp - wall-clock timing shared by the sequential and MPI builds.
//   * Compiled with -DUSE_MPI  -> MPI_Wtime()  (synchronised, high resolution)
//   * Otherwise                -> std::chrono::steady_clock
//
// Stopwatch accumulates elapsed time over many start()/stop() pairs, so each
// rank can keep separate totals for computation and communication.
#pragma once

#include <chrono>

#ifdef USE_MPI
#include <mpi.h>
#endif

inline double wall_time() {
#ifdef USE_MPI
    return MPI_Wtime();
#else
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
#endif
}

class Stopwatch {
public:
    void start() { t0_ = wall_time(); }
    void stop() { total_ += wall_time() - t0_; }
    double total() const { return total_; }
    void reset() { total_ = 0.0; }

private:
    double t0_ = 0.0;
    double total_ = 0.0;
};

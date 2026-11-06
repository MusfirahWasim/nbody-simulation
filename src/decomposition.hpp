// decomposition.hpp - documented data-decomposition strategy.
//
// Strategy: 1-D BLOCK decomposition of the body index range [0, N).
//   base = N / P,  rem = N % P
//   rank r owns  base + 1  bodies if r < rem,  otherwise  base  bodies.
//
// Consequences:
//   * Every body is owned by exactly one rank, blocks are contiguous.
//   * When N is not divisible by P the remainder is spread one body at a time
//     over the first `rem` ranks, so counts differ by at most 1 body
//     (the best possible static balance for equal-cost bodies, because the
//     direct method costs O(N) per body regardless of position).
//   * counts[] / displs[] map directly onto MPI_Allgatherv arguments.
#pragma once

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

struct Decomposition {
    int N = 0;
    int P = 0;
    std::vector<int> counts; // bodies owned by each rank
    std::vector<int> displs; // global index of each rank's first body

    int begin(int rank) const { return displs[rank]; }
    int end(int rank) const { return displs[rank] + counts[rank]; }

    // Same layout scaled by `k` doubles per body (k = 4 for packed positions).
    std::vector<int> scaled_counts(int k) const {
        std::vector<int> v(counts);
        for (int& c : v) c *= k;
        return v;
    }
    std::vector<int> scaled_displs(int k) const {
        std::vector<int> v(displs);
        for (int& d : v) d *= k;
        return v;
    }

    // Load-imbalance of the body distribution: max(count) / mean(count).
    // 1.0 = perfect balance.
    double count_imbalance() const {
        const int mx = *std::max_element(counts.begin(), counts.end());
        const double mean = static_cast<double>(N) / P;
        return mx / mean;
    }
};

inline Decomposition block_decompose(int N, int P) {
    if (N < 1 || P < 1) throw std::invalid_argument("block_decompose: N and P must be >= 1");
    Decomposition d;
    d.N = N;
    d.P = P;
    d.counts.resize(P);
    d.displs.resize(P);
    const int base = N / P;
    const int rem = N % P;
    for (int r = 0; r < P; ++r) d.counts[r] = base + (r < rem ? 1 : 0);
    d.displs[0] = 0;
    for (int r = 1; r < P; ++r) d.displs[r] = d.displs[r - 1] + d.counts[r - 1];
    return d;
}

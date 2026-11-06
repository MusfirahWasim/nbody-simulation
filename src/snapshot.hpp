// snapshot.hpp - binary trajectory writer for visualisation (scripts/animate.py).
//
// File layout (native little-endian):
//   int    N
//   repeat: int step | double x[N] | double y[N] | double z[N]
//
// Only one process (rank 0 in the MPI build) writes. After the per-step
// Allgatherv every rank already holds all positions, so NO extra communication
// is needed. Snapshot I/O must stay OUTSIDE the timed regions (see callers).
#pragma once

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "common.hpp"

class SnapshotWriter {
public:
    bool open(const std::string& path, int N) {
        namespace fs = std::filesystem;
        N_ = N;
        const fs::path parent = fs::path(path).parent_path();
        if (!parent.empty()) {
            std::error_code ec;
            fs::create_directories(parent, ec);
            if (ec) return false;
        }
        out_.open(path, std::ios::binary);
        if (!out_) return false;
        out_.write(reinterpret_cast<const char*>(&N_), sizeof(int));
        buf_.resize(3 * static_cast<size_t>(N));
        return static_cast<bool>(out_);
    }

    // De-interleave packed [x,y,z,m] into contiguous x[], y[], z[] blocks.
    bool write(int step, const System& s) {
        const size_t n = static_cast<size_t>(N_);
        for (size_t i = 0; i < n; ++i) {
            buf_[i] = s.pos[4 * i];
            buf_[n + i] = s.pos[4 * i + 1];
            buf_[2 * n + i] = s.pos[4 * i + 2];
        }
        out_.write(reinterpret_cast<const char*>(&step), sizeof(int));
        out_.write(reinterpret_cast<const char*>(buf_.data()), buf_.size() * sizeof(double));
        return static_cast<bool>(out_);
    }

    void close() { if (out_.is_open()) out_.close(); }

private:
    std::ofstream out_;
    std::vector<double> buf_;
    int N_ = 0;
};

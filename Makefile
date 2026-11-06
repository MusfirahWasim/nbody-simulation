CXX      ?= g++
MPICXX   ?= mpicxx
CXXFLAGS ?= -std=c++17 -O3 -march=native -Wall -Wextra
INC      := -Isrc

HEADERS := $(wildcard src/*.hpp)

.PHONY: all clean dirs

all: dirs bin/nbody_seq bin/nbody_mpi

dirs:
	@mkdir -p bin results/logs results/csv results/plots results/snapshots results/animations

bin/nbody_seq: src/sequential.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INC) $< -o $@

bin/nbody_mpi: src/mpi_nbody.cpp $(HEADERS)
	$(MPICXX) $(CXXFLAGS) -DUSE_MPI -DOMPI_SKIP_MPICXX $(INC) $< -o $@

clean:
	rm -rf bin

CXX ?= g++
MPICXX ?= mpicxx

CXXFLAGS ?= -O2 -std=c++17
MPI_CXXFLAGS ?=
MPI_LDFLAGS ?=

TARGETS = Problem1 Problem2 Problem3 Problem4 Problem5 Problem6 Problem7
KOKKOS_TARGETS = Problem2 Problem3 Problem4 Problem5 Problem6 Problem7

.PHONY: all clean help

all: $(TARGETS)

Problem1: Problem1.cpp
	$(MPICXX) $(CXXFLAGS) $(MPI_CXXFLAGS) -o $@ $< $(MPI_LDFLAGS)

$(KOKKOS_TARGETS): Problem%: Problem%.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	$(RM) $(TARGETS) $(TARGETS:%=%.exe)

help:
	@echo "Build all: make"
	@echo "Build one: make Problem6"
	@echo "Clean:     make clean"
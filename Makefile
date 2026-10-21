CXX = g++
CXXFLAGS = -O3 -std=c++17 -march=haswell -Wall -Wextra -Wno-unused-parameter
HDR = src/board.h src/judge_ref.h src/search.h src/tf.h

all: build/libataxx.so build/engine build/engine-ice

build:
	mkdir -p build

build/engine: src/engine.cpp $(HDR) | build
	$(CXX) $(CXXFLAGS) -static -o $@ $<

build/engine-ice: src/engine.cpp $(HDR) | build
	$(CXX) $(CXXFLAGS) -march=icelake-server -mprefer-vector-width=256 -static -o $@ $<

build/libataxx.so: src/lib.cpp $(HDR) | build
	$(CXX) $(CXXFLAGS) -fopenmp -fPIC -shared -o $@ $<

.PHONY: all

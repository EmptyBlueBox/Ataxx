CXX = g++
CXXFLAGS = -O3 -std=c++17 -march=haswell -Wall -Wextra -Wno-unused-parameter
ORT = .deps/onnxruntime-linux-x64-1.22.0
HDR = src/board.h src/judge_ref.h src/search.h

all: build/libataxx.so build/engine

$(ORT)/lib/libonnxruntime.so:
	mkdir -p .deps
	curl -fL https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz -o .deps/onnxruntime.tgz
	echo "8344d55f93d5bc5021ce342db50f62079daf39aaafb5d311a451846228be49b3  .deps/onnxruntime.tgz" | sha256sum -c -
	tar -xzf .deps/onnxruntime.tgz -C .deps
	rm .deps/onnxruntime.tgz

build:
	mkdir -p build

build/engine: src/engine.cpp src/net.h $(HDR) $(ORT)/lib/libonnxruntime.so | build
	$(CXX) $(CXXFLAGS) -I$(ORT)/include -o $@ $< -L$(ORT)/lib -lonnxruntime -Wl,-rpath,'$$ORIGIN/../$(ORT)/lib'

build/libataxx.so: src/lib.cpp $(HDR) | build
	$(CXX) $(CXXFLAGS) -fopenmp -fPIC -shared -o $@ $<

build/test_rules: src/test_rules.cpp $(HDR) | build
	$(CXX) $(CXXFLAGS) -fopenmp -o $@ $<

test: build/test_rules
	build/test_rules

.PHONY: all test

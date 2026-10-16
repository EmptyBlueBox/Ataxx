# Ataxx

## Python

MCTS bot with a neural network. Linux x86-64, NVIDIA CUDA.

```sh
cd python
make -j
uv run -m ataxx
```

## C++

Highest-ranked C++ bot on the [Ataxx botzone ladder](https://www.botzone.org.cn/game/ranklist/5809c7647f65182b044b5e3b).

```sh
cd cpp
cmake -S . -B build
cmake --build build -j
./build/ataxx
```

# Ataxx

## C++

C++17 and CMake 3.24+.

```sh
cd cpp
cmake -S . -B build
cmake --build build -j
./build/ataxx
```

Botzone: submit `build/botzone.cpp` after `cmake --build build --target submission`.
Evaluation parameters: [Autaxx](https://github.com/kz04px/autaxx), MIT.

## Python

Linux x86-64; NVIDIA CUDA for training and self-play.

```sh
cd python
make -j
uv run -m ataxx
```

The bot reads Botzone JSON from stdin and writes moves to stdout.

Run training and self-play in separate terminals.

```sh
uv run -m ataxx.train --net weights/champion.pt --data run/data --out run/nets
uv run -m ataxx.selfplay --net run/nets/latest.pt --out run/data
uv run -m ataxx.botzone out
```

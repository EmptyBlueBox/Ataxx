# Ataxx

Build and play (CMake, C++17):

```sh
cmake -S . -B build
cmake --build build -j
./build/ataxx
```

Use Clang on macOS or MinGW on Windows. The desktop build downloads raylib 6.0.

Click a piece, then a highlighted square. Single-player Undo returns to your previous turn.
Save and Load use `ataxx.save` in the working directory. Old `info.txt` saves are unsupported.

Build without the desktop dependency:

```sh
cmake -S . -B build-bot -DATAXX_DESKTOP=OFF
cmake --build build-bot --target bot submission -j
```

Submit `build-bot/botzone.cpp` to Botzone as C++17 with simple interaction enabled.

Evaluation parameters are adapted from [Autaxx](https://github.com/kz04px/autaxx) under the MIT license.

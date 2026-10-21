# Ataxx

Ataxx_Python is #1 on the [Botzone Ataxx ladder](https://www.botzone.org.cn/game/ranklist/5809c7647f65182b044b5e3b), Ataxx_CPP is the #1 C++ bot.

```sh
uv run -m ataxx.train --teacher KataAtaxx.onnx   # distil KataAtaxx into weights/ from scratch
uv run -m ataxx.package py|cpp|search            # build a Botzone submission in out/
```

Both online bots run the neural engine; `search` (pure alpha-beta) is not deployed.

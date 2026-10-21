import hashlib
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"


def bundle():
    subprocess.run(["make", "-C", str(ROOT)], check=True)
    phases = sorted(ROOT.glob("weights/p*-*.bin"), key=lambda p: int(p.stem[1:].split("-")[0]))
    files = {"engine": ROOT / "build/engine", "engine-ice": ROOT / "build/engine-ice", "net.bin": ROOT / "weights/champion.bin", **{p.name: p for p in phases}}
    v = hashlib.sha256(b"".join(f.read_bytes() for f in files.values())).hexdigest()[:10]
    with zipfile.ZipFile(OUT / f"v-{v}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name, f in files.items():
            z.write(f, f"Ataxx/v-{v}/{name}")
    return v, [f"{p.name}:{p.stem[1:].replace('-', ':')}" for p in phases]


def py():
    v, specs = bundle()
    (OUT / f"bot-{v}.py").write_text((ROOT / "bot/bot.py").read_text().replace("@@V@@", v).replace("@@P@@", " ".join(specs)))


def cpp():
    v, specs = bundle()
    paths = ", ".join(f'"data/Ataxx/v-{v}/{s}"' for s in specs)
    (OUT / f"bot-{v}.cpp").write_text((ROOT / "bot/bot.cpp").read_text().replace("@@V@@", v).replace("@@P@@", paths))


def search():
    source = '#pragma GCC optimize("O3")\n'
    for name in ["src/board.h", "search/patterns.h", "search/directional.h", "search/wide.h", "search/engine.hpp", "search/bot.cpp"]:
        source += re.sub(r'#include "[^\n]*\n|#pragma once\n', "", (ROOT / name).read_text()) + "\n"
    (OUT / "search.cpp").write_text(source)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    {"py": py, "cpp": cpp, "search": search}[sys.argv[1]]()

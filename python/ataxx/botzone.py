import argparse
import hashlib
import re
import subprocess
import zipfile
from pathlib import Path


def digest(paths):
    h = hashlib.sha256()
    for path in paths:
        h.update(Path(path).read_bytes())
    return h.hexdigest()[:10]


def package(net, out):
    root = Path(__file__).resolve().parent.parent
    engine = root / "build/engine"
    libs = sorted(set(re.findall(r"(/\S+) \(", subprocess.run(["ldd", engine], capture_output=True, text=True, check=True).stdout)))
    notices = [root / ".deps/onnxruntime-linux-x64-1.22.0" / name for name in ("LICENSE", "ThirdPartyNotices.txt")]
    rt, v = digest([*libs, *notices]), digest([engine, net, root / "book.txt"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out / f"rt-{rt}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for lib in [*libs, *notices]:
            z.write(lib, f"Ataxx/rt-{rt}/{Path(lib).name}")
    with zipfile.ZipFile(out / f"v-{v}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(engine, f"Ataxx/v-{v}/engine")
        z.write(net, f"Ataxx/v-{v}/net.onnx")
        z.write(root / "book.txt", f"Ataxx/v-{v}/book.txt")
    source = out / f"bot-{v}.py"
    source.write_text((root / "bot/bot.py").read_text().replace("@@RT@@", rt).replace("@@V@@", v))
    return out / f"rt-{rt}.zip", out / f"v-{v}.zip", source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    for path in package(root / "weights/champion.onnx", args.out):
        print(path)


if __name__ == "__main__":
    main()

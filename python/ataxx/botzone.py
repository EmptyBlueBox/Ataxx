import argparse
import hashlib
import zipfile
from pathlib import Path


def package(net, out):
    root = Path(__file__).resolve().parent.parent
    engine = root / "build/engine"
    v = hashlib.sha256(engine.read_bytes() + Path(net).read_bytes()).hexdigest()[:10]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out / f"v-{v}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(engine, f"Ataxx/v-{v}/engine")
        z.write(net, f"Ataxx/v-{v}/net.bin")
    source = out / f"bot-{v}.py"
    source.write_text((root / "bot/bot.py").read_text().replace("@@V@@", v))
    return out / f"v-{v}.zip", source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    for path in package(Path(__file__).resolve().parent.parent / "weights/champion.bin", args.out):
        print(path)


if __name__ == "__main__":
    main()

import argparse
import hashlib
import json
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


def cpp_package(net, out, phases):
    root = Path(__file__).resolve().parent.parent
    engine = root / "build/engine"
    weights = [Path(net)]
    specs = []
    for phase in phases:
        path, lo, hi = phase.rsplit(":", 2)
        weights.append(Path(path))
        specs.append(f"net{len(weights)-1}.bin:{lo}:{hi}")
    v = hashlib.sha256(engine.read_bytes() + b"".join(p.read_bytes() for p in weights)).hexdigest()[:12]
    folder = f"data/Ataxx/cpp-{v}/"
    command = [folder + "engine", folder + "net0.bin", "simpleio", "1.94", ".97", *[folder + s for s in specs]]
    code = '#include <sys/stat.h>\n#include <unistd.h>\nint main() {\n  const char* command[] = {' + ", ".join(map(json.dumps, command)) + ', nullptr};\n'
    code += '''  chmod(command[0], 0755);
  return execv(command[0], const_cast<char* const*>(command));
}
'''
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    archive = out / f"cpp-{v}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(engine, f"Ataxx/cpp-{v}/engine")
        for i, path in enumerate(weights):
            z.write(path, f"Ataxx/cpp-{v}/net{i}.bin")
    source = out / f"bot-{v}.cpp"
    source.write_text(code)
    return archive, source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--cpp", action="store_true")
    parser.add_argument("--phase", action="append", default=[])
    args = parser.parse_args()
    net = Path(__file__).resolve().parent.parent / "weights/champion.bin"
    paths = cpp_package(net, args.out, args.phase) if args.cpp else package(net, args.out)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

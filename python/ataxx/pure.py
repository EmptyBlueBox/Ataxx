import argparse
import json
import os
import struct
import subprocess
from pathlib import Path

DX = [-2, -1, 0, 1, 2, -2, 2, -2, 2, -2, 2, -2, -1, 0, 1, 2]
DY = [-2, -2, -2, -2, -2, -1, -1, 0, 0, 1, 1, 2, 2, 2, 2, 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--openings", type=Path, default=Path("openings.json"))
    parser.add_argument("--pairs", type=int, default=128)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--variants", type=int, nargs=2, default=[1, 0])
    parser.add_argument("--nodes", type=int, nargs=2, default=[300000, 300000])
    parser.add_argument("--table-bits", type=int, default=18)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--weights", type=Path, nargs=2)
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    openings = json.loads(args.openings.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    input_path = args.out.with_suffix(".bin")
    with input_path.open("wb") as output:
        output.write(struct.pack("<i", 2 * args.pairs))
        for i in range(args.pairs):
            moves = openings[args.offset + i]
            encoded = []
            for x, y, u, v in moves:
                kind = 0 if max(abs(x - u), abs(y - v)) == 1 else list(zip(DX, DY)).index((u - x, v - y)) + 1
                encoded.append(kind * 49 + 7 * v + u)
            line = struct.pack("<i", len(moves)) + struct.pack("<" + "H" * len(encoded), *encoded)
            output.write(line * 2)
    command = [str(args.engine.resolve()), str(input_path), str(args.out), *map(str, args.variants), *map(str, args.nodes), str(args.table_bits), "401", "64"]
    if args.weights:
        command.extend(map(str, args.weights))
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu)}
    if args.sample:
        environment["ATAXX_SAMPLE"] = "1"
    subprocess.run(command, env=environment, check=True)
    results = [list(map(int, line.split())) for line in args.out.read_text().splitlines()]
    wins = sum(winner == i % 2 for i, winner, _ in results)
    draws = sum(winner == -1 for _, winner, _ in results)
    print(json.dumps(dict(variants=args.variants, nodes=args.nodes, pairs=args.pairs, offset=args.offset, wins=wins, draws=draws, losses=len(results) - wins - draws, win_rate=wins / len(results), score=(wins + draws / 2) / len(results))))


if __name__ == "__main__":
    main()

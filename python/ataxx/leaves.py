import argparse
from pathlib import Path

import numpy as np
import torch

from ataxx.distill import Teacher
from ataxx.train import SHIFTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("positions", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--teacher", type=Path, default=Path("run/teacher.onnx"))
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()
    teacher = Teacher(str(args.teacher), gpus=range(args.gpus))
    rows = np.fromfile(args.positions, dtype=[("bb", "<u8", (2,)), ("side", "<i4"), ("ply", "<i4")])
    unique = np.unique(rows[["bb", "side"]], return_index=True)[1]
    rows = rows[np.sort(unique)]
    output = np.empty(len(rows), dtype=[("bb", "<u8", (2,)), ("side", "u1"), ("ply", "<u2"), ("q", "<i2")])
    for key in ("bb", "side", "ply"):
        output[key] = rows[key]
    for begin in range(0, len(rows), 4096 * args.gpus):
        batch = rows[begin:begin + 4096 * args.gpus]
        bb = torch.tensor(batch["bb"].astype(np.int64), device="cuda")
        side = torch.tensor(batch["side"].astype(np.int64), device="cuda")
        bits = (bb[:, :, None] >> SHIFTS.cuda() & 1).float()
        index = torch.arange(len(batch), device="cuda")
        probabilities = teacher.run(bits[index, side], bits[index, 1 - side])[1].softmax(1)
        value = probabilities[:, 0] - probabilities[:, 1]
        output["q"][begin:begin + len(batch)] = (value * 32767).round().cpu().numpy().astype(np.int16)
        if begin % 65536 == 0:
            print(begin, len(rows), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, output)


if __name__ == "__main__":
    main()

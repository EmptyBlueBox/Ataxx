import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ataxx import model
from ataxx.selfplay import NMOVES, library
from ataxx.train import bits, features

PHASES = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "cpp/weights.h")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    rows = np.concatenate([np.load(f) for f in sorted(args.data.glob("*.npy"))])
    squares, moves = np.zeros((8, 49), np.int32), np.zeros((8, NMOVES), np.int32)
    library().symmetry(squares.ctypes.data, moves.ctypes.data)
    orbit = torch.tensor(np.unique(squares.min(0), return_inverse=True)[1], device="cuda")
    t = {k: torch.from_numpy(np.ascontiguousarray(rows[k])).cuda().long() for k in ("bb", "side", "ply")}
    net = model.load(args.net, "cuda")
    with torch.no_grad():
        value = torch.cat([net(features(t["bb"][i, 0], t["bb"][i, 1], t["side"][i], t["ply"][i]))[1] for i in torch.arange(len(rows), device="cuda").split(8192)])
    target = value[:, :3].softmax(1) @ torch.tensor([1.0, 0.0, -1.0], device="cuda")
    cell = torch.zeros(PHASES, 2, 81, device="cuda", requires_grad=True)
    square = torch.zeros(PHASES, 10, device="cuda", requires_grad=True)
    tempo = torch.zeros(PHASES, device="cuda", requires_grad=True)

    def tables():
        empty = cell[:, 0].view(-1, 9, 9)
        own = cell[:, 1].view(-1, 9, 9)
        full = torch.stack([empty - empty.transpose(1, 2), own, -own.transpose(1, 2)], 1).flatten(2)
        return full, torch.stack([torch.zeros_like(square), square, -square], 1)[:, :, orbit]

    def evaluate(i):
        black, white, s = bits(t["bb"][i, 0]), bits(t["bb"][i, 1]), t["side"][i].view(-1, 1, 1, 1).float()
        own, opp = black * (1 - s) + white * s, white * (1 - s) + black * s
        near = lambda x: (F.avg_pool2d(x, 3, 1, 1) * 9 - x).round().long()
        state = (own + 2 * opp).long().flatten(1)
        index = (near(own) * 9 + near(opp)).flatten(1)
        phase = ((49 - own.sum((1, 2, 3)) - opp.sum((1, 2, 3))) * PHASES / 50).long()
        full, placed = tables()
        active = F.max_pool2d(own + opp, 3, 1, 1).flatten(1)
        value = full[phase[:, None], state, index] * active + placed[phase[:, None], state, torch.arange(49, device="cuda")]
        return tempo[phase] + value.sum(1), own.sum((1, 2, 3)) - opp.sum((1, 2, 3))

    optimizer = torch.optim.Adam([cell, square, tempo], 0.01)
    for epoch in range(args.epochs):
        total = 0
        for i in torch.randperm(len(rows), device="cuda").split(65536):
            loss = F.mse_loss(torch.tanh(evaluate(i)[0]), target[i]) + 1e-5 * cell.square().sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(i)
        print(epoch, total / len(rows), target.var().item(), flush=True)
    with torch.no_grad():
        value, material = evaluate(torch.arange(min(len(rows), 1 << 20), device="cuda"))
        scale = 100 * (material * material).sum() / (value * material).sum()
        full, placed = tables()
        full, placed, bias = [(x * scale).round().int().tolist() for x in (full, placed, tempo)]
    text = "#pragma once\n\nnamespace ataxx {\n\nstruct Weights {\n  int tempo, cell[3][81], square[3][49];\n};\n\nconstexpr int PHASES = %d;\n\ninline constexpr Weights weights[PHASES] = {\n" % PHASES
    brace = lambda x: "{" + ", ".join(map(brace, x)) + "}" if isinstance(x, list) else str(x)
    text += "".join("  {%d, %s, %s},\n" % (bias[p], brace(full[p]), brace(placed[p])) for p in range(PHASES))
    args.out.write_text(text + "};\n\n}\n")
    print("scale", scale.item())


if __name__ == "__main__":
    main()

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ataxx.train import SHIFTS


def indices():
    digits = np.arange(19683)[:, None] // (3 ** np.arange(9)) % 3
    grids = digits.reshape(-1, 3, 3)
    transforms = [np.rot90(grids, k, (1, 2)) for k in range(4)]
    transforms += [g.transpose(0, 2, 1) for g in transforms]
    canonical = np.stack([(g.reshape(-1, 9) * 3 ** np.arange(9)).sum(1) for g in transforms]).min(0)
    swap = ((digits == 1) * 2 + (digits == 2)) @ (3 ** np.arange(9))
    return canonical, swap


def encode(bb, side):
    b = (bb[:, :, None] >> SHIFTS.to(bb.device) & 1).float()
    row = torch.arange(len(bb), device=bb.device)
    own, opp = b[row, side], b[row, 1 - side]
    cell = (own + 2 * opp).reshape(-1, 1, 7, 7)
    index = (F.unfold(cell, 3, padding=1).long() * (3 ** torch.arange(9, device=bb.device))[None, :, None]).sum(1)
    near = lambda x, k: F.max_pool2d(x.reshape(-1, 1, 7, 7), k, 1, k // 2).flatten(1)
    empty = 1 - own - opp
    mobility = torch.stack([((near(own, k) - near(opp, k)) * empty).sum(1) for k in (3, 5)], 1)
    frontier = ((near(empty, 3) * (own - opp))).sum(1, keepdim=True)
    material = (own - opp).sum(1, keepdim=True)
    phase = (empty.sum(1).long() * 6 // 50)
    return index, own - opp, torch.cat([material, mobility, frontier], 1), phase


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--init", type=Path)
    parser.add_argument("--rows", type=int, default=20000000)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    rng = np.random.default_rng(2718)
    parts = []
    dtype = [("bb", "<u8", (2,)), ("side", "u1"), ("ply", "<u2"), ("q", "<i2")]
    for directory in args.data:
        files = sorted(directory.rglob("*.npy"))
        take = max(1, args.rows // len(args.data) // len(files))
        for file in files:
            a = np.load(file, mmap_mode="r")
            chosen = a[rng.choice(len(a), min(take, len(a)), replace=False)]
            part = np.empty(len(chosen), dtype=dtype)
            for key in part.dtype.names:
                part[key] = chosen[key]
            parts.append(part)
    rows = np.concatenate(parts)
    del parts
    rng.shuffle(rows)
    split = len(rows) * 19 // 20
    canonical = torch.tensor(np.unique(indices()[0], return_inverse=True)[1], device="cuda").long()
    xy = torch.arange(49, device="cuda")
    x, y = torch.minimum(xy % 7, 6 - xy % 7), torch.minimum(xy // 7, 6 - xy // 7)
    _, orbit = torch.unique(torch.minimum(x, y) * 4 + torch.maximum(x, y), return_inverse=True)
    power = 3 ** torch.arange(7, device="cuda")
    code = torch.arange(2187, device="cuda")
    digits = code[:, None] // power % 3
    reverse = (digits * power.flip(0)).sum(1)
    _, line_canon = torch.unique(torch.minimum(code, reverse), return_inverse=True)
    groups = torch.tensor([min(i, 6 - i) for i in range(7)] * 2, device="cuda")
    table = torch.zeros(6, 2862, device="cuda", requires_grad=True)
    square = torch.zeros(6, 10, device="cuda", requires_grad=True)
    linear = torch.zeros(6, 4, device="cuda", requires_grad=True)
    tempo = torch.zeros(6, device="cuda", requires_grad=True)
    line = torch.zeros(6, 4, 1134, device="cuda", requires_grad=True)
    edge = torch.zeros(6, 3, 3321, device="cuda", requires_grad=True)
    edge_squares = []
    for band in range(3):
        for rotation in range(4):
            for start in range(4):
                cells = []
                for y in range(band, band + 2):
                    for x in range(start, start + 4):
                        xx, yy = x, y
                        for _ in range(rotation):
                            xx, yy = 6 - yy, xx
                        cells.append(yy * 7 + xx)
                edge_squares.append(cells)
    edge_groups = torch.arange(3, device="cuda").repeat_interleave(16)
    edge_squares = torch.tensor(edge_squares, device="cuda")
    edge_power = 3 ** torch.arange(8, device="cuda")
    numbers = torch.arange(6561, device="cuda")
    digits = (numbers[:, None] // edge_power % 3).reshape(-1, 2, 4)
    reverse = (digits.flip(2).flatten(1) * edge_power).sum(1)
    _, edge_canon = torch.unique(torch.minimum(numbers, reverse), return_inverse=True)
    parameters = [table, square, linear, tempo, line, edge]
    if args.init:
        initial = np.load(args.init)
        scale = initial["scale"].item()
        representatives = [int((orbit == i).nonzero()[0]) for i in range(10)]
        with torch.no_grad():
            for key, a in zip(("table", "square", "linear", "tempo", "line", "edge"), parameters):
                values = initial[key][:, representatives] if key == "square" else initial[key]
                a.copy_(torch.tensor(values, device="cuda") / scale)
    optimizer = torch.optim.Adam(parameters, lr=0.003)

    encoded = [[] for _ in range(6)]
    for begin in range(0, len(rows), 16384):
        batch = rows[begin:begin + 16384]
        bb = torch.tensor(batch["bb"].astype(np.int64), device="cuda")
        side = torch.tensor(batch["side"].astype(np.int64), device="cuda")
        index, pieces, scalar, _ = encode(bb, side)
        cells = (pieces + 3 * (pieces < 0)).reshape(-1, 7, 7).long()
        li = torch.cat([(cells * power).sum(2), (cells.transpose(1, 2) * power).sum(2)], 1)
        values = [canonical[index].to(torch.int16), pieces.to(torch.int8), scalar.to(torch.int8), (pieces == 0).sum(1).to(torch.int8), line_canon[li].to(torch.int16)]
        di = (cells.flatten(1)[:, edge_squares] * edge_power).sum(2)
        values.append(edge_canon[di].to(torch.int16))
        for output, value in zip(encoded, values):
            output.append(value)
    index, pieces, scalar, empty_counts, line_index, edge_index = [torch.cat(a) for a in encoded]
    del encoded
    target = torch.tensor(rows["q"].astype(np.float32) / 32767, device="cuda")

    knots = torch.tensor([0, 2, 5, 10, 20, 45], device="cuda")

    def evaluate(i):
        empty = empty_counts[i].long()
        lower = torch.searchsorted(knots, empty, right=True).clamp(1, 5) - 1
        fraction = (empty - knots[lower]) / (knots[lower + 1] - knots[lower])
        value = 0
        for shift in (0, 1):
            phase = lower + shift
            part = table[phase[:, None], index[i].long()].sum(1) + line[phase[:, None], groups, line_index[i].long()].sum(1)
            part += edge[phase[:, None], edge_groups, edge_index[i].long()].sum(1)
            part += (square[phase][:, orbit] * pieces[i]).sum(1) + (linear[phase] * scalar[i]).sum(1) + tempo[phase]
            value = value + part * (fraction if shift else 1 - fraction)
        return value, target[i], scalar[i, 0].float()

    def validation():
        with torch.no_grad():
            error = 0
            for i in torch.arange(split, len(rows), device="cuda").split(16384):
                value, expected, _ = evaluate(i)
                error += F.mse_loss(torch.tanh(value), expected, reduction="sum").item()
            return error / (len(rows) - split)

    best = validation()
    saved = [a.detach().clone() for a in parameters]
    print(json.dumps(dict(rows=len(rows), train=split, initial=best)), flush=True)
    for epoch in range(args.epochs):
        total = 0
        for i in torch.randperm(split, device="cuda").split(16384):
            value, expected, _ = evaluate(i)
            error = F.mse_loss(torch.tanh(value), expected) + 1e-7 * table.square().sum()
            optimizer.zero_grad()
            error.backward()
            optimizer.step()
            total += error.item() * len(i)
        with torch.no_grad():
            error = validation()
            if error < best:
                best = error
                saved = [a.detach().clone() for a in parameters]
        print(json.dumps(dict(epoch=epoch, train=total / split, valid=error, best=best)), flush=True)
    with torch.no_grad():
        for a, b in zip(parameters, saved):
            a.copy_(b)
        if not args.init:
            numerator = denominator = 0
            for i in torch.arange(split, len(rows), device="cuda").split(16384):
                value, _, material = evaluate(i)
                numerator += material.square().sum().item()
                denominator += (value * material).sum().item()
            scale = 100 * numerator / denominator
        arrays = [torch.round(a * scale).to(torch.int16).cpu().numpy() for a in (table, square[:, orbit], linear, tempo, line, edge)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out.with_suffix(".npz"), **dict(zip(("table", "square", "linear", "tempo", "line", "edge"), arrays)), scale=scale, knots=knots.cpu().numpy())
    arrays[0] = arrays[0][:, canonical.cpu().numpy()]
    arrays[-1] = arrays[-1][:, :, edge_canon.cpu().numpy()]
    raw = b"".join(a.astype("<i2").tobytes() for a in arrays)
    text = """#pragma once
#include <cstdint>
namespace ataxx {
template<int N> struct PatternWeightsBase {
std::int16_t table[N][19683], square[N][49], linear[N][4], tempo[N], line[N][4][1134], edge[N][3][6561];
};
using PatternWeights = PatternWeightsBase<46>;
inline const PatternWeights& patternWeights() {
    static const PatternWeights weights = [] {
        const char* data = "@@DATA@@";
        const char* alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        int digits[128]{};
        for (int i = 0; i < 64; ++i) digits[int(alphabet[i])] = i;
        PatternWeightsBase<6> packed;
        auto* out = reinterpret_cast<unsigned char*>(&packed);
        for (unsigned i = 0; i < sizeof(packed) / 3; ++i) {
            unsigned value = (digits[int(data[4*i])] << 18) | (digits[int(data[4*i+1])] << 12) | (digits[int(data[4*i+2])] << 6) | digits[int(data[4*i+3])];
            out[3*i] = value >> 16;
            out[3*i+1] = value >> 8;
            out[3*i+2] = value;
        }
        PatternWeights result;
        const int knots[6] = {0, 2, 5, 10, 20, 45};
        auto expand = [&](auto& target, const auto& source) {
            for (int e = 0; e <= 45; ++e) {
                int lo = 0;
                while (lo < 4 && knots[lo + 1] <= e) ++lo;
                int den = knots[lo + 1] - knots[lo], num = e - knots[lo];
                auto* to = reinterpret_cast<std::int16_t*>(&target[e]);
                const auto* a = reinterpret_cast<const std::int16_t*>(&source[lo]);
                const auto* b = reinterpret_cast<const std::int16_t*>(&source[lo + 1]);
                for (unsigned k = 0; k < sizeof(source[0]) / sizeof(std::int16_t); ++k) {
                    int value = (den - num) * a[k] + num * b[k];
                    to[k] = value >= 0 ? (value + den / 2) / den : -((-value + den / 2) / den);
                }
            }
        };
        expand(result.table, packed.table);
        expand(result.square, packed.square);
        expand(result.linear, packed.linear);
        expand(result.tempo, packed.tempo);
        expand(result.line, packed.line);
        expand(result.edge, packed.edge);
        return result;
    }();
    return weights;
}
}
"""
    args.out.write_text(text.replace("@@DATA@@", base64.b64encode(raw).decode()))
    knots = knots.cpu().numpy()
    empty = np.arange(46)
    lo = np.searchsorted(knots, empty, side="right").clip(1, 5) - 1
    numerator, denominator = empty - knots[lo], knots[lo + 1] - knots[lo]
    expanded = []
    for a in arrays:
        shape = (46,) + (1,) * (a.ndim - 1)
        num, den = numerator.reshape(shape), denominator.reshape(shape)
        value = (den - num) * a[lo].astype(np.int32) + num * a[lo + 1].astype(np.int32)
        expanded.append((np.sign(value) * ((abs(value) + den // 2) // den)).astype("<i2").tobytes())
    args.out.with_suffix(".bin").write_bytes(b"".join(expanded))
    print(json.dumps(dict(scale=scale, bytes=args.out.stat().st_size)), flush=True)


if __name__ == "__main__":
    main()

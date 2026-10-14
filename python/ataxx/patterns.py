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
    local_canonical = np.unique(indices()[0], return_inverse=True)[1]
    digits = np.arange(19683)[:, None] // (3 ** np.arange(9)) % 3
    joint = np.full((49, 19683), 49 * 19683, dtype=np.int32)
    for t in range(8):
        x, y = np.arange(9) % 3, np.arange(9) // 3
        if t & 1: x = 2 - x
        if t & 2: y = 2 - y
        if t & 4: x, y = y, x
        transformed = digits @ (3 ** (3 * y + x))
        x, y = np.arange(49) % 7, np.arange(49) // 7
        if t & 1: x = 6 - x
        if t & 2: y = 6 - y
        if t & 4: x, y = y, x
        joint = np.minimum(joint, (7 * y + x)[:, None] * 19683 + transformed)
    cx, cy = np.arange(49) % 7, np.arange(49) // 7
    xx = cx[:,None] + np.arange(9)[None] % 3 - 1
    yy = cy[:,None] + np.arange(9)[None] // 3 - 1
    outside = (xx < 0) | (xx > 6) | (yy < 0) | (yy > 6)
    valid = ~(outside.astype(np.int32) @ (digits != 0).astype(np.int32).T).astype(bool)
    representatives, inverse = np.unique(joint[valid], return_inverse=True)
    mapped = np.full((49,19683), -1, np.int32)
    mapped[valid] = inverse
    canonical = torch.tensor(mapped, device="cuda").long()
    initial_codes = local_canonical[representatives % 19683]
    xy = torch.arange(49, device="cuda")
    x, y = torch.minimum(xy % 7, 6 - xy % 7), torch.minimum(xy // 7, 6 - xy // 7)
    _, orbit = torch.unique(torch.minimum(x, y) * 4 + torch.maximum(x, y), return_inverse=True)
    power = 3 ** torch.arange(7, device="cuda")
    code = torch.arange(2187, device="cuda")
    digits = code[:, None] // power % 3
    reverse = (digits * power.flip(0)).sum(1)
    _, line_canon = torch.unique(torch.minimum(code, reverse), return_inverse=True)
    groups = torch.tensor([min(i, 6 - i) for i in range(7)] * 2, device="cuda")
    table = torch.zeros(6, len(representatives), device="cuda", requires_grad=True)
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
        square_representatives = [int((orbit == i).nonzero()[0]) for i in range(10)]
        with torch.no_grad():
            for key, a in zip(("table", "square", "linear", "tempo", "line", "edge"), parameters):
                values = initial[key][:, square_representatives] if key == "square" else initial[key]
                if key == "table" and values.shape[1] == 2862: values = values[:, initial_codes]
                a.copy_(torch.tensor(values, device="cuda") / scale)
    anchor = table.detach().clone()
    if args.init:
        for parameter in parameters[1:]: parameter.requires_grad_(False)
        parameters = [table]
    optimizer = torch.optim.Adam(parameters, lr=0.003)

    encoded = [[] for _ in range(6)]
    for begin in range(0, len(rows), 16384):
        batch = rows[begin:begin + 16384]
        bb = torch.tensor(batch["bb"].astype(np.int64), device="cuda")
        side = torch.tensor(batch["side"].astype(np.int64), device="cuda")
        index, pieces, scalar, _ = encode(bb, side)
        cells = (pieces + 3 * (pieces < 0)).reshape(-1, 7, 7).long()
        li = torch.cat([(cells * power).sum(2), (cells.transpose(1, 2) * power).sum(2)], 1)
        values = [canonical[xy[None], index].to(torch.int32), pieces.to(torch.int8), scalar.to(torch.int8), (pieces == 0).sum(1).to(torch.int8), line_canon[li].to(torch.int16)]
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
            error = F.mse_loss(torch.tanh(value), expected) + 1e-7 * (table - anchor).square().sum()
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
    np.savez(args.out.with_suffix(".npz"), **dict(zip(("table", "square", "linear", "tempo", "line", "edge"), arrays)), scale=scale, knots=knots.cpu().numpy(), representatives=representatives)
    arrays[-1] = arrays[-1][:, :, edge_canon.cpu().numpy()]
    raw = b"".join(a.astype("<i2").tobytes() for a in arrays)
    lookup = canonical.cpu().numpy()[[0, 1, 2, 3, 8, 9, 10, 16, 17, 24]]
    arrays[0] = arrays[0][:, lookup]
    arrays[0][:, lookup < 0] = 0
    text = """#pragma once
#include <cstdint>
#include "board.h"
namespace ataxx {
template<int N> struct PatternWeightsBase {
std::int16_t table[N][6][19683], square[N][49], linear[N][4], tempo[N], line[N][4][1134], edge[N][6][6561];
};
using PatternWeights = PatternWeightsBase<46>;
struct PackedPatternWeights {
std::int16_t table[6][65286], square[6][49], linear[6][4], tempo[6], line[6][4][1134], edge[6][3][6561];
};
inline const PatternWeights& patternWeights() {
    static const PatternWeights weights = [] {
        const char* data = "@@DATA@@";
        const char* alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        int digits[128]{};
        for (int i = 0; i < 64; ++i) digits[int(alphabet[i])] = i;
        PackedPatternWeights packed;
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
                    to[k] = 2 * (value >= 0 ? (value + den / 2) / den : -((-value + den / 2) / den));
                }
            }
        };
        const int squares[10] = {0, 1, 2, 3, 8, 9, 10, 16, 17, 24};
        std::vector<int> index(10 * 19683, -1);
        int count = 0;
        for (int r = 0; r < 10; ++r) for (int code = 0; code < 19683; ++code) {
            int s = squares[r], value = code;
            bool valid = true;
            for (int i = 0; i < 9; ++i, value /= 3)
                if (!onBoard(s % 7 + i % 3 - 1, s / 7 + i / 3 - 1) && value % 3) valid = false;
            if (!valid) continue;
            int canonical = code;
            for (int t = 1; t < 8; ++t)
                if (T().symSq[t][s] == s) canonical = std::min(canonical, int(T().patternSym[t][code]));
            index[r * 19683 + code] = canonical == code ? count++ : index[r * 19683 + canonical];
        }
        for (int e = 0; e <= 45; ++e) {
            int lo = 0;
            while (lo < 4 && knots[lo + 1] <= e) ++lo;
            int den = knots[lo + 1] - knots[lo], num = e - knots[lo];
            auto interpolate = [&](int a, int b) {
                int value = (den - num) * a + num * b;
                return value >= 0 ? (value + den / 2) / den : -((-value + den / 2) / den);
            };
            auto local = [&](int r, int code) {
                int k = index[r * 19683 + code];
                return k < 0 ? 0 : interpolate(packed.table[lo][k], packed.table[lo + 1][k]);
            };
            for (int r = 4; r < 10; ++r) for (int code = 0; code < 19683; ++code)
                result.table[e][r - 4][code] = 2 * local(r, code);
            for (int code = 0; code < 6561; ++code) {
                int d[8], value = code;
                for (int j = 0; j < 8; ++j, value /= 3) d[j] = value % 3;
                int one = 27*d[0]+81*d[1]+243*d[2]+729*d[4]+2187*d[5]+6561*d[6];
                int two = 27*d[1]+81*d[2]+243*d[3]+729*d[5]+2187*d[6]+6561*d[7];
                int corner = 81*d[0]+243*d[1]+2187*d[4]+6561*d[5];
                int edge = 2 * interpolate(packed.edge[lo][0][code], packed.edge[lo + 1][0][code]);
                result.edge[e][0][code] = edge + 2*local(1, one) + 2*local(2, two) + local(0, corner);
                result.edge[e][1][code] = edge + local(3, two);
                int reverse = d[3]+3*d[2]+9*d[1]+27*d[0]+81*d[7]+243*d[6]+729*d[5]+2187*d[4];
                result.edge[e][2][reverse] = result.edge[e][1][code];
                result.edge[e][3][reverse] = result.edge[e][0][code];
                for (int band = 1; band < 3; ++band)
                    result.edge[e][band + 3][code] = 2 * interpolate(packed.edge[lo][band][code], packed.edge[lo + 1][band][code]);
            }
        }
        expand(result.square, packed.square);
        expand(result.linear, packed.linear);
        expand(result.tempo, packed.tempo);
        expand(result.line, packed.line);
        for (int e = 0; e <= 45; ++e) for (int r = 0; r < 4; ++r)
            for (int code = 0; code < 2187; ++code) {
                int value = code, reverse = 0, score = 0;
                for (int x = 0; x < 7; ++x, value /= 3) {
                    int digit = value % 3;
                    reverse = 3 * reverse + digit;
                    score += (result.square[e][7*r+x] + result.linear[e][0]) * ((digit == 1) - (digit == 2));
                }
                if (code <= reverse) result.line[e][r][T().lineOrbit[code]] += score / 2;
            }
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
        expanded.append(np.sign(value) * ((abs(value) + den // 2) // den))
    table, square, linear, tempo, line, edge = expanded
    digits = np.arange(6561)[:, None] // (3 ** np.arange(8)) % 3
    reverse = digits.reshape(-1, 2, 4)[:, :, ::-1].reshape(-1, 8) @ (3 ** np.arange(8))
    one = digits[:, [0, 1, 2, 4, 5, 6]] @ (3 ** np.arange(3, 9))
    two = digits[:, [1, 2, 3, 5, 6, 7]] @ (3 ** np.arange(3, 9))
    corner = digits[:, [0, 1, 4, 5]] @ (3 ** np.array([4, 5, 7, 8]))
    strips = np.empty((46, 6, 6561), dtype=np.int32)
    strips[:, 0] = 2 * edge[:, 0] + 2 * table[:, 1, one] + 2 * table[:, 2, two] + table[:, 0, corner]
    strips[:, 1] = 2 * edge[:, 0] + table[:, 3, two]
    strips[:, 2] = strips[:, 1, reverse]
    strips[:, 3] = strips[:, 0, reverse]
    strips[:, 4:] = 2 * edge[:, 1:]
    expanded = [2 * table[:, 4:], 2 * square, 2 * linear, 2 * tempo, 2 * line, strips]
    codes = np.arange(2187)
    digits = codes[:, None] // (3 ** np.arange(7)) % 3
    digits = digits[codes <= digits @ (3 ** np.arange(6, -1, -1))]
    pieces = (digits == 1).astype(np.int32) - (digits == 2)
    for r in range(4):
        expanded[4][:, r] += (square[:, 7*r:7*r+7] + linear[:, :1]) @ pieces.T
    args.out.with_suffix(".bin").write_bytes(b"".join(a.astype("<i2").tobytes() for a in expanded))
    print(json.dumps(dict(scale=scale, bytes=args.out.stat().st_size)), flush=True)


if __name__ == "__main__":
    main()

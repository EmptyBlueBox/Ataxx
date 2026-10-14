import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ataxx.pure import DX, DY
from ataxx.patterns import encode, indices
from ataxx.train import SHIFTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=1000000)
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    rng = np.random.default_rng(31415)
    parts = []
    for directory in args.data:
        files = sorted(directory.rglob("*.npy"))
        take = max(1, args.rows // len(args.data) // len(files))
        for file in files:
            rows = np.load(file, mmap_mode="r")
            parts.append(rows[rng.choice(len(rows), min(take, len(rows)), replace=False)])
    rows = np.concatenate(parts)
    rng.shuffle(rows)
    device = "cuda"
    raw = np.arange(19683)[:, None] // (3 ** np.arange(9)) % 3
    sym = []
    for t in range(8):
        perm = []
        for cell in range(9):
            x, y = cell % 3, cell // 3
            if t & 1: x = 2 - x
            if t & 2: y = 2 - y
            if t & 4: x, y = y, x
            perm.append(3 * y + x)
        sym.append((raw * 3 ** np.array(perm)).sum(1))
    sym = np.array(sym)
    canon = np.stack([indices()[0], np.minimum(np.arange(19683), sym[2]), np.arange(19683), np.minimum(np.arange(19683), sym[4])])
    canonical = torch.tensor(canon, device=device).long()
    transforms = [(int(x < 0) | (int(y < 0) << 1) | (int(abs(x) < abs(y)) << 2)) for x, y in zip(DX, DY)]
    family = torch.tensor([min(abs(x), abs(y)) + 1 for x, y in zip(DX, DY)], device=device)
    rotated = torch.tensor(sym[transforms], device=device).long()
    origin = np.array([[(s // 7 - y) * 7 + s % 7 - x if 0 <= s % 7 - x < 7 and 0 <= s // 7 - y < 7 else -1 for s in range(49)] for x, y in zip(DX, DY)])
    valid = torch.tensor(origin >= 0, device=device)
    origin = torch.tensor(origin.clip(0), device=device).long()
    encoded, ownbits, oppbits, phase = [], [], [], []
    for part in np.array_split(rows, max(1, len(rows) // 16384)):
        bb = torch.tensor(part["bb"].astype(np.int64), device=device)
        side = torch.tensor(part["side"].astype(np.int64), device=device)
        index, _, _, p = encode(bb, side)
        encoded.append(index.to(torch.int16))
        row = torch.arange(len(part), device=device)
        ownbits.append(bb[row, side])
        oppbits.append(bb[row, 1 - side])
        phase.append(p)
    encoded, ownbits, oppbits, phase = [torch.cat(x) for x in (encoded, ownbits, oppbits, phase)]
    target_idx = torch.tensor(rows["idx"].astype(np.int64), device=device)
    target_cnt = torch.tensor(rows["cnt"].astype(np.float32), device=device)
    target_cnt /= target_cnt.sum(1, keepdim=True)
    xy = torch.arange(49, device=device)
    x, y = torch.minimum(xy % 7, 6 - xy % 7), torch.minimum(xy // 7, 6 - xy // 7)
    _, orbit = torch.unique(torch.minimum(x, y) * 4 + torch.maximum(x, y), return_inverse=True)
    destination = torch.zeros(4, 19683, device=device, requires_grad=True)
    source = torch.zeros(3, 19683, device=device, requires_grad=True)
    square = torch.zeros(4, 10, device=device, requires_grad=True)
    bias = torch.zeros(6, 4, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([destination, source, square, bias], lr=0.02)
    split = len(rows) * 19 // 20

    def loss(i):
        idx = encoded[i].long()
        own = (ownbits[i, None] >> SHIFTS.cuda() & 1).bool()
        opp = (oppbits[i, None] >> SHIFTS.cuda() & 1).bool()
        empty = ~(own | opp)
        clone = F.max_pool2d(own.float().reshape(-1, 1, 7, 7), 3, 1, 1).flatten(1).bool() & empty
        jump = own[:, origin] & valid & empty[:, None]
        clone_logits = destination[0, canonical[0, idx]] + square[0, orbit] + bias[phase[i], 0, None]
        transformed = rotated[torch.arange(16, device=device)[None, :, None], idx[:, None, :]]
        transformed = canonical[family[None, :, None], transformed]
        from_codes = rotated[torch.arange(16, device=device)[None, :, None], idx[:, origin]]
        from_codes = canonical[family[None, :, None], from_codes]
        jump_logits = destination[family[None, :, None], transformed] + source[family[None, :, None] - 1, from_codes]
        jump_logits += square[family[:, None], orbit] + bias[phase[i]][:, family, None]
        logits = torch.cat([clone_logits[:, None], jump_logits], 1).flatten(1)
        mask = torch.cat([clone[:, None], jump], 1).flatten(1)
        logits = logits.masked_fill(~mask, -1e9).log_softmax(1)
        target = target_cnt[i]
        return -(logits.gather(1, target_idx[i]) * target).sum(1).mean()

    for epoch in range(args.epochs):
        total = 0
        for i in torch.randperm(split, device=device).split(2048):
            error = loss(i) + 1e-6 * (destination.square().mean() + source.square().mean())
            optimizer.zero_grad()
            error.backward()
            optimizer.step()
            total += error.item() * len(i)
        with torch.no_grad():
            validation = sum(loss(i).item() * len(i) for i in torch.arange(split, len(rows), device=device).split(2048)) / (len(rows) - split)
        print(json.dumps(dict(epoch=epoch, train=total / split, valid=validation)), flush=True)
    with torch.no_grad():
        empty = (torch.arange(19683, device=device) // 81 % 3 == 0)
        own = (torch.arange(19683, device=device) // 81 % 3 == 1)
        arrays = [(a * 256).round().to(torch.int16).cpu().numpy() for a in (destination[torch.arange(4, device=device)[:, None], canonical][:, empty], source[torch.arange(3, device=device)[:, None], canonical[1:]][:, own], square[:, orbit], bias)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out.with_suffix(".npz"), destination=arrays[0], source=arrays[1], square=arrays[2], bias=arrays[3])
    brace = lambda x: "{" + ",".join(map(brace, x)) + "}" if isinstance(x, list) else str(x)
    text = "#pragma once\n#include <cstdint>\nnamespace ataxx {\nstruct DirectionalWeights {\nstd::int16_t destination[4][6561], source[3][6561], square[4][49], bias[6][4];\n};\ninline constexpr DirectionalWeights directionalWeights = "
    args.out.write_text(text + "{" + ",\n".join(brace(a.tolist()) for a in arrays) + "};\n}\n")


if __name__ == "__main__":
    main()

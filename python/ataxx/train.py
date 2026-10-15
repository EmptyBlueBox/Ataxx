import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ataxx import model
from ataxx.selfplay import NMOVES, ROW, library

SHIFTS = torch.tensor([(s // 7) * 8 + s % 7 for s in range(49)])


def tables(device):
    squares, moves = np.zeros((8, 49), np.int32), np.zeros((8, NMOVES), np.int32)
    library().symmetry(squares.ctypes.data, moves.ctypes.data)
    return torch.tensor(squares, device=device).long(), torch.tensor(moves, device=device).long()


def bits(b):
    return (b[:, None] >> SHIFTS.to(b.device) & 1).float().view(-1, 1, 7, 7)


def features(black, white, side, ply):
    b, w, s = bits(black), bits(white), side.view(-1, 1, 1, 1).float()
    own, opp = b * (1 - s) + w * s, w * (1 - s) + b * s
    empty = 1 - own - opp
    near = lambda x: F.max_pool2d(x, 3, 1, 1)
    far = lambda x: F.max_pool2d(x, 5, 1, 2) * (1 - near(x))
    count = lambda x: F.avg_pool2d(x, 3, 1, 1, count_include_pad=True) * 9 / 8
    scalar = lambda x: x.view(-1, 1, 1, 1).expand(-1, 1, 7, 7)
    planes = [own, opp, empty, torch.ones_like(own), near(own) * empty, near(opp) * empty, far(own) * empty,
              far(opp) * empty, count(opp) * empty, count(own) * empty, scalar(ply.float() / 401),
              scalar(own.sum((1, 2, 3)) / 49), scalar(opp.sum((1, 2, 3)) / 49), scalar(empty.sum((1, 2, 3)) / 49)]
    return torch.cat(planes, 1)


def batch(rows, device, squares, moves, mix=0.0, pure=0):
    t = {k: torch.from_numpy(np.ascontiguousarray(rows[k])).to(device) for k in ("bb", "fin", "idx", "cnt", "ply", "side", "z", "score", "q")}
    bb, fin, side = t["bb"].long(), t["fin"].long(), t["side"].long()
    x = features(bb[:, 0], bb[:, 1], side, t["ply"].long())
    policy = torch.zeros(len(rows), NMOVES, device=device).scatter_add_(1, t["idx"].long(), t["cnt"].float())
    policy = policy / policy.sum(1, keepdim=True)
    s = side.view(-1, 1, 1, 1).float()
    own = ((bits(fin[:, 0]) - bits(fin[:, 1])) * (1 - 2 * s)).flatten(1)
    sym = torch.randint(0, 8, (len(rows),), device=device)
    sq, mv = squares[sym], moves[sym]
    x = torch.zeros_like(x).flatten(2).scatter_(2, sq[:, None, :].expand(-1, model.PLANES, -1), x.flatten(2)).view_as(x)
    policy = torch.zeros_like(policy).scatter_(1, mv, policy)
    own = torch.zeros_like(own).scatter_(1, sq, own)
    z, q = t["z"].float(), t["q"].float() / 32767
    known = ((fin[:, 0] | fin[:, 1]) != 0).float()
    known[:pure] = 0
    w = torch.where(known > 0, mix * (q != 0).float(), known + 1)
    win = (1 - w) * (z > 0).float() + w * (1 + q) / 2
    loss = (1 - w) * (z < 0).float() + w * (1 - q) / 2
    value = torch.stack([win, 1 - win - loss, loss], 1)
    return x, policy, value, t["score"].float() / 49, own, known


class Data:
    def __init__(self, root, minimum, fraction, maximum, prune, since, phase=(0, 50)):
        self.root, self.minimum, self.fraction, self.prune, self.since, self.phase = root, minimum, fraction, prune, since, phase
        self.ring = np.zeros(maximum, ROW)
        self.sizes, self.total, self.base = {}, 0, 0

    def refresh(self):
        for path in sorted((p for p in self.root.rglob("*.npy") if ".tmp" not in p.name and p not in self.sizes and p.name >= self.since), key=lambda p: p.name):
            try:
                rows = np.load(path)
            except FileNotFoundError:
                continue
            if self.phase != (0, 50):
                empty = 49 - np.bitwise_count(rows["bb"][:, 0] | rows["bb"][:, 1])
                rows = rows[(empty >= self.phase[0]) & (empty < self.phase[1])]
            self.sizes[path] = len(rows)
            at = (self.total + np.arange(len(rows))) % len(self.ring)
            self.ring[at] = rows
            self.total += len(rows)
        kept = 0
        for path in sorted(self.sizes, key=lambda p: p.name, reverse=True):
            if kept >= self.prune and path.exists():
                path.unlink()
            kept += self.sizes[path]
        self.window = min(len(self.ring), self.total, max(self.minimum, int(self.fraction * (self.total + self.base))))

    def sample(self, n):
        return self.ring[(self.total - 1 - np.random.randint(0, self.window, n)) % len(self.ring)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--net", type=Path)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ratio-mlp", type=int, default=2)
    parser.add_argument("--relu", action="store_true")
    parser.add_argument("--quant", action="store_true")
    parser.add_argument("--key", type=int, default=0)
    parser.add_argument("--value", type=int, default=0)
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ratio", type=float, default=4)
    parser.add_argument("--start", type=int, default=200000)
    parser.add_argument("--window", type=float, nargs=3, default=[2e6, 0.3, 3e7])
    parser.add_argument("--prune", type=float, default=8e7)
    parser.add_argument("--q-mix", type=float, default=0)
    parser.add_argument("--since", default="")
    parser.add_argument("--extra", nargs=2)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--export", type=int, default=1000)
    parser.add_argument("--keep", type=int, default=10000)
    parser.add_argument("--phase", type=int, nargs=2, default=[0, 50])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    state = args.out / "train.pt"
    if state.exists():
        saved = torch.load(state, map_location=device)
        net = model.Net(**saved["config"]).to(device)
        net.load_state_dict(saved["raw"])
    else:
        saved = None
        if args.net:
            start = torch.load(args.net, map_location=device)
            net = model.Net(**{**start["config"], "quant": args.quant}).to(device)
            net.load_state_dict(start["state"], strict=False)
        else:
            net = model.Net(args.blocks, args.channels, args.heads, args.ratio_mlp, not args.relu, args.quant, args.key, args.value, not args.hard).to(device)
    net = net.to(memory_format=torch.channels_last)
    ema = copy.deepcopy(net).eval()
    decay = [p for n, p in net.named_parameters() if p.ndim > 1]
    plain = [p for n, p in net.named_parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=1e-4), dict(params=plain, weight_decay=0)], lr=args.lr, fused=True)
    step = samples = 0
    if saved:
        ema.load_state_dict(saved["state"])
        opt.load_state_dict(saved["opt"])
        step, samples = saved["step"], saved["samples"]
        for group in opt.param_groups:
            group["lr"] = args.lr
    else:
        model.save(ema, args.out / "latest.pt")
    squares, moves = tables(device)
    data = Data(args.data, int(args.window[0]), args.window[1], int(args.window[2]), int(args.prune), args.since, tuple(args.phase))
    data.refresh()
    extra = args.extra and Data(Path(args.extra[0]), 0, 1.0, int(2e7), int(args.prune), "")
    if args.fresh:
        samples = 0
    elif saved:
        data.base = max(0, int(saved.get("rows", samples / args.ratio)) - data.total)
    log = (args.out / "train.jsonl").open("a")
    began = time.time()
    while True:
        data.refresh()
        if extra:
            extra.refresh()
        if data.total < args.start or samples > args.ratio * (data.total + data.base):
            time.sleep(20)
            continue
        net.train()
        total = torch.zeros(5, device=device)
        for _ in range(args.export):
            rows, pure = data.sample(args.batch), int(float(args.extra[1]) * args.batch) if extra else 0
            rows[:pure] = extra.sample(pure) if pure else rows[:0]
            x, policy, value, score, own, known = batch(rows, device, squares, moves, args.q_mix, pure)
            with torch.autocast("cuda", torch.bfloat16):
                p, v, o = net.train_forward(x.contiguous(memory_format=torch.channels_last))
            p, v, o = p.float(), v.float(), o.float()
            losses = torch.stack([-(policy * F.log_softmax(p, 1)).sum(1).mean(), -(value * F.log_softmax(v[:, :3], 1)).sum(1).mean(),
                                  ((torch.tanh(v[:, 3]) - score) ** 2 * known).mean(), ((torch.tanh(o) - own) ** 2 * known[:, None]).mean(),
                                  (p.argmax(1) == policy.argmax(1)).float().mean()])
            loss = losses[0] + 1.5 * losses[1] + 0.5 * losses[2] + 0.5 * losses[3]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            step += 1
            samples += args.batch
            total = total + losses.detach()
            if step % 4 == 0:
                with torch.no_grad():
                    for a, b in zip(ema.parameters(), net.parameters()):
                        a.lerp_(b, 0.004)
        with torch.no_grad():
            for a, b in zip(ema.buffers(), net.buffers()):
                a.copy_(b)
        sums = total.cpu().numpy()
        record = dict(step=step, samples=samples, rows=data.total + data.base, window=data.window, time=time.time() - began,
                      **dict(zip(("policy", "value", "score", "own", "top1"), (sums.astype(float) / args.export).round(4).tolist())))
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(record, flush=True)
        model.save(ema, args.out / "new.pt")
        os.replace(args.out / "new.pt", args.out / "latest.pt")
        if step % args.keep == 0:
            model.save(ema, args.out / f"{step:08d}.pt")
        torch.save(dict(config=net.config, raw=net.state_dict(), state=ema.state_dict(), opt=opt.state_dict(), step=step, samples=samples, rows=data.total + data.base), state)


if __name__ == "__main__":
    main()

import argparse
import os
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ataxx import kata, model, probe
from ataxx.selfplay import NMOVES
from ataxx.train import SHIFTS

DX = [-2, -1, 0, 1, 2, -2, 2, -2, 2, -2, 2, -2, -1, 0, 1, 2]
DY = [-2, -2, -2, -2, -2, -1, -1, 0, 0, 1, 1, 2, 2, 2, 2, 2]


class Teacher:
    def __init__(self, path, device="cuda", floor=0.01, batch=4096, gpus=(0,), student=None):
        self.device, self.floor, self.batch, self.student = device, floor, batch, student
        xy = torch.arange(49)
        dx, dy = xy[None] % 7 - xy[:, None] % 7, xy[None] // 7 - xy[:, None] // 7
        d = torch.maximum(dx.abs(), dy.abs())
        self.near, self.far = (d == 1).float().to(device), (d == 2).to(device)
        kind = torch.zeros(5, 5, dtype=torch.long)
        for k in range(16):
            kind[DX[k] + 2, DY[k] + 2] = k + 1
        self.move = (kind[(dx + 2).clamp(0, 4), (dy + 2).clamp(0, 4)] * 49 + xy[None]).to(device)
        self.models = [(torch.compile(kata.load(path).half().to(f"cuda:{g}")), f"cuda:{g}") for g in gpus]
        self.pool = ThreadPoolExecutor(len(gpus))

    def forward(self, k, x, g):
        model, dev = self.models[k % len(self.models)]
        n = len(x)
        x, g = F.pad(x, (0, 0, 0, 0, 0, 0, 0, -n % 1024)), F.pad(g, (0, 0, 0, -n % 1024))
        with torch.no_grad():
            policy, value = model(x.to(dev).half(), g.to(dev).half())[:2]
        return policy[:n, 0, :49].float().to(self.device), value[:n].float().to(self.device)

    def run(self, own, opp, chosen=None):
        n = len(own)
        x = torch.zeros(n, 22, 49, device=self.device)
        g = torch.zeros(n, 19, device=self.device)
        x[:, 0], x[:, 1], x[:, 2] = 1, own, opp
        g[:, 3] = g[:, 9] = 1
        if chosen is not None:
            x[torch.arange(n, device=self.device), 4, chosen] = 1
            g[:, 0] = 1
        x = x.view(-1, 22, 7, 7)
        step = min(self.batch, -(-n // len(self.models)))
        parts = [(k, x[i:i + step], g[i:i + step]) for k, i in enumerate(range(0, n, step))]
        out = [self.forward(*a) for a in parts] if len(self.models) == 1 else list(self.pool.map(lambda a: self.forward(*a), parts))
        return torch.cat([p for p, v in out]), torch.cat([v for p, v in out])

    def __call__(self, bb, side):
        bits = (bb[:, :, None] >> SHIFTS.to(self.device) & 1).bool()
        s = side.view(-1, 1).bool()
        return self.evaluate(torch.where(s, bits[:, 1], bits[:, 0]), torch.where(s, bits[:, 0], bits[:, 1]))

    def net(self, x):
        jump = None
        if self.student:
            with torch.autocast("cuda", torch.float16), torch.no_grad():
                jump = self.student(x.to(self.device))[0].float().softmax(1)[:, self.move]
        policy, q = self.evaluate(x[:, 0].flatten(1) > 0.5, x[:, 1].flatten(1) > 0.5, jump)
        zero = torch.zeros_like(q)
        return (policy + 1e-30).log(), torch.stack([(1 + q).clamp_min(1e-6).log(), zero - 30, (1 - q).clamp_min(1e-6).log(), zero], 1)

    def evaluate(self, own, opp, jump=None):
        own, opp = own.to(self.device), opp.to(self.device)
        empty = ~(own | opp)
        clone = empty & (own.float() @ self.near > 0)
        origin = own & (empty.float() @ self.far.float() > 0)
        logits, value = self.run(own.float(), opp.float())
        first = logits.masked_fill(~(clone | origin), -1e9).softmax(1)
        joint = torch.zeros(len(own), NMOVES, device=self.device)
        joint[:, :49] = first * clone
        if jump is not None:
            second = jump * (empty[:, None] & self.far) + 1e-9
            joint.scatter_add_(1, self.move.expand(len(own), -1, -1).flatten(1), (first[:, :, None] * origin[:, :, None] * second / second.sum(2, keepdim=True)).flatten(1))
        else:
            row, sq = ((first > self.floor) & origin).nonzero(as_tuple=True)
            if len(row):
                reach = empty[row] & self.far[sq]
                second = self.run(own[row].float(), opp[row].float(), sq)[0].masked_fill(~reach, -1e9).softmax(1) * reach
                joint.index_put_((row[:, None].expand(-1, 49), self.move[sq]), first[row, sq, None] * second, accumulate=True)
        v = value.softmax(1)
        return joint / joint.sum(1, keepdim=True), v[:, 0] - v[:, 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--shard", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--mix", type=float, default=1)
    parser.add_argument("--q-mix", type=float, default=1)
    parser.add_argument("--evals", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=16384)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--student", type=Path)
    args = parser.parse_args()
    teacher = Teacher(args.teacher, device="cuda:0", gpus=range(args.gpus), student=args.student and model.load(args.student, "cuda:0"))
    while True:
        for path in sorted(p for p in args.src.rglob("*.npy") if ".tmp" not in p.name and zlib.crc32(p.name.encode()) % args.shard[1] == args.shard[0]):
            rows = np.load(path)
            for i in range(0, len(rows), args.chunk):
                part = rows[i:i + args.chunk]
                bb = torch.from_numpy(part["bb"].astype(np.int64)).cuda()
                policy, q = teacher(bb, torch.from_numpy(part["side"].astype(np.int64)).cuda())
                if args.evals:
                    search, deep = (torch.from_numpy(t).cuda() for t in probe.search(teacher.net, part["bb"], part["side"], part["ply"], args.evals))
                else:
                    search = torch.zeros_like(policy).scatter_add_(1, torch.from_numpy(part["idx"].astype(np.int64)).cuda(), torch.from_numpy(part["cnt"].astype(np.float32)).cuda())
                    deep = torch.from_numpy(part["q"].astype(np.float32)).cuda() / 32767
                policy = args.mix * policy + (1 - args.mix) * search / search.sum(1, keepdim=True)
                top = policy.topk(32, 1)
                cnt = (top.values * 65535).round()
                part["idx"], part["cnt"], part["n"] = top.indices.cpu().numpy(), cnt.cpu().numpy(), (cnt > 0).sum(1).cpu().numpy()
                q = args.q_mix * q + (1 - args.q_mix) * deep
                q = (q * 32767).round().clamp(-32767, 32767)
                part["q"] = torch.where(q == 0, 1, q).cpu().numpy()
            out = args.dst / path.relative_to(args.src)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out.with_suffix(".tmp.npy"), rows)
            os.rename(out.with_suffix(".tmp.npy"), out)
            path.unlink()
            print(out, len(rows), flush=True)
        time.sleep(30)


if __name__ == "__main__":
    main()

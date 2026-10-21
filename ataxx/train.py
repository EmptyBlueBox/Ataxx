import argparse
import copy
import json
import multiprocessing
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ataxx import kata, model, selfplay
from ataxx.model import NMOVES

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "run/train"


def tables(device):
    squares, moves = np.zeros((8, 49), np.int32), np.zeros((8, NMOVES), np.int32)
    selfplay.library().symmetry(squares.ctypes.data, moves.ctypes.data)
    return torch.tensor(squares, device=device).long(), torch.tensor(moves, device=device).long()


def bits(b):
    return (b[:, None] >> model.SHIFTS.to(b.device) & 1).float().view(-1, 1, 7, 7)


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


def batch(rows, device, squares, moves, mix):
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
    w = torch.where(known > 0, mix * (q != 0).float(), known + 1)
    win = (1 - w) * (z > 0).float() + w * (1 + q) / 2
    loss = (1 - w) * (z < 0).float() + w * (1 - q) / 2
    value = torch.stack([win, 1 - win - loss, loss], 1)
    return x, policy, value, t["score"].float() / 49, own, known


class Data:
    def __init__(self, root, minimum, fraction, maximum, phase):
        self.root, self.minimum, self.fraction, self.phase = root, minimum, fraction, phase
        self.ring = np.zeros(maximum, selfplay.ROW)
        self.seen, self.total = set(), 0

    def refresh(self):
        for path in sorted((p for p in self.root.glob("*.npy") if ".tmp" not in p.name and p not in self.seen), key=lambda p: p.name):
            rows = np.load(path)
            empty = 49 - np.bitwise_count(rows["bb"][:, 0] | rows["bb"][:, 1])
            rows = rows[(empty >= self.phase[0]) & (empty < self.phase[1])]
            self.seen.add(path)
            self.ring[(self.total + np.arange(len(rows))) % len(self.ring)] = rows
            self.total += len(rows)
        self.window = min(len(self.ring), self.total, max(self.minimum, int(self.fraction * self.total)))

    def sample(self, n):
        return self.ring[(self.total - 1 - np.random.randint(0, self.window, n)) % len(self.ring)]


def fit(out, gpu, steps, lr, window, quant=True, init=None, phase=(0, 50), ratio=float("inf"), start=0, q_mix=0.75, size=2048, block=1000):
    torch.cuda.set_device(gpu)
    device = "cuda"
    out.mkdir(parents=True, exist_ok=True)
    state = out / "train.pt"
    net = model.Net(quant).to(device)
    if state.exists():
        saved = torch.load(state, map_location=device)
        net.load_state_dict(saved["raw"])
    elif init:
        net.load_state_dict(torch.load(init, map_location=device)["state"], strict=False)
    net = net.to(memory_format=torch.channels_last)
    ema = copy.deepcopy(net).eval()
    decay = [p for p in net.parameters() if p.ndim > 1]
    plain = [p for p in net.parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=1e-4), dict(params=plain, weight_decay=0)], lr=lr, fused=True)
    step = samples = 0
    if state.exists():
        ema.load_state_dict(saved["state"])
        opt.load_state_dict(saved["opt"])
        step, samples = saved["step"], saved["samples"]
    squares, moves = tables(device)
    data = Data(RUN / "kd", int(window[0]), window[1], int(window[2]), phase)
    log = (out / "train.jsonl").open("a")
    began = time.time()
    while step < steps:
        data.refresh()
        if data.total < start or samples > ratio * data.total:
            time.sleep(20)
            continue
        net.train()
        total = torch.zeros(5, device=device)
        for _ in range(block):
            x, policy, value, score, own, known = batch(data.sample(size), device, squares, moves, q_mix)
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
            samples += size
            total = total + losses.detach()
            if step % 4 == 0:
                with torch.no_grad():
                    for a, b in zip(ema.parameters(), net.parameters()):
                        a.lerp_(b, 0.004)
        with torch.no_grad():
            for a, b in zip(ema.buffers(), net.buffers()):
                a.copy_(b)
        record = dict(step=step, rows=data.total, time=round(time.time() - began),
                      **dict(zip(("policy", "value", "score", "own", "top1"), (total.cpu().numpy().astype(float) / block).round(4).tolist())))
        log.write(json.dumps(record) + "\n")
        log.flush()
        model.save(ema, out / "new.pt")
        os.replace(out / "new.pt", out / "latest.pt")
        torch.save(dict(raw=net.state_dict(), state=ema.state_dict(), opt=opt.state_dict(), step=step, samples=samples), state)


def launch(jobs):
    context = multiprocessing.get_context("spawn")
    procs = [context.Process(target=target, kwargs=kwargs) for target, kwargs in jobs]
    for p in procs:
        p.start()
    return procs


def wait(procs, helpers=()):
    try:
        while any(p.is_alive() for p in procs):
            assert all(h.is_alive() for h in helpers)
            time.sleep(10)
        assert all(p.exitcode == 0 for p in procs)
    finally:
        for p in [*procs, *helpers]:
            p.terminate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    args = parser.parse_args()
    subprocess.run(["make", "-C", str(ROOT)], check=True)
    gpus = args.gpus
    (RUN / "base").mkdir(parents=True, exist_ok=True)
    if not (RUN / "base/latest.pt").exists():
        model.save(model.Net(), RUN / "base/latest.pt")
    helpers = launch([(selfplay.run, dict(path=RUN / "base/latest.pt", out=RUN / "raw", gpu=g)) for g in range(gpus)]
                     + [(kata.label, dict(teacher=args.teacher, src=RUN / "raw", dst=RUN / "kd", gpus=gpus))])
    wait(launch([(fit, dict(out=RUN / "base", gpu=0, steps=420000, lr=1e-3, window=(2e6, 1.0, 1.2e8), quant=False, ratio=4, start=200000))]), helpers)
    wait(launch([(fit, dict(out=RUN / "qat", gpu=0, steps=100000, lr=1e-4, window=(2e6, 1.0, 3e7), init=RUN / "base/latest.pt"))]))
    for bands in [[("qat", 0, 15, 100000), ("qat", 15, 30, 800000), ("qat", 30, 50, 800000)], [("p0-15", 0, 8, 800000), ("p0-15", 8, 15, 800000)]]:
        wait(launch([(fit, dict(out=RUN / f"p{lo}-{hi}", gpu=i % gpus, steps=steps, lr=1e-3, window=(2e6, 1.0, 6e7), init=RUN / f"{parent}/latest.pt", phase=(lo, hi)))
                     for i, (parent, lo, hi, steps) in enumerate(bands)]))
    model.export(model.load(RUN / "qat/latest.pt"), ROOT / "weights/champion.bin")
    for lo, hi in [(0, 8), (8, 15), (15, 30), (30, 50)]:
        model.export(model.load(RUN / f"p{lo}-{hi}/latest.pt"), ROOT / f"weights/p{lo}-{hi}.bin")


if __name__ == "__main__":
    main()

import argparse
import ctypes
import os
import time
from pathlib import Path

import numpy as np
import torch

from ataxx import model

ROW = np.dtype([("bb", "<u8", 2), ("fin", "<u8", 2), ("idx", "<u2", 32), ("cnt", "<u2", 32), ("ply", "<u2"),
                ("side", "u1"), ("n", "u1"), ("z", "i1"), ("score", "i1"), ("q", "<i2")])
NMOVES = 833


def library():
    lib = ctypes.CDLL(str(Path(__file__).resolve().parent.parent / "build/libataxx.so"))
    lib.create.argtypes = [ctypes.c_int] * 3 + [ctypes.c_float, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_float, ctypes.c_ulong, ctypes.c_int]
    lib.collect.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.feed.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    lib.drain.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.stats.argtypes = [ctypes.c_void_p]
    lib.features.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.status.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    lib.adjudicate.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    lib.symmetry.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    return lib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games", type=int, default=2048)
    parser.add_argument("--full", type=int, default=400)
    parser.add_argument("--fast", type=int, default=64)
    parser.add_argument("--p-full", type=float, default=0.25)
    parser.add_argument("--temp-plies", type=int, default=20)
    parser.add_argument("--late-temp", type=float, default=0.25)
    parser.add_argument("--gumbel", type=int, default=0)
    parser.add_argument("--p-opening", type=float, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--shard", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=int(time.time() * 1000) % 2**31)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lib = library()
    lib.create(args.games, args.full, args.fast, args.p_full, args.temp_plies, args.late_temp, args.gumbel, args.p_opening, args.seed, args.threads)
    half = args.games // 2
    banks = [(0, half), (half, args.games)]
    feats = [torch.zeros(half, model.PLANES, 7, 7).pin_memory() for _ in banks]
    net, loaded = None, 0
    pending = [None, None]
    rows, buf, stats = [], np.zeros(1 << 16, ROW), np.zeros(4, np.int64)
    started, steps, last = time.time(), 0, time.time()

    def submit(b):
        lib.collect(*banks[b], feats[b].numpy().ctypes.data)
        with torch.autocast("cuda", torch.float16), torch.no_grad():
            policy, value = net(feats[b].cuda(non_blocking=True).contiguous(memory_format=torch.channels_last))
        pending[b] = (policy.float(), value.float())

    def fetch(b):
        policy, value = (np.ascontiguousarray(t.cpu().numpy()) for t in pending[b])
        lib.feed(*banks[b], policy.ctypes.data, value.ctypes.data)

    while True:
        mtime = args.net.stat().st_mtime
        if mtime != loaded:
            net, loaded = model.load(args.net, "cuda").to(memory_format=torch.channels_last), mtime
        if pending[0] is None:
            submit(0)
            submit(1)
        for _ in range(256):
            for b in (0, 1):
                fetch(b)
                submit(b)
        steps += 512
        while (n := lib.drain(buf.ctypes.data, len(buf))):
            rows.append(buf[:n].copy())
        count = sum(map(len, rows))
        if count >= args.shard:
            name = f"{int(time.time() * 1000)}-{os.getpid()}"
            np.save(args.out / f"{name}.tmp.npy", np.concatenate(rows))
            os.rename(args.out / f"{name}.tmp.npy", args.out / f"{name}.npy")
            rows = []
        if time.time() - last > 60:
            last = time.time()
            lib.stats(stats.ctypes.data)
            print(f"t={last - started:.0f}s evals/s={steps * half / (last - started):.0f} games={stats[0]} "
                  f"plies/game={stats[1] / max(1, stats[0]):.1f} draws={stats[2]} black={stats[3]} buffered={count}", flush=True)


if __name__ == "__main__":
    main()

import ctypes
import os
import time
from pathlib import Path

import numpy as np
import torch

from ataxx import model

ROW = np.dtype([("bb", "<u8", 2), ("fin", "<u8", 2), ("idx", "<u2", 32), ("cnt", "<u2", 32), ("ply", "<u2"),
                ("side", "u1"), ("n", "u1"), ("z", "i1"), ("score", "i1"), ("q", "<i2")])


def library():
    lib = ctypes.CDLL(str(Path(__file__).resolve().parent.parent / "build/libataxx.so"))
    lib.create.argtypes = [ctypes.c_int] * 3 + [ctypes.c_float, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_ulong, ctypes.c_int]
    lib.collect.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.feed.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    lib.drain.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.symmetry.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    return lib


def run(path, out, gpu, games=2048, full=400, fast=64, p_full=0.25, temp_plies=20, late_temp=0.25, p_opening=0.15, threads=8, shard=50000):
    torch.cuda.set_device(gpu)
    out.mkdir(parents=True, exist_ok=True)
    lib = library()
    lib.create(games, full, fast, p_full, temp_plies, late_temp, p_opening, (int(time.time() * 1000) + gpu) % 2**31, threads)
    half = games // 2
    banks = [(0, half), (half, games)]
    feats = [torch.zeros(half, model.PLANES, 7, 7).pin_memory() for _ in banks]
    pending = [None, None]
    rows, buf, loaded = [], np.zeros(1 << 16, ROW), None

    def submit(b):
        lib.collect(*banks[b], feats[b].numpy().ctypes.data)
        with torch.autocast("cuda", torch.float16), torch.no_grad():
            policy, value = net(feats[b].cuda(non_blocking=True).contiguous(memory_format=torch.channels_last))
        pending[b] = policy.float(), value.float()

    def fetch(b):
        policy, value = (np.ascontiguousarray(t.cpu().numpy()) for t in pending[b])
        lib.feed(*banks[b], policy.ctypes.data, value.ctypes.data)

    while True:
        if path.stat().st_mtime_ns != loaded:
            loaded = path.stat().st_mtime_ns
            net = model.load(path, "cuda").to(memory_format=torch.channels_last)
        if pending[0] is None:
            submit(0)
            submit(1)
        for _ in range(256):
            for b in (0, 1):
                fetch(b)
                submit(b)
        while (n := lib.drain(buf.ctypes.data, len(buf))):
            rows.append(buf[:n].copy())
        if sum(map(len, rows)) >= shard:
            name = f"{int(time.time() * 1000)}-{os.getpid()}"
            np.save(out / f"{name}.tmp.npy", np.concatenate(rows))
            os.rename(out / f"{name}.tmp.npy", out / f"{name}.npy")
            rows = []

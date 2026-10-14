import ctypes

import numpy as np
import torch

from ataxx import model
from ataxx.selfplay import NMOVES, library


def run(lib, net, count, batch):
    lib.probe_collect.argtypes = [ctypes.c_void_p]
    lib.probe_feed.argtypes = [ctypes.c_void_p] * 2
    feat = torch.zeros(count * batch, model.PLANES, 7, 7)
    while (n := lib.probe_collect(feat.numpy().ctypes.data)):
        with torch.autocast("cuda", torch.float16), torch.no_grad():
            policy, value = net(feat[:n].cuda())
        policy, value = (np.ascontiguousarray(t.float().cpu().numpy()) for t in (policy, value))
        lib.probe_feed(policy.ctypes.data, value.ctypes.data)


def search(net, bb, side, ply, evals, batch=8, threads=16):
    lib = library()
    lib.probe_boards.argtypes = [ctypes.c_int, ctypes.c_long, ctypes.c_int] + [ctypes.c_void_p] * 3 + [ctypes.c_int]
    lib.probe_root.argtypes = [ctypes.c_int, ctypes.c_void_p]
    lib.probe_root.restype = ctypes.c_float
    bb, side, ply = np.ascontiguousarray(bb, np.uint64), np.ascontiguousarray(side, np.uint8), np.ascontiguousarray(ply, np.uint16)
    lib.probe_boards(len(bb), evals, batch, bb.ctypes.data, side.ctypes.data, ply.ctypes.data, threads)
    run(lib, net, len(bb), batch)
    visits, q = np.zeros((len(bb), NMOVES), np.float32), np.zeros(len(bb), np.float32)
    for i in range(len(bb)):
        q[i] = lib.probe_root(i, visits[i].ctypes.data)
    return visits, q


def probe(net, lines, evals, batch=64, threads=8):
    lib = library()
    lib.probe_create.argtypes = [ctypes.c_int, ctypes.c_long, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.probe_result.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 4
    moves = np.array([m for line in lines for m in line], np.int32).reshape(-1, 4)
    lengths = np.array([len(line) for line in lines], np.int32)
    lib.probe_create(len(lines), evals, batch, moves.ctypes.data, lengths.ctypes.data, threads)
    run(lib, net, len(lines), batch)
    out = []
    mv, visits, q, prior = np.zeros((NMOVES, 4), np.int32), np.zeros(NMOVES, np.float32), np.zeros(NMOVES, np.float32), np.zeros(NMOVES, np.float32)
    for i in range(len(lines)):
        k = lib.probe_result(i, mv.ctypes.data, visits.ctypes.data, q.ctypes.data, prior.ctypes.data)
        out.append([(tuple(map(int, mv[j])), float(visits[j]), float(q[j]), float(prior[j])) for j in np.argsort(-visits[:k])])
    return out

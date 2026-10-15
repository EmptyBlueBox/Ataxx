import argparse
import ctypes
import json

import numpy as np
import torch

from ataxx import model
from ataxx.distill import Teacher
from ataxx.match import ROOT, summary
from ataxx.selfplay import NMOVES, library


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--evals", type=int, nargs=2, default=[600, 600])
    parser.add_argument("--pairs", type=int, default=512)
    parser.add_argument("--openings", default=str(ROOT / "openings.json"))
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--params-a", type=float, nargs=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--late-a")
    parser.add_argument("--elo", type=float, nargs=2, default=[0, 15])
    args = parser.parse_args()
    lib = library()
    lib.arena_create.argtypes = [ctypes.c_int, ctypes.c_long, ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.arena_collect.argtypes = [ctypes.c_void_p] * 3
    lib.arena_feed.argtypes = [ctypes.c_void_p] * 4
    lib.arena_scores.argtypes = [ctypes.c_void_p]
    n = 2 * args.pairs
    nets = [Teacher(path).net if path.endswith(".onnx") else model.load(path, "cuda").to(memory_format=torch.channels_last) for path in (args.a, args.b)]
    late = args.late_a and model.load(args.late_a, "cuda").to(memory_format=torch.channels_last)
    feats = [torch.zeros(n, model.PLANES, 7, 7).pin_memory() for _ in nets]
    counts, scores = np.zeros(2, np.int32), np.zeros(n, np.float32)
    book = json.load(open(args.openings))
    chosen = [book[(args.offset + k) % len(book)] for k in range(args.pairs)]
    moves = np.array([m for o in chosen for m in o], np.int32)
    lengths = np.array([len(o) for o in chosen], np.int32)
    lib.arena_create(args.pairs, *args.evals, moves.ctypes.data, lengths.ctypes.data, args.threads)
    if args.params_a:
        lib.arena_params.argtypes = [ctypes.c_int] + [ctypes.c_float] * 4
        lib.arena_params(0, *args.params_a)
    while lib.arena_collect(feats[0].numpy().ctypes.data, feats[1].numpy().ctypes.data, counts.ctypes.data):
        outputs = []
        for k, (net, feat, count) in enumerate(zip(nets, feats, counts)):
            if not count:
                outputs += [np.zeros((1, NMOVES), np.float32), np.zeros((1, 4), np.float32)]
                continue
            with torch.autocast("cuda", torch.float16), torch.no_grad():
                x = feat[:count].cuda().contiguous(memory_format=torch.channels_last)
                policy, value = net(x)
                if k == 0 and late:
                    m = x[:, 2].sum((1, 2)) < 14.5
                    if m.any():
                        policy[m], value[m] = late(x[m])
            outputs += [np.ascontiguousarray(policy.float().cpu().numpy()), np.ascontiguousarray(value.float().cpu().numpy())]
        lib.arena_feed(*(o.ctypes.data for o in outputs))
    lib.arena_scores(scores.ctypes.data)
    pairs = [(dict(score=float(scores[2 * k])), dict(score=float(scores[2 * k + 1]))) for k in range(args.pairs)]
    print(json.dumps(dict(a=args.a, b=args.b, evals=args.evals, params=args.params_a, **summary(pairs, *args.elo))))


if __name__ == "__main__":
    main()

import argparse
import json
import math
import random
import subprocess
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ataxx import judge
from ataxx.selfplay import library

ROOT = Path(__file__).resolve().parent.parent
STATES = {1: "black", 2: "white", 3: "draw"}
players = {}


class Engine:
    def __init__(self, binary, net, budget, seed, explore=0, share=1, symmetric=0):
        self.unit, self.budget = ("time", float(budget[:-1])) if budget.endswith("s") else ("evals", int(budget))
        self.proc = subprocess.Popen([binary, net], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.send(f"explore {explore} {share} {seed} {symmetric}")

    def close(self):
        self.send("quit")
        self.proc.wait()

    def send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def reset(self):
        self.send("new")
        self.first = True

    def play(self, move):
        self.send("play %d %d %d %d" % move)

    def genmove(self, legal):
        self.send(f"go {self.unit} {(3 if self.first else 1) * self.budget}")
        self.first = False
        return tuple(map(int, self.proc.stdout.readline().split()[:4]))


class Bot:
    def __init__(self, path, scale=1):
        self.args, self.proc = [path, str(scale)], None

    def close(self):
        if self.proc:
            self.proc.kill()
            self.proc.wait()
        self.proc = None

    def reset(self):
        self.close()
        self.moves = []

    def play(self, move):
        self.moves.append(move)

    def genmove(self, legal):
        if self.proc:
            self.proc.stdin.write("%d %d %d %d\n" % self.moves[-1])
        else:
            self.proc = subprocess.Popen(self.args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            lines = ["-1 -1 -1 -1"] * (len(self.moves) % 2 == 0) + ["%d %d %d %d" % m for m in self.moves]
            self.proc.stdin.write("\n".join([str((len(lines) + 1) // 2), *lines]) + "\n")
        self.proc.stdin.flush()
        move = tuple(map(int, self.proc.stdout.readline().split()))
        self.info = self.proc.stdout.readline().strip()
        for line in iter(self.proc.stdout.readline, ">>>BOTZONE_REQUEST_KEEP_RUNNING<<<\n"):
            if not line:
                self.close()
                break
        if max(abs(move[0] - move[2]), abs(move[1] - move[3])) == 1:
            move = next(m for m in legal if m[2:] == move[2:] and max(abs(m[0] - m[2]), abs(m[1] - m[3])) == 1)
        self.moves.append(move)
        return move


class Uniform:
    def close(self):
        pass

    def reset(self):
        pass

    def play(self, move):
        pass

    def genmove(self, legal):
        return random.choice(legal)


def make(spec, seed):
    kind, *rest = spec.split(":")
    if kind.startswith("engine"):
        return Engine(kind[7:] or ROOT / "build/engine", rest[0], rest[1], seed, *rest[2:])
    return {"uniform": Uniform, "bot": Bot}[kind](*rest)


def position(lib, moves):
    flat = np.array(moves, np.int32).reshape(-1)
    legal, count = np.zeros(4 * 833, np.int32), np.zeros(1, np.int32)
    state = lib.status(flat.ctypes.data, len(moves), legal.ctypes.data, count.ctypes.data)
    return state, [tuple(map(int, m)) for m in legal[:4 * count[0]].reshape(-1, 4)]


def pair(task):
    index, specs, book, sides = task
    lib = library()
    opening = [tuple(m) for m in book[index % len(book)]]
    results = []
    for a_side in sides:
        for i, spec in enumerate(specs):
            if (i, spec) in players and (a_side == 0 or spec.count(":") > 2):
                players.pop((i, spec)).close()
            if (i, spec) not in players:
                players[i, spec] = make(spec, 1000 + 2 * index + a_side)
        both = [players[0, specs[0]], players[1, specs[1]]]
        moves = list(opening)
        for player in both:
            player.reset()
            for move in moves:
                player.play(move)
        failed = None
        while True:
            state, legal = position(lib, moves)
            if state:
                break
            side = len(moves) % 2
            move = both[side ^ a_side].genmove(legal)
            if move not in legal:
                state, failed = 2 - side, (side ^ a_side, move)
                for key in [k for k in players if k[0] == side ^ a_side]:
                    del players[key]
                break
            both[side ^ a_side ^ 1].play(move)
            moves.append(move)
        verdict = judge.status(moves)
        assert failed or verdict == STATES[state] or (isinstance(verdict, tuple) and not legal), (verdict, state, moves)
        score = 0.5 if state == 3 else float(state - 1 == a_side)
        results.append(dict(opening=index, a_side=a_side, score=score, plies=len(moves), failed=failed, moves=moves))
    return results


def summary(pairs, elo0, elo1):
    x = np.array([sum(g["score"] for g in p) / len(p) for p in pairs])
    n, mean, var = len(x), x.mean(), max(x.var(), 1e-9)
    expected = lambda e: 1 / (1 + 10 ** (-e / 400))
    s0, s1 = expected(elo0), expected(elo1)
    llr = n / (2 * var) * (s1 - s0) * (2 * mean - s0 - s1)
    elo = lambda s: -400 * math.log10(1 / min(max(s, 1e-6), 1 - 1e-6) - 1)
    half = 1.96 * math.sqrt(var / n)
    return dict(games=sum(len(p) for p in pairs), score=round(float(mean), 4), elo=round(elo(mean), 1), low=round(elo(mean - half), 1),
                high=round(elo(mean + half), 1), llr=round(float(llr), 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--openings", default=str(ROOT / "openings.json"))
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sides", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--elo", type=float, nargs=2, default=[0, 15])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    book = [[]] if args.openings == "start" else json.load(open(args.openings))
    tasks = [(args.offset + i, (args.a, args.b), book, args.sides) for i in range(args.pairs)]
    done = []
    with Pool(args.workers) as pool:
        for results in pool.imap_unordered(pair, tasks):
            done.append(results)
            if args.out:
                with args.out.open("a") as f:
                    for r in results:
                        f.write(json.dumps(dict(a=args.a, b=args.b, **r)) + "\n")
            if len(done) % 10 == 0 or len(done) == len(tasks):
                print(json.dumps(summary(done, *args.elo)), flush=True)


if __name__ == "__main__":
    main()

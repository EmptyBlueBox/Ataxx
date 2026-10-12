#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <vector>
#include <omp.h>
#include "board.h"
#include "judge_ref.h"

using namespace ataxx;
using Seq = std::vector<std::array<int, 4>>;

static std::atomic<long> checks{0}, failures{0};

static bool same(const Verdict& a, const Verdict& b) {
  return a.finish == b.finish && (!a.finish || (a.winner == b.winner && a.err == b.err && a.errorturn == b.errorturn));
}

static void check(const Seq& mv, const Verdict& expect, const char* what) {
  Verdict j = JudgeRef().run(mv);
  checks++;
  if (!same(j, expect)) {
    failures++;
    printf("MISMATCH %s plies=%zu judge(f%d w%d e%d t%d) expect(f%d w%d e%d t%d)\n", what, mv.size(), j.finish, j.winner, j.err, j.errorturn, expect.finish, expect.winner, expect.err, expect.errorturn);
    for (auto& m : mv) printf("%d%d%d%d ", m[0], m[1], m[2], m[3]);
    printf("\n");
  }
}

enum Policy { RANDOM, SHUFFLE, BLOCKADE, PROLONG };

static bool captures(const Pos& p, Move m) { return T().ring1[m % SQ] & p.bb[p.side ^ 1]; }

static Move choose(const Pos& p, Move* mv, int n, Policy pol, std::mt19937_64& rng, const std::vector<Move>& mine) {
  std::vector<Move> pref;
  if (pol == SHUFFLE) {
    for (int i = 0; i < n; i++)
      if (mv[i] >= SQ && !captures(p, mv[i]))
        for (Move last : mine)
          if (last >= SQ && T().jumpOrigin[last % SQ][last / SQ - 1] == bitOfSq(mv[i] % SQ)) pref.push_back(mv[i]);
    if (!pref.empty() && rng() % 10 < 8) return pref[rng() % pref.size()];
    for (int i = 0; i < n; i++) if (mv[i] >= SQ) pref.push_back(mv[i]);
    if (!pref.empty() && rng() % 10 < 7) return pref[rng() % pref.size()];
  }
  if (pol == BLOCKADE) {
    for (int i = 0; i < n; i++) if (mv[i] < SQ) pref.push_back(mv[i]);
    if (!pref.empty()) return pref[rng() % pref.size()];
  }
  if (pol == PROLONG) {
    for (int i = 0; i < n; i++) if (mv[i] >= SQ && !captures(p, mv[i])) pref.push_back(mv[i]);
    if (!pref.empty()) return pref[rng() % pref.size()];
  }
  return mv[rng() % n];
}

struct Game {
  Seq seq;
  Verdict expect;
  Seq probe;
  Verdict probeExpect;
};

static Game generate(Policy pol, std::mt19937_64& rng, bool circularProbe) {
  Game g;
  Pos p = start();
  History h;
  h.reset(p);
  std::vector<Move> mine[2];
  Move mv[NMOVES];
  while (true) {
    int ng = genGeometric(p, mv), n = filterLegal(p, h, mv, ng);
    if (n < ng && circularProbe && g.probe.empty() && rng() % 3 == 0) {
      Move all[NMOVES];
      int na = genGeometric(p, all);
      for (int i = 0; i < na; i++) {
        int t = circularTurn(p, h, all[i]);
        if (t) {
          g.probe = g.seq;
          Coord c = toCoord(p, all[i]);
          g.probe.push_back({c.x0, c.y0, c.x1, c.y1});
          g.probeExpect = {true, p.side ^ 1, 2, t};
          break;
        }
      }
    }
    if (n == 0) {
      g.expect = {true, p.side ^ 1, 2, 0};
      Coord c = toCoord(p, mv[0]);
      g.expect.errorturn = circularTurn(p, h, mv[0]);
      g.seq.push_back({c.x0, c.y0, c.x1, c.y1});
      return g;
    }
    Move m = choose(p, mv, n, pol, rng, mine[p.side]);
    Coord c = toCoord(p, m);
    g.seq.push_back({c.x0, c.y0, c.x1, c.y1});
    mine[p.side].push_back(m);
    if (mine[p.side].size() > 2) mine[p.side].erase(mine[p.side].begin());
    h.push(p, m);
    play(p, m);
    Outcome o;
    if (terminal(p, o)) {
      g.expect = {true, o.winner, 0, 0};
      return g;
    }
  }
}

static void illegalProbes(const Game& g, std::mt19937_64& rng) {
  if (g.seq.size() < 6) return;
  size_t k = 1 + rng() % (g.seq.size() - 2);
  Pos p = start();
  for (size_t i = 0; i < k; i++) {
    auto& m = g.seq[i];
    play(p, fromCoord(m[0], m[1], m[2], m[3]));
  }
  u64 own = p.bb[p.side], opp = p.bb[p.side ^ 1], empty = ON & ~(own | opp);
  int ob = lsb(own), ox = ob & 7, oy = ob >> 3;
  std::vector<std::array<int, 4>> bad;
  bad.push_back({ox, oy, ox, oy});
  if (opp) { int b = lsb(opp); bad.push_back({ox, oy, b & 7, b >> 3}); bad.push_back({b & 7, b >> 3, ((b & 7) + 1) % 7, b >> 3}); }
  if (empty) { int e = lsb(empty); bad.push_back({e & 7, e >> 3, (e & 7) ^ 1, e >> 3}); bad.push_back({-1, oy, e & 7, e >> 3}); bad.push_back({ox, oy, 7, e >> 3}); }
  for (int d = 3; d <= 6; d++)
    if (onBoard(ox + d, oy) && (empty >> bitOf(ox + d, oy) & 1)) { bad.push_back({ox, oy, ox + d, oy}); break; }
  for (auto& b : bad) {
    Seq s(g.seq.begin(), g.seq.begin() + k);
    s.push_back(b);
    check(s, {true, p.side ^ 1, 1, 0}, "illegal");
  }
}

static void invariants(std::mt19937_64& rng) {
  Pos p = start();
  History h;
  h.reset(p);
  for (int step = 0; step < 20000; step++) {
    Move mv[NMOVES];
    int n = genGeometric(p, mv);
    std::set<Move> gen(mv, mv + n), naive;
    for (u64 o = p.bb[p.side]; o; o &= o - 1) {
      int ob = lsb(o);
      for (int dx = -2; dx <= 2; dx++)
        for (int dy = -2; dy <= 2; dy++) {
          int x = (ob & 7) + dx, y = (ob >> 3) + dy;
          if ((!dx && !dy) || !onBoard(x, y) || ((p.bb[0] | p.bb[1]) >> bitOf(x, y) & 1)) continue;
          naive.insert(fromCoord(ob & 7, ob >> 3, x, y));
        }
    }
    if (gen != naive) { printf("MOVEGEN MISMATCH ply %d\n", p.ply); failures++; }
    for (int t = 0; t < 8; t++) {
      Pos q{{transform(p.bb[0], t), transform(p.bb[1], t)}, p.side, p.ply};
      Move qm[NMOVES];
      int qn = genGeometric(q, qm);
      std::set<Move> a(qm, qm + qn), b;
      for (Move m : gen) b.insert(transformMove(m, t));
      if (a != b) { printf("SYMMETRY MISMATCH t=%d ply %d\n", t, p.ply); failures++; }
      Move m = mv[rng() % n];
      Pos p2 = p, q2 = q;
      play(p2, m);
      play(q2, transformMove(m, t));
      if (transform(p2.bb[0], t) != q2.bb[0] || transform(p2.bb[1], t) != q2.bb[1]) { printf("PLAY SYMMETRY MISMATCH\n"); failures++; }
    }
    for (int i = 0; i < n; i++) {
      Coord c = toCoord(p, mv[i]);
      if (fromCoord(c.x0, c.y0, c.x1, c.y1) != mv[i]) { printf("COORD MISMATCH\n"); failures++; }
    }
    n = filterLegal(p, h, mv, n);
    Outcome o;
    if (n == 0) { p = start(); h.reset(p); continue; }
    Move m = mv[rng() % n];
    h.push(p, m);
    play(p, m);
    if (terminal(p, o)) { p = start(); h.reset(p); }
  }
}

int main(int argc, char** argv) {
  int scale = argc > 1 ? atoi(argv[1]) : 1;
  std::mt19937_64 r0(12345);
  invariants(r0);
  printf("invariants done, failures=%ld\n", failures.load());
  struct Job { Policy pol; int count; const char* name; };
  Job jobs[] = {{RANDOM, 50 * scale, "random"}, {SHUFFLE, 50 * scale, "shuffle"}, {BLOCKADE, 20 * scale, "blockade"}, {PROLONG, 5 * scale, "prolong"}};
  for (auto& job : jobs) {
    std::atomic<long> plies{0}, circ{0}, caps{0};
#pragma omp parallel for schedule(dynamic)
    for (int gi = 0; gi < job.count; gi++) {
      std::mt19937_64 rng(1000003ULL * gi + job.pol * 7919 + 1);
      Game g = generate(job.pol, rng, true);
      plies += g.seq.size();
      if (g.seq.size() >= PLYCAP) caps++;
      for (size_t k = 1; k < g.seq.size(); k++) check(Seq(g.seq.begin(), g.seq.begin() + k), {false, 0, 0, 0}, job.name);
      check(g.seq, g.expect, job.name);
      if (!g.probe.empty()) { circ++; check(g.probe, g.probeExpect, "circular"); }
      if (gi % 3 == 0) illegalProbes(g, rng);
    }
    printf("%s: %d games, %ld plies, %ld circular probes, %ld capped, checks=%ld failures=%ld\n", job.name, job.count, plies.load(), circ.load(), caps.load(), checks.load(), failures.load());
  }
  printf("%s\n", failures ? "FAILED" : "PASSED");
  return failures ? 1 : 0;
}

#include <omp.h>
#include <mutex>
#include "search.h"

using namespace ataxx;

constexpr int TOP = 32;

struct Row {
  u64 bb[2], fin[2];
  uint16_t idx[TOP], cnt[TOP];
  uint16_t ply;
  uint8_t side, n;
  int8_t z, score;
  int16_t q;
};

struct Game {
  Search s;
  std::mt19937_64 rng;
  std::vector<Row> rows;
  uint32_t budget = 0;
  bool full = false, noised = false;
};

struct Driver {
  std::vector<Game> games;
  std::vector<Row> out;
  std::mutex lock;
  int fullVisits, fastVisits, tempPlies;
  float pFull, lateTemp, pOpening = 0;

  void begin(Game& g) {
    g.full = std::uniform_real_distribution<float>(0, 1)(g.rng) < pFull;
    g.budget = g.full ? fullVisits : fastVisits;
    g.noised = false;
    g.s.forced = g.full;
  }

  void restart(Game& g) {
    History h;
    Pos p = start();
    h.reset(p);
    g.s.reset(p, h);
    g.rows.clear();
    if (std::uniform_real_distribution<float>(0, 1)(g.rng) < pOpening) {
      Move mv[NMOVES];
      Outcome o;
      for (int k = 1 + g.rng() % 8; k > 0; k--) {
        int n = genLegal(g.s.root, g.s.hist, mv);
        Pos next = g.s.root;
        Move m = mv[g.rng() % n];
        play(next, m);
        if (terminal(next, o)) break;
        g.s.advance(m);
      }
    }
    begin(g);
  }

  void record(Game& g) {
    const Node& r = g.s.nodes[0];
    Row row{};
    row.bb[0] = g.s.root.bb[0];
    row.bb[1] = g.s.root.bb[1];
    row.ply = g.s.root.ply;
    row.side = g.s.root.side;
    float t[NMOVES];
    if (r.proven) {
      std::fill(t, t + r.n, 0.f);
      t[g.s.best()] = 1;
    } else {
      g.s.target(t);
    }
    for (int k = 0; k < TOP; k++) {
      int b = std::max_element(t, t + r.n) - t;
      if (t[b] <= 0) break;
      row.idx[k] = g.s.edges[r.edge + b].move;
      row.cnt[k] = t[b];
      row.n = k + 1;
      t[b] = 0;
    }
    g.rows.push_back(row);
  }

  int sample(Game& g) {
    const Node& r = g.s.nodes[0];
    if (r.proven) return g.s.best();
    float inv = 1 / (g.s.root.ply < tempPlies ? 1.f : lateTemp), w[NMOVES], total = 0;
    for (int j = 0; j < r.n; j++) total += w[j] = std::pow((float)g.s.childVisits(j), inv);
    float x = std::uniform_real_distribution<float>(0, total)(g.rng);
    for (int j = 0; j < r.n; j++)
      if ((x -= w[j]) <= 0) return j;
    return g.s.best();
  }

  void finish(Game& g, Outcome o) {
    for (Row& r : g.rows) {
      r.z = o.winner < 0 ? 0 : o.winner == r.side ? 1 : -1;
      r.score = o.count[r.side] - o.count[r.side ^ 1];
      r.fin[0] = g.s.root.bb[0];
      r.fin[1] = g.s.root.bb[1];
    }
    std::lock_guard<std::mutex> guard(lock);
    out.insert(out.end(), g.rows.begin(), g.rows.end());
  }

  void move(Game& g) {
    if (g.full) record(g);
    int j = sample(g);
    if (g.full) {
      const Node& r = g.s.nodes[0];
      g.rows.back().q = std::max(-32767.f, std::min(32767.f, 32767 * (r.proven ? r.pv : g.s.improvedValue())));
    }
    g.s.advance(g.s.edges[g.s.nodes[0].edge + j].move);
    Outcome o;
    Move mv[NMOVES];
    if (terminal(g.s.root, o)) {
      finish(g, o);
      restart(g);
    } else if (!genLegal(g.s.root, g.s.hist, mv)) {
      int s = g.s.root.side;
      o.winner = s ^ 1;
      o.count[0] = popcnt(g.s.root.bb[0]);
      o.count[1] = popcnt(g.s.root.bb[1]);
      finish(g, o);
      restart(g);
    } else {
      begin(g);
    }
  }

  void step(Game& g, float* feat) {
    for (;;) {
      const Node& r = g.s.nodes[0];
      if (r.expanded && (r.proven || r.visits >= g.budget)) {
        move(g);
        continue;
      }
      if (g.full && !g.noised && r.expanded) {
        g.s.noise(g.rng);
        g.noised = true;
      }
      if (g.s.select(feat)) return;
    }
  }
};

static Driver D;

extern "C" {

void create(int games, int fullVisits, int fastVisits, float pFull, int tempPlies, float lateTemp, float pOpening, unsigned long seed, int threads) {
  omp_set_num_threads(threads);
  D.games.resize(games);
  D.fullVisits = fullVisits;
  D.fastVisits = fastVisits;
  D.pFull = pFull;
  D.tempPlies = tempPlies;
  D.lateTemp = lateTemp;
  D.pOpening = pOpening;
  for (int i = 0; i < games; i++) {
    D.games[i].rng.seed(seed * 1000003 + i);
    D.restart(D.games[i]);
  }
}

void collect(int begin, int end, float* feat) {
#pragma omp parallel for schedule(dynamic, 16)
  for (int i = begin; i < end; i++) D.step(D.games[i], feat + (long)(i - begin) * PLANES * SQ);
}

void feed(int begin, int end, const float* policy, const float* value) {
#pragma omp parallel for schedule(dynamic, 16)
  for (int i = begin; i < end; i++) D.games[i].s.feed(policy + (long)(i - begin) * NMOVES, value + (long)(i - begin) * 4);
}

int drain(Row* buf, int cap) {
  std::lock_guard<std::mutex> guard(D.lock);
  int n = std::min<long>(cap, D.out.size());
  std::copy(D.out.end() - n, D.out.end(), buf);
  D.out.resize(D.out.size() - n);
  return n;
}

void symmetry(int* squares, int* moves) {
  for (int t = 0; t < 8; t++) {
    for (int s = 0; s < SQ; s++) squares[t * SQ + s] = T().symSq[t][s];
    for (int m = 0; m < NMOVES; m++) moves[t * NMOVES + m] = transformMove(m, t);
  }
}
}

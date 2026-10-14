#include <omp.h>
#include <mutex>
#include "search.h"
#include "judge_ref.h"

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
  int fullVisits, fastVisits, tempPlies, gumbelM = 0;
  float pFull, lateTemp, pOpening = 0;
  long finished = 0, plies = 0, draws = 0, blackWins = 0;

  void begin(Game& g) {
    g.full = gumbelM || std::uniform_real_distribution<float>(0, 1)(g.rng) < pFull;
    g.budget = g.full ? fullVisits : fastVisits;
    g.noised = false;
    g.s.forced = g.full && !gumbelM;
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
    } else if (gumbelM) {
      g.s.improved(t);
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
    if (gumbelM) return g.s.cand[0];
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
    finished++;
    plies += g.s.root.ply;
    draws += o.winner < 0;
    blackWins += o.winner == 0;
  }

  void move(Game& g) {
    if (g.full) record(g);
    int j = sample(g);
    if (g.full) {
      const Node& r = g.s.nodes[0];
      g.rows.back().q = std::max(-32767.f, std::min(32767.f, 32767 * g.s.q(r, g.s.edges[r.edge + j], r.sum / r.visits)));
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
      if (r.expanded && (r.proven || (gumbelM ? g.s.gActive && g.s.gDone : r.visits >= g.budget))) {
        move(g);
        continue;
      }
      if (gumbelM && r.expanded && !g.s.gActive) {
        g.s.gumbelStart(g.rng, gumbelM, fullVisits);
        continue;
      }
      if (!gumbelM && g.full && !g.noised && r.expanded) {
        g.s.noise(g.rng);
        g.noised = true;
      }
      if (g.s.select(feat)) return;
    }
  }
};

static Driver D;

struct Duel {
  Search s[2];
  long used = 0, sims = 0;
  int aSide = 0, moves[2] = {0, 0}, pending = -1;
  float score = -1;
};

struct Arena {
  std::vector<Duel> games;
  std::vector<int> order[2];
  long evals[2];

  int mover(const Duel& g) const { return g.s[0].root.side ^ g.aSide; }

  void apply(Duel& g, Move m) {
    g.s[0].advance(m);
    g.s[1].advance(m);
    g.used = g.sims = 0;
    Outcome o;
    Move mv[NMOVES];
    if (terminal(g.s[0].root, o)) g.score = o.winner < 0 ? 0.5f : o.winner == g.aSide;
    else if (!genLegal(g.s[0].root, g.s[0].hist, mv)) g.score = g.s[0].root.side != g.aSide;
  }

  void step(Duel& g, float* feat) {
    g.pending = -1;
    while (g.score < 0) {
      int w = mover(g);
      Search& s = g.s[w];
      long budget = g.moves[w] ? evals[w] : 3 * evals[w];
      if (s.nodes[0].expanded && (s.nodes[0].proven || s.nodes[0].n == 1 || g.used >= budget || g.sims >= 64 * budget)) {
        g.moves[w]++;
        apply(g, s.edges[s.nodes[0].edge + s.best()].move);
        continue;
      }
      g.sims++;
      if (s.select(feat)) {
        g.pending = w;
        g.used++;
        return;
      }
    }
  }
};

static Arena A;

struct Probe {
  std::vector<Search> s;
  std::vector<int> count;
  long evals;
  int batch;
};

static Probe P;

extern "C" {

void create(int games, int fullVisits, int fastVisits, float pFull, int tempPlies, float lateTemp, int gumbel, float pOpening, unsigned long seed, int threads) {
  omp_set_num_threads(threads);
  D.games.resize(games);
  D.fullVisits = fullVisits;
  D.fastVisits = fastVisits;
  D.pFull = pFull;
  D.tempPlies = tempPlies;
  D.lateTemp = lateTemp;
  D.gumbelM = gumbel;
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

void stats(long* s) {
  std::lock_guard<std::mutex> guard(D.lock);
  s[0] = D.finished;
  s[1] = D.plies;
  s[2] = D.draws;
  s[3] = D.blackWins;
}

void arena_create(int pairs, long evalsA, long evalsB, const int* moves, const int* lengths, int threads) {
  omp_set_num_threads(threads);
  A.games.assign(2 * pairs, Duel{});
  A.evals[0] = evalsA;
  A.evals[1] = evalsB;
  for (int k = 0; k < pairs; k++) {
    for (int c = 0; c < 2; c++) {
      Duel& g = A.games[2 * k + c];
      g.aSide = c;
      Pos p = start();
      History h;
      h.reset(p);
      for (Search& s : g.s) {
        s.reset(p, h);
        for (int i = 0; i < lengths[k]; i++) s.advance(fromCoord(moves[4 * i], moves[4 * i + 1], moves[4 * i + 2], moves[4 * i + 3]));
      }
    }
    moves += 4 * lengths[k];
  }
}

int arena_collect(float* featA, float* featB, int* counts) {
  float* feat[2] = {featA, featB};
  int n = A.games.size();
  std::vector<float> tmp((long)n * PLANES * SQ);
#pragma omp parallel for schedule(dynamic, 16)
  for (int i = 0; i < n; i++) A.step(A.games[i], tmp.data() + (long)i * PLANES * SQ);
  A.order[0].clear();
  A.order[1].clear();
  int live = 0;
  for (int i = 0; i < n; i++) {
    int w = A.games[i].pending;
    live += A.games[i].score < 0;
    if (w < 0) continue;
    std::copy(tmp.begin() + (long)i * PLANES * SQ, tmp.begin() + (long)(i + 1) * PLANES * SQ, feat[w] + (long)A.order[w].size() * PLANES * SQ);
    A.order[w].push_back(i);
  }
  counts[0] = A.order[0].size();
  counts[1] = A.order[1].size();
  return live;
}

void arena_feed(const float* polA, const float* valA, const float* polB, const float* valB) {
  const float* pol[2] = {polA, polB};
  const float* val[2] = {valA, valB};
  for (int w = 0; w < 2; w++) {
    int n = A.order[w].size();
#pragma omp parallel for schedule(dynamic, 16)
    for (int j = 0; j < n; j++) A.games[A.order[w][j]].s[w].feed(pol[w] + (long)j * NMOVES, val[w] + (long)j * 4);
  }
}

void arena_params(int side, float cpuct, float cpuctLog, float fpu, float scoreWeight) {
  for (Duel& g : A.games) {
    g.s[side].cpuct = cpuct;
    g.s[side].cpuctLog = cpuctLog;
    g.s[side].fpu = fpu;
    g.s[side].scoreWeight = scoreWeight;
  }
}

void probe_create(int n, long evals, int batch, const int* moves, const int* lengths, int threads) {
  omp_set_num_threads(threads);
  P.s.assign(n, Search{});
  P.count.assign(n, 0);
  P.evals = evals;
  P.batch = batch;
  for (int k = 0; k < n; k++) {
    Pos p = start();
    History h;
    h.reset(p);
    P.s[k].reset(p, h);
    for (int i = 0; i < lengths[k]; i++) P.s[k].advance(fromCoord(moves[4 * i], moves[4 * i + 1], moves[4 * i + 2], moves[4 * i + 3]));
    moves += 4 * lengths[k];
  }
}

void probe_boards(int n, long evals, int batch, const uint64_t* bb, const uint8_t* side, const uint16_t* ply, int threads) {
  omp_set_num_threads(threads);
  P.s.assign(n, Search{});
  P.count.assign(n, 0);
  P.evals = evals;
  P.batch = batch;
  for (int k = 0; k < n; k++) {
    Pos p{{bb[2 * k], bb[2 * k + 1]}, side[k], ply[k]};
    History h;
    h.reset(p);
    P.s[k].reset(p, h);
  }
}

float probe_root(int i, float* visits) {
  const Search& s = P.s[i];
  const Node& r = s.nodes[0];
  std::fill(visits, visits + NMOVES, 0.f);
  for (int j = 0; j < r.n; j++) visits[s.edges[r.edge + j].move] = s.childVisits(j);
  return r.proven ? r.pv : r.sum / r.visits;
}

int probe_collect(float* feat) {
  int n = P.s.size(), total = 0;
#pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < n; i++) {
    Search& s = P.s[i];
    P.count[i] = 0;
    while (!s.nodes[0].proven && s.evals < P.evals && !P.count[i]) P.count[i] = s.gather(feat + (long)i * P.batch * PLANES * SQ, s.nodes[0].expanded ? P.batch : 1);
  }
  for (int i = 0; i < n; i++) {
    std::copy(feat + (long)i * P.batch * PLANES * SQ, feat + ((long)i * P.batch + P.count[i]) * PLANES * SQ, feat + (long)total * PLANES * SQ);
    total += P.count[i];
  }
  return total;
}

void probe_feed(const float* policy, const float* value) {
  int n = P.s.size();
  std::vector<int> first(n + 1, 0);
  for (int i = 0; i < n; i++) first[i + 1] = first[i] + P.count[i];
#pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < n; i++)
    for (int j = 0; j < P.count[i]; j++) P.s[i].resolve(j, policy + (long)(first[i] + j) * NMOVES, value + (long)(first[i] + j) * 4);
}

int probe_result(int i, int* moves, float* visits, float* q, float* prior) {
  const Search& s = P.s[i];
  const Node& r = s.nodes[0];
  for (int j = 0; j < r.n; j++) {
    const Edge& e = s.edges[r.edge + j];
    Coord c = toCoord(s.root, e.move);
    int v[4] = {c.x0, c.y0, c.x1, c.y1};
    std::copy(v, v + 4, moves + 4 * j);
    visits[j] = s.childVisits(j);
    q[j] = s.q(r, e, -2);
    prior[j] = e.prior;
  }
  return r.n;
}

void arena_scores(float* out) {
  for (size_t i = 0; i < A.games.size(); i++) out[i] = A.games[i].score;
}

int status(const int* moves, int n, int* legal, int* count) {
  Pos p = start();
  History h;
  h.reset(p);
  for (int i = 0; i < n; i++) {
    Move m = fromCoord(moves[4 * i], moves[4 * i + 1], moves[4 * i + 2], moves[4 * i + 3]);
    h.push(p, m);
    play(p, m);
  }
  Outcome o;
  *count = 0;
  if (terminal(p, o)) return o.winner < 0 ? 3 : o.winner + 1;
  Move mv[NMOVES];
  *count = genLegal(p, h, mv);
  for (int j = 0; j < *count; j++) {
    Coord c = toCoord(p, mv[j]);
    int v[4] = {c.x0, c.y0, c.x1, c.y1};
    std::copy(v, v + 4, legal + 4 * j);
  }
  return *count ? 0 : 2 - p.side;
}

void adjudicate(const int* moves, int n, int* out) {
  std::vector<std::array<int, 4>> history(n);
  for (int i = 0; i < n; i++) std::copy(moves + 4 * i, moves + 4 * i + 4, history[i].begin());
  auto v = JudgeRef().run(history);
  out[0] = v.finish;
  out[1] = v.winner;
  out[2] = v.err;
  out[3] = v.errorturn;
}

void features(u64 black, u64 white, int side, int ply, float* out) { encode(Pos{{black, white}, side, ply}, out); }

void symmetry(int* squares, int* moves) {
  for (int t = 0; t < 8; t++) {
    for (int s = 0; s < SQ; s++) squares[t * SQ + s] = T().symSq[t][s];
    for (int m = 0; m < NMOVES; m++) moves[t * NMOVES + m] = transformMove(m, t);
  }
}
}

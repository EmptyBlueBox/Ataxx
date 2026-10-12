#pragma once
#include <cmath>
#include <random>
#include <unordered_map>
#include "board.h"

namespace ataxx {

struct Node {
  uint32_t edge = 0, visits = 0;
  float sum = 0, pv = 0;
  uint16_t n = 0;
  bool expanded = false, proven = false;
};

struct Edge {
  uint32_t child;
  float prior;
  Move move;
};

struct Search {
  std::vector<Node> nodes;
  std::vector<Edge> edges;
  std::vector<uint32_t> path;
  Pos root, leaf;
  History hist;
  int depth = 0;
  long evals = 0;
  bool forced = false;
  std::vector<int> cand;
  float gumbel[NMOVES];
  int gSims = 0, gPer = 0, gRound = 0, gNext = 0, gPhases = 0;
  bool gActive = false, gDone = false;
  size_t cacheLimit = 0;
  std::unordered_map<u64, std::vector<float>> cache;
  float cpuct = 1.0f, cpuctLog = 0.45f, fpu = 0.2f, scoreWeight = 0.0f, scoreScale = 5.0f;

  void reset(const Pos& p, const History& h) {
    root = p;
    hist = h;
    nodes.assign(1, Node{});
    edges.clear();
    cache.clear();
    gActive = false;
  }

  void advance(Move m) {
    uint32_t c = 0;
    for (int j = 0; j < nodes[0].n; j++)
      if (edges[nodes[0].edge + j].move == m) c = edges[nodes[0].edge + j].child;
    hist.push(root, m);
    play(root, m);
    gActive = false;
    if (!c) {
      nodes.assign(1, Node{});
      edges.clear();
      return;
    }
    std::vector<Node> nn;
    std::vector<Edge> ne;
    nn.push_back(nodes[c]);
    for (size_t i = 0; i < nn.size(); i++) {
      uint32_t old = nn[i].edge;
      int n = nn[i].n;
      nn[i].edge = ne.size();
      for (int j = 0; j < n; j++) {
        Edge e = edges[old + j];
        if (e.child) {
          nn.push_back(nodes[e.child]);
          e.child = nn.size() - 1;
        }
        ne.push_back(e);
      }
    }
    nodes.swap(nn);
    edges.swap(ne);
  }

  float outcomeValue(const Outcome& o, int side) const {
    float w = o.winner < 0 ? 0.f : o.winner == side ? 1.f : -1.f;
    return (1 - scoreWeight) * w + scoreWeight * std::tanh(scoreScale * (o.count[side] - o.count[side ^ 1]) / 49.f);
  }

  float q(const Node& parent, const Edge& e, float fpuQ) const {
    if (!e.child) return fpuQ;
    const Node& c = nodes[e.child];
    return c.proven ? -c.pv : c.visits ? -c.sum / c.visits : fpuQ;
  }

  float explore(const Node& n) const { return (cpuct + cpuctLog * std::log((n.visits + 500.f) / 500.f)) * std::sqrt((float)n.visits); }

  float fpuValue(const Node& n, bool isRoot) const {
    float mass = 0;
    for (int j = 0; j < n.n; j++)
      if (edges[n.edge + j].child) mass += edges[n.edge + j].prior;
    return n.sum / n.visits - (isRoot ? 0.5f : 1.f) * fpu * std::sqrt(mass);
  }

  float sigma(const Node& r, float value) const {
    uint32_t top = 0;
    for (int j = 0; j < r.n; j++) top = std::max(top, childVisits(j));
    return (50.f + top) * (value + 1) / 2;
  }

  float completed(const Node& r, int j) const {
    const Edge& e = edges[r.edge + j];
    return std::log(e.prior + 1e-30f) + sigma(r, q(r, e, r.sum / r.visits));
  }

  template <class Rng>
  void gumbelStart(Rng& rng, int m, int sims) {
    const Node& r = nodes[0];
    std::uniform_real_distribution<float> uniform(1e-12f, 1.f);
    cand.resize(r.n);
    for (int j = 0; j < r.n; j++) cand[j] = j, gumbel[j] = -std::log(-std::log(uniform(rng)));
    std::sort(cand.begin(), cand.end(), [&](int a, int b) { return gumbel[a] + std::log(edges[r.edge + a].prior + 1e-30f) > gumbel[b] + std::log(edges[r.edge + b].prior + 1e-30f); });
    cand.resize(std::min<int>(m, r.n));
    gPhases = std::max(1, (int)std::ceil(std::log2((float)cand.size())));
    gSims = sims;
    gPer = std::max<int>(1, sims / (gPhases * cand.size()));
    gRound = gNext = 0;
    gActive = true;
    gDone = cand.size() == 1;
  }

  int gumbelPick() {
    int j = cand[gNext++];
    if (gNext == (int)cand.size()) {
      gNext = 0;
      if (++gRound == gPer) {
        const Node& r = nodes[0];
        std::sort(cand.begin(), cand.end(), [&](int a, int b) { return gumbel[a] + completed(r, a) > gumbel[b] + completed(r, b); });
        if (cand.size() <= 2) gDone = true;
        cand.resize((cand.size() + 1) / 2);
        gPer = std::max<int>(1, gSims / (gPhases * cand.size()));
        gRound = 0;
      }
    }
    return j;
  }

  void improved(float* out) const {
    const Node& r = nodes[0];
    float mx = -1e30f, total = 0;
    for (int j = 0; j < r.n; j++) mx = std::max(mx, out[j] = completed(r, j));
    for (int j = 0; j < r.n; j++) total += out[j] = std::exp(out[j] - mx);
    for (int j = 0; j < r.n; j++) out[j] *= 65535 / total;
  }

  int pick(const Node& n, bool isRoot) {
    if (isRoot && gActive && !gDone) return gumbelPick();
    float c = explore(n), fq = fpuValue(n, isRoot), best = -1e9f;
    int bi = 0;
    for (int j = 0; j < n.n; j++) {
      const Edge& e = edges[n.edge + j];
      uint32_t cv = e.child ? nodes[e.child].visits : 0;
      if (isRoot && forced && cv && !nodes[e.child].proven && cv < std::sqrt(2 * e.prior * n.visits)) return j;
      float u = q(n, e, fq) + c * e.prior / (1 + cv);
      if (u > best) best = u, bi = j;
    }
    return bi;
  }

  void prove(Node& n) {
    float best = -2;
    bool all = true;
    for (int j = 0; j < n.n; j++) {
      uint32_t c = edges[n.edge + j].child;
      if (!c || !nodes[c].proven) all = false;
      else best = std::max(best, -nodes[c].pv);
    }
    if (best > 0 || all) n.proven = true, n.pv = best;
  }

  void backup(float v) {
    for (int i = (int)path.size() - 1; i >= 0; i--) {
      Node& n = nodes[path[i]];
      n.visits++;
      n.sum += v;
      if (i + 1 < (int)path.size() && nodes[path[i + 1]].proven && !n.proven) prove(n);
      v = -v;
    }
    for (; depth; depth--) hist.pop();
  }

  bool select(float* feat) {
    path.clear();
    Pos p = root;
    uint32_t ni = 0;
    for (;;) {
      path.push_back(ni);
      if (nodes[ni].proven) {
        backup(nodes[ni].pv);
        return false;
      }
      if (!nodes[ni].expanded) {
        leaf = p;
        auto hit = cache.find(cacheKey(p));
        if (hit != cache.end()) {
          Move mv[NMOVES];
          float policy[NMOVES];
          int n = genGeometric(p, mv);
          for (int j = 0; j < n; j++) policy[mv[j]] = hit->second[4 + j];
          expand(policy, hit->second.data());
          return false;
        }
        encode(p, feat);
        return true;
      }
      uint32_t ei = nodes[ni].edge + pick(nodes[ni], ni == 0);
      hist.push(p, edges[ei].move);
      play(p, edges[ei].move);
      depth++;
      if (!edges[ei].child) {
        Node c;
        Outcome o;
        if (terminal(p, o)) c.proven = true, c.pv = outcomeValue(o, p.side);
        nodes.push_back(c);
        edges[ei].child = nodes.size() - 1;
      }
      ni = edges[ei].child;
    }
  }

  static u64 cacheKey(const Pos& p) { return mix(mix(p.bb[0], p.bb[1]), p.side * 512 + p.ply); }

  void feed(const float* policy, const float* value) {
    evals++;
    if (cacheLimit) {
      if (cache.size() >= cacheLimit) cache.clear();
      Move mv[NMOVES];
      int n = genGeometric(leaf, mv);
      std::vector<float>& c = cache[cacheKey(leaf)];
      c.assign(value, value + 4);
      for (int j = 0; j < n; j++) c.push_back(policy[mv[j]]);
    }
    expand(policy, value);
  }

  void expand(const float* policy, const float* value) {
    Move mv[NMOVES];
    int n = genLegal(leaf, hist, mv);
    uint32_t ni = path.back();
    if (!n) {
      nodes[ni].proven = true;
      nodes[ni].pv = -1;
      backup(-1);
      return;
    }
    float mx = -1e30f, total = 0, p[NMOVES];
    for (int j = 0; j < n; j++) mx = std::max(mx, policy[mv[j]]);
    for (int j = 0; j < n; j++) total += p[j] = std::exp(policy[mv[j]] - mx);
    nodes[ni].edge = edges.size();
    nodes[ni].n = n;
    nodes[ni].expanded = true;
    for (int j = 0; j < n; j++) edges.push_back({0, p[j] / total, mv[j]});
    float m = std::max(value[0], std::max(value[1], value[2]));
    float w = std::exp(value[0] - m), d = std::exp(value[1] - m), l = std::exp(value[2] - m);
    backup((1 - scoreWeight) * (w - l) / (w + d + l) + scoreWeight * std::tanh(scoreScale * std::tanh(value[3])));
  }

  uint32_t childVisits(int j) const {
    uint32_t c = edges[nodes[0].edge + j].child;
    return c ? nodes[c].visits : 0;
  }

  int best() const {
    const Node& r = nodes[0];
    int bi = 0;
    float bs = -1e30f;
    for (int j = 0; j < r.n; j++) {
      const Edge& e = edges[r.edge + j];
      float s = childVisits(j) + (e.child && nodes[e.child].visits ? -nodes[e.child].sum / nodes[e.child].visits : -1.f) * 0.5f;
      if (r.proven) s = e.child && nodes[e.child].proven ? 1e6f * (-nodes[e.child].pv + 2) + s : s;
      if (s > bs) bs = s, bi = j;
    }
    return bi;
  }

  template <class Rng>
  void noise(Rng& rng, float weight = 0.25f, float concentration = 10.83f) {
    Node& r = nodes[0];
    std::gamma_distribution<float> gamma(concentration / r.n, 1.f);
    float g[NMOVES], total = 0;
    for (int j = 0; j < r.n; j++) total += g[j] = gamma(rng) + 1e-12f;
    for (int j = 0; j < r.n; j++) edges[r.edge + j].prior = (1 - weight) * edges[r.edge + j].prior + weight * g[j] / total;
  }

  void target(float* out) const {
    const Node& r = nodes[0];
    int b = 0;
    for (int j = 1; j < r.n; j++)
      if (childVisits(j) > childVisits(b)) b = j;
    float c = explore(r), fq = fpuValue(r, true);
    float bestU = q(r, edges[r.edge + b], fq) + c * edges[r.edge + b].prior / (1 + childVisits(b));
    for (int j = 0; j < r.n; j++) {
      const Edge& e = edges[r.edge + j];
      float cv = childVisits(j);
      if (j != b && cv > 0 && forced) {
        float gap = bestU - q(r, e, fq);
        float need = gap > 0 ? c * e.prior / gap - 1 : cv;
        cv = std::min(cv, std::max(need, cv - std::sqrt(2 * e.prior * r.visits)));
        if (cv < 1) cv = 0;
      }
      out[j] = cv;
    }
  }
};

}

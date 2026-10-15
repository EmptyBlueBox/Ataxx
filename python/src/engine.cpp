#include <array>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include "judge_ref.h"
#include "tf.h"
#include "search.h"

using namespace ataxx;

static double cpu() {
  timespec t;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

struct Phase {
  Transformer* net;
  int lo, hi;
};

struct Engine {
  Transformer net;
  std::vector<Phase> phases;
  Search s;
  std::vector<std::array<int, 4>> log;
  long evals = 0, sims = 0;
  int explorePlies = 0;
  bool symmetric = false;
  float exploreShare = 1;
  std::mt19937_64 rng;

  explicit Engine(const char* path, const std::vector<std::string>& specs = {}) : net(path) {
    for (const std::string& spec : specs) {
      size_t a = spec.find(':'), b = spec.find(':', a + 1);
      phases.push_back({new Transformer(spec.substr(0, a).c_str()), std::stoi(spec.substr(a + 1, b - a - 1)), std::stoi(spec.substr(b + 1))});
    }
    s.cacheLimit = 200000;
    reset();
  }

  void reset() {
    Pos p = start();
    History h;
    h.reset(p);
    s.reset(p, h);
    log.clear();
  }

  void apply(int x0, int y0, int x1, int y1) {
    log.push_back({x0, y0, x1, y1});
    s.advance(fromCoord(x0, y0, x1, y1));
  }

  bool accepted(Move m) {
    Coord c = toCoord(s.root, m);
    auto moves = log;
    moves.push_back({c.x0, c.y0, c.x1, c.y1});
    return JudgeRef().run(moves).err == 0;
  }

  Coord go(long maxEvals, double deadline) {
    float feat[PLANES * SQ], policy[NMOVES], value[4];
    Move mv[NMOVES];
    int n = genLegal(s.root, s.hist, mv);
    evals = sims = 0;
    Move chosen = mv[0];
    if (n > 1) {
      while (!s.nodes[0].proven && evals < maxEvals && sims < 64 * maxEvals && cpu() < deadline) {
        sims++;
        if (s.select(feat)) {
          int empty = popcnt(ON & ~(s.leaf.bb[0] | s.leaf.bb[1]));
          Transformer* t0 = &net;
          for (Phase& p : phases)
            if (empty >= p.lo && empty < p.hi) t0 = p.net;
          if (symmetric) {
            int t = rng() % 8;
            float raw[NMOVES];
            encode(Pos{{transform(s.leaf.bb[0], t), transform(s.leaf.bb[1], t)}, s.leaf.side, s.leaf.ply}, feat);
            t0->eval(feat, raw, value);
            for (int m = 0; m < NMOVES; m++) policy[m] = raw[transformMove(m, t)];
          } else {
            t0->eval(feat, policy, value);
          }
          s.feed(policy, value);
          evals++;
        }
      }
      chosen = s.edges[s.nodes[0].edge + s.best()].move;
      if ((int)log.size() < explorePlies && !s.nodes[0].proven) {
        const Node& r = s.nodes[0];
        uint32_t top = 0;
        std::vector<Move> near;
        for (int j = 0; j < r.n; j++) top = std::max(top, s.childVisits(j));
        for (int j = 0; j < r.n; j++)
          if (s.childVisits(j) >= exploreShare * top) near.push_back(s.edges[r.edge + j].move);
        chosen = near[rng() % near.size()];
      }
    }
    if (!accepted(chosen))
      for (int j = 0; j < n; j++)
        if (accepted(mv[j])) chosen = mv[j];
    return toCoord(s.root, chosen);
  }

  float rootQ() const { return s.nodes[0].visits ? s.nodes[0].sum / s.nodes[0].visits : 0; }
};

static int field(const std::string& text, size_t from, const char* key) {
  size_t at = text.find(std::string("\"") + key + "\"", from);
  return std::stoi(text.substr(text.find(':', at) + 1));
}

static std::vector<std::array<int, 4>> objects(const std::string& text, const char* key) {
  std::vector<std::array<int, 4>> out;
  size_t at = text.find(std::string("\"") + key + "\"");
  size_t end = text.find(']', at);
  for (size_t o = text.find('{', at); o != std::string::npos && o < end; o = text.find('{', o + 1))
    out.push_back({field(text, o, "x0"), field(text, o, "y0"), field(text, o, "x1"), field(text, o, "y1")});
  return out;
}

static void botzone(Engine& e, double firstBudget, double turnBudget) {
  std::string line;
  bool first = true;
  while (std::getline(std::cin, line)) {
    double began = first ? 0 : cpu();
    if (first) {
      auto requests = objects(line, "requests"), responses = objects(line, "responses");
      for (size_t i = 0; i < requests.size(); i++) {
        if (requests[i][0] >= 0) e.apply(requests[i][0], requests[i][1], requests[i][2], requests[i][3]);
        if (i < responses.size()) e.apply(responses[i][0], responses[i][1], responses[i][2], responses[i][3]);
      }
    } else {
      e.apply(field(line, 0, "x0"), field(line, 0, "y0"), field(line, 0, "x1"), field(line, 0, "y1"));
    }
    Coord c = e.go(1L << 40, began + (first ? firstBudget : turnBudget));
    e.apply(c.x0, c.y0, c.x1, c.y1);
    printf("{\"response\":{\"x0\":%d,\"y0\":%d,\"x1\":%d,\"y1\":%d},\"debug\":\"e%ld s%ld v%u q%.3f t%.0f\"}\n>>>BOTZONE_REQUEST_KEEP_RUNNING<<<\n", c.x0, c.y0, c.x1, c.y1, e.evals, e.sims, e.s.nodes[0].visits, e.rootQ(), (cpu() - began) * 1000);
    fflush(stdout);
    first = false;
  }
}

static void simplebotzone(Engine& e, double firstBudget, double turnBudget) {
  int turns, x0, y0, x1, y1;
  std::cin >> turns;
  for (int i = 0; i < 2 * turns - 1; ++i) {
    std::cin >> x0 >> y0 >> x1 >> y1;
    if (x0 >= 0) e.apply(x0, y0, x1, y1);
  }
  double began = 0;
  bool first = true;
  for (;;) {
    Coord c = e.go(1L << 40, began + (first ? firstBudget : turnBudget));
    e.apply(c.x0, c.y0, c.x1, c.y1);
    printf("%d %d %d %d\ne%ld s%ld q%.3f t%.0f k%d\n\n\n>>>BOTZONE_REQUEST_KEEP_RUNNING<<<\n", c.x0, c.y0, c.x1, c.y1, e.evals, e.sims, e.rootQ(), (cpu() - began) * 1000, VNNI);
    fflush(stdout);
    if (!(std::cin >> x0 >> y0 >> x1 >> y1)) break;
    began = cpu();
    first = false;
    e.apply(x0, y0, x1, y1);
  }
}

int main(int argc, char** argv) {
  Engine e(argv[1], argc > 5 ? std::vector<std::string>(argv + 5, argv + argc) : std::vector<std::string>{});
  if (argc > 2 && std::string(argv[2]) == "simpleio") {
    e.s.cacheLimit = 65536;
    simplebotzone(e, std::stod(argv[3]), std::stod(argv[4]));
    return 0;
  }
  if (argc > 2 && std::string(argv[2]) == "botzone") {
    botzone(e, std::stod(argv[3]), std::stod(argv[4]));
    return 0;
  }
  std::string line, cmd;
  while (std::getline(std::cin, line)) {
    std::istringstream in(line);
    in >> cmd;
    if (cmd == "new") {
      e.reset();
    } else if (cmd == "play") {
      int x0, y0, x1, y1;
      in >> x0 >> y0 >> x1 >> y1;
      e.apply(x0, y0, x1, y1);
    } else if (cmd == "legal") {
      Move mv[NMOVES];
      Outcome o;
      int n = terminal(e.s.root, o) ? 0 : genLegal(e.s.root, e.s.hist, mv);
      for (int j = 0; j < n; j++) {
        Coord c = toCoord(e.s.root, mv[j]);
        printf("%d %d %d %d ", c.x0, c.y0, c.x1, c.y1);
      }
      printf("\n");
    } else if (cmd == "go") {
      std::string kind;
      double amount;
      in >> kind >> amount;
      double began = cpu();
      Coord c = kind == "evals" ? e.go((long)amount, 1e18) : e.go(1L << 40, began + amount);
      double q = e.rootQ();
      e.apply(c.x0, c.y0, c.x1, c.y1);
      printf("%d %d %d %d %ld %ld %.4f %.4f\n", c.x0, c.y0, c.x1, c.y1, e.evals, e.sims, q, cpu() - began);
    } else if (cmd == "bench") {
      int n;
      in >> n;
      float feat[PLANES * SQ], policy[NMOVES], value[4];
      encode(e.s.root, feat);
      e.net.eval(feat, policy, value);
      double began = cpu();
      for (int i = 0; i < n; i++) e.net.eval(feat, policy, value);
      printf("%.4f\n", (cpu() - began) * 1000 / n);
    } else if (cmd == "explore") {
      unsigned long seed;
      in >> e.explorePlies >> e.exploreShare >> seed >> e.symmetric;
      e.rng.seed(seed);
    } else if (cmd == "fpu") {
      in >> e.s.fpu;
    } else if (cmd == "quit") {
      break;
    }
    fflush(stdout);
  }
}

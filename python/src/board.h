#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>
#ifdef __x86_64__
#pragma GCC target("popcnt")
#endif

namespace ataxx {

using u64 = uint64_t;
using Move = uint16_t;
constexpr int SQ = 49, KINDS = 17, NMOVES = KINDS * SQ, PLYCAP = 401;
constexpr u64 ROW0 = 0x7F;
constexpr u64 ON = ROW0 | ROW0 << 8 | ROW0 << 16 | ROW0 << 24 | ROW0 << 32 | ROW0 << 40 | ROW0 << 48;
constexpr u64 COL0 = 0x0001010101010101ULL & ON;
constexpr u64 COL6 = COL0 << 6;
constexpr int DX[16] = {-2, -1, 0, 1, 2, -2, 2, -2, 2, -2, 2, -2, -1, 0, 1, 2};
constexpr int DY[16] = {-2, -2, -2, -2, -2, -1, -1, 0, 0, 1, 1, 2, 2, 2, 2, 2};

inline int bitOf(int x, int y) { return y * 8 + x; }
inline int sqOf(int x, int y) { return y * 7 + x; }
inline int sqOfBit(int b) { return (b >> 3) * 7 + (b & 7); }
inline int bitOfSq(int s) { return (s / 7) * 8 + s % 7; }
inline int popcnt(u64 b) { return __builtin_popcountll(b); }
inline int lsb(u64 b) { return __builtin_ctzll(b); }
inline u64 east(u64 b) { return (b & ~COL6) << 1; }
inline u64 west(u64 b) { return (b & ~COL0) >> 1; }
inline u64 dilate1(u64 b) { u64 h = b | east(b) | west(b); return (h | h >> 8 | h << 8) & ON; }
inline u64 dilate2(u64 b) { return dilate1(dilate1(b)); }
inline bool onBoard(int x, int y) { return x >= 0 && x < 7 && y >= 0 && y < 7; }

struct Tables {
  u64 ring1[SQ];
  int jumpOrigin[SQ][16];
  int kindOf[5][5];
  int symSq[8][SQ], symKind[8][KINDS];
  u64 zob[2][SQ];
  Tables() {
    for (int s = 0; s < SQ; s++) {
      u64 b = 1ULL << bitOfSq(s);
      ring1[s] = dilate1(b) & ~b;
      for (int k = 0; k < 16; k++) {
        int x = s % 7 - DX[k], y = s / 7 - DY[k];
        jumpOrigin[s][k] = onBoard(x, y) ? bitOf(x, y) : -1;
      }
    }
    for (int k = 0; k < 16; k++) kindOf[DX[k] + 2][DY[k] + 2] = k + 1;
    for (int t = 0; t < 8; t++) {
      for (int s = 0; s < SQ; s++) {
        int x = s % 7, y = s / 7;
        symSq[t][s] = sqOf(tx(x, y, t), ty(x, y, t));
      }
      symKind[t][0] = 0;
      for (int k = 0; k < 16; k++) {
        int dx = DX[k], dy = DY[k];
        if (t & 1) dx = -dx;
        if (t & 2) dy = -dy;
        if (t & 4) std::swap(dx, dy);
        symKind[t][k + 1] = kindOf[dx + 2][dy + 2];
      }
    }
    u64 x = 0x9E3779B97F4A7C15ULL;
    for (int c = 0; c < 2; c++)
      for (int s = 0; s < SQ; s++) {
        x += 0x9E3779B97F4A7C15ULL;
        u64 z = x;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        zob[c][s] = z ^ (z >> 31);
      }
  }
  static int tx(int x, int y, int t) { int a = t & 1 ? 6 - x : x, b = t & 2 ? 6 - y : y; return t & 4 ? b : a; }
  static int ty(int x, int y, int t) { int a = t & 1 ? 6 - x : x, b = t & 2 ? 6 - y : y; return t & 4 ? a : b; }
};
inline const Tables& T() { static Tables t; return t; }

struct Pos {
  u64 bb[2];
  int side, ply;
};

inline Pos start() {
  Pos p{};
  p.bb[0] = 1ULL << bitOf(0, 0) | 1ULL << bitOf(6, 6);
  p.bb[1] = 1ULL << bitOf(6, 0) | 1ULL << bitOf(0, 6);
  return p;
}

inline int genGeometric(const Pos& p, Move* out) {
  u64 own = p.bb[p.side], empty = ON & ~(p.bb[0] | p.bb[1]);
  int n = 0;
  for (u64 c = dilate1(own) & empty; c; c &= c - 1) out[n++] = sqOfBit(lsb(c));
  for (u64 o = own; o; o &= o - 1) {
    int ob = lsb(o), ox = ob & 7, oy = ob >> 3;
    for (int k = 0; k < 16; k++) {
      int x = ox + DX[k], y = oy + DY[k];
      if (onBoard(x, y) && (empty >> bitOf(x, y) & 1)) out[n++] = (k + 1) * SQ + sqOf(x, y);
    }
  }
  return n;
}

inline void placement(const Pos& p, Move m, u64 s[2]) {
  int k = m / SQ, dsq = m % SQ;
  s[0] = p.bb[0];
  s[1] = p.bb[1];
  if (k) s[p.side] &= ~(1ULL << T().jumpOrigin[dsq][k - 1]);
  s[p.side] |= 1ULL << bitOfSq(dsq);
}

inline void play(Pos& p, Move m) {
  int k = m / SQ, dsq = m % SQ;
  if (k) p.bb[p.side] &= ~(1ULL << T().jumpOrigin[dsq][k - 1]);
  p.bb[p.side] |= 1ULL << bitOfSq(dsq);
  u64 f = T().ring1[dsq] & p.bb[p.side ^ 1];
  p.bb[p.side ^ 1] ^= f;
  p.bb[p.side] |= f;
  p.side ^= 1;
  p.ply++;
}

struct Snap {
  u64 b[2];
  bool operator==(const Snap& o) const { return b[0] == o.b[0] && b[1] == o.b[1]; }
};

struct History {
  std::vector<Snap> s;
  std::vector<int> lastClone;
  void reset(const Pos& p) {
    s.assign(1, Snap{{p.bb[0], p.bb[1]}});
    lastClone.assign(1, 0);
  }
  void push(const Pos& before, Move m) {
    Snap a;
    placement(before, m, a.b);
    s.push_back(a);
    lastClone.push_back(m < SQ ? (int)s.size() - 1 : lastClone.back());
  }
  void pop() {
    s.pop_back();
    lastClone.pop_back();
  }
  int ply() const { return (int)s.size() - 1; }
};

inline int circularTurn(const Pos& p, const History& h, Move m) {
  int k = h.ply();
  if (k + 1 < 4) return 0;
  Snap c;
  placement(p, m, c.b);
  for (int i = h.lastClone[k]; i <= k - 2; i++)
    if (h.s[i] == h.s[k] && h.s[i + 1] == c) return i + 1;
  return 0;
}

inline int filterLegal(const Pos& p, const History& h, Move* moves, int n) {
  int k = h.ply();
  if (k + 1 < 4) return n;
  int idx[PLYCAP + 1], ni = 0;
  for (int i = h.lastClone[k]; i <= k - 2; i++)
    if (h.s[i] == h.s[k]) idx[ni++] = i;
  if (!ni) return n;
  int w = 0;
  for (int j = 0; j < n; j++) {
    Snap c;
    placement(p, moves[j], c.b);
    bool circ = false;
    for (int t = 0; t < ni && !circ; t++) circ = h.s[idx[t] + 1] == c;
    if (!circ) moves[w++] = moves[j];
  }
  return w;
}

inline int genLegal(const Pos& p, const History& h, Move* out) { return filterLegal(p, h, out, genGeometric(p, out)); }

struct Outcome {
  int winner;
  int count[2];
};

inline bool terminal(const Pos& p, Outcome& o) {
  int c[2] = {popcnt(p.bb[0]), popcnt(p.bb[1])};
  if (!c[p.side]) {
    o.winner = p.side ^ 1;
    o.count[p.side ^ 1] = 49;
    o.count[p.side] = 0;
    return true;
  }
  u64 empty = ON & ~(p.bb[0] | p.bb[1]);
  bool has0 = dilate2(p.bb[0]) & empty, has1 = dilate2(p.bb[1]) & empty;
  if (has0 && has1 && p.ply < PLYCAP) return false;
  if (has0 && !has1) c[0] = 49 - c[1];
  if (!has0 && has1) c[1] = 49 - c[0];
  o.winner = c[0] > c[1] ? 0 : c[0] < c[1] ? 1 : -1;
  o.count[0] = c[0];
  o.count[1] = c[1];
  return true;
}

struct Coord {
  int x0, y0, x1, y1;
};

inline Coord toCoord(const Pos& p, Move m) {
  int k = m / SQ, dsq = m % SQ;
  int ob = k ? T().jumpOrigin[dsq][k - 1] : lsb(T().ring1[dsq] & p.bb[p.side]);
  return {ob & 7, ob >> 3, dsq % 7, dsq / 7};
}

inline Move fromCoord(int x0, int y0, int x1, int y1) {
  int dx = x1 - x0, dy = y1 - y0;
  int k = std::max(std::abs(dx), std::abs(dy)) == 1 ? 0 : T().kindOf[dx + 2][dy + 2];
  return k * SQ + sqOf(x1, y1);
}

inline u64 transform(u64 b, int t) {
  u64 r = 0;
  for (; b; b &= b - 1) {
    int bi = lsb(b);
    r |= 1ULL << bitOfSq(T().symSq[t][sqOfBit(bi)]);
  }
  return r;
}

inline Move transformMove(Move m, int t) { return T().symKind[t][m / SQ] * SQ + T().symSq[t][m % SQ]; }

inline u64 mix(u64 a, u64 b) {
  u64 z = a * 0x9E3779B97F4A7C15ULL ^ (b + 0x632BE59BD9B4E019ULL);
  z = (z ^ (z >> 29)) * 0xBF58476D1CE4E5B9ULL;
  return z ^ (z >> 32);
}

inline u64 canonicalKey(const Pos& p) {
  u64 best = ~0ULL;
  for (int t = 0; t < 8; t++) best = std::min(best, mix(transform(p.bb[p.side], t), transform(p.bb[p.side ^ 1], t)));
  return best;
}

constexpr int PLANES = 14;

inline void encode(const Pos& p, float* f) {
  u64 own = p.bb[p.side], opp = p.bb[p.side ^ 1], empty = ON & ~(own | opp);
  u64 planes[8] = {own, opp, empty, ON, dilate1(own) & empty, dilate1(opp) & empty, dilate2(own) & ~dilate1(own) & empty, dilate2(opp) & ~dilate1(opp) & empty};
  for (int i = 0; i < 8; i++)
    for (int s = 0; s < SQ; s++) f[i * SQ + s] = planes[i] >> bitOfSq(s) & 1;
  for (int s = 0; s < SQ; s++) {
    bool e = empty >> bitOfSq(s) & 1;
    f[8 * SQ + s] = e ? popcnt(T().ring1[s] & opp) / 8.f : 0.f;
    f[9 * SQ + s] = e ? popcnt(T().ring1[s] & own) / 8.f : 0.f;
  }
  float g[4] = {p.ply / 401.f, popcnt(own) / 49.f, popcnt(opp) / 49.f, popcnt(empty) / 49.f};
  for (int i = 0; i < 4; i++)
    for (int s = 0; s < SQ; s++) f[(10 + i) * SQ + s] = g[i];
}

}

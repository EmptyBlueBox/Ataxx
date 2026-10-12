#pragma once
#include <cstdlib>
#include <vector>

namespace ataxx {

struct Verdict {
  bool finish;
  int winner;
  int err;
  int errorturn;
};

struct JudgeRef {
  int grid[7][7];
  int blackCount, whiteCount;
  char history[1002][7][7];
  static constexpr int delta[24][2] = {{1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}, {1, 0}, {2, 0}, {2, 1}, {2, 2}, {1, 2}, {0, 2}, {-1, 2}, {-2, 2}, {-2, 1}, {-2, 0}, {-2, -1}, {-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2}, {2, -1}};

  void save(int g) {
    for (int r = 0; r < 7; r++)
      for (int c = 0; c < 7; c++) history[g][r][c] = grid[r][c] == 0 ? '0' : '2' - (grid[r][c] + 1) / 2;
  }
  int repeat(int g) {
    if (g < 4) return 0;
    for (int bef = g - 2; bef >= 0; bef--)
      for (int i = 0; i < bef; i++) {
        bool same = true;
        for (int r = 0; r < 7 && same; r++)
          for (int c = 0; c < 7; c++)
            if (history[i][r][c] != history[g - 1][r][c]) { same = false; break; }
        if (!same) continue;
        for (int r = 0; r < 7 && same; r++)
          for (int c = 0; c < 7; c++)
            if (history[i + 1][r][c] != history[g][r][c]) { same = false; break; }
        if (same) return -i - 1;
      }
    return 0;
  }
  static bool in(int x, int y) { return x >= 0 && x <= 6 && y >= 0 && y <= 6; }
  int step(int x0, int y0, int x1, int y1, int color, int g) {
    if (!in(x0, y0) || !in(x1, y1)) return 0;
    if (grid[x0][y0] != color) return 0;
    int dx = std::abs(x0 - x1), dy = std::abs(y0 - y1);
    if ((dx == 0 && dy == 0) || dx > 2 || dy > 2) return 0;
    if (grid[x1][y1] != 0) return 0;
    if (dx == 2 || dy == 2) grid[x0][y0] = 0;
    else (color == 1 ? blackCount : whiteCount)++;
    grid[x1][y1] = color;
    save(g);
    int cnt = 0;
    for (int d = 0; d < 8; d++) {
      int x = x1 + delta[d][0], y = y1 + delta[d][1];
      if (in(x, y) && grid[x][y] == -color) { grid[x][y] = color; cnt++; }
    }
    if (color == 1) { blackCount += cnt; whiteCount -= cnt; } else { whiteCount += cnt; blackCount -= cnt; }
    int r = repeat(g);
    return r == 0 ? 1 : r;
  }
  bool hasMove(int color, int g) {
    for (int y0 = 0; y0 < 7; y0++)
      for (int x0 = 0; x0 < 7; x0++) {
        if (grid[x0][y0] != color) continue;
        for (int d = 0; d < 24; d++) {
          int x = x0 + delta[d][0], y = y0 + delta[d][1];
          if (!in(x, y) || grid[x][y] != 0) continue;
          history[g][x][y] = '2' - (color + 1) / 2;
          bool rep = repeat(g);
          history[g][x][y] = '0';
          if (!rep) return true;
        }
      }
    return false;
  }
  Verdict run(const std::vector<std::array<int, 4>>& moves) {
    for (auto& r : grid) for (int& c : r) c = 0;
    grid[0][0] = grid[6][6] = 1;
    grid[6][0] = grid[0][6] = -1;
    blackCount = whiteCount = 2;
    save(0);
    int color = 1, size = 2 * (int)moves.size();
    Verdict v{false, 0, 0, 0};
    for (int i = 1; i < size; i += 2) {
      bool last = i == size - 1;
      auto& m = moves[(i - 1) / 2];
      if (color == 1) save((i + 1) / 2);
      int res = step(m[0], m[1], m[2], m[3], color, (i + 1) / 2);
      if (last) {
        int me = color == 1 ? 0 : 1, other = 1 - me;
        if (!res) return {true, other, 1, 0};
        if (res < 0) return {true, other, 2, -res};
        if ((color == 1 ? whiteCount : blackCount) == 0) return {true, me, 0, 0};
        int g = color == 1 ? i : i + 1;
        bool b = hasMove(1, g), w = hasMove(-1, g);
        if (!b || !w || size > 800) {
          if (b && !w) blackCount = 49 - whiteCount;
          if (!b && w) whiteCount = 49 - blackCount;
          return {true, blackCount > whiteCount ? 0 : blackCount < whiteCount ? 1 : -1, 0, 0};
        }
      }
      color = -color;
    }
    return v;
  }
};

}

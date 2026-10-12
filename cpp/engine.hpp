#ifndef ATAXX_ENGINE_HPP
#define ATAXX_ENGINE_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <utility>
#include <vector>

namespace ataxx {

inline constexpr char evaluationNotice[] = R"NOTICE(Evaluation adapted from Autaxx by kz04px.
MIT License

Copyright (c) 2019 kz04px

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
)NOTICE";


using Bits = std::uint64_t;
constexpr Bits boardMask = (Bits{1} << 49) - 1;
constexpr int infinity = 1000000;
inline std::array<Bits, 49> adjacent{}, jumps{};

inline int count(Bits bits) {
#if defined(__POPCNT__) || defined(__aarch64__)
    return __builtin_popcountll(bits);
#else
    bits -= (bits >> 1) & 0x5555555555555555ULL;
    bits = (bits & 0x3333333333333333ULL) + ((bits >> 2) & 0x3333333333333333ULL);
    bits = (bits + (bits >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
    return static_cast<int>((bits * 0x0101010101010101ULL) >> 56);
#endif
}
inline int first(Bits bits) { return __builtin_ctzll(bits); }

inline void initialize() {
    for (int a = 0; a < 49; ++a)
        for (int b = 0; b < 49; ++b) {
            int distance = std::max(std::abs(a / 7 - b / 7), std::abs(a % 7 - b % 7));
            if (distance == 1) adjacent[a] |= Bits{1} << b;
            if (distance == 2) jumps[a] |= Bits{1} << b;
        }
}

struct Move {
    int from = -1, to = -1;
    bool operator==(Move other) const { return from == other.from && to == other.to; }
};

struct Position {
    Bits us, them;
    Position play(Move move) const {
        Bits own = us | (Bits{1} << move.to);
        if (jumps[move.from] & (Bits{1} << move.to)) own &= ~(Bits{1} << move.from);
        Bits captured = them & adjacent[move.to];
        return {them ^ captured, own | captured};
    }
    bool operator==(Position other) const { return us == other.us && them == other.them; }
};

struct PastMove {
    Position position;
    Move move;
    int side;
};

inline bool isLegal(Position position, Move move, int side, const std::vector<PastMove>& history) {
    if (move.from < 0 || move.from >= 49 || move.to < 0 || move.to >= 49) return false;
    Bits target = Bits{1} << move.to;
    if (!(position.us & (Bits{1} << move.from)) || ((position.us | position.them) & target)) return false;
    if (!((adjacent[move.from] | jumps[move.from]) & target)) return false;
    for (auto it = history.rbegin(); it != history.rend(); ++it) {
        if (adjacent[it->move.from] & (Bits{1} << it->move.to)) break;
        if (it->side == side && it->position == position && it->move == move) return false;
    }
    return true;
}

inline Bits expand(Bits bits) {
    constexpr Bits left = 0x40810204081ULL;
    constexpr Bits right = left << 6;
    Bits horizontal = bits | ((bits & ~left) >> 1) | ((bits & ~right) << 1);
    return (horizontal | (horizontal << 7) | (horizontal >> 7)) & boardMask;
}

inline int result(int difference) {
    return difference > 0 ? 100000 + difference : difference < 0 ? -100000 + difference : 0;
}

inline int terminal(Position position, int ply) {
    int own = count(position.us), opponent = count(position.them);
    if (!own || !opponent) return result(own - opponent);
    Bits empty = boardMask & ~(position.us | position.them);
    bool ownMoves = expand(expand(position.us)) & empty;
    bool opponentMoves = expand(expand(position.them)) & empty;
    if (!ownMoves || !opponentMoves) {
        int remaining = count(empty);
        return result(own - opponent + (ownMoves ? remaining : 0) - (opponentMoves ? remaining : 0));
    }
    if (ply >= 400) return result(own - opponent);
    return infinity;
}

inline constexpr int middlePst[49] = {100, 73, 15, 12, 33, 57, 56, 86, 19, 2, 1, 2, 18, 41, 18, 3, 0, 4, 15, 1, 15, 15, 5, 2, 1, 4, 3, 28, 18, 1, 14, 5, 2, 5, 16, 65, 25, 9, 2, 3, 25, 56, 78, 37, 27, 20, 26, 73, 83};
inline constexpr int latePst[49] = {95, 65, 87, 37, 84, 74, 96, 57, 37, 18, 0, 13, 27, 74, 74, 15, 21, 2, 29, 23, 85, 44, 2, 0, 0, 5, 2, 26, 81, 19, 9, 4, 28, 19, 83, 67, 33, 33, 1, 19, 34, 69, 100, 78, 72, 45, 80, 65, 85};
inline constexpr auto positionalWeights = [] {
    std::array<std::array<std::array<int, 2>, 128>, 7> weights{};
    for (int row = 0; row < 7; ++row)
        for (int mask = 1; mask < 128; ++mask) {
            int bit = 0;
            while (!(mask & (1 << bit))) ++bit;
            weights[row][mask][0] = weights[row][mask & (mask - 1)][0] + middlePst[row * 7 + bit];
            weights[row][mask][1] = weights[row][mask & (mask - 1)][1] + latePst[row * 7 + bit];
        }
    return weights;
}();

inline std::array<Bits, 4> neighborBits(Bits own) {
    constexpr Bits left = 0x40810204081ULL;
    Bits west = (own & ~left) >> 1, east = (own & ~(left << 6)) << 1;
    auto add = [](Bits a, Bits b, Bits c, Bits& low, Bits& high) {
        Bits pair = a ^ b;
        low = pair ^ c;
        high = (a & b) | (pair & c);
    };
    Bits s0, c0, s1, c1, s2, c2, s3, c3;
    add(west, east, own << 7, s0, c0);
    add(own >> 7, west << 7, west >> 7, s1, c1);
    add(s0, s1, east << 7, s2, c2);
    Bits one = s2 ^ (east >> 7), carry = s2 & (east >> 7);
    add(c0, c1, c2, s3, c3);
    Bits two = s3 ^ carry, next = s3 & carry;
    return {one, two, c3 ^ next, c3 & next};
}

inline int evaluate(Position position) {
    constexpr int middleHoles[9] = {-17, 4, -28, -88, -125, -200, -322, -446, -534};
    constexpr int lateHoles[9] = {23, 28, 23, 0, -64, -129, -192, -351, -440};
    Bits empty = boardMask & ~(position.us | position.them);
    int phase = count(position.us | position.them);
    int material = count(position.us) - count(position.them);
    int middle = material * 100 + 243, late = material * 79 + 394;
    int safe = count(position.us & ~expand(empty)) - count(position.them & ~expand(empty));
    middle += safe * 149;
    late += safe * 27;
    for (int side : {1, -1}) {
        Bits own = side == 1 ? position.us : position.them;
        Bits enemy = side == 1 ? position.them : position.us;
        for (int row = 0; row < 7; ++row) {
            const auto& weights = positionalWeights[row][(own >> (row * 7)) & 127];
            middle += side * weights[0];
            late += side * weights[1];
        }
        Bits holes = expand(own) & empty & expand(expand(enemy));
        auto bits = neighborBits(own);
        Bits low0 = ~bits[0] & ~bits[1], low1 = bits[0] & ~bits[1];
        Bits low2 = ~bits[0] & bits[1], low3 = bits[0] & bits[1];
        auto add = [&](Bits mask, int neighbors) {
            if (mask) {
                int cells = side * count(mask);
                middle += cells * middleHoles[neighbors];
                late += cells * lateHoles[neighbors];
            }
        };
        add(holes & low1 & ~bits[2], 1);
        add(holes & low2 & ~bits[2], 2);
        add(holes & low3 & ~bits[2], 3);
        add(holes & low0 & bits[2], 4);
        add(holes & low1 & bits[2], 5);
        add(holes & low2 & bits[2], 6);
        add(holes & low3 & bits[2], 7);
        add(holes & bits[3], 8);
    }
    return (middle * (49 - phase) + late * phase) / 49;
}

template<class AddMove>
inline void generate(Position position, int side, const std::vector<PastMove>& history, AddMove add) {
    std::array<Bits, 49> banned{};
    for (auto it = history.rbegin(); it != history.rend(); ++it) {
        if (adjacent[it->move.from] & (Bits{1} << it->move.to)) break;
        if (it->side == side && it->position == position) banned[it->move.from] |= Bits{1} << it->move.to;
    }
    Bits empty = boardMask & ~(position.us | position.them);
    Bits clones = 0;
    for (Bits pieces = position.us; pieces; pieces &= pieces - 1) {
        int from = first(pieces);
        Bits targets = ((adjacent[from] & ~clones) | jumps[from]) & empty & ~banned[from];
        for (; targets; targets &= targets - 1) {
            int to = first(targets);
            add(Move{from, to});
            if (adjacent[from] & (Bits{1} << to)) clones |= Bits{1} << to;
        }
    }
}

inline std::vector<Move> generate(Position position, int side, const std::vector<PastMove>& history) {
    std::vector<Move> moves;
    moves.reserve(256);
    generate(position, side, history, [&](Move move) { moves.push_back(move); });
    return moves;
}

struct Entry {
    Bits us = 0, them = 0;
    Move move;
    std::uint64_t historyKey = 0;
    int ply = 0, side = 0, depth = -1, score = 0, bound = 0;
};

struct SearchStats {
    std::uint64_t nodes = 0;
    int depth = 0, score = 0;
    int moveDepth = 0, moveScore = 0;
    double seconds = 0;
};

class Search {
    struct Candidate { Move move; int score; };
    struct Frame {
        std::array<Candidate, 49 * 24> moves;
        Move best;
    };
    std::vector<Entry> table = std::vector<Entry>(1 << 18);
    std::vector<Frame> frames;
    int rootPly = 0;
    std::clock_t start, deadline;
    SearchStats statistics;
    bool stopped = false;
    Move rootBest;
    int rootScore = 0;
    std::vector<PastMove> history;
    std::uint64_t historyKey = 0;

    static std::uint64_t signature(const PastMove& past) {
        Bits key = past.position.us * 0x9e3779b97f4a7c15ULL ^ past.position.them * 0xbf58476d1ce4e5b9ULL;
        key ^= (past.move.from * 49 + past.move.to) * 0x94d049bb133111ebULL;
        key ^= past.side == 1 ? 0xd6e8feb86659fd93ULL : 0xa5a3564e27f8862bULL;
        key = (key ^ (key >> 30)) * 0xbf58476d1ce4e5b9ULL;
        key = (key ^ (key >> 27)) * 0x94d049bb133111ebULL;
        return key ^ (key >> 31);
    }

    Entry& entry(Position position) {
        Bits key = position.us * 0x9e3779b97f4a7c15ULL ^ position.them * 0xbf58476d1ce4e5b9ULL;
        return table[(key ^ (key >> 32)) & (table.size() - 1)];
    }

    int negamax(Position position, int side, int ply, int depth, int alpha, int beta, bool allowNull = true) {
        if ((statistics.nodes++ & 255) == 0 && std::clock() >= deadline) stopped = true;
        if (stopped) return 0;
        int end = terminal(position, ply);
        if (end != infinity) return end;
        if (!depth) return evaluate(position);
        if (allowNull && depth >= 3 && beta == alpha + 1 && beta < 100000
            && count(boardMask & ~(position.us | position.them)) >= 10
            && count(position.us) >= 4 && count(position.them) >= 4 && evaluate(position) >= beta) {
            int score = -negamax({position.them, position.us}, -side, ply + 1, depth - 3, -beta, -beta + 1, false);
            if (stopped) return 0;
            if (score >= beta) return score;
        }
        int originalAlpha = alpha;
        auto& cached = entry(position);
        if (cached.us == position.us && cached.them == position.them && cached.historyKey == historyKey
            && cached.ply == ply && cached.side == side && cached.depth >= depth
            && (!cached.bound || (cached.bound > 0 && cached.score >= beta) || (cached.bound < 0 && cached.score <= alpha))) {
            frames[depth - 1].best = cached.move;
            if (ply == rootPly) { rootBest = cached.move; rootScore = cached.score; }
            return cached.score;
        }
        Move preferred = cached.us == position.us && cached.them == position.them ? cached.move : Move{};
        if (preferred.from < 0 && depth >= 4) {
            negamax(position, side, ply, depth - 2, alpha, beta);
            if (stopped) return 0;
            preferred = cached.us == position.us && cached.them == position.them ? cached.move : Move{};
        }
        auto& frame = frames[depth - 1];
        int size = 0;
        auto prepare = [&] {
            std::array<int, 49> captures;
            captures.fill(-1);
            generate(position, side, history, [&](Move move) {
                if (captures[move.to] < 0) captures[move.to] = 200 * count(position.them & adjacent[move.to]);
                int score = captures[move.to];
                score += (adjacent[move.from] & (Bits{1} << move.to)) ? 100 : 0;
                if (move == preferred) score += infinity;
                frame.moves[size++] = {move, score};
            });
        };
        bool deferred = preferred.from >= 0 && isLegal(position, preferred, side, history);
        if (deferred) frame.moves[size++] = {preferred, infinity};
        else prepare();
        if (!size) return -100049;
        int best = -infinity;
        Move bestMove;
        for (int i = 0; i < size; ++i) {
            auto begin = frame.moves.begin() + i;
            auto next = std::max_element(begin, frame.moves.begin() + size,
                [](Candidate a, Candidate b) { return a.score < b.score; });
            std::iter_swap(begin, next);
            Move move = begin->move;
            int score;
            if (depth == 1) score = -negamax(position.play(move), -side, ply + 1, 0, -beta, -alpha);
            else {
                history.push_back({position, move, side});
                Bits oldHistoryKey = historyKey;
                historyKey = (adjacent[move.from] & (Bits{1} << move.to)) ? 0 : historyKey ^ signature(history.back());
                if (!i) score = -negamax(position.play(move), -side, ply + 1, depth - 1, -beta, -alpha);
                else {
                    score = -negamax(position.play(move), -side, ply + 1, depth - 1, -alpha - 1, -alpha);
                    if (!stopped && score > alpha && score < beta)
                        score = -negamax(position.play(move), -side, ply + 1, depth - 1, -beta, -alpha);
                }
                history.pop_back();
                historyKey = oldHistoryKey;
            }
            if (stopped) return 0;
            if (score > best) {
                best = score;
                bestMove = move;
                if (ply == rootPly) { rootBest = move; rootScore = score; }
            }
            alpha = std::max(alpha, score);
            if (alpha >= beta) break;
            if (!i && deferred) {
                size = 0;
                prepare();
                auto first = std::find_if(frame.moves.begin(), frame.moves.begin() + size,
                    [&](Candidate candidate) { return candidate.move == preferred; });
                std::iter_swap(frame.moves.begin(), first);
            }
        }
        frame.best = bestMove;
        if ((cached.us == position.us && cached.them == position.them) || cached.depth <= depth + 2)
            cached = {position.us, position.them, bestMove, historyKey, ply, side, depth, best, best <= originalAlpha ? -1 : best >= beta ? 1 : 0};
        return best;
    }

public:
    Search(std::clock_t start, double seconds, std::vector<PastMove> past)
        : start(start), deadline(start + static_cast<std::clock_t>(seconds * CLOCKS_PER_SEC)), history(std::move(past)) {
        for (const auto& h : history)
            historyKey = (adjacent[h.move.from] & (Bits{1} << h.move.to)) ? 0 : historyKey ^ signature(h);
    }

    const SearchStats& stats() const { return statistics; }

    Move choose(Position position, int side, int ply, int maxDepth = 64) {
        statistics = {};
        rootPly = ply;
        stopped = false;
        auto finish = [&](Move move) {
            statistics.seconds = static_cast<double>(std::clock() - start) / CLOCKS_PER_SEC;
            return move;
        };
        if (terminal(position, ply) != infinity) return finish({});
        auto moves = generate(position, side, history);
        if (moves.empty()) return finish({});
        Move best = moves.front();
        if (moves.size() == 1) return finish(best);
        maxDepth = std::min(maxDepth, 400 - ply);
        frames.resize(maxDepth);
        history.reserve(history.size() + maxDepth);
        for (int depth = 1; depth <= maxDepth; ++depth) {
            if (std::clock() >= deadline) break;
            rootBest = {};
            int score = negamax(position, side, ply, depth, -infinity, infinity);
            if (rootBest.from >= 0) {
                best = rootBest;
                statistics.moveDepth = depth;
                statistics.moveScore = rootScore;
            }
            if (stopped) break;
            best = frames[depth - 1].best;
            statistics.depth = depth;
            statistics.score = score;
            if (std::abs(score) >= 100000 || std::clock() >= deadline) break;
        }
        return finish(best);
    }
};

struct Game {
    Position position{(Bits{1} << 0) | (Bits{1} << 48), (Bits{1} << 6) | (Bits{1} << 42)};
    std::vector<PastMove> history;
    int side() const { return history.size() % 2 ? -1 : 1; }
    int ply() const { return static_cast<int>(history.size()); }

    bool legal(Move move) const { return isLegal(position, move, side(), history); }

    int outcome() const {
        int end = terminal(position, ply());
        if (end != infinity) return end;
        return generate(position, side(), history).empty() ? -100049 : infinity;
    }

    void play(Move move) {
        history.push_back({position, move, side()});
        position = position.play(move);
    }

    void undo() {
        position = history.back().position;
        history.pop_back();
    }
};


}

#endif

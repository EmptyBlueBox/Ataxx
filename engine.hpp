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

using Bits = std::uint64_t;
constexpr Bits boardMask = (Bits{1} << 49) - 1;
constexpr int infinity = 1000000;
inline std::array<Bits, 49> adjacent{}, jumps{};

inline int count(Bits bits) { return __builtin_popcountll(bits); }
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
    for (const auto& past : history)
        if (past.side == side && past.position == position && past.move == move) return false;
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

inline int evaluate(Position position) {
    Bits empty = boardMask & ~(position.us | position.them);
    Bits ownReach = expand(expand(position.us)) & empty;
    Bits opponentReach = expand(expand(position.them)) & empty;
    int material = count(position.us) - count(position.them);
    int mobility = count(ownReach) - count(opponentReach);
    int exposed = count(position.them & expand(ownReach)) - count(position.us & expand(opponentReach));
    return material * 100 + mobility * 12 + exposed * 8;
}

template<class AddMove>
inline void generate(Position position, int side, const std::vector<PastMove>& history, AddMove add) {
    std::array<Bits, 49> banned{};
    for (const auto& past : history)
        if (past.side == side && past.position == position) banned[past.move.from] |= Bits{1} << past.move.to;
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

    Entry& entry(Position position) {
        Bits key = position.us * 0x9e3779b97f4a7c15ULL ^ position.them * 0xbf58476d1ce4e5b9ULL;
        return table[(key ^ (key >> 32)) & (table.size() - 1)];
    }

    int negamax(Position position, int side, int ply, int depth, int alpha, int beta) {
        if ((statistics.nodes++ & 255) == 0 && std::clock() >= deadline) stopped = true;
        if (stopped) return 0;
        int end = terminal(position, ply);
        if (end != infinity) return end;
        if (!depth) return evaluate(position);
        auto& cached = entry(position);
        Move preferred = cached.us == position.us && cached.them == position.them ? cached.move : Move{};
        if (preferred.from < 0 && depth >= 4) {
            negamax(position, side, ply, depth - 2, alpha, beta);
            if (stopped) return 0;
            preferred = cached.us == position.us && cached.them == position.them ? cached.move : Move{};
        }
        auto& frame = frames[depth - 1];
        std::array<int, 49> captures;
        captures.fill(-1);
        int size = 0;
        generate(position, side, history, [&](Move move) {
            if (captures[move.to] < 0) captures[move.to] = 200 * count(position.them & adjacent[move.to]);
            int score = captures[move.to];
            score += (adjacent[move.from] & (Bits{1} << move.to)) ? 100 : 0;
            if (move == preferred) score += infinity;
            frame.moves[size++] = {move, score};
        });
        if (!size) return -100049;
        int best = -infinity;
        Move bestMove;
        for (int i = 0; i < size; ++i) {
            auto begin = frame.moves.begin() + i;
            auto next = std::max_element(begin, frame.moves.begin() + size,
                [](Candidate a, Candidate b) { return a.score < b.score; });
            std::iter_swap(begin, next);
            Move move = begin->move;
            history.push_back({position, move, side});
            int score;
            if (!i) score = -negamax(position.play(move), -side, ply + 1, depth - 1, -beta, -alpha);
            else {
                score = -negamax(position.play(move), -side, ply + 1, depth - 1, -alpha - 1, -alpha);
                if (!stopped && score > alpha && score < beta)
                    score = -negamax(position.play(move), -side, ply + 1, depth - 1, -beta, -alpha);
            }
            history.pop_back();
            if (stopped) return 0;
            if (score > best) {
                best = score;
                bestMove = move;
                if (ply == rootPly) { rootBest = move; rootScore = score; }
            }
            alpha = std::max(alpha, score);
            if (alpha >= beta) break;
        }
        frame.best = bestMove;
        cached = {position.us, position.them, bestMove};
        return best;
    }

public:
    Search(std::clock_t start, double seconds, std::vector<PastMove> past)
        : start(start), deadline(start + static_cast<std::clock_t>(seconds * CLOCKS_PER_SEC)), history(std::move(past)) {}

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

#ifndef ATAXX_ENGINE_HPP
#define ATAXX_ENGINE_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <vector>
#include "board.h"
#include "weights.h"

namespace ataxx {

constexpr int infinity = 1000000, win = 100000;

inline int evaluate(const Pos& p) {
    u64 own = p.bb[p.side], opp = p.bb[p.side ^ 1], all = own | opp;
    const Weights& w = weights[popcnt(ON & ~all) * PHASES / 50];
    const Tables& t = T();
    int score = w.tempo;
    for (u64 near = dilate1(all); near; near &= near - 1) {
        int bit = lsb(near), square = sqOfBit(bit), state = own >> bit & 1 ? 1 : opp >> bit & 1 ? 2 : 0;
        score += w.cell[state][popcnt(t.ring1[square] & own) * 9 + popcnt(t.ring1[square] & opp)] + w.square[state][square];
    }
    return score;
}

struct Game {
    Pos pos = start();
    History hist;
    std::vector<Pos> past;
    std::vector<Move> moves;

    Game() { hist.reset(pos); }

    int legal(Move* out) const {
        Outcome o;
        return terminal(pos, o) ? 0 : genLegal(pos, hist, out);
    }

    bool legal(int from, int to) const {
        int dx = to % 7 - from % 7, dy = to / 7 - from / 7;
        if (!(pos.bb[pos.side] >> bitOfSq(from) & 1) || std::max(std::abs(dx), std::abs(dy)) > 2) return false;
        Move moves[NMOVES];
        return std::count(moves, moves + legal(moves), fromCoord(from % 7, from / 7, to % 7, to / 7));
    }

    int winner() const {
        Outcome o;
        Move moves[NMOVES];
        return terminal(pos, o) ? o.winner : genLegal(pos, hist, moves) ? 2 : pos.side ^ 1;
    }

    void play(Move move) {
        past.push_back(pos);
        moves.push_back(move);
        hist.push(pos, move);
        ataxx::play(pos, move);
    }

    void undo() {
        pos = past.back();
        past.pop_back();
        moves.pop_back();
        hist.pop();
    }
};

struct Entry {
    u64 own = 0, opp = 0;
    std::int32_t score = 0;
    Move move = NMOVES;
    std::int8_t depth = -1, bound = 0;
    std::uint8_t age = 0;
};

struct SearchStats {
    std::uint64_t nodes = 0;
    int depth = 0, score = 0;
    double seconds = 0;
};

class Search {
    struct Frame {
        Move moves[NMOVES], killer;
        int scores[NMOVES];
    };
    std::vector<Entry> table = std::vector<Entry>(1 << 21);
    std::vector<Frame> frames = std::vector<Frame>(66);
    std::array<int, NMOVES> cutoffs{};
    std::array<std::array<int, 64>, 64> reductions;
    int rootPly = 0;
    std::clock_t deadline;
    SearchStats statistics;
    bool stopped = false;
    Move rootBest;
    History hist;
    std::uint8_t age = 0;

    int negamax(const Pos& p, int depth, int alpha, int beta, bool allowNull = true) {
        if ((statistics.nodes++ & 1023) == 0 && std::clock() >= deadline) stopped = true;
        if (stopped) return 0;
        u64 own = p.bb[p.side], opp = p.bb[p.side ^ 1];
        Outcome o;
        if (terminal(p, o)) {
            int margin = o.count[p.side] - o.count[p.side ^ 1];
            return margin > 0 ? win + margin : margin < 0 ? margin - win : 0;
        }
        if (depth <= 0) return evaluate(p);
        bool root = p.ply == rootPly, pv = beta > alpha + 1, exact = p.ply + depth < PLYCAP;
        Entry& cached = table[(own * 0x9e3779b97f4a7c15ULL ^ opp) * 0xbf58476d1ce4e5b9ULL >> 43];
        bool hit = cached.own == own && cached.opp == opp;
        if (hit && exact && !root && cached.depth >= depth
            && (!cached.bound || (cached.bound > 0 && cached.score >= beta) || (cached.bound < 0 && cached.score <= alpha)))
            return cached.score;
        if (!pv && beta < win && alpha > -win) {
            int standing = evaluate(p);
            if (depth <= 6 && standing - 120 * depth >= beta) return standing;
            if (allowNull && depth >= 3 && standing >= beta && popcnt(ON & ~(own | opp)) >= 10 && popcnt(own) >= 4 && popcnt(opp) >= 4) {
                Pos pass = p;
                pass.side ^= 1;
                pass.ply++;
                int score = -negamax(pass, depth - 3 - depth / 6, -beta, -beta + 1, false);
                if (stopped) return 0;
                if (score >= beta) return score;
            }
        }
        Move preferred = root ? rootBest : hit ? cached.move : NMOVES;
        if (preferred == NMOVES && depth >= 4) --depth;
        Frame& frame = frames[p.ply - rootPly];
        int size = genLegal(p, hist, frame.moves);
        if (!size) return -win - 49;
        const Tables& t = T();
        for (int i = 0; i < size; ++i) {
            Move move = frame.moves[i];
            u64 ring = t.ring1[move % SQ];
            int score = 1000 * popcnt(ring & opp) + cutoffs[move];
            score += move < SQ ? 500 + 100 * popcnt(ring & own) : -200 * popcnt(t.ring1[sqOfBit(t.jumpOrigin[move % SQ][move / SQ - 1])] & own);
            if (move == frame.killer) score += 400;
            if (move == preferred) score += infinity;
            frame.scores[i] = score;
        }
        int originalAlpha = alpha, best = -infinity;
        Move bestMove = preferred;
        for (int i = 0; i < size; ++i) {
            int next = std::max_element(frame.scores + i, frame.scores + size) - frame.scores;
            std::swap(frame.scores[i], frame.scores[next]);
            std::swap(frame.moves[i], frame.moves[next]);
            Move move = frame.moves[i];
            if (!pv && depth <= 2 && i >= 6 * depth && best > -win) break;
            Pos child = p;
            hist.push(p, move);
            play(child, move);
            int score;
            if (!i || depth == 1) score = -negamax(child, depth - 1, -beta, -alpha);
            else {
                int reduction = depth >= 3 ? std::max(0, std::min(reductions[std::min(depth, 63)][std::min(i, 63)] - pv, depth - 2)) : 0;
                score = -negamax(child, depth - 1 - reduction, -alpha - 1, -alpha);
                if (reduction && score > alpha) score = -negamax(child, depth - 1, -alpha - 1, -alpha);
                if (score > alpha && score < beta) score = -negamax(child, depth - 1, -beta, -alpha);
            }
            hist.pop();
            if (stopped) return 0;
            if (score > best) {
                best = score;
                if (score > alpha || !i) {
                    bestMove = move;
                    if (root) rootBest = move;
                }
            }
            alpha = std::max(alpha, score);
            if (alpha >= beta) {
                int bonus = std::min(depth * depth, 256);
                cutoffs[move] += bonus - cutoffs[move] * bonus / 512;
                for (int j = 0; j < i; ++j) cutoffs[frame.moves[j]] -= bonus + cutoffs[frame.moves[j]] * bonus / 512;
                frame.killer = move;
                break;
            }
        }
        if (hit || cached.age != age || cached.depth <= depth + 2)
            cached = {own, opp, best, bestMove, static_cast<std::int8_t>(exact ? depth : 0),
                static_cast<std::int8_t>(best <= originalAlpha ? -1 : best >= beta ? 1 : 0), age};
        return best;
    }

public:
    Search() {
        for (int depth = 1; depth < 64; ++depth)
            for (int i = 1; i < 64; ++i) reductions[depth][i] = static_cast<int>(0.5 + std::log(depth) * std::log(i) / 1.5);
    }

    const SearchStats& stats() const { return statistics; }

    Move choose(const Game& game, std::clock_t start, double seconds) {
        statistics = {};
        rootPly = game.pos.ply;
        hist = game.hist;
        deadline = start + static_cast<std::clock_t>(seconds * CLOCKS_PER_SEC);
        stopped = false;
        ++age;
        for (int& value : cutoffs) value /= 4;
        Move moves[NMOVES]{};
        int size = game.legal(moves), score = 0;
        rootBest = moves[0];
        for (int depth = 1; size > 1 && depth <= std::min(64, PLYCAP - rootPly) && !stopped && std::abs(score) < win; ++depth) {
            for (int window = depth < 4 ? infinity : 80; !stopped; window *= 4) {
                int alpha = std::max(score - window, -infinity), beta = std::min(score + window, infinity);
                int value = negamax(game.pos, depth, alpha, beta);
                if (stopped) break;
                score = value;
                if (value > alpha && value < beta) {
                    statistics.depth = depth;
                    statistics.score = score;
                    break;
                }
            }
        }
        statistics.seconds = static_cast<double>(std::clock() - start) / CLOCKS_PER_SEC;
        return rootBest;
    }
};

}

#endif

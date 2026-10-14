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
#include "patterns.h"
#include "directional.h"

namespace ataxx {

constexpr int infinity = 1000000, win = 100000;


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

ATAXX_HD inline int tableEvaluate(const Pos& p, const Weights* weights) {
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

ATAXX_HD inline void patternCodes(u64 own, u64 opp, int* codes) {
    const int ternary[8] = {0, 1, 3, 4, 9, 10, 12, 13};
    int rows[9][7]{};
    for (int y = 0; y < 7; ++y) {
        int a = (own >> (8 * y)) & 127, b = (opp >> (8 * y)) & 127;
        for (int x = 0; x < 7; ++x) {
            int aa = (x ? a >> (x - 1) : a << 1) & 7;
            int bb = (x ? b >> (x - 1) : b << 1) & 7;
            rows[y + 1][x] = ternary[aa] + 2 * ternary[bb];
        }
    }
    for (int y = 0; y < 7; ++y)
        for (int x = 0; x < 7; ++x)
            codes[y * 7 + x] = rows[y][x] + 27 * rows[y + 1][x] + 729 * rows[y + 2][x];
}

ATAXX_HD inline int patternEvaluate(const Pos& p, const PatternWeights& w) {
    u64 own = p.bb[p.side], opp = p.bb[p.side ^ 1], empty = ON & ~(own | opp);
    int phase = popcnt(empty);
    const int ternary[8] = {0, 1, 3, 4, 9, 10, 12, 13};
    int rows[9][7]{};
    int score = w.tempo[phase], columns[7]{}, rowCodes[7], power = 1;
    for (int y = 0; y < 7; ++y, power *= 3) {
        int a = (own >> (8 * y)) & 127, b = (opp >> (8 * y)) & 127;
        rowCodes[y] = T().ternaryRow[a] + 2 * T().ternaryRow[b];
        score += w.line[phase][std::min(y, 6 - y)][T().lineOrbit[rowCodes[y]]];
        for (int x = 0; x < 7; ++x) {
            int aa = (x ? a >> (x - 1) : a << 1) & 7;
            int bb = (x ? b >> (x - 1) : b << 1) & 7;
            rows[y + 1][x] = ternary[aa] + 2 * ternary[bb];
            columns[x] += (((a >> x) & 1) + 2 * ((b >> x) & 1)) * power;
            score += w.square[phase][y * 7 + x] * (((a >> x) & 1) - ((b >> x) & 1));
        }
    }
    for (int y = 0; y < 7; ++y)
        for (int x = 0; x < 7; ++x) {
            score += w.table[phase][rows[y][x] + 27 * rows[y + 1][x] + 729 * rows[y + 2][x]];
        }
    for (int x = 0; x < 7; ++x) score += w.line[phase][std::min(x, 6 - x)][T().lineOrbit[columns[x]]];
    for (int band = 0; band < 3; ++band) {
        int outside[4] = {rowCodes[band], columns[6 - band], rowCodes[6 - band], columns[band]};
        int inside[4] = {rowCodes[band + 1], columns[5 - band], rowCodes[5 - band], columns[band + 1]};
        for (int side = 0; side < 4; ++side)
            for (int j = 0; j < 4; ++j) {
                score += w.edge[phase][band][outside[side] % 81 + 81 * (inside[side] % 81)];
                outside[side] /= 3;
                inside[side] /= 3;
            }
    }
    u64 frontier = dilate1(empty);
    score += w.linear[phase][0] * (popcnt(own) - popcnt(opp));
    score += w.linear[phase][1] * (popcnt(dilate1(own) & empty) - popcnt(dilate1(opp) & empty));
    score += w.linear[phase][2] * (popcnt(dilate2(own) & empty) - popcnt(dilate2(opp) & empty));
    score += w.linear[phase][3] * (popcnt(frontier & own) - popcnt(frontier & opp));
    return score;
}
struct Entry {
    u64 own = 0, opp = 0, context = 0;
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

template<class H>
struct SearchCore {
    struct Frame {
        Move moves[NMOVES], killer;
        int scores[NMOVES];
    };
    Entry* table;
    int tableBits = 21;
    bool legacy = false;
    const Weights* evalWeights = nullptr;
    const PatternWeights* patterns = nullptr;
    const DirectionalWeights* directional = nullptr;
    std::uint64_t nodeLimit = 0;
    Frame* frames;
    Pos* samples = nullptr;
    int sampleCount = 0;
    std::uint64_t sampleSeen = 0;
    int cutoffs[NMOVES]{};
    int reductions[64][64];
    u64 contexts[PLYCAP + 1];
    int rootPly = 0;
    std::clock_t deadline;
    SearchStats statistics;
    bool stopped = false;
    Move rootBest;
    H hist;
    std::uint8_t age = 0;

    ATAXX_HD int negamax(const Pos& p, int depth, int alpha, int beta, bool allowNull = true, bool synthetic = false, Move excluded = NMOVES) {
        ++statistics.nodes;
        if (nodeLimit && statistics.nodes > nodeLimit) stopped = true;
#ifndef __CUDA_ARCH__
        if (!nodeLimit && (statistics.nodes & 1023) == 0 && std::clock() >= deadline) stopped = true;
#endif
        if (stopped) return 0;
        u64 own = p.bb[p.side], opp = p.bb[p.side ^ 1];
        Outcome o;
        if (terminal(p, o)) {
            int margin = o.count[p.side] - o.count[p.side ^ 1];
            return margin > 0 ? win + margin : margin < 0 ? margin - win : 0;
        }
        if (!legacy && !synthetic) {
            int k = hist.ply();
            for (int i = hist.lastClone[k]; i <= k - 2; ++i)
                if (hist.s[i] == hist.s[k]) {
                    if (!genLegal(p, hist, frames[p.ply - rootPly].moves)) return -win - 49;
                    break;
                }
        }
        if (depth <= 0) return evaluate(p);
        bool root = p.ply == rootPly, pv = beta > alpha + 1, exact = p.ply + depth < PLYCAP;
        u64 context = 0;
        if (!legacy && !synthetic) {
            context = mix(contexts[hist.ply()], p.ply + 1);
        }
        Entry& cached = table[(own * 0x9e3779b97f4a7c15ULL ^ opp) * 0xbf58476d1ce4e5b9ULL >> (64 - tableBits)];
        bool boardHit = cached.own == own && cached.opp == opp;
        bool hit = boardHit && (legacy || (!synthetic && cached.context == context));
        if (excluded == NMOVES && hit && exact && !root && cached.depth >= depth
            && (!cached.bound || (cached.bound > 0 && cached.score >= beta) || (cached.bound < 0 && cached.score <= alpha)))
            return cached.score;
        if (excluded == NMOVES && !pv && beta < win && alpha > -win) {
            int standing = evaluate(p);
            if (depth <= 6 && standing - 120 * depth >= beta) return standing;
            if (allowNull && depth >= 3 && standing >= beta && popcnt(ON & ~(own | opp)) >= 10 && popcnt(own) >= 4 && popcnt(opp) >= 4) {
                Pos pass = p;
                pass.side ^= 1;
                pass.ply++;
                int score = -negamax(pass, depth - 3 - depth / 6, -beta, -beta + 1, false, !legacy || synthetic);
                if (stopped) return 0;
                if (score >= beta) return score;
            }
        }
        Move preferred = root ? rootBest : (legacy ? hit : boardHit) ? cached.move : NMOVES;
        Move singularMove = NMOVES;
        if (!legacy && excluded == NMOVES && !root && !synthetic && depth >= 8 && p.ply - rootPly + depth < 64 && hit && cached.bound >= 0 && cached.depth >= depth - 3 && std::abs(cached.score) < win && preferred != NMOVES) {
            int bound = cached.score - 4 * depth;
            int alternative = negamax(p, (depth - 1) / 2, bound - 1, bound, false, false, preferred);
            if (stopped) return 0;
            if (alternative < bound) singularMove = preferred;
        }
        if (excluded != NMOVES) preferred = NMOVES;
        Frame& frame = frames[p.ply - rootPly];
        int size = synthetic ? genGeometric(p, frame.moves) : genLegal(p, hist, frame.moves);
        if (excluded != NMOVES)
            for (int i = 0; i < size; ++i)
                if (frame.moves[i] == excluded) frame.moves[i--] = frame.moves[--size];
        if (!size) return -win - 49;
        if (!legacy) {
            bool found = false;
            for (int i = 0; i < size; ++i) found |= frame.moves[i] == preferred;
            if (!found) preferred = NMOVES;
        }
        if (excluded == NMOVES && preferred == NMOVES && depth >= 4) --depth;
        const Tables& t = T();
        int codes[SQ], phase = popcnt(ON & ~(own | opp)) * 6 / 50;
        int preferredIndex = -1;
        for (int i = 0; i < size; ++i) {
            frame.scores[i] = cutoffs[frame.moves[i]];
            if (frame.moves[i] == preferred) preferredIndex = i;
        }
        bool ordered = false;
        int originalAlpha = alpha, best = -infinity;
        Move bestMove = preferred;
        for (int i = 0; i < size; ++i) {
            int next = i;
            if (!i && preferredIndex >= 0) next = preferredIndex;
            else {
                if (!ordered) {
                    if (!legacy) patternCodes(own, opp, codes);
                    for (int j = i; j < size; ++j) {
                        Move move = frame.moves[j];
                        u64 ring = t.ring1[move % SQ];
                        int score = 1000 * popcnt(ring & opp) + frame.scores[j];
                        score += move < SQ ? 500 + 100 * popcnt(ring & own) : -200 * popcnt(t.ring1[sqOfBit(t.jumpOrigin[move % SQ][move / SQ - 1])] & own);
                        if (!legacy) {
                            int to = move % SQ, kind = move / SQ, family = t.moveFamily[kind], transform = t.moveTransform[kind];
                            int code = t.patternSym[transform][codes[to]];
                            score = directional->destination[family][code % 81 + code / 243 * 81] + directional->square[family][to] + directional->bias[phase][family] + frame.scores[j];
                            if (kind) {
                                int source = t.patternSym[transform][codes[sqOfBit(t.jumpOrigin[to][kind - 1])]];
                                score += directional->source[family - 1][source % 81 + source / 243 * 81];
                            }
                        }
                        if (move == frame.killer) score += 400;
                        frame.scores[j] = score;
                    }
                    ordered = true;
                }
                for (int j = i + 1; j < size; ++j) if (frame.scores[j] > frame.scores[next]) next = j;
            }
            int savedScore = frame.scores[i];
            frame.scores[i] = frame.scores[next];
            frame.scores[next] = savedScore;
            Move savedMove = frame.moves[i];
            frame.moves[i] = frame.moves[next];
            frame.moves[next] = savedMove;
            Move move = frame.moves[i];
            if (!pv && depth <= 2 && i >= 6 * depth && best > -win) break;
            Pos child = p;
            hist.push(p, move);
            int k = hist.ply();
            if (!legacy) contexts[k] = mix((move < SQ ? 0 : contexts[k - 1]) ^ hist.s[k].b[0], hist.s[k].b[1]);
            play(child, move);
            int score, nextDepth = depth - 1 + (move == singularMove);
            if (!i || depth == 1) score = -negamax(child, nextDepth, -beta, -alpha, true, synthetic);
            else {
                int reduction = depth >= 3 ? std::max(0, std::min(reductions[std::min(depth, 63)][std::min(i, 63)] - pv, depth - 2)) : 0;
                score = -negamax(child, nextDepth - reduction, -alpha - 1, -alpha, true, synthetic);
                if (reduction && score > alpha) score = -negamax(child, nextDepth, -alpha - 1, -alpha, true, synthetic);
                if (score > alpha && score < beta) score = -negamax(child, nextDepth, -beta, -alpha, true, synthetic);
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
        if (excluded == NMOVES && !synthetic && (hit || cached.age != age || cached.depth <= depth + 2))
            cached = {own, opp, context, best, bestMove, static_cast<std::int8_t>(exact ? depth : 0),
                static_cast<std::int8_t>(best <= originalAlpha ? -1 : best >= beta ? 1 : 0), age};
        return best;
    }
    ATAXX_HD void init() {
        for (int depth = 1; depth < 64; ++depth)
            for (int i = 1; i < 64; ++i)
                reductions[depth][i] = static_cast<int>(0.5 + log(double(depth)) * log(double(i)) / 1.5);
    }
    ATAXX_HD int evaluate(const Pos& p) {
        if (samples && statistics.nodes % 251 == 0) {
            ++sampleSeen;
            std::uint64_t index = sampleSeen <= 4096 ? sampleSeen - 1 : mix(sampleSeen, 0x9e3779b97f4a7c15ULL) % sampleSeen;
            if (index < 4096) samples[index] = p;
            sampleCount = std::min(sampleSeen, std::uint64_t(4096));
        }
        int score = legacy ? tableEvaluate(p, evalWeights) : patternEvaluate(p, *patterns);
        return score;
    }
    ATAXX_HD Move choose(const Pos& pos, const H& history, std::clock_t start, double seconds, std::uint64_t nodes = 0, int maxDepth = 64) {
        statistics = {};
        rootPly = pos.ply;
        hist = history;
        if (!legacy)
            for (int i = 0; i <= hist.ply(); ++i)
                contexts[i] = mix((i == hist.lastClone[i] ? 0 : contexts[i - 1]) ^ hist.s[i].b[0], hist.s[i].b[1]);
        nodeLimit = nodes;
        deadline = start + static_cast<std::clock_t>(seconds * CLOCKS_PER_SEC);
        stopped = false;
        ++age;
        for (int& value : cutoffs) value /= 4;
        Move moves[NMOVES]{};
        Outcome outcome;
        int size = terminal(pos, outcome) ? 0 : genLegal(pos, hist, moves), score = 0;
        rootBest = moves[0];
        for (int depth = 1; size > 1 && depth <= std::min(maxDepth, PLYCAP - rootPly) && !stopped && std::abs(score) < win; ++depth) {
            for (int window = depth < 4 ? infinity : 80; !stopped; window *= 4) {
                int alpha = std::max(score - window, -infinity), beta = std::min(score + window, int(infinity));
                int value = negamax(pos, depth, alpha, beta);
                if (stopped) break;
                score = value;
                if (value > alpha && value < beta) {
                    statistics.depth = depth;
                    statistics.score = score;
                    break;
                }
            }
        }
#ifndef __CUDA_ARCH__
        statistics.seconds = static_cast<double>(std::clock() - start) / CLOCKS_PER_SEC;
#endif
        return rootBest;
    }
};

class Search {
    std::vector<Entry> table = std::vector<Entry>(1 << 21);
    std::vector<SearchCore<History>::Frame> frames = std::vector<SearchCore<History>::Frame>(66);
public:
    SearchCore<History> core;
    Search() {
        core.evalWeights = weights;
        core.patterns = &patternWeights();
        core.directional = &directionalWeights;
        core.table = table.data();
        core.frames = frames.data();
        core.init();
    }
    const SearchStats& stats() const { return core.statistics; }
    Move choose(const Game& game, std::clock_t start, double seconds) {
        return core.choose(game.pos, game.hist, start, seconds);
    }
};

}

#endif

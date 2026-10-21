#ifndef ATAXX_ENGINE_HPP
#define ATAXX_ENGINE_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <vector>
#include "board.h"
#include "patterns.h"
#include "wide.h"
#include "directional.h"

namespace ataxx {

constexpr int infinity = 1000000, win = 100000;


struct Game {
    Pos pos = start();
    History hist;

    Game() { hist.reset(pos); }

    void play(Move move) {
        hist.push(pos, move);
        ataxx::play(pos, move);
    }
};

inline void patternCodes(u64 own, u64 opp, int* codes) {
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

inline u64 transposeBoard(u64 b) {
    u64 t = (b ^ (b >> 7)) & 0x00aa00aa00aa00aaULL;
    b ^= t ^ (t << 7);
    t = (b ^ (b >> 14)) & 0x0000cccc0000ccccULL;
    b ^= t ^ (t << 14);
    t = (b ^ (b >> 28)) & 0x00000000f0f0f0f0ULL;
    return b ^ t ^ (t << 28);
}

struct ExpandedPatternWeights {
    std::int16_t table[46][25][19683], line[46][4][2187];
};

inline const ExpandedPatternWeights& expandedPatternWeights() {
    static const auto* weights = [] {
        auto* result = new ExpandedPatternWeights;
        const auto& w = patternWeights();
        const auto& t = T();
        const int regions[25] = {0,1,2,1,0,1,3,4,3,1,2,4,5,4,2,1,3,4,3,1,0,1,2,1,0};
        const int transforms[25] = {0,0,0,1,1,4,0,0,1,5,4,4,0,5,5,6,2,2,3,7,2,2,2,3,3};
        for (int phase = 0; phase < 46; ++phase) {
            for (int square = 0; square < 25; ++square)
                for (int code = 0; code < 19683; ++code)
                    result->table[phase][square][code] = w.table[phase][regions[square]][t.patternSym[transforms[square]][code]];
            for (int region = 0; region < 4; ++region)
                for (int code = 0; code < 2187; ++code)
                    result->line[phase][region][code] = w.line[phase][region][t.lineOrbit[code]];
        }
        return result;
    }();
    return *weights;
}

inline int patternEvaluate(const Pos& p, const PatternWeights& w, const ExpandedPatternWeights* expanded, const WideWeights* wide) {
    u64 own = p.bb[p.side], opp = p.bb[p.side ^ 1], empty = ON & ~(own | opp);
    int phase = popcnt(empty);
    u64 rows[9][2]{}, wideRows[7][3];
    int score = w.tempo[phase], columns[7], rowCodes[7];
    u64 ownT = transposeBoard(own), oppT = transposeBoard(opp);
    for (int y = 0; y < 7; ++y) {
        int a = (own >> (8 * y)) & 127, b = (opp >> (8 * y)) & 127;
        rowCodes[y] = T().ternaryRow[a] + 2 * T().ternaryRow[b];
        for (int k = 0; k < 3; ++k) wideRows[y][k] = wide->rows[a][k] + 2 * wide->rows[b][k];
        score += expanded->line[phase][std::min(y, 6 - y)][rowCodes[y]];
        for (int j = 0; j < 2; ++j) rows[y + 1][j] = T().neighborRow[a][j] + 2 * T().neighborRow[b][j];
        columns[y] = T().ternaryRow[(ownT >> (8 * y)) & 127] + 2 * T().ternaryRow[(oppT >> (8 * y)) & 127];
    }
    for (int y = 1; y < 6; ++y) {
        u64 a = rows[y][0] + 27 * rows[y + 1][0] + 729 * rows[y + 2][0];
        u64 b = rows[y][1] + 27 * rows[y + 1][1] + 729 * rows[y + 2][1];
        for (int x = 1; x < 3; ++x) score += expanded->table[phase][5 * (y - 1) + 2 * x - 1][(a >> (16 * x)) & 65535];
        for (int x = 0; x < 3; ++x) score += expanded->table[phase][5 * (y - 1) + 2 * x][(b >> (16 * x)) & 65535];
    }
    for (int x = 0; x < 7; ++x) score += expanded->line[phase][std::min(x, 6 - x)][columns[x]];
    for (int band = 0; band < 3; ++band) {
        int outside[4] = {rowCodes[band], columns[6 - band], rowCodes[6 - band], columns[band]};
        int inside[4] = {rowCodes[band + 1], columns[5 - band], rowCodes[5 - band], columns[band + 1]};
        for (int side = 0; side < 4; ++side) {
            u64 codes = T().stripRow[outside[side]] + 81 * T().stripRow[inside[side]];
            for (int j = 0; j < 4; ++j) score += w.edge[phase][band ? band + 3 : j][(codes >> (16 * j)) & 65535];
        }
    }
    for (int y = 0; y < 3; ++y) {
        u64 codes = wideRows[y][0] + 27 * wideRows[y+1][1] + 243 * wideRows[y+2][2] + 2187 * wideRows[y+3][1] + 19683 * wideRows[y+4][0];
        for (int x = 0; x < 3; ++x) score += wide->table[phase][wide->map[(codes >> (20*x)) & 1048575]];
    }
    u64 frontier = dilate1(empty);
    score += w.linear[phase][1] * (popcnt(dilate1(own) & empty) - popcnt(dilate1(opp) & empty));
    score += w.linear[phase][2] * (popcnt(dilate2(own) & empty) - popcnt(dilate2(opp) & empty));
    score += w.linear[phase][3] * (popcnt(frontier & own) - popcnt(frontier & opp));
    return score / 2;
}
struct alignas(32) Entry {
    u64 own = 0, opp = 0, context = 0;
    std::int32_t score : 21, bound : 2;
    std::uint32_t age : 8;
    Move move = NMOVES;
    std::int8_t depth = -1;
};

struct EvalEntry {
    u64 own = 0, opp = 0;
    int score = 0;
};

struct SearchStats {
    std::uint64_t nodes = 0;
    int depth = 0, score = 0;
    double seconds = 0;
};

struct ExpandedDirectionalWeights {
    std::int16_t destination[KINDS][19683], source[KINDS - 1][19683];
};

inline const ExpandedDirectionalWeights& expandedDirectionalWeights() {
    static const ExpandedDirectionalWeights weights = [] {
        ExpandedDirectionalWeights result;
        const Tables& t = T();
        for (int kind = 0; kind < KINDS; ++kind)
            for (int code = 0; code < 19683; ++code) {
                int oriented = t.patternSym[t.moveTransform[kind]][code];
                int index = oriented % 81 + oriented / 243 * 81;
                int family = t.moveFamily[kind];
                result.destination[kind][code] = directionalWeights.destination[family][index];
                if (kind) result.source[kind - 1][code] = directionalWeights.source[family - 1][index];
            }
        return result;
    }();
    return weights;
}

struct OrderTree {
    int size;
    std::int64_t values[2048];
};

struct SearchCore {
    struct Frame {
        Move moves[NMOVES], killer;
        int scores[NMOVES];
    };
    Entry* table;
    EvalEntry* evalTable;
    int tableBits = 21;
    const PatternWeights* patterns = nullptr;
    const WideWeights* wide = nullptr;
    const ExpandedPatternWeights* expandedPatterns = nullptr;
    const DirectionalWeights* directional = nullptr;
    const ExpandedDirectionalWeights* expandedDirectional = nullptr;
    Frame* frames;
    OrderTree* orders = nullptr;
    int cutoffs[NMOVES]{};
    int reductions[64][64];
    u64 contexts[PLYCAP + 1];
    int rootPly = 0;
    std::clock_t deadline;
    SearchStats statistics;
    bool stopped = false;
    Move rootBest;
    History hist;
    std::uint8_t age = 0;

    int negamax(const Pos& p, int depth, int alpha, int beta, bool allowNull = true, bool synthetic = false, Move excluded = NMOVES) {
        ++statistics.nodes;
        if ((statistics.nodes & 1023) == 0 && std::clock() >= deadline) stopped = true;
        if (stopped) return 0;
        u64 own = p.bb[p.side], opp = p.bb[p.side ^ 1];
        Outcome o;
        if (terminal(p, o)) {
            int margin = o.count[p.side] - o.count[p.side ^ 1];
            return margin > 0 ? win + margin : margin < 0 ? margin - win : 0;
        }
        if (!synthetic) {
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
        if (!synthetic) context = mix(contexts[hist.ply()], p.ply + 1);
        Entry& cached = table[(own * 0x9e3779b97f4a7c15ULL ^ opp) * 0xbf58476d1ce4e5b9ULL >> (64 - tableBits)];
        bool boardHit = cached.own == own && cached.opp == opp;
        bool hit = boardHit && !synthetic && cached.context == context;
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
                int score = -negamax(pass, depth - 3 - depth / 6, -beta, -beta + 1, false, true);
                if (stopped) return 0;
                if (score >= beta) return score;
            }
        }
        Move preferred = root ? rootBest : boardHit ? cached.move : NMOVES;
        Move singularMove = NMOVES;
        if (excluded == NMOVES && !root && !synthetic && depth >= 8 && p.ply - rootPly + depth < 64 && hit && cached.bound >= 0 && cached.depth >= depth - 3 && std::abs(cached.score) < win && preferred != NMOVES) {
            int bound = cached.score - 4 * depth;
            int alternative = negamax(p, (depth - 1) / 2, bound - 1, bound, false, false, preferred);
            if (stopped) return 0;
            if (alternative < bound) singularMove = preferred;
            else if (!pv && bound >= beta && std::abs(alternative) < win) return alternative;
        }
        if (excluded != NMOVES) preferred = NMOVES;
        Frame& frame = frames[p.ply - rootPly];
        int size = synthetic ? genGeometric(p, frame.moves) : genLegal(p, hist, frame.moves);
        if (excluded != NMOVES)
            for (int i = 0; i < size; ++i)
                if (frame.moves[i] == excluded) frame.moves[i--] = frame.moves[--size];
        if (!size) return -win - 49;
        bool found = false;
        for (int i = 0; i < size; ++i) found |= frame.moves[i] == preferred;
        if (!found) preferred = NMOVES;
        if (excluded == NMOVES && preferred == NMOVES && depth >= 4) --depth;
        const Tables& t = T();
        int codes[SQ], phase = popcnt(ON & ~(own | opp)) * 6 / 50;
        int preferredIndex = -1;
        for (int i = 0; i < size; ++i) {
            frame.scores[i] = cutoffs[frame.moves[i]];
            if (frame.moves[i] == preferred) preferredIndex = i;
        }
        OrderTree& order = orders[p.ply - rootPly];
        bool ordered = false;
        int originalAlpha = alpha, best = -infinity;
        Move bestMove = preferred;
        for (int i = 0; i < size; ++i) {
            int next = i;
            if (!i && preferredIndex >= 0) next = preferredIndex;
            else {
                if (!ordered) {
                    patternCodes(own, opp, codes);
                    for (int j = i; j < size; ++j) {
                        Move move = frame.moves[j];
                        int to = move % SQ, kind = move / SQ, family = t.moveFamily[kind];
                        int score = expandedDirectional->destination[kind][codes[to]];
                        if (kind) score += expandedDirectional->source[kind - 1][codes[sqOfBit(t.jumpOrigin[to][kind - 1])]];
                        score += directional->square[family][to] + directional->bias[phase][family] + frame.scores[j];
                        if (move == frame.killer) score += 400;
                        frame.scores[j] = score;
                    }
                    order.size = 1;
                    while (order.size < size) order.size *= 2;
                    for (int j = 0; j < order.size; ++j)
                        order.values[order.size + j] = j >= i && j < size ? std::int64_t(frame.scores[j]) * 1024 + 1023 - j : -(1LL << 60);
                    for (int j = order.size - 1; j; --j) order.values[j] = std::max(order.values[2 * j], order.values[2 * j + 1]);
                    ordered = true;
                }
                next = 1023 - int(order.values[1] & 1023);
            }
            int savedScore = frame.scores[i];
            frame.scores[i] = frame.scores[next];
            frame.scores[next] = savedScore;
            Move savedMove = frame.moves[i];
            frame.moves[i] = frame.moves[next];
            frame.moves[next] = savedMove;
            if (ordered) {
                order.values[order.size + next] = std::int64_t(frame.scores[next]) * 1024 + 1023 - next;
                order.values[order.size + i] = -(1LL << 60);
                for (int j = (order.size + i) / 2; j; j /= 2) order.values[j] = std::max(order.values[2 * j], order.values[2 * j + 1]);
                for (int j = (order.size + next) / 2; j; j /= 2) order.values[j] = std::max(order.values[2 * j], order.values[2 * j + 1]);
            }
            Move move = frame.moves[i];
            if (!pv && depth <= 2 && i >= 6 * depth && best > -win) break;
            Pos child = p;
            hist.push(p, move);
            int k = hist.ply();
            contexts[k] = mix((move < SQ ? 0 : contexts[k - 1]) ^ hist.s[k].b[0], hist.s[k].b[1]);
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
            cached = {own, opp, context, best, best <= originalAlpha ? -1 : best >= beta ? 1 : 0, age,
                bestMove, static_cast<std::int8_t>(exact ? depth : 0)};
        return best;
    }
    void init() {
        for (int depth = 1; depth < 64; ++depth)
            for (int i = 1; i < 64; ++i)
                reductions[depth][i] = static_cast<int>(0.5 + log(double(depth)) * log(double(i)) / 1.5);
    }
    int evaluate(const Pos& p) {
        u64 own = p.bb[p.side], opp = p.bb[p.side ^ 1];
        EvalEntry& entry = evalTable[((own * 0x9e3779b97f4a7c15ULL ^ opp) * 0xbf58476d1ce4e5b9ULL) >> 46];
        if (entry.own == own && entry.opp == opp) return entry.score;
        int score = patternEvaluate(p, *patterns, expandedPatterns, wide);
        entry = {own, opp, score};
        return score;
    }
    Move choose(const Pos& pos, const History& history, std::clock_t start, double seconds) {
        statistics = {};
        rootPly = pos.ply;
        hist = history;
        for (int i = 0; i <= hist.ply(); ++i)
            contexts[i] = mix((i == hist.lastClone[i] ? 0 : contexts[i - 1]) ^ hist.s[i].b[0], hist.s[i].b[1]);
        deadline = start + static_cast<std::clock_t>(seconds * CLOCKS_PER_SEC);
        stopped = false;
        ++age;
        for (int& value : cutoffs) value /= 4;
        Move moves[NMOVES]{};
        Outcome outcome;
        int size = terminal(pos, outcome) ? 0 : genLegal(pos, hist, moves), score = 0;
        rootBest = moves[0];
        for (int depth = 1; size > 1 && depth <= std::min(64, PLYCAP - rootPly) && !stopped && std::abs(score) < win; ++depth) {
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
        statistics.seconds = static_cast<double>(std::clock() - start) / CLOCKS_PER_SEC;
        return rootBest;
    }
};

class Search {
    std::vector<Entry> table = std::vector<Entry>(1 << 21);
    std::vector<EvalEntry> evaluations = std::vector<EvalEntry>(1 << 18);
    std::vector<SearchCore::Frame> frames = std::vector<SearchCore::Frame>(66);
    std::vector<OrderTree> orders = std::vector<OrderTree>(66);
public:
    SearchCore core;
    Search() {
        core.patterns = &patternWeights();
        core.wide = &wideWeights();
        core.expandedPatterns = &expandedPatternWeights();
        core.directional = &directionalWeights;
        core.expandedDirectional = &expandedDirectionalWeights();
        core.table = table.data();
        core.evalTable = evaluations.data();
        core.frames = frames.data();
        core.orders = orders.data();
        core.init();
    }
    const SearchStats& stats() const { return core.statistics; }
    Move choose(const Game& game, std::clock_t start, double seconds) {
        return core.choose(game.pos, game.hist, start, seconds);
    }
};

}

#endif

#include "engine.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <chrono>
#include <stdexcept>

namespace ataxx {
__device__ const Tables* deviceTables;

struct FixedHistory {
    Snap s[PLYCAP + 1];
    int lastClone[PLYCAP + 1];
    int count;
    ATAXX_HD void reset(const Pos& p) {
        count = 1;
        s[0] = {{p.bb[0], p.bb[1]}};
        lastClone[0] = 0;
    }
    ATAXX_HD int ply() const { return count - 1; }
    ATAXX_HD void push(const Pos& p, Move m) {
        placement(p, m, s[count].b);
        lastClone[count] = m < SQ ? count : lastClone[count - 1];
        ++count;
    }
    ATAXX_HD void pop() { --count; }
};

using Core = SearchCore<FixedHistory>;

struct ArenaGame {
    Pos pos;
    FixedHistory history;
    int result = 2;
    int lastMove = -1, score = 0, depth = 0;
    unsigned long long nodes = 0;
};

__global__ void initialize(Core* cores, Entry* entries, Core::Frame* frames, const Weights* w, const PatternWeights* p, const PatternWeights* pb, const DirectionalWeights* directional, const WideWeights* wide, int a, int b, Pos* samples, int count, int bits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    Core& c = cores[i];
    c.table = entries + (size_t(i) << bits);
    c.frames = frames + size_t(i) * 66;
    c.evalWeights = w;
    c.patterns = (i & 1) == ((i / 2) & 1) ? p : pb;
    c.directional = directional;
    c.wide = (((i & 1) == ((i / 2) & 1)) ? a : b) == 2 ? nullptr : wide;
    c.tableBits = bits;
    c.samples = samples ? samples + size_t(i) * 4096 : nullptr;
    c.init();
}

__global__ void step(Core* cores, ArenaGame* games, int count, int a, int b, unsigned long long na, unsigned long long nb, int maxDepth) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count || games[i].result != 2) return;
    ArenaGame& g = games[i];
    Core& c = cores[2 * i + g.pos.side];
    bool candidate = g.pos.side == (i & 1);
    c.legacy = (candidate ? a : b) == 0;
    Move m = c.choose(g.pos, g.history, 0, 0, candidate ? na : nb, maxDepth);
    g.lastMove = m;
    g.score = c.statistics.score;
    g.depth = c.statistics.depth;
    g.nodes += c.statistics.nodes;
    g.history.push(g.pos, m);
    play(g.pos, m);
    Outcome o;
    if (terminal(g.pos, o)) g.result = o.winner;
    else {
        Move legal[NMOVES];
        if (!genLegal(g.pos, g.history, legal)) g.result = g.pos.side ^ 1;
    }
}
}

void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

template<class T> T* allocate(size_t n) {
    T* p;
    check(cudaMalloc(&p, n * sizeof(T)));
    check(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}

int main(int argc, char** argv) {
    using namespace ataxx;
    std::ifstream input(argv[1], std::ios::binary);
    int count, a = atoi(argv[3]), b = atoi(argv[4]), bits = atoi(argv[7]);
    unsigned long long na = strtoull(argv[5], nullptr, 10), nb = strtoull(argv[6], nullptr, 10);
    int maxTurns = argc > 8 ? atoi(argv[8]) : PLYCAP;
    int maxDepth = argc > 9 ? atoi(argv[9]) : 64;
    input.read(reinterpret_cast<char*>(&count), sizeof(count));
    std::vector<ArenaGame> host(count);
    for (auto& g : host) {
        g.pos = start();
        g.history.reset(g.pos);
        int length;
        input.read(reinterpret_cast<char*>(&length), sizeof(length));
        for (int j = 0; j < length; ++j) {
            Move m;
            input.read(reinterpret_cast<char*>(&m), sizeof(m));
            g.history.push(g.pos, m);
            play(g.pos, m);
        }
    }
    check(cudaDeviceSetLimit(cudaLimitStackSize, 65536));
    auto tables = allocate<Tables>(1);
    check(cudaMemcpy(tables, &T(), sizeof(Tables), cudaMemcpyHostToDevice));
    check(cudaMemcpyToSymbol(deviceTables, &tables, sizeof(tables)));
    auto w = allocate<Weights>(PHASES);
    auto p = allocate<PatternWeights>(1);
    auto pb = allocate<PatternWeights>(1);
    auto directional = allocate<DirectionalWeights>(1);
    check(cudaMemcpy(directional, &directionalWeights, sizeof(DirectionalWeights), cudaMemcpyHostToDevice));
    check(cudaMemcpy(w, weights, sizeof(weights), cudaMemcpyHostToDevice));
    std::vector<char> model(sizeof(PatternWeights));
    std::memcpy(model.data(), &patternWeights(), model.size());
    if (argc > 10) {
        std::ifstream file(argv[10], std::ios::binary);
        file.read(model.data(), model.size());
        if (!file) throw std::runtime_error("Invalid pattern weights");
    }
    check(cudaMemcpy(p, model.data(), model.size(), cudaMemcpyHostToDevice));
    if (argc > 11) {
        std::ifstream file(argv[11], std::ios::binary);
        file.read(model.data(), model.size());
        if (!file) throw std::runtime_error("Invalid opponent pattern weights");
    }
    check(cudaMemcpy(pb, model.data(), model.size(), cudaMemcpyHostToDevice));
    auto wide = allocate<WideWeights>(1);
    check(cudaMemcpy(wide, &wideWeights(), sizeof(WideWeights), cudaMemcpyHostToDevice));
    auto cores = allocate<Core>(2 * count);
    auto entries = allocate<Entry>(size_t(2 * count) << bits);
    auto frames = allocate<Core::Frame>(size_t(2 * count) * 66);
    auto games = allocate<ArenaGame>(count);
    auto samples = getenv("ATAXX_SAMPLE") ? allocate<Pos>(size_t(2 * count) * 4096) : nullptr;
    check(cudaMemcpy(games, host.data(), count * sizeof(ArenaGame), cudaMemcpyHostToDevice));
    initialize<<<(2 * count + 31) / 32, 32>>>(cores, entries, frames, w, p, pb, directional, wide, a, b, samples, 2 * count, bits);
    check(cudaDeviceSynchronize());
    int lanes = getenv("ATAXX_LANES") ? atoi(getenv("ATAXX_LANES")) : 1;
    auto start = std::chrono::steady_clock::now();
    std::ofstream trace(std::string(argv[2]) + ".moves");
    for (int turn = 0; turn < maxTurns; ++turn) {
        step<<<(count + lanes - 1) / lanes, lanes>>>(cores, games, count, a, b, na, nb, maxDepth);
        check(cudaDeviceSynchronize());
        check(cudaMemcpy(host.data(), games, count * sizeof(ArenaGame), cudaMemcpyDeviceToHost));
        int active = 0, wins = 0, losses = 0, draws = 0;
        unsigned long long nodes = 0;
        for (int i = 0; i < count; ++i) {
            auto& g = host[i];
            nodes += g.nodes;
            active += g.result == 2;
            wins += g.result == (i & 1);
            losses += g.result >= 0 && g.result != 2 && g.result != (i & 1);
            draws += g.result == -1;
            trace << turn << ' ' << i << ' ' << g.lastMove << ' ' << g.score << ' ' << g.depth << ' ' << g.result << '\n';
        }
        if (turn % 16 == 0 || !active || turn + 1 == maxTurns) {
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            printf("{\"turn\":%d,\"active\":%d,\"wins\":%d,\"losses\":%d,\"draws\":%d,\"nodes\":%llu,\"seconds\":%.3f}\n", turn + 1, active, wins, losses, draws, nodes, seconds);
            fflush(stdout);
        }
        if (!active) break;
    }
    if (samples) {
        std::vector<Core> state(2 * count);
        check(cudaMemcpy(state.data(), cores, state.size() * sizeof(Core), cudaMemcpyDeviceToHost));
        std::vector<Pos> positions(4096);
        std::ofstream data(std::string(argv[2]) + ".positions", std::ios::binary);
        for (auto& c : state) {
            check(cudaMemcpy(positions.data(), c.samples, c.sampleCount * sizeof(Pos), cudaMemcpyDeviceToHost));
            data.write(reinterpret_cast<char*>(positions.data()), c.sampleCount * sizeof(Pos));
        }
    }
    std::ofstream output(argv[2]);
    for (int i = 0; i < count; ++i) output << i << ' ' << host[i].result << ' ' << host[i].pos.ply << '\n';
}

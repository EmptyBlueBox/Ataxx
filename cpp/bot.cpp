#include "engine.hpp"
#include <iostream>

int main(int argc, char** argv) {
    std::clock_t start = std::clock();
    double scale = argc > 1 ? std::atof(argv[1]) : 1;
    ataxx::Game game;
    ataxx::Search search;
    int turns, x0, y0, x1, y1;
    std::cin >> turns;
    for (int i = 0; i < turns * 2 - 1; ++i) {
        std::cin >> x0 >> y0 >> x1 >> y1;
        if (x0 >= 0) game.play(ataxx::fromCoord(x0, y0, x1, y1));
    }
    for (;;) {
        ataxx::Move move = search.choose(game, start, (game.pos.ply < 2 ? 1.990 : 0.995) * scale);
        ataxx::Coord c = ataxx::toCoord(game.pos, move);
        const auto& stats = search.stats();
        std::cout << c.x0 << ' ' << c.y0 << ' ' << c.x1 << ' ' << c.y1 << "\nd" << stats.depth << " s" << stats.score << " n" << stats.nodes
                  << " t" << stats.seconds << "\n\n\n>>>BOTZONE_REQUEST_KEEP_RUNNING<<<" << std::endl;
        game.play(move);
        if (!(std::cin >> x0 >> y0 >> x1 >> y1)) break;
        start = std::clock();
        game.play(ataxx::fromCoord(x0, y0, x1, y1));
    }
}

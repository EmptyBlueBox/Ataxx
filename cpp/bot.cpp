#include "bot_io.hpp"
#include <iostream>

int main() {
    std::clock_t start = std::clock();
    ataxx::initialize();
    ataxx::Game game = ataxx::readGame(std::cin);
    ataxx::Search search(start, game.ply() < 2 ? 1.990 : 0.995, game.history);
    ataxx::Move move = search.choose(game.position, game.side(), game.ply());
    ataxx::writeMove(std::cout, move);
}

#ifndef ATAXX_BOT_IO_HPP
#define ATAXX_BOT_IO_HPP

#include "engine.hpp"
#include <istream>
#include <ostream>

namespace ataxx {

inline Game readGame(std::istream& input) {
    Game game;
    int turns;
    input >> turns;
    for (int i = 0; i < turns * 2 - 1; ++i) {
        int x0, y0, x1, y1;
        input >> x0 >> y0 >> x1 >> y1;
        if (i == 0 && x0 == -1) continue;
        Move move{x0 * 7 + y0, x1 * 7 + y1};
        game.play(move);
    }
    return game;
}

inline void writeMove(std::ostream& output, Move move) {
    if (move.from < 0) output << "-1 -1 -1 -1\n";
    else output << move.from / 7 << ' ' << move.from % 7 << ' ' << move.to / 7 << ' ' << move.to % 7 << '\n';
}

}

#endif

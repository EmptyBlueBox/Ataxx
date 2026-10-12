#ifndef ATAXX_SESSION_HPP
#define ATAXX_SESSION_HPP

#include "engine.hpp"
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>

namespace ataxx {

struct BoardLayout {
    float x, y, cell;

    int square(float mouseX, float mouseY) const {
        float column = (mouseX - x) / cell, row = (mouseY - y) / cell;
        if (column < 0 || column >= 7 || row < 0 || row >= 7) return -1;
        return static_cast<int>(row) * 7 + static_cast<int>(column);
    }
};

struct Session {
    Game game;
    bool singlePlayer = true;
    int humanSide = 0;
    int selected = -1;

    bool humanTurn() const { return !singlePlayer || game.pos.side == humanSide; }

    void play(Move move) {
        game.play(move);
        selected = -1;
    }

    void click(int square) {
        if (square < 0 || !humanTurn() || game.winner() != 2) return;
        if (game.pos.bb[game.pos.side] >> bitOfSq(square) & 1) selected = square;
        else if (selected >= 0 && game.legal(selected, square)) play(fromCoord(selected % 7, selected / 7, square % 7, square / 7));
    }

    bool canUndo() const { return singlePlayer && static_cast<int>(game.past.size()) > humanSide; }

    void undo() {
        do { game.undo(); } while (game.pos.side != humanSide);
        selected = -1;
    }
};

inline void saveSession(std::ostream& output, const Session& session) {
    output << "ATAXX 2\n" << session.singlePlayer << ' ' << session.humanSide << ' ' << session.game.moves.size() << '\n';
    for (Move move : session.game.moves) output << move << '\n';
}

inline Session loadSession(std::istream& input) {
    Session session;
    std::string magic;
    int version, moves;
    if (!(input >> magic >> version >> session.singlePlayer >> session.humanSide >> moves) || magic != "ATAXX" || version != 2)
        throw std::runtime_error("Invalid Ataxx save file.");
    for (int i = 0; i < moves; ++i) {
        Move move, legal[NMOVES];
        if (!(input >> move) || !std::count(legal, legal + session.game.legal(legal), move))
            throw std::runtime_error("Invalid move history in save file.");
        session.play(move);
    }
    return session;
}

}

#endif

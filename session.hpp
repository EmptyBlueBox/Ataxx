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
    int humanSide = 1;
    int selected = -1;

    bool humanTurn() const { return !singlePlayer || game.side() == humanSide; }

    void play(Move move) {
        game.play(move);
        selected = -1;
    }

    void click(int square) {
        if (square < 0 || square >= 49 || !humanTurn() || game.outcome() != infinity) return;
        if (game.position.us & (Bits{1} << square)) selected = square;
        else if (selected >= 0 && game.legal({selected, square})) play({selected, square});
    }

    bool canUndo() const {
        return singlePlayer && std::any_of(game.history.begin(), game.history.end(),
            [this](const PastMove& past) { return past.side == humanSide; });
    }

    void undo() {
        do { game.undo(); } while (game.side() != humanSide);
        selected = -1;
    }
};

inline void saveSession(std::ostream& output, const Session& session) {
    output << "ATAXX 1\n" << (session.singlePlayer ? 1 : 2) << ' ' << session.humanSide << '\n';
    output << session.game.history.size() << '\n';
    for (const auto& past : session.game.history) output << past.move.from << ' ' << past.move.to << '\n';
    if (!output) throw std::runtime_error("Could not write the save file.");
}

inline Session loadSession(std::istream& input) {
    Session session;
    std::string magic;
    int version, mode, moves;
    if (!(input >> magic >> version >> mode >> session.humanSide >> moves) || magic != "ATAXX" || version != 1 ||
        (mode != 1 && mode != 2) || (session.humanSide != 1 && session.humanSide != -1) || moves < 0 || moves > 400)
        throw std::runtime_error("Invalid Ataxx save file.");
    session.singlePlayer = mode == 1;
    for (int i = 0; i < moves; ++i) {
        Move move;
        if (!(input >> move.from >> move.to) || session.game.outcome() != infinity || !session.game.legal(move))
            throw std::runtime_error("Invalid move history in save file.");
        session.play(move);
    }
    input >> std::ws;
    if (!input.eof()) throw std::runtime_error("Unexpected data in save file.");
    return session;
}

}

#endif

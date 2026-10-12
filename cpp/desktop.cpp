#include "session.hpp"
#include <raylib.h>
#include <chrono>
#include <fstream>
#include <future>
#include <string>

namespace {

constexpr Color background{18, 27, 35, 255};
constexpr Color panel{30, 43, 52, 255};
constexpr Color ink{233, 239, 234, 255};
constexpr Color muted{145, 166, 169, 255};
constexpr Color accent{133, 215, 187, 255};
constexpr Color red{235, 114, 103, 255};
constexpr Color yellow{240, 200, 111, 255};
constexpr ataxx::BoardLayout board{80, 208, 80};

void centered(const std::string& text, int center, int y, int size, Color color) {
    DrawText(text.c_str(), center - MeasureText(text.c_str(), size) / 2, y, size, color);
}

bool button(Rectangle bounds, const char* label, bool enabled = true, bool active = false) {
    bool hover = enabled && CheckCollisionPointRec(GetMousePosition(), bounds);
    Color fill = active ? accent : hover ? Color{49, 67, 75, 255} : panel;
    DrawRectangleRounded(bounds, 0.18f, 8, enabled ? fill : Fade(fill, 0.45f));
    centered(label, static_cast<int>(bounds.x + bounds.width / 2),
        static_cast<int>(bounds.y + (bounds.height - 20) / 2), 20,
        !enabled ? Fade(muted, 0.4f) : active ? background : ink);
    return hover && IsMouseButtonPressed(MOUSE_BUTTON_LEFT);
}

void drawBoard(const ataxx::Session& session, bool interactive) {
    const auto& game = session.game;
    for (int row = 0; row < 7; ++row) {
        centered(std::to_string(7 - row), 58, static_cast<int>(board.y + row * board.cell + 32), 18, muted);
        centered(std::string(1, static_cast<char>('A' + row)), static_cast<int>(board.x + row * board.cell + 40), 774, 16, muted);
        for (int column = 0; column < 7; ++column) {
            int square = row * 7 + column;
            float x = board.x + column * board.cell, y = board.y + row * board.cell;
            Rectangle cell{x + 2, y + 2, board.cell - 4, board.cell - 4};
            Color fill = (row + column) % 2 ? Color{37, 54, 61, 255} : Color{44, 63, 69, 255};
            if (interactive && CheckCollisionPointRec(GetMousePosition(), cell)) fill = Color{54, 76, 80, 255};
            DrawRectangleRounded(cell, 0.12f, 4, fill);
            int bit = ataxx::bitOfSq(square), side = game.pos.bb[0] >> bit & 1 ? 1 : game.pos.bb[1] >> bit & 1 ? 2 : 0;
            Vector2 center{x + 40, y + 40};
            if (side) {
                DrawCircleV({center.x, center.y + 3}, 26, Fade(BLACK, 0.2f));
                DrawCircleV(center, 26, side == 1 ? red : yellow);
                DrawCircleV({center.x - 7, center.y - 8}, 5, Fade(WHITE, 0.18f));
                if (square == session.selected) DrawRing(center, 30, 33, 0, 360, 40, ink);
            } else if (interactive && session.selected >= 0 && game.legal(session.selected, square)) {
                if (ataxx::T().ring1[session.selected] >> bit & 1) DrawCircleV(center, 6, accent);
                else DrawRing(center, 6, 8, 0, 360, 24, accent);
            }
        }
    }
}

void scoreCard(Rectangle bounds, const char* name, int score, Color color, bool turn) {
    DrawRectangleRounded(bounds, 0.18f, 8, panel);
    DrawCircle(static_cast<int>(bounds.x + 26), static_cast<int>(bounds.y + 29), 9, color);
    DrawText(name, static_cast<int>(bounds.x + 47), static_cast<int>(bounds.y + 19), 20, turn ? ink : muted);
    std::string value = std::to_string(score);
    DrawText(value.c_str(), static_cast<int>(bounds.x + bounds.width - 46), static_cast<int>(bounds.y + 16), 26, color);
    if (turn) DrawRectangle(static_cast<int>(bounds.x + 16), static_cast<int>(bounds.y + bounds.height - 3),
        static_cast<int>(bounds.width - 32), 3, color);
}

}

int main() {
    SetConfigFlags(FLAG_WINDOW_HIGHDPI | FLAG_MSAA_4X_HINT);
    InitWindow(720, 900, "Ataxx");
    SetExitKey(KEY_NULL);
    SetTargetFPS(60);

    ataxx::Session session;
    ataxx::Search search;
    std::future<ataxx::Move> pending;
    bool menu = true, hasGame = false, singlePlayer = true;
    int humanSide = 0;
    std::string message;

    auto load = [&] {
        try {
            std::ifstream input("ataxx.save");
            if (!input) throw std::runtime_error("Could not open ataxx.save.");
            session = ataxx::loadSession(input);
            singlePlayer = session.singlePlayer;
            humanSide = session.humanSide;
            menu = false;
            hasGame = true;
            message = "Game loaded.";
        } catch (const std::runtime_error& error) { message = error.what(); }
    };

    while (!WindowShouldClose()) {
        if (pending.valid() && pending.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            session.play(pending.get());
            message.clear();
        }
        int end = session.game.winner();
        if (!menu && !pending.valid() && end == 2 && !session.humanTurn()) {
            pending = std::async(std::launch::async, [&search, game = session.game] {
                return search.choose(game, std::clock(), 0.85);
            });
        }
        bool busy = pending.valid();
        bool interactive = !menu && !busy && end == 2 && session.humanTurn();
        if (interactive && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            Vector2 mouse = GetMousePosition();
            int square = board.square(mouse.x, mouse.y);
            if (square >= 0) { session.click(square); message.clear(); }
            end = session.game.winner();
            interactive = end == 2 && session.humanTurn();
        }

        BeginDrawing();
        ClearBackground(background);
        DrawText("ATAXX", 80, 35, 40, ink);
        DrawText("SEVEN BY SEVEN", 82, 81, 14, accent);

        if (menu) {
            DrawText("Choose your game", 80, 168, 28, ink);
            DrawText("Copy one square, or jump two.", 80, 212, 20, muted);
            DrawText("Adjacent enemy pieces turn your color.", 80, 243, 20, muted);
            if (button({80, 315, 560, 62}, "Play against AI", true, singlePlayer)) singlePlayer = true;
            if (button({80, 393, 560, 62}, "Two players", true, !singlePlayer)) singlePlayer = false;
            DrawText(singlePlayer ? "PLAY AS" : "RED MOVES FIRST", 80, 500, 16, muted);
            if (singlePlayer) {
                if (button({80, 536, 272, 54}, "Red / first", true, humanSide == 0)) humanSide = 0;
                if (button({368, 536, 272, 54}, "Yellow / second", true, humanSide == 1)) humanSide = 1;
            }
            if (button({80, 638, 560, 62}, "Start game", true, true)) {
                session = {};
                session.singlePlayer = singlePlayer;
                session.humanSide = humanSide;
                menu = false;
                hasGame = true;
                message.clear();
            }
            if (button({80, 720, 272, 54}, "Load saved game")) load();
            if (button({368, 720, 272, 54}, "Resume", hasGame)) { menu = false; message.clear(); }
            centered(message, 360, 820, 18, muted);
        } else {
            if (button({520, 40, 120, 44}, "Menu", !busy)) { menu = true; session.selected = -1; message.clear(); }
            const auto& game = session.game;
            scoreCard({80, 115, 272, 58}, "RED", ataxx::popcnt(game.pos.bb[0]), red, end == 2 && game.pos.side == 0);
            scoreCard({368, 115, 272, 58}, "YELLOW", ataxx::popcnt(game.pos.bb[1]), yellow, end == 2 && game.pos.side == 1);
            std::string status;
            if (end != 2) status = end < 0 ? "Draw" : end == 0 ? "Red wins" : "Yellow wins";
            else if (busy || !session.humanTurn()) status = "AI is thinking...";
            else status = std::string(game.pos.side == 0 ? "Red" : "Yellow") + (session.selected < 0 ? ": choose a piece" : ": choose a destination");
            DrawText(status.c_str(), 80, 182, 18, end == 2 ? muted : accent);
            std::string moves = "Moves: " + std::to_string(game.pos.ply);
            DrawText(moves.c_str(), 640 - MeasureText(moves.c_str(), 16), 184, 16, muted);
            drawBoard(session, interactive);
            float width = session.singlePlayer ? 176 : 272;
            float saveX = session.singlePlayer ? 272 : 80;
            if (session.singlePlayer && button({80, 808, 176, 48}, "Undo", !busy && session.canUndo())) {
                session.undo();
                message = "Returned to your previous turn.";
            }
            if (button({saveX, 808, width, 48}, "Save", !busy)) {
                try {
                    std::ofstream output("ataxx.save");
                    ataxx::saveSession(output, session);
                    output.close();
                    if (!output) throw std::runtime_error("Could not write the save file.");
                    message = "Saved to ataxx.save.";
                } catch (const std::runtime_error& error) { message = error.what(); }
            }
            if (button({session.singlePlayer ? 464.0f : 368.0f, 808, width, 48}, "Load", !busy)) load();
            centered(message.empty() ? "Filled dots: copy    /    Rings: jump" : message, 360, 873, 16, muted);
        }
        EndDrawing();
    }
    CloseWindow();
}

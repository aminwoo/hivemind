/*
 * Hivemind - Bughouse Chess Engine
 * GUI State Writer - Outputs game state to JSON for the web GUI
 */

#pragma once

#include <atomic>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

#include "../board.h"
#include "gamepgn.h"

/**
 * @brief Writes game state to a JSON file for the web GUI.
 * Thread-safe and designed for real-time updates.
 */
class GuiStateWriter {
public:
    GuiStateWriter(const std::string& outputPath = "./gui/game_state.json")
        : outputPath(outputPath), enabled(true) {}
    
    void setEnabled(bool enable) { enabled = enable; }
    bool isEnabled() const { return enabled; }
    
    /**
     * @brief Update the game state for the GUI.
     * Should be called after each move.
     */
    void update(const Board& board, 
                const std::vector<BughouseGamePGN::MoveRecord>& moves,
                size_t gameNumber,
                size_t totalGames,
                size_t ply,
                Stockfish::Color sideToMove,
                const std::string& whiteTeam,
                const std::string& blackTeam,
                const std::string& result,
                size_t player1Wins,
                size_t player1Losses,
                size_t draws)
    {
        if (!enabled) return;
        
        std::lock_guard<std::mutex> lock(writeMutex);
        
        std::ofstream file(outputPath);
        if (!file.is_open()) return;
        
        file << "{\n";
        file << "  \"fenA\": \"" << escapeJson(board.pos[0]->fen()) << "\",\n";
        file << "  \"fenB\": \"" << escapeJson(board.pos[1]->fen()) << "\",\n";
        file << "  \"sideToMove\": \"" << (sideToMove == Stockfish::WHITE ? "w" : "b") << "\",\n";
        file << "  \"ply\": " << ply << ",\n";
        file << "  \"gameNumber\": " << gameNumber << ",\n";
        file << "  \"totalGames\": " << totalGames << ",\n";
        file << "  \"whiteTeam\": \"" << escapeJson(whiteTeam) << "\",\n";
        file << "  \"blackTeam\": \"" << escapeJson(blackTeam) << "\",\n";
        file << "  \"result\": \"" << escapeJson(result) << "\",\n";
        file << "  \"player1Wins\": " << player1Wins << ",\n";
        file << "  \"player1Losses\": " << player1Losses << ",\n";
        file << "  \"draws\": " << draws << ",\n";
        
        // Write moves array
        file << "  \"moves\": [\n";
        for (size_t i = 0; i < moves.size(); i++) {
            const auto& m = moves[i];
            file << "    {";
            file << "\"board\": " << m.boardNum << ", ";
            file << "\"san\": \"" << escapeJson(m.move) << "\", ";
            file << "\"moveNum\": " << m.moveNumber << ", ";
            file << "\"isWhite\": " << (m.isWhite ? "true" : "false");
            file << "}";
            if (i < moves.size() - 1) file << ",";
            file << "\n";
        }
        file << "  ],\n";
        
        // Timestamp for cache busting
        file << "  \"timestamp\": " << std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count() << "\n";
        
        file << "}\n";
        file.close();
    }
    
    /**
     * @brief Clear the game state (e.g., when starting a new game).
     */
    void clear() {
        if (!enabled) return;
        
        std::lock_guard<std::mutex> lock(writeMutex);
        
        std::ofstream file(outputPath);
        if (!file.is_open()) return;
        
        file << "{\n";
        file << "  \"fenA\": \"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\",\n";
        file << "  \"fenB\": \"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\",\n";
        file << "  \"sideToMove\": \"w\",\n";
        file << "  \"ply\": 0,\n";
        file << "  \"gameNumber\": 0,\n";
        file << "  \"totalGames\": 0,\n";
        file << "  \"whiteTeam\": \"-\",\n";
        file << "  \"blackTeam\": \"-\",\n";
        file << "  \"result\": \"ongoing\",\n";
        file << "  \"player1Wins\": 0,\n";
        file << "  \"player1Losses\": 0,\n";
        file << "  \"draws\": 0,\n";
        file << "  \"moves\": [],\n";
        file << "  \"timestamp\": 0\n";
        file << "}\n";
        file.close();
    }

private:
    std::string outputPath;
    std::atomic<bool> enabled;
    std::mutex writeMutex;
    
    static std::string escapeJson(const std::string& s) {
        std::ostringstream o;
        for (char c : s) {
            switch (c) {
                case '"': o << "\\\""; break;
                case '\\': o << "\\\\"; break;
                case '\b': o << "\\b"; break;
                case '\f': o << "\\f"; break;
                case '\n': o << "\\n"; break;
                case '\r': o << "\\r"; break;
                case '\t': o << "\\t"; break;
                default:
                    if ('\x00' <= c && c <= '\x1f') {
                        o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)c;
                    } else {
                        o << c;
                    }
            }
        }
        return o.str();
    }
};

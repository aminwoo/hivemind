/*
 * Hivemind - Bughouse Chess Engine
 * PGN output for Bughouse games
 * Adapted from CrazyAra's GamePGN
 */

#pragma once

#include <string>
#include <vector>
#include <iostream>

/**
 * @brief Game result enumeration for Bughouse.
 */
enum class GameResult {
    NO_RESULT,      // Game in progress
    WHITE_WINS,     // White team wins (checkmate or flag)
    BLACK_WINS,     // Black team wins (checkmate or flag)
    DRAW,           // Draw (stalemate, repetition, etc.)
    WHITE_RESIGNS,  // White team resigns
    BLACK_RESIGNS   // Black team resigns
};

/**
 * @brief Structure for exporting a Bughouse game in PGN format.
 * 
 * Bughouse PGN uses board prefixes:
 * - Moves on Board A are prefixed with "1"
 * - Moves on Board B are prefixed with "2"
 * 
 * Example: 1e4 2d4 1e5 2d5 means:
 *   Board A: e4, e5
 *   Board B: d4, d5
 */
struct BughouseGamePGN {
    std::string variant = "bughouse";
    std::string event = "SelfPlay";
    std::string site = "Hivemind";
    std::string date = "????.??.??";
    std::string round = "?";
    
    // Team names (White team = Board A White + Board B Black)
    std::string whiteTeam = "TeamA";
    std::string blackTeam = "TeamB";
    
    // Individual players
    std::string whiteBoardA = "EngineA1";    // White pieces on Board A
    std::string blackBoardA = "EngineA2";    // Black pieces on Board A (partner of whiteBoardA)
    std::string whiteBoardB = "EngineB1";    // White pieces on Board B
    std::string blackBoardB = "EngineB2";    // Black pieces on Board B (partner of whiteBoardB)
    
    // Starting FEN for both boards (standard if empty)
    std::string fenBoardA = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    std::string fenBoardB = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    
    std::string result = "*";
    std::string timeControl = "-";
    
    // Moves with board prefix: "1e2e4", "2d2d4", etc.
    std::vector<std::string> gameMoves;
    
    // Track which team had time advantage (for training purposes)
    bool whiteTeamHadTimeAdvantage = false;
    
    /**
     * @brief Clear moves and reset for new game.
     */
    void new_game();
    
    /**
     * @brief Add a move with the appropriate board prefix.
     * @param boardNum 0 for Board A, 1 for Board B
     * @param moveStr The move in UCI format (e.g., "e2e4", "N@f3")
     */
    void add_move(int boardNum, const std::string& moveStr);
    
    /**
     * @brief Set the game result.
     */
    void set_result(GameResult res);
    
    /**
     * @brief Get result string for PGN.
     */
    std::string result_string(GameResult res) const;
};

/**
 * @brief Output operator for PGN format.
 */
std::ostream& operator<<(std::ostream& os, const BughouseGamePGN& pgn);

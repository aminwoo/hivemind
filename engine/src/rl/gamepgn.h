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
 * Uses BPGN (Bughouse PGN) format:
 * - "1A." = Move 1, White, Board A
 * - "1a." = Move 1, Black, Board A  
 * - "1B." = Move 1, White, Board B
 * - "1b." = Move 1, Black, Board B
 * 
 * Example: 1A. e4 {179.9} 1a. e6 {179.9} 1B. e4 {179.9} 1b. e5 {179.9}
 */
struct BughouseGamePGN {
    std::string variant = "bughouse";
    std::string event = "Hivemind Self-Play";
    std::string site = "Hivemind Engine";
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
    std::string timeControl = "180";  // Default 3 minutes
    std::string termination = "";     // How the game ended (e.g., "White won by checkmate")
    
    // Move storage with timing
    struct MoveRecord {
        int boardNum;        // 0 = Board A, 1 = Board B
        bool isWhite;        // true = White's move, false = Black's move
        int moveNumber;      // Full move number
        std::string move;    // UCI move string
        float clockTime;     // Time remaining after move
    };
    std::vector<MoveRecord> gameMoves;
    
    // Track move numbers for each board
    int moveNumberBoardA = 1;
    int moveNumberBoardB = 1;
    bool whiteToMoveBoardA = true;
    bool whiteToMoveBoardB = true;
    
    // Track which team had time advantage (for training purposes)
    bool whiteTeamHadTimeAdvantage = false;
    
    /**
     * @brief Clear moves and reset for new game.
     */
    void new_game();
    
    /**
     * @brief Add a move with timing information.
     * @param boardNum 0 for Board A, 1 for Board B
     * @param moveStr The move in UCI format (e.g., "e2e4", "N@f3")
     * @param clockTime Time remaining after move (default 0.0)
     */
    void add_move(int boardNum, const std::string& moveStr, float clockTime = 0.0f);
    
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

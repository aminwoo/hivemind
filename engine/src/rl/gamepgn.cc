/*
 * Hivemind - Bughouse Chess Engine
 * PGN output for Bughouse games (BPGN format)
 * Adapted from CrazyAra's GamePGN
 */

#include "gamepgn.h"
#include "../constants.h"
#include <ctime>
#include <iomanip>
#include <sstream>

void BughouseGamePGN::new_game() {
    gameMoves.clear();
    result = "*";
    termination = "";
    
    // Reset move counters
    moveNumberBoardA = 1;
    moveNumberBoardB = 1;
    whiteToMoveBoardA = true;
    whiteToMoveBoardB = true;
    
    // Update date
    time_t now = time(nullptr);
    struct tm* tstruct = localtime(&now);
    char date_buf[80];
    strftime(date_buf, sizeof(date_buf), "%Y.%m.%d", tstruct);
    date = date_buf;
}

void BughouseGamePGN::add_move(int boardNum, const std::string& moveStr, float clockTime) {
    MoveRecord record;
    record.boardNum = boardNum;
    record.move = moveStr;
    record.clockTime = clockTime;
    
    if (boardNum == BOARD_A) {
        // Board A
        record.isWhite = whiteToMoveBoardA;
        record.moveNumber = moveNumberBoardA;
        
        if (!whiteToMoveBoardA) {
            moveNumberBoardA++;  // Increment after Black's move
        }
        whiteToMoveBoardA = !whiteToMoveBoardA;
    } else {
        // Board B
        record.isWhite = whiteToMoveBoardB;
        record.moveNumber = moveNumberBoardB;
        
        if (!whiteToMoveBoardB) {
            moveNumberBoardB++;  // Increment after Black's move
        }
        whiteToMoveBoardB = !whiteToMoveBoardB;
    }
    
    gameMoves.push_back(record);
}

void BughouseGamePGN::set_result(GameResult res) {
    result = result_string(res);
    
    // Set termination message
    switch (res) {
        case GameResult::WHITE_WINS:
            termination = whiteTeam + " won by checkmate";
            break;
        case GameResult::BLACK_WINS:
            termination = blackTeam + " won by checkmate";
            break;
        case GameResult::WHITE_RESIGNS:
            termination = whiteTeam + " resigned";
            break;
        case GameResult::BLACK_RESIGNS:
            termination = blackTeam + " resigned";
            break;
        case GameResult::DRAW:
            termination = "Game drawn";
            break;
        default:
            termination = "";
            break;
    }
}

std::string BughouseGamePGN::result_string(GameResult res) const {
    switch (res) {
        case GameResult::WHITE_WINS:
        case GameResult::BLACK_RESIGNS:
            return "1-0";
        case GameResult::BLACK_WINS:
        case GameResult::WHITE_RESIGNS:
            return "0-1";
        case GameResult::DRAW:
            return "1/2-1/2";
        case GameResult::NO_RESULT:
        default:
            return "*";
    }
}

std::ostream& operator<<(std::ostream& os, const BughouseGamePGN& pgn) {
    // BPGN headers
    os << "[Event \"" << pgn.event << "\"]\n"
       << "[Site \"" << pgn.site << "\"]\n"
       << "[Date \"" << pgn.date << "\"]\n"
       << "[Round \"" << pgn.round << "\"]\n"
       << "[Variant \"" << pgn.variant << "\"]\n"
       << "[TimeControl \"" << pgn.timeControl << "\"]\n"
       << "[WhiteTeam \"" << pgn.whiteTeam << "\"]\n"
       << "[BlackTeam \"" << pgn.blackTeam << "\"]\n"
       << "[WhiteA \"" << pgn.whiteBoardA << "\"]\n"
       << "[BlackA \"" << pgn.blackBoardA << "\"]\n"
       << "[WhiteB \"" << pgn.whiteBoardB << "\"]\n"
       << "[BlackB \"" << pgn.blackBoardB << "\"]\n"
       << "[TimeAdvantage \"" << (pgn.whiteTeamHadTimeAdvantage ? pgn.whiteTeam : pgn.blackTeam) << "\"]\n"
       << "[PlyCount \"" << pgn.plyCount << "\"]\n"
       << "[Result \"" << pgn.result << "\"]\n";
    
    if (!pgn.termination.empty()) {
        os << "[Termination \"" << pgn.termination << "\"]\n";
    }
    
    os << "\n";
    
    // Output moves in BPGN format
    // Format: 1A. e4 {179.9} 1a. e6 {179.8} 1B. d4 {179.7} 1b. d5 {179.6}
    for (size_t i = 0; i < pgn.gameMoves.size(); ++i) {
        const auto& move = pgn.gameMoves[i];
        
        // Build move prefix: "1A." for White Board A, "1a." for Black Board A, etc.
        os << move.moveNumber;
        
        char boardLetter = (move.boardNum == BOARD_A) ? 'A' : 'B';
        if (!move.isWhite) {
            boardLetter = (move.boardNum == BOARD_A) ? 'a' : 'b';
        }
        os << boardLetter << ". ";
        
        // Output move
        os << move.move;
        
        // Output clock time in braces
        os << " {" << std::fixed << std::setprecision(1) << move.clockTime << "} ";
    }
    
    // Add termination comment if present
    if (!pgn.termination.empty()) {
        os << "{C:" << pgn.termination << " " << pgn.result << "}\n";
        os << "{" << pgn.termination << "} ";
    }
    
    // Final result marker
    os << "*\n\n";
    
    return os;
}

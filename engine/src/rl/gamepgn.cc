/*
 * Hivemind - Bughouse Chess Engine
 * PGN output for Bughouse games
 * Adapted from CrazyAra's GamePGN
 */

#include "gamepgn.h"
#include <ctime>
#include <iomanip>

void BughouseGamePGN::new_game() {
    gameMoves.clear();
    result = "*";
    
    // Update date
    time_t now = time(nullptr);
    struct tm* tstruct = localtime(&now);
    char date_buf[80];
    strftime(date_buf, sizeof(date_buf), "%Y.%m.%d", tstruct);
    date = date_buf;
}

void BughouseGamePGN::add_move(int boardNum, const std::string& moveStr) {
    // Prefix with 1 for Board A, 2 for Board B
    std::string prefixedMove = std::to_string(boardNum + 1) + moveStr;
    gameMoves.push_back(prefixedMove);
}

void BughouseGamePGN::set_result(GameResult res) {
    result = result_string(res);
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
    // Standard PGN headers
    os << "[Event \"" << pgn.event << "\"]\n"
       << "[Site \"" << pgn.site << "\"]\n"
       << "[Date \"" << pgn.date << "\"]\n"
       << "[Round \"" << pgn.round << "\"]\n"
       << "[Variant \"" << pgn.variant << "\"]\n";
    
    // Bughouse-specific headers
    os << "[WhiteTeam \"" << pgn.whiteTeam << "\"]\n"
       << "[BlackTeam \"" << pgn.blackTeam << "\"]\n"
       << "[WhiteA \"" << pgn.whiteBoardA << "\"]\n"
       << "[BlackA \"" << pgn.blackBoardA << "\"]\n"
       << "[WhiteB \"" << pgn.whiteBoardB << "\"]\n"
       << "[BlackB \"" << pgn.blackBoardB << "\"]\n"
       << "[FENBoardA \"" << pgn.fenBoardA << "\"]\n"
       << "[FENBoardB \"" << pgn.fenBoardB << "\"]\n"
       << "[Result \"" << pgn.result << "\"]\n"
       << "[PlyCount \"" << pgn.gameMoves.size() << "\"]\n"
       << "[TimeControl \"" << pgn.timeControl << "\"]\n";
    
    // Custom header for RL training
    os << "[WhiteTeamTimeAdvantage \"" << (pgn.whiteTeamHadTimeAdvantage ? "true" : "false") << "\"]\n";
    
    os << "\n";
    
    // Output moves with line wrapping
    size_t lineLength = 0;
    for (size_t i = 0; i < pgn.gameMoves.size(); ++i) {
        const std::string& move = pgn.gameMoves[i];
        
        if (lineLength + move.length() + 1 > 80) {
            os << "\n";
            lineLength = 0;
        }
        
        if (lineLength > 0) {
            os << " ";
            lineLength++;
        }
        
        os << move;
        lineLength += move.length();
    }
    
    // Final result
    if (lineLength > 0) {
        os << " ";
    }
    os << pgn.result << "\n\n";
    
    return os;
}

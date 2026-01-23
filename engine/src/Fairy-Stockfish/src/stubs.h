#ifndef STUBS_H_INCLUDED
#define STUBS_H_INCLUDED

#include <string>
#include <vector>
#include "types.h"

namespace Stockfish {

class Position;

namespace PSQT {
    constexpr Score psq[PIECE_NB][SQUARE_NB + 1] = {};
}

namespace Eval {
    constexpr bool useNNUE = false;
    
    namespace NNUE {
        struct Accumulator {
            bool computed[COLOR_NB];
        };
    }
}

namespace UCI {
    inline std::string square(const Position& pos, Square s) {
        return std::string(1, 'a' + file_of(s)) + std::string(1, '1' + rank_of(s));
    }
    
    inline std::string move(const Position& pos, Move m) {
        if (m == MOVE_NONE)
            return "(none)";
        if (m == MOVE_NULL)
            return "0000";
            
        Square from = from_sq(m);
        Square to = to_sq(m);
        
        auto square_to_str = [](Square s) {
            return std::string(1, 'a' + file_of(s)) + std::string(1, '1' + rank_of(s));
        };
        
        if (type_of(m) == DROP) {
            PieceType pt = dropped_piece_type(m);
            const char piece_chars[] = ".PNBRQ................K";
            return std::string(1, tolower(piece_chars[pt])) + "@" + square_to_str(to);
        }
        
        std::string move_str = square_to_str(from) + square_to_str(to);
        
        if (type_of(m) == PROMOTION) {
            PieceType pt = promotion_type(m);
            const char piece_chars[] = ".pnbrq................k";
            move_str += piece_chars[pt];
        }
            
        return move_str;
    }
    
    Move to_move(const Position& pos, std::string str);
}

namespace Pawns {
    struct Table {};
}

namespace Material {
    struct Table {};
}

namespace Search {
    struct RootMoves : std::vector<Move> {};
    struct LimitsType {};
}

using CounterMoveHistory = int;
using ButterflyHistory = int;
using LowPlyHistory = int;
using CapturePieceToHistory = int;
using ContinuationHistory = int;

}

#endif

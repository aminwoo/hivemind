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
        constexpr size_t CacheLineSize = 64;
        struct Accumulator {
            bool computed[COLOR_NB];
        };
    }
}

namespace UCI {
    inline std::string square(const Position& pos, Square s) {
        return std::string{char('a' + file_of(s)), char('1' + rank_of(s))};
    }

    inline std::string dropped_piece(const Position& pos, Move m);
    
    std::string move(const Position& pos, Move m);
    
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

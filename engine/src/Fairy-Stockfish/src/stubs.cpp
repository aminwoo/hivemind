#include "stubs.h"
#include "types.h"
#include "position.h"
#include "movegen.h"

namespace Stockfish {

Value PieceValue[PHASE_NB][PIECE_NB] = {
    { VALUE_ZERO, PawnValueMg, KnightValueMg, BishopValueMg, RookValueMg, QueenValueMg },
    { VALUE_ZERO, PawnValueEg, KnightValueEg, BishopValueEg, RookValueEg, QueenValueEg }
};

Value CapturePieceValue[PHASE_NB][PIECE_NB] = {
    { VALUE_ZERO, PawnValueMg, KnightValueMg, BishopValueMg, RookValueMg, QueenValueMg },
    { VALUE_ZERO, PawnValueEg, KnightValueEg, BishopValueEg, RookValueEg, QueenValueEg }
};

namespace UCI {
    Move to_move(const Position& pos, std::string str) {
        if (str.length() == 5) {
            if (str[4] == '=')
                str.pop_back();
            else
                str[4] = char(tolower(str[4]));
        }
        
        for (const auto& m : MoveList<LEGAL>(pos))
            if (str == move(pos, m) || (is_pass(m) && str == square(pos, from_sq(m)) + square(pos, to_sq(m))))
                return m;
        
        return MOVE_NONE;
    }
}

}

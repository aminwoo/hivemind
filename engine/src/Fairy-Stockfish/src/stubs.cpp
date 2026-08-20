#include "stubs.h"
#include "types.h"
#include "position.h"
#include "movegen.h"
#include <cctype>

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

std::string move(const Position& pos, Move m) {
    Square from = from_sq(m);
    Square to = to_sq(m);

    if (m == MOVE_NONE)
        return "(none)";

    if (m == MOVE_NULL)
        return "0000";

    if (is_gating(m) && gating_square(m) == to)
        from = to_sq(m), to = from_sq(m);
    else if (type_of(m) == CASTLING && !pos.is_chess960())
    {
        to = make_square(to > from ? pos.castling_kingside_file() : pos.castling_queenside_file(), rank_of(from));
        // If the castling move is ambiguous with a normal king move, switch to 960 notation
        if (pos.pseudo_legal(make_move(from, to)))
            to = to_sq(m);
    }

    std::string moveStr = (type_of(m) == DROP ? std::string{pos.piece_to_char()[dropped_piece_type(m)]} + '@'
                                              : UCI::square(pos, from)) + UCI::square(pos, to);

    if (type_of(m) == PROMOTION)
        moveStr += pos.piece_to_char()[make_piece(BLACK, promotion_type(m))];
    else if (type_of(m) == PIECE_PROMOTION)
        moveStr += '+';
    else if (type_of(m) == PIECE_DEMOTION)
        moveStr += '-';
    else if (is_gating(m))
    {
        moveStr += pos.piece_to_char()[make_piece(BLACK, gating_type(m))];
        if (gating_square(m) != from)
            moveStr += UCI::square(pos, gating_square(m));
    }

    return moveStr;
}

Move to_move(const Position& pos, std::string str) {
    if (str.length() == 5)
    {
        if (str[4] == '=')
            str.pop_back();
        else if (str[1] != '@')
            str[4] = char(tolower(str[4]));
    }

    for (const auto& m : MoveList<LEGAL>(pos))
        if (str == UCI::move(pos, m) || (is_pass(m) && str == UCI::square(pos, from_sq(m)) + UCI::square(pos, to_sq(m))))
            return m;

    return MOVE_NONE;
}

} // namespace UCI

} // namespace Stockfish

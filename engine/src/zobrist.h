#pragma once

#include "Fairy-Stockfish/src/types.h"

namespace Stockfish {
  namespace Zobrist {
    const int MAX_PLY = 1024;
    extern Key ply[MAX_PLY];
    extern Key psq[PIECE_NB][SQUARE_NB];
    extern Key enpassant[FILE_NB];
    extern Key castling[CASTLING_RIGHT_NB];
    extern Key side, noPawns;
    extern Key inHand[PIECE_NB][SQUARE_NB];
    extern Key checks[COLOR_NB][CHECKS_NB];
  }
}
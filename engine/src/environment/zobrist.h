#pragma once

#include "Fairy-Stockfish/src/types.h"

namespace Stockfish {
  namespace Zobrist {
    // Ply-based Zobrist keys for position hashing (specific to this project)
    // Other Zobrist keys are defined in Fairy-Stockfish's position.cpp
    const int MAX_PLY = 1024;
    extern Key ply[MAX_PLY];
    
    // Time advantage key for MCGS transposition detection
    // Positions with different time advantage states are treated as distinct
    extern Key timeAdvantage;

    // Board-only repetition keys. These cover exactly the fields that define
    // repetition identity for a single bughouse board - piece placement with
    // promotion markers, side to move, castling rights and the en passant file -
    // and deliberately exclude the pockets, which repetition claims ignore.
    extern Key boardPsq[PIECE_NB][SQUARE_NB];
    extern Key boardPromoted[SQUARE_NB];
    extern Key boardCastling[16];
    extern Key boardEnPassant[FILE_NB];
    extern Key boardSide;
  }
}
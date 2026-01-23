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
  }
}
#ifndef PLANES_H
#define PLANES_H

#include "board.h"
#include "utils.h"
#include "Fairy-Stockfish/src/types.h"

void board_to_planes(Board& board, float* inputPlanes, Stockfish::Color us, bool pass);  

#endif 

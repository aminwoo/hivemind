#ifndef PLANES_H
#define PLANES_H

#include "bugboard.h"
#include "utils.h"
#include "Fairy-Stockfish/src/types.h"

void board_to_planes(Bugboard& board, float* inputPlanes, Stockfish::Color us);  

#endif 
#include <iostream>

#include "Fairy-Stockfish/src/bitboard.h"
#include "Fairy-Stockfish/src/position.h"
#include "Fairy-Stockfish/src/thread.h"
#include "Fairy-Stockfish/src/piece.h"
#include "Fairy-Stockfish/src/types.h"

#include "utils.h"
#include "constants.h"
#include "engine.h"
#include "bugboard.h"
#include "planes.h"
#include "node.h"
#include "agent.h"

std::unordered_map<std::string, int> POLICY_INDEX; 

using namespace std; 

int main() {
    Stockfish::pieceMap.init();
    Stockfish::variants.init();
    Stockfish::Bitboards::init();
    Stockfish::Position::init();
    Stockfish::Threads.set(1);

    for (int i = 0; i < NUM_POLICY_VALUES; i++) {
        POLICY_INDEX[UCI_MOVES[i]] = i; 
    }

    Bugboard board; 
    Agent agent;
    agent.set_is_running(true);
    agent.run_search(board, 1000);
}
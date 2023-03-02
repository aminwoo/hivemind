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

    board.set("r1bq1b1r/ppp1k1pp/3npp2/4N1B1/3Q4/2N2N2/PPP2KPP/R6R[NQqbnPPP] w | r1b1k2r/ppp2ppp/2p1p3/6B1/B2nn3/2P1P3/P4PPP/R1B1K2R[pPP] b kq"); 
    board.set_time(0, 1800); 
    board.set_time(1, 1700); 
    board.set_time(2, 1700);
    board.set_time(3, 1800);

    Agent agent;
    agent.set_is_running(true);
    agent.run_search(board, 1000);
    //Node* curr = new Node(Stockfish::WHITE);
}
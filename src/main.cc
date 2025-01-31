#include "uci.h"
#include "constants.h"
#include "Fairy-Stockfish/src/bitboard.h"
#include "Fairy-Stockfish/src/position.h"
#include "Fairy-Stockfish/src/thread.h"
#include "Fairy-Stockfish/src/piece.h"
#include "Fairy-Stockfish/src/types.h"
#include <iostream>

std::unordered_map<std::string, int> POLICY_INDEX; 

using namespace std; 

int main() {
    Stockfish::pieceMap.init();
    Stockfish::variants.init();
    Stockfish::Bitboards::init();
    Stockfish::Position::init();
    Stockfish::Threads.set(1);

    for (int i = 0; i < NB_POLICY_VALUES(); i++) {
        if (POLICY_INDEX.find(UCI_MOVES[i]) == POLICY_INDEX.end()) {
            POLICY_INDEX[UCI_MOVES[i]] = i; 
        }
    }

    UCI uci;
    uci.loop();
}

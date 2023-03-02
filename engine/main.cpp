#include <iostream>

#include "Fairy-Stockfish/src/bitboard.h"
#include "Fairy-Stockfish/src/position.h"
#include "Fairy-Stockfish/src/thread.h"
#include "Fairy-Stockfish/src/piece.h"

#include "utils.h"
#include "constants.h"
#include "engine.h"
#include "bugboard.h"
#include "planes.h"
#include "agent.h"
#include "uci.h"

std::unordered_map<std::string, int> POLICY_INDEX; 

using namespace std; 

void print_board(float* inputPlanes) {
    for (int i = 0; i < NUM_BUGHOUSE_VALUES(); i++) {
        cout << inputPlanes[i] << ' ';
        if (i > 0 && (i + 1) % 16 == 0) {
            cout << endl; 
        }
        if (i > 0 && (i + 1) % 128 == 0) {
            cout << endl; 
        }
    }
}

int main() {
    Stockfish::pieceMap.init();
    Stockfish::variants.init();
    Stockfish::Bitboards::init();
    Stockfish::Position::init();
    Stockfish::Threads.set(1);

    for (int i = 0; i < NUM_POLICY_VALUES; i++) {
        POLICY_INDEX[UCI_MOVES[i]] = i; 
    }

    //board.set("rnbqkbnr/pp1ppppp/8/8/1PpPP3/8/P1P2PPP/RNBQKBNR b KQkq b3 0 3 | r~3kb1r~/pppbqppp/4P3/3pN~3/3Pn3/8/PPP2PPP/R1BQK2R[] b Kq"); 
    /*board.set("r1bq1b1r/ppp1k1pp/3npp2/4N1B1/3Q4/2N2N2/PPP2KPP/R6R[NQqbnPPP] w | r1b1k2r/ppp2ppp/2p1p3/6B1/B2nn3/2P1P3/P4PPP/R1B1K2R[pPP] b kq"); 
    float* inputPlanes = new float[NUM_BUGHOUSE_VALUES()]; 
    board_to_planes(board, inputPlanes, Stockfish::WHITE);

    Engine engine(1); 
    float* valueOutput = new float[1]; 
    float* policyOutput = new float[2*NUM_POLICY_VALUES]; 

    engine.build("../../models/model.onnx", "trt.engine");
    engine.loadNetwork();
    engine.bind_executor_input(NUM_BUGHOUSE_VALUES()); 
    engine.bind_executor_policy(2*NUM_POLICY_VALUES); 
    engine.bind_executor_value(1); 
    engine.predict(inputPlanes, valueOutput, policyOutput); // Throwaway predict since first predict is always slow 

    print_board(inputPlanes);

    cout << valueOutput[0] << endl; 
    cout << policyOutput[0] << endl; */
    
    Bugboard board; 
    Agent agent;
    agent.set_is_running(true);

    /*board.set("r2q1bkr/ppp3p1/2n4p/3pP1N1/6b1/5N2/PPP2PPP/R1BQK2R[] w KQ - 1 10 | rnbqkb1r/ppp1pppp/5n2/3p4/3P4/5N2/PPPNPPPP/R1BQKB1R[BPnpp] b KQkq - 3 3");
    board.set_time(0, 1114); 
    board.set_time(1, 1149); 
    board.set_time(2, 1189);
    board.set_time(3, 1154);  */

    /*board.set("r1bq1b1r/ppp1k1pp/3npp2/4N1B1/3Q4/2N2N2/PPP2KPP/R6R[NQqbnPPP] w | r1b1k2r/ppp2ppp/2p1p3/6B1/B2nn3/2P1P3/P4PPP/R1B1K2R[pPP] b kq"); 
    board.set_time(0, 1800); 
    board.set_time(1, 1700); 
    board.set_time(2, 1700);
    board.set_time(3, 1800);  */
    
    /*board.set("rr6/4kppp/p1p1p3/2Pp4/3b4/3NPPn1/PPP1Q1P1/R4RK1[Nqb] w - - 0 27 | rn2k1nr/ppq2ppp/4pp2/bN1p1b2/8/2B1PN2/PPP2PPP/R2QKB1R[BBPPnp] b KQkq - 3 16"); 

    board.set_time(0, 933); 
    board.set_time(1, 1029); 
    board.set_time(2, 928);
    board.set_time(3, 1033);  

    board.set_move_time(5);
    agent.run_search(board, 500);*/

    /*board.set("1r3rk1/p4p1p/2pp1BpP/8/4P1b1/1Pq5/P1P2nPP/4RR1K[p] w - - 2 20 | r1bqkb1r/p1p2ppp/2p1pn2/3pN2b/3Pp2n/P1N1PP2/1P3PPP/R1BQKBNR[QBNPn] b KQkq - 1 11"); 

    board.set_time(0, 1048); 
    board.set_time(1, 1049); 
    board.set_time(2, 1048);
    board.set_time(3, 1048);  

    board.set_move_time(5);
    agent.run_search(board, 500);*/
    //UCI uci;
    //uci.loop();
    

    string token, cmd;
    do {
        if (!getline(cin, cmd)) 
            cmd = "quit";

        istringstream is(cmd);

        token.clear(); 
        is >> skipws >> token;

        if (token == "uci") cout << "uciok"  << endl;
        else if (token == "go")  {
            is >> token >> token;
            int move_time = std::stoi(token); 
            board.set_move_time(move_time / 100);
            agent.run_search(board, move_time);
        }
        else if (token == "position") {
            is >> token;
            string fen; 
            while (is >> token) {
                fen += token + " ";
            }
            board.set(fen); 
        }
        else if (token == "time") {
            int idx = 0; 
            while (is >> token) {
                board.set_time(idx++, stoi(token)); 
            }
        }

    } while (token != "quit"); 
}
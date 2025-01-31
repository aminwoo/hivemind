#include <iostream>
#include <sstream>
#include <string>

#include "uci.h"

using namespace std;


void UCI::stop() {
    if (ongoingSearch) {
        agent.set_is_running(false);
        mainSearchThread->join();
        ongoingSearch = false;
    } 
}

void UCI::position(istringstream& is) {
    string token, fen;
    is >> token;

    while (is >> token) {
        fen += token + " ";
    }

    board.set(fen); 
}

void UCI::go(istringstream& is) {
    string token;
    is >> token >> token;

    int move_time = std::stoi(token); 
    ongoingSearch = true; 
    agent.set_is_running(true);
    mainSearchThread = new thread{&Agent::run_search, ref(agent), ref(board), ref(engine), move_time};
}

void UCI::loop() {
    if (!engine.loadNetwork("./networks/model.onnx", "engine.trt")) {
        std::cerr << "Failed to load network" << std::endl;
        return;
    }

    string token, cmd;

    do {
        if (!getline(cin, cmd)) // Block here waiting for input or EOF
            cmd = "quit";

        istringstream is(cmd);

        token.clear(); // Avoid a stale if getline() returns empty or blank line
        is >> skipws >> token;

        if (token == "uci")             cout << "uciok"  << endl;
        else if (token == "go")         go(is);
        else if (token == "position")   position(is);
        else if (token == "stop")       stop();

    } while (token != "quit"); // Command line args are one-shot
}

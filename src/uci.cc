#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <sstream>
#include <vector>
#include <memory>

#include "uci.h"

using namespace std;


UCI::UCI() : mainSearchThread(nullptr) {

}

UCI::~UCI() {
    // Join the main search thread if running.
    if (mainSearchThread && mainSearchThread->joinable()) {
        mainSearchThread->join();
        delete mainSearchThread;
    }
    // Engines will automatically be cleaned up when the vector is destroyed.
}

void UCI::initializeEngines(const std::vector<int>& deviceIds) {
    // Clear any existing engines.
    engines.clear();

    // Define the file names for the network.
    // Adjust these filenames as needed.
    const std::string onnxFile = "./model.onnx";
    const std::string engineFile = "engine.trt";

    // For each device ID, create a new Engine, load the network, and store it.
    for (int deviceId : deviceIds) {
        // Create a new engine instance on the given GPU.
        auto enginePtr = std::make_unique<Engine>(deviceId);
        
        // Attempt to load the network (build or deserialize).
        if (!enginePtr->loadNetwork(onnxFile, engineFile)) {
            std::cerr << "Error: Failed to load engine on device " << deviceId << std::endl;
            // Depending on your error-handling strategy, you might choose to:
            // - continue to try loading for other devices, or
            // - abort the initialization (e.g., by returning or throwing an exception).
        } else {
            std::cout << "Engine successfully loaded on device " << deviceId << std::endl;
            engines.push_back(std::move(enginePtr));
        }
    }

    // Now that we know how many GPUs (and therefore engines) are available,
    // create the Agent with a number of threads equal to the number of GPUs.
    agent = std::make_unique<Agent>(static_cast<int>(deviceIds.size()));

}


void UCI::stop() {
    if (ongoingSearch) {
        agent->set_is_running(false);
        mainSearchThread->join();
        ongoingSearch = false;
    } 
}

void UCI::position(istringstream& is) {
    std::string token;
    is >> token;
    
    // Set the board position
    if (token == "startpos") {
        // Use a predefined starting FEN for the initial position.
        board.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1|rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    }
    else if (token == "fen") {
        // Build the FEN string from the next six tokens.
        std::string fen;
        while (is >> token) {
            fen += token + " ";
        }
        board.set(fen);
    }
    else {
        // Unrecognized position command; you might want to handle this error.
        return;
    }
    
    // Apply moves if they are provided.
    if (is >> token && token == "moves") {
        // Parse move list (if any)
        while (is >> token) {
            int boardNum = token[0] - '1'; // '1' becomes 0, '2' becomes 1.
            std::string moveStr = token.substr(1); // Extract move string without board indicator
            Stockfish::Move m = Stockfish::UCI::to_move(*board.pos[boardNum], moveStr);
            if (m == Stockfish::MOVE_NONE)
                break;  // Stop if an invalid move is encountered.
            board.push_move(boardNum, m);
        }
    }
}

void UCI::go(std::istringstream& is) {
    std::string token;
    // Consume the first two tokens (e.g. "go" and "movetime")
    is >> token >> token;
    
    int moveTime = std::stoi(token);
    ongoingSearch = true;
    agent->set_is_running(true);

    // Ensure that engines have been initialized.
    if (engines.empty()) {
        std::cerr << "Error: No engines have been initialized!" << std::endl;
        return;
    }

    // Build a vector of raw Engine pointers from the unique_ptr collection.
    std::vector<Engine*> enginePtrs;
    enginePtrs.reserve(engines.size());
    for (const auto& eng : engines) {
        enginePtrs.push_back(eng.get());
    }

    // Launch the search thread using a lambda that calls Agent::run_search,
    // passing in the board, the collection of engines, and the move time.
    mainSearchThread = new std::thread([this, enginePtrs, moveTime]() {
        agent->run_search(board, enginePtrs, moveTime, teamSide, canSit);
    });
}

void UCI::team(std::istringstream& is) {
  // Expect input in the format: "team <side>"
  std::string side;
  is >> side; 

  // Set team side based on the side token.
  if (side == "white") {
    teamSide = Stockfish::WHITE;
  } else if (side == "black") {
    teamSide = Stockfish::BLACK;
  } else {
    return;
  }

  //std::cout << "Team side set to: " << teamSide << std::endl;
}

void UCI::mode(std::istringstream& is) {
  // Expect input in the format: "set mode <go/sit>"
  std::string token;
  is >> token >> token; 

  if (token == "sit") {
    canSit = true;
  } else if (token == "go") {
    canSit = false;
  } else {
    return;
  }
}


void UCI::loop() {
    string token, cmd;

    do {
        if (!getline(cin, cmd)) // Block here waiting for input or EOF
            cmd = "quit";

        istringstream is(cmd);

        token.clear(); // Avoid a stale if getline() returns empty or blank line
        is >> skipws >> token;

        if (token == "uci")             cout << "uciok"  << endl;
        else if (token == "go")         go(is);
        else if (token == "team")       team(is);
        else if (token == "set")        mode(is);
        else if (token == "position")   position(is);
        else if (token == "stop")       stop();

    } while (token != "quit"); // Command line args are one-shot
}

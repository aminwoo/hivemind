#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <sstream>
#include <vector>

#include <memory>
#include "onnx_utils.h"
#include "planes.h"
#include "utils.h"

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

    // Automatically find the latest ONNX file in the networks directory.
    const std::string onnxFile = findLatestOnnxFile("./networks");
    if (onnxFile.empty()) {
        std::cerr << "Error: No ONNX file found in ./networks directory." << std::endl;
        return;
    }
    // For each device ID, create a new Engine, load the network, and store it.
    for (int deviceId : deviceIds) {
        const std::string engineFile = getEnginePath(onnxFile, "fp16", BATCH_SIZE, deviceId, "v1");
        
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

    // Create the single-threaded Agent
    agent = std::make_unique<Agent>();
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
        agent->run_search(board, enginePtrs, moveTime, teamSide, teamHasTimeAdvantage);
    });
}

void UCI::setoption(std::istringstream& is) {
    std::string token;
    is >> token; 
    if (token != "name") return;
    std::string name;
    is >> name;
    is >> token; 
    if (token != "value") return;
    std::string value;
    is >> value;
    if (name == "Team") {
        if (value == "white") {
            teamSide = Stockfish::WHITE;
        } else if (value == "black") {
            teamSide = Stockfish::BLACK;
        }
    } else if (name == "Mode") {
        if (value == "sit") {
            teamHasTimeAdvantage = true;
        } else if (value == "go") {
            teamHasTimeAdvantage = false;
        }
    }
}

void UCI::send_uci_response() {
    cout << "id name hivemind" << endl;
    cout << "id author aminwoo\n" << endl;
    cout << "option name Team type combo default white var white var black" << endl;
    cout << "option name Mode type combo default go var sit var go" << endl;
    cout << "uciok" << endl;
}

void UCI::policy() {
    if (engines.empty()) {
        cerr << "Error: No engines have been initialized!" << endl;
        return;
    }

    // Allocate inference buffers
    float* obs = new float[BATCH_SIZE * NB_INPUT_VALUES()];
    float* value = new float[BATCH_SIZE];
    float* piA = new float[BATCH_SIZE * NB_POLICY_VALUES()];
    float* piB = new float[BATCH_SIZE * NB_POLICY_VALUES()];

    // Convert board to planes
    board_to_planes(board, obs, teamSide, teamHasTimeAdvantage);

    // Run inference
    Engine* engine = engines[0].get();
    if (!engine->runInference(obs, value, piA, piB)) {
        cerr << "Inference failed" << endl;
        delete[] obs;
        delete[] value;
        delete[] piA;
        delete[] piB;
        return;
    }

    cout << "Value: " << value[0] << endl;
    cout << endl;

    // Board A policy
    cout << "Board A (" << board.fen(BOARD_A) << "):" << endl;
    if (board.side_to_move(BOARD_A) == teamSide) {
        vector<Stockfish::Move> actionsA = board.legal_moves(BOARD_A);
        actionsA.push_back(Stockfish::MOVE_NULL);  // Add sit option
        vector<float> priorsA = get_normalized_probability(piA, actionsA, BOARD_A, board);
        
        // Sort by probability (descending)
        vector<size_t> indices = argsort(priorsA);
        for (size_t idx : indices) {
            string moveStr = (actionsA[idx] == Stockfish::MOVE_NULL) 
                            ? "pass" : board.uci_move(BOARD_A, actionsA[idx]);
            cout << "  " << moveStr << ": " << priorsA[idx] << endl;
        }
    } else {
        cout << "  (not our turn)" << endl;
    }
    cout << endl;

    // Board B policy
    cout << "Board B (" << board.fen(BOARD_B) << "):" << endl;
    if (board.side_to_move(BOARD_B) == ~teamSide) {
        vector<Stockfish::Move> actionsB = board.legal_moves(BOARD_B);
        actionsB.push_back(Stockfish::MOVE_NULL);  // Add sit option
        vector<float> priorsB = get_normalized_probability(piB, actionsB, BOARD_B, board);
        
        // Sort by probability (descending)
        vector<size_t> indices = argsort(priorsB);
        for (size_t idx : indices) {
            string moveStr = (actionsB[idx] == Stockfish::MOVE_NULL) 
                            ? "pass" : board.uci_move(BOARD_B, actionsB[idx]);
            cout << "  " << moveStr << ": " << priorsB[idx] << endl;
        }
    } else {
        cout << "  (not our turn)" << endl;
    }

    delete[] obs;
    delete[] value;
    delete[] piA;
    delete[] piB;
}


void UCI::loop() {
    string token, cmd;

    do {
        if (!getline(cin, cmd)) // Block here waiting for input or EOF
            cmd = "quit";

        istringstream is(cmd);

        token.clear(); // Avoid a stale if getline() returns empty or blank line
        is >> skipws >> token;

        if (token == "uci")             send_uci_response();
        else if (token == "isready")    cout << "readyok" << endl;
        else if (token == "go")         go(is);
        else if (token == "setoption")  setoption(is);
        else if (token == "position")   position(is);
        else if (token == "stop")       stop();
        else if (token == "policy")     policy();

    } while (token != "quit"); // Command line args are one-shot
}

/*
 * Hivemind - Bughouse Chess Engine
 * Model Evaluation - Compare two neural network models
 * Plays tournament games with no exploration noise
 */

#include "model_eval.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <cuda_runtime.h>

#include "../constants.h"
#include "../globals.h"
#include "../planes.h"
#include "../search_params.h"
#include "../onnx_utils.h"
#include "../utils.h"

using namespace std;

ModelEvaluator::ModelEvaluator(const vector<Engine*>& newModelEngines,
                               const vector<Engine*>& oldModelEngines,
                               const EvalSettings& settings)
    : newModelEngines(newModelEngines), 
      oldModelEngines(oldModelEngines),
      settings(settings),
      running(false)
{
    // Create agents for both models with configured thread counts
    int player1Threads = settings.usePlayerConfigs ? settings.player1.numSearchThreads : 0;
    int player2Threads = settings.usePlayerConfigs ? settings.player2.numSearchThreads : 0;
    newModelAgent = make_unique<Agent>(player1Threads);
    oldModelAgent = make_unique<Agent>(player2Threads);
    
    // Create PGN output file if specified
    if (!settings.outputPgnPath.empty()) {
        filesystem::create_directories(filesystem::path(settings.outputPgnPath).parent_path());
        ofstream pgnFile(settings.outputPgnPath, ios::trunc);
        pgnFile << "; Model Evaluation Tournament\n";
        pgnFile << "; Player 1 vs Player 2\n";
        pgnFile << "; " << settings.numGames << " games\n\n";
        pgnFile.close();
    }
    
    // Create GUI state writer if enabled
    if (settings.enableGui) {
        filesystem::create_directories(filesystem::path(settings.guiStatePath).parent_path());
        guiWriter = make_unique<GuiStateWriter>(settings.guiStatePath);
        guiWriter->clear();
        cout << "GUI enabled: " << settings.guiStatePath << endl;
    }
}

ModelEvaluator::~ModelEvaluator() {
    stop();
}

void ModelEvaluator::stop() {
    running = false;
}

EvalStats ModelEvaluator::getStats() const {
    lock_guard<mutex> lock(statsMutex);
    return stats;
}

EvalStats ModelEvaluator::run() {
    running = true;
    startTime = chrono::steady_clock::now();
    
    cout << "========================================" << endl;
    cout << "     Model Evaluation Tournament" << endl;
    cout << "========================================" << endl;
    cout << "Games to play: " << settings.numGames << endl;
    
    if (settings.usePlayerConfigs) {
        cout << "--- Player 1: " << settings.player1.name << " ---" << endl;
        if (settings.player1.moveTimeMs > 0) {
            cout << "  Time/move: " << settings.player1.moveTimeMs << "ms" << endl;
        } else {
            cout << "  Nodes/move: " << settings.player1.nodesPerMove << endl;
        }
        cout << "  Batch size: " << settings.player1.batchSize << endl;
        cout << "  CPUCT: " << settings.player1.cpuctInit << endl;
        cout << "--- Player 2: " << settings.player2.name << " ---" << endl;
        if (settings.player2.moveTimeMs > 0) {
            cout << "  Time/move: " << settings.player2.moveTimeMs << "ms" << endl;
        } else {
            cout << "  Nodes/move: " << settings.player2.nodesPerMove << endl;
        }
        cout << "  Batch size: " << settings.player2.batchSize << endl;
        cout << "  CPUCT: " << settings.player2.cpuctInit << endl;
    } else {
        if (settings.moveTimeMs > 0) {
            cout << "Time per move: " << settings.moveTimeMs << "ms" << endl;
        } else {
            cout << "Nodes per move: " << settings.nodesPerMove << endl;
        }
        cout << "Temperature: " << settings.temperature << " (decays to 0 after " << settings.temperatureDecayMoves << " moves)" << endl;
        cout << "Dirichlet: disabled (deterministic eval)" << endl;
    }
    
    if (settings.enableGui) {
        cout << "GUI: enabled at " << settings.guiStatePath << endl;
    }
    
    cout << "Max game length: " << settings.maxGameLength << " plies" << endl;
    cout << "========================================" << endl << endl;
    
    for (size_t gameNum = 0; gameNum < settings.numGames && running; gameNum++) {
        // Alternate colors: even games = player 1 is white, odd = player 1 is black
        bool player1IsWhite = (gameNum % 2 == 0);
        
        GameResult result = playGame(player1IsWhite, gameNum + 1);
        
        // Update statistics
        {
            lock_guard<mutex> lock(statsMutex);
            
            if (result == GameResult::WHITE_WINS) {
                if (player1IsWhite) {
                    stats.player1Wins++;
                    stats.player1WinsAsWhite++;
                } else {
                    stats.player1Losses++;
                    stats.player1LossesAsBlack++;
                }
            } else if (result == GameResult::BLACK_WINS) {
                if (player1IsWhite) {
                    stats.player1Losses++;
                    stats.player1LossesAsWhite++;
                } else {
                    stats.player1Wins++;
                    stats.player1WinsAsBlack++;
                }
            } else {
                stats.draws++;
            }
        }
        
        // Print progress every 10 games or at end
        if ((gameNum + 1) % 10 == 0 || gameNum == settings.numGames - 1) {
            printStats();
        }
    }
    
    running = false;
    
    cout << endl << "========================================" << endl;
    cout << "     Final Results" << endl;
    cout << "========================================" << endl;
    printStats();
    
    return getStats();
}

GameResult ModelEvaluator::playGame(bool newModelIsWhite, size_t gameNumber) {
    Board board;
    BughouseGamePGN pgn;
    vector<pair<int, string>> gameMoves;  // For opening tracking
    
    pgn.new_game();
    
    // Set player names based on configuration
    string player1Name = settings.usePlayerConfigs ? settings.player1.name : "NewModel";
    string player2Name = settings.usePlayerConfigs ? settings.player2.name : "OldModel";
    
    pgn.whiteTeam = newModelIsWhite ? player1Name : player2Name;
    pgn.blackTeam = newModelIsWhite ? player2Name : player1Name;
    pgn.whiteBoardA = newModelIsWhite ? "P1-A" : "P2-A";
    pgn.blackBoardA = newModelIsWhite ? "P2-B" : "P1-B";
    pgn.whiteBoardB = newModelIsWhite ? "P2-A" : "P1-A";
    pgn.blackBoardB = newModelIsWhite ? "P1-B" : "P2-B";
    pgn.round = to_string(gameNumber);
    
    GameResult result = GameResult::NO_RESULT;
    size_t ply = 0;
    Stockfish::Color currentSide = Stockfish::WHITE;
    
    // Alternate which MODEL has time advantage between games
    // Game 1,3,5...: Player1 (New Model) has advantage
    // Game 2,4,6...: Player2 (Old Model) has advantage
    bool player1HasTimeAdvantage = (gameNumber % 2 == 1);
    
    // Translate model advantage to color advantage
    // If new model is white and has advantage, white has advantage
    // If new model is black and doesn't have advantage (old has it), white has advantage
    bool whiteHasTimeAdvantage = (newModelIsWhite == player1HasTimeAdvantage);
    pgn.whiteTeamHadTimeAdvantage = whiteHasTimeAdvantage;
    
    // Track time for both sides (starting at 180.0, decreasing by 0.1 per move)
    float whiteTime = 180.0f;
    float blackTime = 180.0f;
    
    // Allocate planes buffer
    vector<float> inputPlanes(NB_INPUT_VALUES());
    
    // Helper to populate RLSettings from PlayerConfig
    auto populateRLSettings = [&](RLSettings& rl, const PlayerConfig& pc) {
        rl.nodesPerMove = pc.nodesPerMove;
        rl.moveTimeMs = pc.moveTimeMs;
        rl.temperature = pc.temperature;
        rl.temperatureDecayMoves = pc.temperatureDecayMoves;
        rl.dirichletEpsilon = pc.dirichletEpsilon;
        rl.dirichletAlpha = pc.dirichletAlpha;
        rl.nodeRandomFactor = 0.0f;  // No randomization in eval
        rl.maxGameLength = settings.maxGameLength;
        
        // PUCT parameters
        rl.cpuctInit = pc.cpuctInit;
        rl.cpuctBase = pc.cpuctBase;
        
        // First Play Urgency
        rl.fpuReduction = pc.fpuReduction;
        
        // MCGS settings
        rl.enableMCGS = pc.enableMCGS;
        rl.enableTranspositions = pc.enableTranspositions;
        
        // Draw contempt
        rl.drawContempt = pc.drawContempt;
        
        // Progressive widening
        rl.pwCoefficient = pc.pwCoefficient;
        rl.pwExponent = pc.pwExponent;
        
        // Q-value settings
        rl.qValueWeight = pc.qValueWeight;
        rl.qVetoDelta = pc.qVetoDelta;
    };
    
    // Create per-player RL settings
    RLSettings player1Settings;
    RLSettings player2Settings;
    
    if (settings.usePlayerConfigs) {
        populateRLSettings(player1Settings, settings.player1);
        populateRLSettings(player2Settings, settings.player2);
    } else {
        // Default settings with temperature for game variety
        player1Settings.nodesPerMove = settings.nodesPerMove;
        player1Settings.moveTimeMs = settings.moveTimeMs;  // Fixed movetime support
        player1Settings.temperature = settings.temperature;
        player1Settings.temperatureDecayMoves = settings.temperatureDecayMoves;
        player1Settings.dirichletEpsilon = 0.0f;     // No Dirichlet noise for eval
        player1Settings.dirichletAlpha = 0.2f;
        player1Settings.nodeRandomFactor = 0.0f;
        player1Settings.maxGameLength = settings.maxGameLength;
        
        player2Settings = player1Settings;  // Same settings for both
    }
    
    while (result == GameResult::NO_RESULT && ply < settings.maxGameLength) {
        // Determine which agent/engines/settings to use based on current side
        Agent* agent;
        const vector<Engine*>* engines;
        RLSettings* currentSettings;
        
        bool isPlayer1Turn = (currentSide == Stockfish::WHITE) == newModelIsWhite;
        
        if (isPlayer1Turn) {
            // Player 1's turn (new model in classic eval)
            agent = newModelAgent.get();
            engines = &newModelEngines;
            currentSettings = &player1Settings;
        } else {
            // Player 2's turn (old model in classic eval)
            agent = oldModelAgent.get();
            engines = &oldModelEngines;
            currentSettings = &player2Settings;
        }
        
        // Determine if current side has time advantage
        bool teamHasTimeAdvantage = (currentSide == Stockfish::WHITE) ? whiteHasTimeAdvantage : !whiteHasTimeAdvantage;
        
        // Get legal moves for current side
        auto legalMoves = board.legal_moves(currentSide, teamHasTimeAdvantage);
        
        if (legalMoves.empty()) {
            result = (currentSide == Stockfish::WHITE) ? GameResult::BLACK_WINS : GameResult::WHITE_WINS;
            break;
        }
        
        // Check for draw conditions
        if (board.is_draw()) {
            result = GameResult::DRAW;
            break;
        }
        
        // Get search limits for current player
        size_t nodesForThisMove = currentSettings->nodesPerMove;
        int timeForThisMove = currentSettings->moveTimeMs;
        
        // Asymmetric node allocation (like selfplay):
        // - Attacker (time advantage): 0.5x nodes to force efficient play
        // - Defender (time disadvantage): 1.5x nodes for better defense
        constexpr float ATTACKER_NODE_MULT = 0.5f;
        constexpr float DEFENDER_NODE_MULT = 1.5f;
        if (teamHasTimeAdvantage) {
            nodesForThisMove = static_cast<size_t>(nodesForThisMove * ATTACKER_NODE_MULT);
        } else {
            nodesForThisMove = static_cast<size_t>(nodesForThisMove * DEFENDER_NODE_MULT);
        }
        nodesForThisMove = max(static_cast<size_t>(1), nodesForThisMove);
        
        // Apply temperature decay like selfplay: start at configured temp, decay to 0
        float tempForThisMove = currentSettings->temperature;
        size_t decayMoves = currentSettings->temperatureDecayMoves;
        if (decayMoves > 0 && ply < decayMoves) {
            float progress = static_cast<float>(ply) / decayMoves;
            tempForThisMove = currentSettings->temperature * (1.0f - progress);
        } else if (decayMoves > 0) {
            tempForThisMove = 0.0f;  // Decay complete
        }
        
        // Run MCTS search with player-specific settings
        // If moveTimeMs > 0, search uses time; otherwise uses nodes
        SearchOptions opts;
        if (timeForThisMove > 0) {
            opts = SearchOptions::selfplay(timeForThisMove, *currentSettings);
        } else {
            opts = SearchOptions::selfplay(nodesForThisMove, *currentSettings);
        }
        JointActionCandidate bestAction = agent->run_search(
            board, *engines, currentSide, teamHasTimeAdvantage, opts);
        
        // Sample action with temperature from visit distribution (like selfplay)
        // This allows for more varied play and realistic draw rates
        auto rootNode = agent->get_root_node();
        if (rootNode && rootNode->is_expanded() && tempForThisMove > 0.0f) {
            auto childActionVisits = rootNode->get_child_action_visits();
            if (!childActionVisits.empty()) {
                bestAction = sample_action_with_temperature(childActionVisits, tempForThisMove);
            }
        }
        
        Stockfish::Move moveA = bestAction.moveA;
        Stockfish::Move moveB = bestAction.moveB;
        
        // Record moves for PGN and opening tracking
        // Decrement time for current side before recording
        float currentTime = (currentSide == Stockfish::WHITE) ? whiteTime : blackTime;
        
        if (moveA != Stockfish::MOVE_NONE) {
            string moveStr = board.san_move(0, moveA);
            pgn.add_move(0, moveStr, currentTime);
            if (ply < settings.openingMovesToTrack * 2) {
                gameMoves.push_back({0, moveStr});
            }
        }
        if (moveB != Stockfish::MOVE_NONE) {
            string moveStr = board.san_move(1, moveB);
            pgn.add_move(1, moveStr, currentTime);
            if (ply < settings.openingMovesToTrack * 2) {
                gameMoves.push_back({1, moveStr});
            }
        }
        
        // Decrement time after move is recorded
        if (currentSide == Stockfish::WHITE) {
            whiteTime -= 0.1f;
        } else {
            blackTime -= 0.1f;
        }
        
        // Apply the joint move
        board.make_moves(moveA, moveB);
        ply++;
        
        // Update GUI if enabled
        if (guiWriter) {
            EvalStats currentStats = getStats();
            guiWriter->update(board, pgn.gameMoves, gameNumber, settings.numGames, ply,
                              ~currentSide,  // Next side to move
                              pgn.whiteTeam, pgn.blackTeam, "ongoing",
                              currentStats.player1Wins, currentStats.player1Losses, currentStats.draws);
        }
        
        // Check for mate after move
        Stockfish::Color opponentSide = ~currentSide;
        bool opponentTimeAdvantage = !teamHasTimeAdvantage;
        
        if (board.is_checkmate(opponentSide, opponentTimeAdvantage)) {
            result = (currentSide == Stockfish::WHITE) ? GameResult::WHITE_WINS : GameResult::BLACK_WINS;
        }
        
        // Switch sides
        currentSide = ~currentSide;
    }
    
    // Handle max game length
    if (result == GameResult::NO_RESULT && ply >= settings.maxGameLength) {
        result = GameResult::DRAW;
    }
    
    pgn.set_result(result);
    
    // Final GUI update with result
    if (guiWriter) {
        EvalStats currentStats = getStats();
        string resultStr = pgn.result;
        guiWriter->update(board, pgn.gameMoves, gameNumber, settings.numGames, ply,
                          currentSide, pgn.whiteTeam, pgn.blackTeam, resultStr,
                          currentStats.player1Wins, currentStats.player1Losses, currentStats.draws);
    }
    
    // Update game length stats
    {
        lock_guard<mutex> lock(statsMutex);
        stats.totalPlies += ply;
        stats.minGameLength = min(stats.minGameLength, ply);
        stats.maxGameLength = max(stats.maxGameLength, ply);
    }
    
    // Record opening moves
    recordOpeningMoves(gameMoves);
    
    // Write to PGN if configured
    if (!settings.outputPgnPath.empty()) {
        writeGameToPgn(pgn);
    }
    
    // Print game result if verbose
    if (settings.verbose) {
        string resultStr;
        if (result == GameResult::WHITE_WINS) {
            resultStr = newModelIsWhite ? "New wins" : "Old wins";
        } else if (result == GameResult::BLACK_WINS) {
            resultStr = newModelIsWhite ? "Old wins" : "New wins";
        } else {
            resultStr = "Draw";
        }
        
        cout << "Game " << gameNumber << ": " << resultStr 
             << " (" << ply << " plies)" << endl;
    }
    
    return result;
}

void ModelEvaluator::recordOpeningMoves(const vector<pair<int, string>>& moves) {
    if (moves.empty()) return;
    
    // Create opening string from first N moves
    stringstream ss;
    size_t count = 0;
    for (const auto& [boardNum, moveStr] : moves) {
        if (count >= settings.openingMovesToTrack) break;
        if (count > 0) ss << " ";
        ss << (boardNum == BOARD_A ? "A:" : "B:") << moveStr;
        count++;
    }
    
    string opening = ss.str();
    
    lock_guard<mutex> lock(statsMutex);
    stats.openingMoves[opening]++;
}

void ModelEvaluator::writeGameToPgn(const BughouseGamePGN& pgn) {
    lock_guard<mutex> lock(pgnMutex);
    
    ofstream pgnFile(settings.outputPgnPath, ios::app);
    pgnFile << pgn;
    pgnFile.close();
}

vector<pair<string, size_t>> ModelEvaluator::getTopOpenings(size_t n) const {
    vector<pair<string, size_t>> openings;
    
    {
        lock_guard<mutex> lock(statsMutex);
        for (const auto& [opening, count] : stats.openingMoves) {
            openings.push_back({opening, count});
        }
    }
    
    // Sort by count descending
    sort(openings.begin(), openings.end(), 
         [](const auto& a, const auto& b) { return a.second > b.second; });
    
    if (openings.size() > n) {
        openings.resize(n);
    }
    
    return openings;
}

void ModelEvaluator::printStats() const {
    EvalStats s = getStats();
    
    auto now = chrono::steady_clock::now();
    float elapsedMin = chrono::duration_cast<chrono::milliseconds>(now - startTime).count() / 60000.0f;
    size_t totalGames = s.player1Wins + s.player1Losses + s.draws;
    
    // Determine player names for display
    string player1Name = settings.usePlayerConfigs ? settings.player1.name : "Player 1";
    string player2Name = settings.usePlayerConfigs ? settings.player2.name : "Player 2";
    
    cout << endl;
    cout << "--- Evaluation Progress (" << totalGames << "/" << settings.numGames << " games) ---" << endl;
    cout << endl;
    
    // W/D/L statistics
    cout << "Win/Draw/Loss (" << player1Name << " perspective):" << endl;
    cout << "  Wins:   " << setw(5) << s.player1Wins << " (" << fixed << setprecision(1) << s.winRate() << "%)" << endl;
    cout << "  Draws:  " << setw(5) << s.draws << " (" << fixed << setprecision(1) << s.drawRate() << "%)" << endl;
    cout << "  Losses: " << setw(5) << s.player1Losses << " (" << fixed << setprecision(1) << s.lossRate() << "%)" << endl;
    cout << endl;
    
    // Elo estimate
    float elo = s.eloDifference();
    cout << "Estimated Elo difference: " << showpos << fixed << setprecision(0) << elo << noshowpos << endl;
    cout << endl;
    
    // Color-specific stats
    cout << "By color (" << player1Name << "):" << endl;
    cout << "  As White: +" << s.player1WinsAsWhite << " -" << s.player1LossesAsWhite << endl;
    cout << "  As Black: +" << s.player1WinsAsBlack << " -" << s.player1LossesAsBlack << endl;
    cout << endl;
    
    // Game length stats
    cout << "Game length:" << endl;
    cout << "  Average: " << fixed << setprecision(1) << s.avgGameLength() << " plies" << endl;
    if (s.minGameLength != SIZE_MAX) {
        cout << "  Min:     " << s.minGameLength << " plies" << endl;
        cout << "  Max:     " << s.maxGameLength << " plies" << endl;
    }
    cout << endl;
    
    // Top openings
    auto topOpenings = getTopOpenings(5);
    if (!topOpenings.empty()) {
        cout << "Most common opening sequences:" << endl;
        for (size_t i = 0; i < topOpenings.size(); i++) {
            float pct = (totalGames > 0) ? (100.0f * topOpenings[i].second / totalGames) : 0.0f;
            cout << "  " << (i + 1) << ". " << topOpenings[i].first 
                 << " (" << topOpenings[i].second << " games, " 
                 << fixed << setprecision(1) << pct << "%)" << endl;
        }
        cout << endl;
    }
    
    // Timing
    float gpm = (elapsedMin > 0.01f) ? totalGames / elapsedMin : 0.0f;
    cout << "Elapsed: " << fixed << setprecision(1) << elapsedMin << " min, " 
         << fixed << setprecision(2) << gpm << " games/min" << endl;
    cout << endl;
}

void run_model_eval(const string& newModelPath,
                    const string& oldModelPath,
                    const EvalSettings& settings) {
    
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        cerr << "No CUDA devices found" << endl;
        return;
    }
    
    cout << "Loading models..." << endl;
    cout << "  New model: " << newModelPath << endl;
    cout << "  Old model: " << oldModelPath << endl;
    
    // Load new model engines
    vector<Engine*> newModelEngines;
    for (int i = 0; i < deviceCount; i++) {
        Engine* engine = new Engine(i);
        string engineFile = getEnginePath(newModelPath, "fp16", SearchParams::BATCH_SIZE, i, "v1");
        if (!engine->loadNetwork(newModelPath, engineFile)) {
            cerr << "Failed to load new model on GPU " << i << endl;
            for (auto* e : newModelEngines) delete e;
            return;
        }
        newModelEngines.push_back(engine);
    }
    
    // Load old model engines
    vector<Engine*> oldModelEngines;
    for (int i = 0; i < deviceCount; i++) {
        Engine* engine = new Engine(i);
        string engineFile = getEnginePath(oldModelPath, "fp16", SearchParams::BATCH_SIZE, i, "v1");
        if (!engine->loadNetwork(oldModelPath, engineFile)) {
            cerr << "Failed to load old model on GPU " << i << endl;
            for (auto* e : newModelEngines) delete e;
            for (auto* e : oldModelEngines) delete e;
            return;
        }
        oldModelEngines.push_back(engine);
    }
    
    cout << "Models loaded successfully." << endl << endl;
    
    // Run evaluation
    ModelEvaluator evaluator(newModelEngines, oldModelEngines, settings);
    EvalStats results = evaluator.run();
    
    // Cleanup
    for (auto* e : newModelEngines) delete e;
    for (auto* e : oldModelEngines) delete e;
}

void run_param_eval(const string& modelPath,
                    const EvalSettings& settings) {
    
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        cerr << "No CUDA devices found" << endl;
        return;
    }
    
    cout << "========================================" << endl;
    cout << "   Parameter Evaluation (Same Model)" << endl;
    cout << "========================================" << endl;
    cout << "Model: " << modelPath << endl;
    cout << endl;
    
    // Helper to print player config
    auto printPlayerConfig = [](const string& label, const PlayerConfig& p) {
        cout << label << ": " << p.name << endl;
        if (p.moveTimeMs > 0) {
            cout << "  Time/move: " << p.moveTimeMs << "ms" << endl;
        } else {
            cout << "  Nodes/move: " << p.nodesPerMove << endl;
        }
        cout << "  Batch size: " << p.batchSize << endl;
        cout << "  Threads: " << p.numSearchThreads << endl;
        cout << "  CPUCT: " << p.cpuctInit << " (base: " << p.cpuctBase << ")" << endl;
        cout << "  FPU reduction: " << p.fpuReduction << endl;
        cout << "  Draw contempt: " << p.drawContempt << endl;
        cout << "  Progressive widening: coef=" << p.pwCoefficient << ", exp=" << p.pwExponent << endl;
        cout << "  MCGS: " << (p.enableMCGS ? "on" : "off") 
             << ", Transpositions: " << (p.enableTranspositions ? "on" : "off") << endl;
        cout << "  Q-value: weight=" << p.qValueWeight << ", veto=" << p.qVetoDelta << endl;
    };
    
    printPlayerConfig("Player 1", settings.player1);
    cout << endl;
    printPlayerConfig("Player 2", settings.player2);
    cout << "========================================" << endl << endl;
    
    // Load engines for player 1 (may use different batch size)
    cout << "Loading engines for Player 1..." << endl;
    vector<Engine*> player1Engines;
    for (int i = 0; i < deviceCount; i++) {
        Engine* engine = new Engine(i);
        string engineFile = getEnginePath(modelPath, "fp16", settings.player1.batchSize, i, "v1");
        if (!engine->loadNetwork(modelPath, engineFile)) {
            cerr << "Failed to load model for Player 1 on GPU " << i << endl;
            for (auto* e : player1Engines) delete e;
            return;
        }
        player1Engines.push_back(engine);
    }
    
    // Load engines for player 2 (may use different batch size)
    cout << "Loading engines for Player 2..." << endl;
    vector<Engine*> player2Engines;
    for (int i = 0; i < deviceCount; i++) {
        Engine* engine = new Engine(i);
        string engineFile = getEnginePath(modelPath, "fp16", settings.player2.batchSize, i, "v1");
        if (!engine->loadNetwork(modelPath, engineFile)) {
            cerr << "Failed to load model for Player 2 on GPU " << i << endl;
            for (auto* e : player1Engines) delete e;
            for (auto* e : player2Engines) delete e;
            return;
        }
        player2Engines.push_back(engine);
    }
    
    cout << "Engines loaded successfully." << endl << endl;
    
    // Create settings with usePlayerConfigs enabled
    EvalSettings paramSettings = settings;
    paramSettings.usePlayerConfigs = true;
    
    // Run evaluation
    ModelEvaluator evaluator(player1Engines, player2Engines, paramSettings);
    EvalStats results = evaluator.run();
    
    // Cleanup
    for (auto* e : player1Engines) delete e;
    for (auto* e : player2Engines) delete e;
}

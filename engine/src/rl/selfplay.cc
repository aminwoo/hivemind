/*
 * Hivemind - Bughouse Chess Engine
 * Self-play for reinforcement learning
 * Adapted from CrazyAra's SelfPlay
 * 
 * Generates AlphaZero-style training data with MCTS visit distributions
 */

#include "selfplay.h"

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <unordered_map>

#include "../globals.h"
#include "../planes.h"
#include "../search_params.h"

using namespace std;

SelfPlay::SelfPlay(const RLSettings& settings, const vector<Engine*>& engines)
    : settings(settings), engines(engines), running(false), 
      gamesGenerated(0), samplesGenerated(0), gamesPerMinute(0.0f)
{
    // Create output directory if it doesn't exist
    filesystem::create_directories(settings.outputDirectory);
    pgnFilePath = settings.outputDirectory + "/" + settings.pgnFileName;
    
    // Clear/create the PGN file
    ofstream pgnFile(pgnFilePath, ios::trunc);
    pgnFile.close();
    
    // Create agents for both teams
    agentWhiteTeam = make_unique<Agent>();
    agentBlackTeam = make_unique<Agent>();
    
    // Create training data writer
    string trainingDataDir = settings.outputDirectory + "/training_data";
    trainingDataWriter = make_unique<TrainingDataWriter>(trainingDataDir);
    
    // Seed random number generator
    srand(static_cast<unsigned>(time(nullptr)));
}

SelfPlay::~SelfPlay() {
    stop();
    // Flush any remaining training data
    if (trainingDataWriter) {
        trainingDataWriter->flush();
    }
}

void SelfPlay::go(size_t numberOfGames) {
    if (numberOfGames == 0) {
        numberOfGames = settings.numberOfGames;
    }
    
    running = true;
    gamesGenerated = 0;
    samplesGenerated = 0;
    startTime = chrono::steady_clock::now();
    
    cout << "=== Hivemind Self-Play ===" << endl;
    cout << "Generating " << (numberOfGames == 0 ? "unlimited" : to_string(numberOfGames)) << " games" << endl;
    cout << "Output PGN: " << pgnFilePath << endl;
    cout << "Output training data: " << settings.outputDirectory << "/training_data/" << endl;
    cout << "Nodes per move: " << settings.nodesPerMove << endl;
    cout << "=========================" << endl << endl;
    
    size_t gameCount = 0;
    while (running && (numberOfGames == 0 || gameCount < numberOfGames)) {
        // Alternate time advantage between teams for balanced training
        bool whiteHasTimeAdvantage = (gameCount % 2 == 0);
        
        (void)generate_game(whiteHasTimeAdvantage, true);
        
        gameCount++;
        gamesGenerated = gameCount;
        
        // Print statistics every 10 games
        if (gameCount % 10 == 0) {
            print_statistics();
        }
    }
    
    running = false;
    
    // Flush remaining training data
    if (trainingDataWriter) {
        trainingDataWriter->flush();
    }
    
    cout << endl << "=== Self-Play Complete ===" << endl;
    print_statistics();
}

void SelfPlay::stop() {
    running = false;
}

GameResult SelfPlay::generate_game(bool whiteHasTimeAdvantage, bool verbose) {
    Board board;
    BughouseGamePGN pgn;
    GameSampleBuffer gameSamples;  // Buffer to collect samples until game ends
    
    pgn.new_game();
    pgn.whiteTeamHadTimeAdvantage = whiteHasTimeAdvantage;
    pgn.whiteTeam = whiteHasTimeAdvantage ? "TimeAdvTeam" : "NoAdvTeam";
    pgn.blackTeam = whiteHasTimeAdvantage ? "NoAdvTeam" : "TimeAdvTeam";
    pgn.round = to_string(gamesGenerated + 1);
    
    GameResult result = GameResult::NO_RESULT;
    size_t ply = 0;
    Stockfish::Color currentSide = Stockfish::WHITE;
    
    // Allocate planes buffer once
    vector<float> inputPlanes(NB_INPUT_VALUES());
    
    while (result == GameResult::NO_RESULT && ply < settings.maxGameLength) {
        // Determine time advantage based on which team is moving
        bool teamHasTimeAdvantage = (currentSide == Stockfish::WHITE) ? whiteHasTimeAdvantage : !whiteHasTimeAdvantage;
        
        // Get legal moves for current side
        auto legalMoves = board.legal_moves(currentSide, teamHasTimeAdvantage);
        
        if (legalMoves.empty()) {
            // Check for checkmate
            if (board.is_checkmate(currentSide, teamHasTimeAdvantage)) {
                result = (currentSide == Stockfish::WHITE) ? GameResult::BLACK_WINS : GameResult::WHITE_WINS;
            } else {
                // Stalemate or other draw
                result = GameResult::DRAW;
            }
            break;
        }
        
        // Check for draw conditions
        if (board.is_draw()) {
            result = GameResult::DRAW;
            break;
        }
        
        // Generate input planes BEFORE making the move (for training data)
        board_to_planes(board, inputPlanes.data(), currentSide, teamHasTimeAdvantage);
        
        // Use MCTS search with Agent for move selection
        Agent* agent = (currentSide == Stockfish::WHITE) ? agentWhiteTeam.get() : agentBlackTeam.get();
        size_t targetNodes = settings.nodesPerMove;
        
        // Randomize node count for variety
        if (settings.nodeRandomFactor > 0.01f) {
            size_t maxDelta = static_cast<size_t>(targetNodes * settings.nodeRandomFactor);
            if (maxDelta > 0) {
                int delta = (rand() % (2 * maxDelta + 1)) - static_cast<int>(maxDelta);
                targetNodes = max(static_cast<size_t>(1), static_cast<size_t>(static_cast<int>(targetNodes) + delta));
            }
        }
        
        // Calculate temperature for this ply (decays to 0)
        float temperature = get_temperature(ply);
        
        JointActionCandidate bestAction = agent->run_search_silent(board, engines, targetNodes, currentSide, teamHasTimeAdvantage, settings, temperature);
        
        // Extract policy distributions from MCTS tree
        vector<PolicyEntry> policyA, policyB;
        auto rootNode = agent->get_root_node();
        if (rootNode && rootNode->is_expanded()) {
            extract_policy_distributions(rootNode, board, policyA, policyB);
        }
        
        // Add training sample (value will be filled in when game ends)
        bool isWhiteTeam = (currentSide == Stockfish::WHITE);
        gameSamples.add_position(inputPlanes.data(), policyA, policyB, isWhiteTeam);
        
        // Apply moves from the joint action
        // In Bughouse, we can have moves on both boards or just one
        Stockfish::Move moveA = bestAction.moveA;
        Stockfish::Move moveB = bestAction.moveB;
        
        // Handle case where search returned no valid moves
        if (moveA == Stockfish::MOVE_NONE && moveB == Stockfish::MOVE_NONE) {
            // Fall back to first legal move
            if (!legalMoves.empty()) {
                auto [boardNum, move] = legalMoves[0];
                if (boardNum == 0) {
                    moveA = move;
                } else {
                    moveB = move;
                }
            } else {
                break;  // No legal moves, game over
            }
        }
        
        // Record moves to PGN
        if (moveA != Stockfish::MOVE_NONE) {
            string moveStr = board.uci_move(0, moveA);
            pgn.add_move(0, moveStr);
        }
        if (moveB != Stockfish::MOVE_NONE) {
            string moveStr = board.uci_move(1, moveB);
            pgn.add_move(1, moveStr);
        }
        
        if (g_logLevel >= LOG_DEBUG) {
            cout << "Ply " << ply << ": " 
                 << (moveA != Stockfish::MOVE_NONE ? board.uci_move(0, moveA) : "pass") << ", "
                 << (moveB != Stockfish::MOVE_NONE ? board.uci_move(1, moveB) : "pass") << endl;
        }
        
        // Apply the joint move
        board.make_moves(moveA, moveB);
        ply++;
        
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
    write_game_to_pgn(pgn, verbose);
    
    // Finalize training samples with game result
    float gameValue = 0.0f;
    if (result == GameResult::WHITE_WINS) {
        gameValue = 1.0f;
    } else if (result == GameResult::BLACK_WINS) {
        gameValue = -1.0f;
    }
    // gameValue is from white team's perspective
    
    // Write samples to training data (value adjusted per sample based on which team moved)
    size_t numSamples = gameSamples.size();
    gameSamples.finalize_game(gameValue, *trainingDataWriter);
    samplesGenerated += numSamples;
    
    return result;
}

void SelfPlay::write_game_to_pgn(const BughouseGamePGN& pgn, bool verbose) {
    lock_guard<mutex> lock(pgnFileMutex);
    
    ofstream pgnFile(pgnFilePath, ios::app);
    pgnFile << pgn;
    pgnFile.close();
    
    if (verbose && g_logLevel >= LOG_INFO) {
        cout << "Game " << gamesGenerated << ": " << pgn.result 
             << " (" << pgn.gameMoves.size() << " moves)"
             << " TimeAdv: " << (pgn.whiteTeamHadTimeAdvantage ? "White" : "Black") << endl;
    }
}

bool SelfPlay::should_resign(float eval, bool allowResignation) const {
    if (!allowResignation) {
        return false;
    }
    return eval < settings.resignThreshold;
}

float SelfPlay::get_temperature(size_t ply) const {
    if (ply >= settings.temperatureDecayMoves) {
        return 0.0f;
    }
    // Linear decay
    float progress = static_cast<float>(ply) / settings.temperatureDecayMoves;
    return settings.temperature * (1.0f - progress);
}

size_t SelfPlay::randomize_nodes(size_t baseNodes) const {
    if (settings.nodeRandomFactor < 0.01f) {
        return baseNodes;
    }
    
    size_t maxDelta = static_cast<size_t>(baseNodes * settings.nodeRandomFactor);
    if (maxDelta == 0) {
        return baseNodes;
    }
    
    int delta = (rand() % (2 * maxDelta + 1)) - static_cast<int>(maxDelta);
    return max(static_cast<size_t>(1), static_cast<size_t>(static_cast<int>(baseNodes) + delta));
}

void SelfPlay::print_statistics() const {
    auto now = chrono::steady_clock::now();
    float elapsedMin = chrono::duration_cast<chrono::milliseconds>(now - startTime).count() / 60000.0f;
    
    float gpm = (elapsedMin > 0.01f) ? gamesGenerated / elapsedMin : 0.0f;
    float spm = (elapsedMin > 0.01f) ? samplesGenerated / elapsedMin : 0.0f;
    
    cout << endl;
    cout << "    games    |  games/min  | samples/min | elapsed (min)" << endl;
    cout << "-------------+-------------+-------------+---------------" << endl;
    cout << setw(13) << gamesGenerated << "|"
         << setw(13) << fixed << setprecision(2) << gpm << "|"
         << setw(13) << fixed << setprecision(1) << spm << "|"
         << setw(15) << fixed << setprecision(2) << elapsedMin << endl;
    cout << endl;
}

void SelfPlay::extract_policy_distributions(
    const shared_ptr<Node>& rootNode,
    Board& board,
    vector<PolicyEntry>& policyA,
    vector<PolicyEntry>& policyB
) {
    policyA.clear();
    policyB.clear();
    
    if (!rootNode || !rootNode->is_expanded()) {
        return;
    }
    
    // Get visit counts for each child action
    auto childActionVisits = rootNode->get_child_action_visits();
    
    if (childActionVisits.empty()) {
        return;
    }
    
    // Marginalize joint action visits to per-board distributions
    // Each joint action (moveA, moveB) contributes its visits to both board A and board B policies
    unordered_map<uint16_t, int> visitsA;
    unordered_map<uint16_t, int> visitsB;
    int totalVisitsA = 0;
    int totalVisitsB = 0;
    
    for (const auto& [jointAction, visits] : childActionVisits) {
        if (visits <= 0) continue;
        
        // Process move on board A
        Stockfish::Move moveA = jointAction.moveA;
        if (moveA != Stockfish::MOVE_NONE) {
            // Convert move to UCI string using board and look up policy index
            string uciA = board.uci_move(0, moveA);
            auto itA = POLICY_INDEX.find(uciA);
            if (itA != POLICY_INDEX.end()) {
                visitsA[itA->second] += visits;
                totalVisitsA += visits;
            }
        }
        
        // Process move on board B
        Stockfish::Move moveB = jointAction.moveB;
        if (moveB != Stockfish::MOVE_NONE) {
            string uciB = board.uci_move(1, moveB);
            auto itB = POLICY_INDEX.find(uciB);
            if (itB != POLICY_INDEX.end()) {
                visitsB[itB->second] += visits;
                totalVisitsB += visits;
            }
        }
    }
    
    // Normalize and build sparse policy distributions
    if (totalVisitsA > 0) {
        policyA.reserve(visitsA.size());
        for (const auto& [idx, visits] : visitsA) {
            float prob = static_cast<float>(visits) / static_cast<float>(totalVisitsA);
            policyA.push_back({idx, prob});
        }
    }
    
    if (totalVisitsB > 0) {
        policyB.reserve(visitsB.size());
        for (const auto& [idx, visits] : visitsB) {
            float prob = static_cast<float>(visits) / static_cast<float>(totalVisitsB);
            policyB.push_back({idx, prob});
        }
    }
}

void run_selfplay(const RLSettings& settings, const vector<Engine*>& engines, size_t numberOfGames) {
    SelfPlay selfplay(settings, engines);
    selfplay.go(numberOfGames);
}

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
#include <utility>

#include "../globals.h"
#include "../planes.h"
#include "../search_params.h"
#include "../utils.h"

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
    cout << "=== Asymmetric Self-Play ==="<< endl;
    cout << "Attacker node multiplier: " << settings.attackerNodeMultiplier << "x" << endl;
    cout << "Defender node multiplier: " << settings.defenderNodeMultiplier << "x" << endl;
    cout << "Mate speed penalty: " << settings.mateSpeedPenalty << endl;
    cout << "Survival bonus: " << settings.survivalBonus << endl;
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
    pgn.whiteTeam = whiteHasTimeAdvantage ? "Alice" : "Bob";
    pgn.blackTeam = whiteHasTimeAdvantage ? "Bob" : "Alice";
    pgn.whiteBoardA = whiteHasTimeAdvantage ? "Alice1" : "Bob1";
    pgn.blackBoardA = whiteHasTimeAdvantage ? "Bob2" : "Alice2";
    pgn.whiteBoardB = whiteHasTimeAdvantage ? "Bob1" : "Alice1";
    pgn.blackBoardB = whiteHasTimeAdvantage ? "Alice2" : "Bob2";
    pgn.round = to_string(gamesGenerated + 1);
    
    GameResult result = GameResult::NO_RESULT;
    size_t ply = 0;
    Stockfish::Color currentSide = Stockfish::WHITE;
    
    // Clock times for each player (start at 180.0, decrement by 0.1 per move)
    float clockTime = 180.0f;
    
    // Allocate planes buffer once
    vector<float> inputPlanes(NB_INPUT_VALUES());
    
    while (result == GameResult::NO_RESULT && ply < settings.maxGameLength) {
        // Determine time advantage based on which team is moving
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
        
        // Generate input planes BEFORE making the move (for training data)
        board_to_planes(board, inputPlanes.data(), currentSide, teamHasTimeAdvantage);
        
        // Use MCTS search with Agent for move selection
        Agent* agent = (currentSide == Stockfish::WHITE) ? agentWhiteTeam.get() : agentBlackTeam.get();
        
        // Asymmetric node allocation:
        // - Attacker (time advantage): may have reduced nodes to force efficient play
        // - Defender (time disadvantage): may have increased nodes for better defense
        size_t targetNodes = settings.nodesPerMove;
        if (teamHasTimeAdvantage) {
            // Attacker - apply attacker multiplier (can be < 1.0 to handicap)
            if (settings.attackerNodeMultiplier != 1.0f) {
                targetNodes = static_cast<size_t>(targetNodes * settings.attackerNodeMultiplier);
            }
        } else {
            // Defender - apply defender multiplier (can be > 1.0 to help defense)
            if (settings.defenderNodeMultiplier != 1.0f) {
                targetNodes = static_cast<size_t>(targetNodes * settings.defenderNodeMultiplier);
            }
        }
        targetNodes = max(static_cast<size_t>(1), targetNodes);  // Ensure at least 1 node
        
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
        
        // Use time-based search if moveTimeMs > 0, otherwise node-based
        SearchOptions opts;
        if (settings.moveTimeMs > 0) {
            opts = SearchOptions::selfplay(settings.moveTimeMs, settings);
        } else {
            opts = SearchOptions::selfplay(targetNodes, settings);
        }
        JointActionCandidate bestAction = agent->run_search(board, engines, currentSide, teamHasTimeAdvantage, opts);
        
        // Extract policy distributions from MCTS tree
        vector<PolicyEntry> policyA, policyB;
        auto rootNode = agent->get_root_node();
        if (rootNode && rootNode->is_expanded()) {
            auto childActionVisits = extract_policy_distributions(rootNode, board, policyA, policyB);
            
            // Sample action with temperature from visit distribution
            if (!childActionVisits.empty()) {
                bestAction = sample_action_with_temperature(childActionVisits, temperature);
            } else {
                cerr << "Warning: No child actions found in MCTS root node!" << endl;
            }
        }
        
        // Add training sample (value will be filled in when game ends)
        bool isWhiteTeam = (currentSide == Stockfish::WHITE);
        gameSamples.add_position(inputPlanes.data(), policyA, policyB, isWhiteTeam);
        
        // Apply moves from the joint action
        // In Bughouse, we can have moves on both boards, just one, or neither (both pass)
        Stockfish::Move moveA = bestAction.moveA;
        Stockfish::Move moveB = bestAction.moveB;
        
        // Note: (MOVE_NONE, MOVE_NONE) is valid - it means both boards are passing/sitting
        // This can happen when the team has time advantage and chooses to wait
        
        // Record moves to PGN (SAN format) with decrementing clock
        if (moveA != Stockfish::MOVE_NONE) {
            string moveStr = board.san_move(0, moveA);
            pgn.add_move(0, moveStr, clockTime);
            clockTime -= 0.1f;
        }
        if (moveB != Stockfish::MOVE_NONE) {
            string moveStr = board.san_move(1, moveB);
            pgn.add_move(1, moveStr, clockTime);
            clockTime -= 0.1f;
        }
        
        if (g_logLevel >= LOG_DEBUG) {
            cout << "Ply " << ply << ": " 
                 << (moveA != Stockfish::MOVE_NONE ? board.uci_move(BOARD_A, moveA) : "pass") << ", "
                 << (moveB != Stockfish::MOVE_NONE ? board.uci_move(BOARD_B, moveB) : "pass") << endl;
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
    
    pgn.plyCount = ply;
    pgn.set_result(result);
    write_game_to_pgn(pgn, verbose);
    
    // Update game length statistics
    totalPlies += ply;
    
    // Update min/max game length
    size_t currentMin = minGameLength.load();
    while (ply < currentMin && !minGameLength.compare_exchange_weak(currentMin, ply)) {}
    
    size_t currentMax = maxGameLength.load();
    while (ply > currentMax && !maxGameLength.compare_exchange_weak(currentMax, ply)) {}
    
    // Update Alice/Bob statistics
    // Alice is the team with time advantage, Bob is without
    bool aliceIsWhite = whiteHasTimeAdvantage;
    if (result == GameResult::WHITE_WINS) {
        if (aliceIsWhite) {
            aliceWins++;
        } else {
            aliceLosses++;
        }
    } else if (result == GameResult::BLACK_WINS) {
        if (aliceIsWhite) {
            aliceLosses++;
        } else {
            aliceWins++;
        }
    } else {
        aliceDraws++;
    }
    
    // Calculate asymmetric rewards based on game outcome and length
    auto [whiteTeamValue, blackTeamValue] = calculate_asymmetric_rewards(
        result, ply, whiteHasTimeAdvantage);
    
    // Write samples to training data with asymmetric values
    size_t numSamples = gameSamples.size();
    gameSamples.finalize_game(whiteTeamValue, blackTeamValue, *trainingDataWriter);
    samplesGenerated += numSamples;
    
    if (verbose && g_logLevel >= LOG_DEBUG) {
        cout << "  Rewards: White=" << whiteTeamValue << ", Black=" << blackTeamValue 
             << " (ply=" << ply << ")" << endl;
    }
    
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
    
    size_t aWins = aliceWins.load();
    size_t aLosses = aliceLosses.load();
    size_t aDraws = aliceDraws.load();
    size_t totalGames = aWins + aLosses + aDraws;
    
    float aliceWinRate = (totalGames > 0) ? (100.0f * aWins / totalGames) : 0.0f;
    float bobWinRate = (totalGames > 0) ? (100.0f * aLosses / totalGames) : 0.0f;
    float drawRate = (totalGames > 0) ? (100.0f * aDraws / totalGames) : 0.0f;
    
    cout << endl;
    cout << "    games    |  games/min  | samples/min | elapsed (min)" << endl;
    cout << "-------------+-------------+-------------+---------------" << endl;
    cout << setw(13) << gamesGenerated << "|"
         << setw(13) << fixed << setprecision(2) << gpm << "|"
         << setw(13) << fixed << setprecision(1) << spm << "|"
         << setw(15) << fixed << setprecision(2) << elapsedMin << endl;
    cout << endl;
    float avgGameLength = (totalGames > 0) ? (static_cast<float>(totalPlies.load()) / totalGames) : 0.0f;
    size_t minLen = minGameLength.load();
    size_t maxLen = maxGameLength.load();
    
    cout << "Alice (attacker) vs Bob (defender):" << endl;
    cout << "  Alice wins: " << aWins << " (" << fixed << setprecision(1) << aliceWinRate << "%)" << endl;
    cout << "  Bob wins:   " << aLosses << " (" << fixed << setprecision(1) << bobWinRate << "%)" << endl;
    cout << "  Draws:      " << aDraws << " (" << fixed << setprecision(1) << drawRate << "%)" << endl;
    cout << "  Avg length: " << fixed << setprecision(1) << avgGameLength << " plies" << endl;
    cout << "  Min length: " << (minLen == 999999 ? 0 : minLen) << " plies" << endl;
    cout << "  Max length: " << maxLen << " plies" << endl;
    cout << endl;
}

pair<float, float> SelfPlay::calculate_asymmetric_rewards(
    GameResult result, size_t ply, bool whiteHadTimeAdvantage) const {
    
    // Determine which team was Alice (attacker with time advantage)
    // and which was Bob (defender without time advantage)
    bool whiteIsAlice = whiteHadTimeAdvantage;
    
    // Game length ratio for reward scaling
    float lengthRatio = static_cast<float>(ply) / static_cast<float>(settings.maxGameLength);
    lengthRatio = min(1.0f, lengthRatio);  // Cap at 1.0
    
    float whiteValue = 0.0f;
    float blackValue = 0.0f;
    
    if (result == GameResult::WHITE_WINS) {
        if (whiteIsAlice) {
            // Alice (white) won - apply time-to-mate penalty for slow wins
            // Faster wins get higher rewards
            float penalty = lengthRatio * settings.mateSpeedPenalty;
            whiteValue = max(settings.minWinReward, 1.0f - penalty);
            
            // Bob (black) lost - apply survival bonus for lasting longer
            float bonus = lengthRatio * settings.survivalBonus;
            blackValue = min(settings.maxLossReward, -1.0f + bonus);
        } else {
            // Bob (white) won against Alice (black) - this is unexpected!
            // Give Bob a full reward for winning despite disadvantage
            whiteValue = 1.0f;
            // Alice lost to Bob - full penalty
            blackValue = -1.0f;
        }
    } else if (result == GameResult::BLACK_WINS) {
        if (!whiteIsAlice) {
            // Alice (black) won - apply time-to-mate penalty
            float penalty = lengthRatio * settings.mateSpeedPenalty;
            blackValue = max(settings.minWinReward, 1.0f - penalty);
            
            // Bob (white) lost - apply survival bonus
            float bonus = lengthRatio * settings.survivalBonus;
            whiteValue = min(settings.maxLossReward, -1.0f + bonus);
        } else {
            // Bob (black) won against Alice (white) - unexpected!
            blackValue = 1.0f;
            whiteValue = -1.0f;
        }
    } else {
        // Draw - both teams get 0
        // Could also give slight bonus to Bob for surviving
        if (whiteIsAlice) {
            // Alice (white) drew - slight penalty for not winning
            whiteValue = -0.1f;
            // Bob (black) drew - slight bonus for surviving
            blackValue = 0.1f;
        } else {
            // Alice (black) drew
            blackValue = -0.1f;
            // Bob (white) drew
            whiteValue = 0.1f;
        }
    }
    
    return {whiteValue, blackValue};
}

vector<pair<JointActionCandidate, int>> SelfPlay::extract_policy_distributions(
    const shared_ptr<Node>& rootNode,
    Board& board,
    vector<PolicyEntry>& policyA,
    vector<PolicyEntry>& policyB
) {
    policyA.clear();
    policyB.clear();
    
    if (!rootNode || !rootNode->is_expanded()) {
        return {};
    }
    
    // Get visit counts for each child action
    auto childActionVisits = rootNode->get_child_action_visits();
    
    if (childActionVisits.empty()) {
        return {};
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
        string uciA;
        if (moveA != Stockfish::MOVE_NONE) {
            // Convert move to UCI string using board and look up policy index
            uciA = board.uci_move(BOARD_A, moveA);
            // Mirror move for black's perspective - policy index expects player-relative coordinates
            if (board.side_to_move(BOARD_A) == Stockfish::BLACK) {
                uciA = mirror_move(uciA);
            }
        } else {
            // Board is not on turn, treat as pass
            uciA = "pass";
        }
        
        auto itA = POLICY_INDEX.find(uciA);
        if (itA != POLICY_INDEX.end()) {
            visitsA[itA->second] += visits;
            totalVisitsA += visits;
        }
        
        // Process move on board B
        Stockfish::Move moveB = jointAction.moveB;
        string uciB;
        if (moveB != Stockfish::MOVE_NONE) {
            uciB = board.uci_move(BOARD_B, moveB);
            // Mirror move for black's perspective - policy index expects player-relative coordinates
            if (board.side_to_move(BOARD_B) == Stockfish::BLACK) {
                uciB = mirror_move(uciB);
            }
        } else {
            // Board is not on turn, treat as pass
            uciB = "pass";
        }
        
        auto itB = POLICY_INDEX.find(uciB);
        if (itB != POLICY_INDEX.end()) {
            visitsB[itB->second] += visits;
            totalVisitsB += visits;
        }
    }
    
    // Normalize and build sparse policy distributions
    // Apply lowPolicyClipThreshold to remove noise from low-visit moves
    float clipThreshold = settings.lowPolicyClipThreshold;
    
    if (totalVisitsA > 0) {
        policyA.reserve(visitsA.size());
        float sumAfterClip = 0.0f;
        
        // First pass: calculate probabilities and filter
        vector<pair<uint16_t, float>> tempA;
        for (const auto& [idx, visits] : visitsA) {
            float prob = static_cast<float>(visits) / static_cast<float>(totalVisitsA);
            if (prob >= clipThreshold) {
                tempA.push_back({idx, prob});
                sumAfterClip += prob;
            }
        }
        
        // Second pass: renormalize after clipping
        if (sumAfterClip > 0.0f) {
            for (const auto& [idx, prob] : tempA) {
                policyA.push_back({idx, prob / sumAfterClip});
            }
        }
    }
    
    if (totalVisitsB > 0) {
        policyB.reserve(visitsB.size());
        float sumAfterClip = 0.0f;
        
        // First pass: calculate probabilities and filter
        vector<pair<uint16_t, float>> tempB;
        for (const auto& [idx, visits] : visitsB) {
            float prob = static_cast<float>(visits) / static_cast<float>(totalVisitsB);
            if (prob >= clipThreshold) {
                tempB.push_back({idx, prob});
                sumAfterClip += prob;
            }
        }
        
        // Second pass: renormalize after clipping
        if (sumAfterClip > 0.0f) {
            for (const auto& [idx, prob] : tempB) {
                policyB.push_back({idx, prob / sumAfterClip});
            }
        }
    }
    
    return childActionVisits;
}

void run_selfplay(const RLSettings& settings, const vector<Engine*>& engines, size_t numberOfGames) {
    SelfPlay selfplay(settings, engines);
    selfplay.go(numberOfGames);
}

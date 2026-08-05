#include "searchthread.h"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <string>

#include <cuda_runtime_api.h>

#include "joint_action.h"
#include "utils.h"

using namespace std;

TerminalOutcome classify_terminal_position(Board& board,
                                             Stockfish::Color teamToPlay,
                                             Stockfish::Color rootTeam,
                                             bool rootTeamHasTimeAdvantage,
                                             int searchPly) {
    const bool teamToPlayHasTimeAdvantage = teamToPlay == rootTeam
        ? rootTeamHasTimeAdvantage
        : !rootTeamHasTimeAdvantage;

    if (board.is_checkmate(~teamToPlay, !teamToPlayHasTimeAdvantage)) {
        return TerminalOutcome::WIN;
    }
    if (board.is_checkmate(teamToPlay, teamToPlayHasTimeAdvantage)) {
        return TerminalOutcome::LOSS;
    }
    if (board.is_draw(searchPly)) {
        return TerminalOutcome::DRAW;
    }
    return TerminalOutcome::NONE;
}

SearchThread::SearchThread() : transpositionTable(nullptr), currentBatchSize(0) {
    // Buffers are allocated lazily in ensureBufferSize() when run_iteration is called
}

SearchThread::~SearchThread() {
    if (obs) cudaFreeHost(obs);
    if (value) cudaFreeHost(value);
    if (piA) cudaFreeHost(piA);
    if (piB) cudaFreeHost(piB);
    if (wdl) cudaFreeHost(wdl);
    if (movesLeft) cudaFreeHost(movesLeft);
}

void SearchThread::ensureBufferSize(int batchSize) {
    if (batchSize == currentBatchSize) return;

    // Free old buffers
    if (obs) cudaFreeHost(obs);
    if (value) cudaFreeHost(value);
    if (piA) cudaFreeHost(piA);
    if (piB) cudaFreeHost(piB);
    if (wdl) cudaFreeHost(wdl);
    if (movesLeft) cudaFreeHost(movesLeft);
    obs = value = piA = piB = wdl = movesLeft = nullptr;

    auto allocatePinned = [](float** buffer, size_t count) {
        cudaError_t result = cudaMallocHost(
            reinterpret_cast<void**>(buffer), count * sizeof(float));
        if (result != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaMallocHost failed: ") + cudaGetErrorString(result));
        }
    };

    allocatePinned(&obs, batchSize * NB_INPUT_VALUES());
    allocatePinned(&value, batchSize);
    allocatePinned(&piA, batchSize * NB_POLICY_VALUES());
    allocatePinned(&piB, batchSize * NB_POLICY_VALUES());
    allocatePinned(&wdl, batchSize * 3);
    allocatePinned(&movesLeft, batchSize);
    
    batchContexts.reserve(batchSize);
    currentBatchSize = batchSize;
}

void SearchThread::set_search_info(SearchInfo* info) {
    searchInfo = info;
}

void SearchThread::set_root_node(Node* node) {
    root = node;
}

void SearchThread::set_transposition_table(TranspositionTable* table) {
    transpositionTable = table;
}

void SearchThread::set_runtime_config(const SearchParams::RuntimeConfig& config) {
    runtimeConfig = config;
}

void SearchThread::backup(vector<TrajectoryEntry>& trajectory, 
                          Board& board, float valueToBackup) {
    // Process nodes in reverse order (from leaf to root)
    NodeType childType = NodeType::UNSOLVED;
    
    for (auto it = trajectory.rbegin(); it != trajectory.rend(); ++it) {
        Node* node = it->node;
        int childIdx = it->selectedChildIdx;

        if (childIdx >= 0) {
            // Internal node - use the stored child index
            node->update_and_remove_virtual_loss(childIdx, valueToBackup);
            
            // MCTS Solver: propagate terminal states
            if (SearchParams::ENABLE_MCTS_SOLVER && childType != NodeType::UNSOLVED) {
                node->init_child_node_types();
                node->update_child_node_type(childIdx, childType);
                childType = node->get_node_type();
            } else {
                childType = NodeType::UNSOLVED;
            }
        } else {
            // Root or leaf without child selection
            node->update_terminal(valueToBackup);
        }
        valueToBackup = -valueToBackup;
    }
    
    // Note: moves are already undone during batch collection, no need to undo here
}

void SearchThread::cancel_virtual_losses(const vector<TrajectoryEntry>& trajectory) {
    for (const TrajectoryEntry& entry : trajectory) {
        if (entry.selectedChildIdx >= 0) {
            entry.node->remove_virtual_loss(entry.selectedChildIdx);
        }
    }
}

/**
 * @brief Runs a minibatch of MCTS iterations.
 * 
 * This collects leaves based on the engine's batch size, runs batched neural network inference,
 * then expands and backs up all leaves. This better utilizes GPU parallelism.
 */
void SearchThread::run_iteration(Board& board, Engine* engine, bool teamHasTimeAdvantage) {
    // Get batch size from engine and ensure buffers are properly sized
    int batchSize = engine->getBatchSize();
    ensureBufferSize(batchSize);
    
    batchContexts.clear();
    int validInferenceCount = 0;
    int sameBatchCollisions = 0;
    int reservationCollisions = 0;
    
    // Phase 1: Collect batchSize leaf nodes
    constexpr int MAX_SELECTION_ATTEMPTS_PER_SLOT = 2;
    const int maxSelectionAttempts = batchSize * MAX_SELECTION_ATTEMPTS_PER_SLOT;
    int selectionAttempts = 0;
    while (static_cast<int>(batchContexts.size()) < batchSize &&
           selectionAttempts < maxSelectionAttempts) {
        selectionAttempts++;
        LeafContext ctx;
        trajectoryBuffer.clear();
        
        // Select and expand to get a leaf node (MCGS: with transposition lookup)
        Node* leaf = select_and_expand(board, teamHasTimeAdvantage);
        
        // A repeated leaf is not an independent simulation. Revert its
        // temporary selection state and leave the first trajectory responsible
        // for evaluating and backing up this position.
        bool isCollision = std::any_of(
            batchContexts.begin(), batchContexts.end(),
            [leaf](const LeafContext& previous) { return previous.leaf == leaf; });
        if (isCollision) {
            sameBatchCollisions++;
            cancel_virtual_losses(trajectoryBuffer);
            for (auto it = trajectoryBuffer.rbegin(); it != trajectoryBuffer.rend(); ++it) {
                const JointActionCandidate& action = it->action;
                if (action.moveA != Stockfish::MOVE_NONE || action.moveB != Stockfish::MOVE_NONE) {
                    board.unmake_moves(action.moveA, action.moveB);
                }
            }
            continue;
        }
        
        // Store trajectory
        ctx.trajectory = trajectoryBuffer;
        ctx.leaf = leaf;
        ctx.teamToPlay = leaf->get_team_to_play();
        ctx.sitPlaneActive = (ctx.teamToPlay == root->get_team_to_play()) == teamHasTimeAdvantage;
        
        // Check for terminal states
        // Pass the tree depth (trajectory size minus 1) as ply so that 2-fold repetitions
        // within the search tree are correctly detected as draws.
        // We subtract 1 because trajectoryBuffer includes the root entry at index 0,
        // and the root position should use ply=0 (requiring 3-fold repetition, not 2-fold).
        // Without this, a root position with only 2-fold repetition is incorrectly
        // treated as a draw on every iteration, preventing the root from ever expanding.
        int searchPly = static_cast<int>(trajectoryBuffer.size()) - 1;
        searchInfo->set_max_depth(searchPly);
        const TerminalOutcome terminalOutcome = classify_terminal_position(
            board, ctx.teamToPlay, root->get_team_to_play(),
            teamHasTimeAdvantage, searchPly);
        if (terminalOutcome != TerminalOutcome::NONE) {
            ctx.isTerminal = true;
            if (terminalOutcome == TerminalOutcome::WIN) {
                ctx.terminalValue = 1.0f;
                if (SearchParams::ENABLE_MCTS_SOLVER) {
                    ctx.leaf->mark_as_win(1);
                }
            } else if (terminalOutcome == TerminalOutcome::LOSS) {
                ctx.terminalValue = -1.0f;
                if (SearchParams::ENABLE_MCTS_SOLVER) {
                    ctx.leaf->mark_as_loss(1);
                }
            } else {
                ctx.terminalValue = ctx.teamToPlay == root->get_team_to_play()
                    ? -runtimeConfig.drawContempt
                    : runtimeConfig.drawContempt;
            }

            batchContexts.push_back(std::move(ctx));
            for (auto it = trajectoryBuffer.rbegin(); it != trajectoryBuffer.rend(); ++it) {
                const JointActionCandidate& action = it->action;
                if (action.moveA != Stockfish::MOVE_NONE || action.moveB != Stockfish::MOVE_NONE) {
                    board.unmake_moves(action.moveA, action.moveB);
                }
            }
            continue;
        }
        
        // This leaf needs neural network inference
        ctx.isTerminal = false;
        if (!ctx.leaf->try_reserve_evaluation()) {
            reservationCollisions++;
            cancel_virtual_losses(trajectoryBuffer);
            for (auto it = trajectoryBuffer.rbegin(); it != trajectoryBuffer.rend(); ++it) {
                const JointActionCandidate& action = it->action;
                if (action.moveA != Stockfish::MOVE_NONE || action.moveB != Stockfish::MOVE_NONE) {
                    board.unmake_moves(action.moveA, action.moveB);
                }
            }
            continue;
        }
        ctx.hasEvaluationReservation = true;
        bool leafTeamHasTimeAdvantage = (ctx.teamToPlay == root->get_team_to_play())
            ? teamHasTimeAdvantage
            : !teamHasTimeAdvantage;
        ctx.leafHash = board.hash_key(leafTeamHasTimeAdvantage);

        // Convert board to planes for this batch slot
        board_to_planes(board, obs + validInferenceCount * NB_INPUT_VALUES(), 
                        ctx.teamToPlay, ctx.sitPlaneActive);
        validInferenceCount++;
        
        batchContexts.push_back(std::move(ctx));
        
        // Undo moves to restore board for next selection
        for (auto it = trajectoryBuffer.rbegin(); it != trajectoryBuffer.rend(); ++it) {
            const JointActionCandidate& action = it->action;
            if (action.moveA != Stockfish::MOVE_NONE || action.moveB != Stockfish::MOVE_NONE) {
                board.unmake_moves(action.moveA, action.moveB);
            }
        }
    }
    
    // Phase 2: Run batched neural network inference (only if we have non-terminal leaves)
    if (validInferenceCount > 0) {
        if (!engine->runInference(obs, value, piA, piB, wdl, movesLeft)) {
            cerr << "Batch inference failed" << endl;
            // Backup all as 0.0 and remove virtual loss
            for (auto& ctx : batchContexts) {
                if (ctx.hasEvaluationReservation) {
                    ctx.leaf->release_evaluation_reservation();
                }
                backup(ctx.trajectory, board, 0.0f);
            }
            searchInfo->increment_nodes(static_cast<int>(batchContexts.size()));
            searchInfo->increment_same_batch_collisions(sameBatchCollisions);
            searchInfo->increment_reservation_collisions(reservationCollisions);
            return;
        }
    }
    
    // Phase 3: Process results and backup
    int inferenceIdx = 0;
    for (auto& ctx : batchContexts) {
        if (ctx.isTerminal) {
            // Terminal node - backup the terminal value and remove virtual loss
            // Also propagate solver state up the tree
            float val = ctx.terminalValue;
            NodeType childType = ctx.leaf->get_node_type();
            
            for (auto it = ctx.trajectory.rbegin(); it != ctx.trajectory.rend(); ++it) {
                Node* node = it->node;
                int childIdx = it->selectedChildIdx;
                if (childIdx >= 0) {
                    node->update_and_remove_virtual_loss(childIdx, val);
                    
                    // MCTS Solver: propagate terminal state
                    if (SearchParams::ENABLE_MCTS_SOLVER && childType != NodeType::UNSOLVED) {
                        node->init_child_node_types();
                        node->update_child_node_type(childIdx, childType);
                        childType = node->get_node_type();
                    } else {
                        childType = NodeType::UNSOLVED;
                    }
                } else {
                    node->update_terminal(val);
                }
                val = -val;
            }
        } else {
            for (const TrajectoryEntry& entry : ctx.trajectory) {
                const JointActionCandidate& action = entry.action;
                if (action.moveA != Stockfish::MOVE_NONE || action.moveB != Stockfish::MOVE_NONE) {
                    board.make_moves(action.moveA, action.moveB);
                }
            }

            // Non-terminal node - process NN output and expand
            float* batchValue = value + inferenceIdx;
            float* batchPiA = piA + inferenceIdx * NB_POLICY_VALUES();
            float* batchPiB = piB + inferenceIdx * NB_POLICY_VALUES();

            Board& leafBoard = board;
            
            // Compute whether the team at this leaf has time advantage
            // Time advantage alternates: if root team has it, opponent team doesn't
            Stockfish::Color rootTeam = root->get_team_to_play();
            bool leafTeamHasTimeAdvantage = (ctx.teamToPlay == rootTeam) ? teamHasTimeAdvantage : !teamHasTimeAdvantage;
            
            // Get legal moves for each board
            vector<Stockfish::Move> actionsA;
            vector<Stockfish::Move> actionsB;
            
            // Track which boards are on turn for the team
            bool boardAOnTurn = (leafBoard.side_to_move(BOARD_A) == ctx.teamToPlay);
            bool boardBOnTurn = (leafBoard.side_to_move(BOARD_B) == ~ctx.teamToPlay);
            const float passPriorFloor = get_pass_prior_floor(
                leafTeamHasTimeAdvantage, boardAOnTurn, boardBOnTurn, runtimeConfig);
            
            if (boardAOnTurn) {
                actionsA = leafBoard.legal_moves(BOARD_A);
                std::erase_if(actionsA, [&leafBoard](Stockfish::Move move) {
                    return !is_policy_move_representable(leafBoard, BOARD_A, move);
                });
            }
            if (boardBOnTurn) {
                actionsB = leafBoard.legal_moves(BOARD_B);
                std::erase_if(actionsB, [&leafBoard](Stockfish::Move move) {
                    return !is_policy_move_representable(leafBoard, BOARD_B, move);
                });
            }
            
            vector<float> priorsA;
            vector<float> priorsB;
            
            // MOVE_NONE is always a valid option because it doesn't change the board state.
            // Even if in check, a team can "pass" on a board - the check persists until dealt with.
            // Double-pass requires time advantage and is invalid when both boards are on turn.
            
            if (actionsA.empty()) {
                // Not on turn or no legal moves - MOVE_NONE is only option
                actionsA.push_back(Stockfish::MOVE_NONE);
                priorsA.push_back(1.0f);
            } else {
                // On turn with legal moves - can also pass (MOVE_NONE)
                actionsA.push_back(Stockfish::MOVE_NONE);
                priorsA = get_normalized_probability(
                    batchPiA, actionsA, BOARD_A, leafBoard, passPriorFloor);
            }
            
            if (actionsB.empty()) {
                // Not on turn or no legal moves - MOVE_NONE is only option
                actionsB.push_back(Stockfish::MOVE_NONE);
                priorsB.push_back(1.0f);
            } else {
                // On turn with legal moves - can also pass (MOVE_NONE)
                actionsB.push_back(Stockfish::MOVE_NONE);
                priorsB = get_normalized_probability(
                    batchPiB, actionsB, BOARD_B, leafBoard, passPriorFloor);
            }

            // Expand leaf node and register in transposition table (MCGS)
            // Use leafTeamHasTimeAdvantage for the team making the move at this leaf.
            // The generator creates joint actions that THIS team will play.
            expand_leaf_node(ctx.leaf, actionsA, actionsB, priorsA, priorsB,
                             leafTeamHasTimeAdvantage, boardAOnTurn, boardBOnTurn,
                             ctx.leafHash);
            ctx.leaf->release_evaluation_reservation();
                
            // Backup value
            float neuralValue = std::isfinite(*batchValue)
                ? std::clamp(*batchValue, -1.0f, 1.0f)
                : 0.0f;
            backup(ctx.trajectory, leafBoard, neuralValue);

            for (auto it = ctx.trajectory.rbegin(); it != ctx.trajectory.rend(); ++it) {
                const JointActionCandidate& action = it->action;
                if (action.moveA != Stockfish::MOVE_NONE || action.moveB != Stockfish::MOVE_NONE) {
                    board.unmake_moves(action.moveA, action.moveB);
                }
            }

            inferenceIdx++;
        }
    }
    
    searchInfo->increment_nodes(static_cast<int>(batchContexts.size()));
    searchInfo->increment_same_batch_collisions(sameBatchCollisions);
    searchInfo->increment_reservation_collisions(reservationCollisions);
}

/**
 * @brief Selects a leaf node and expands all actions in policy order with MCGS.
 * 
 * MCGS Enhancement: When expanding a new child, checks the transposition table
 * first. If the resulting position already exists, reuses that node instead of
 * creating a new one. This transforms the tree into a DAG for better convergence.
 * 
 * @param board The current board state (will be modified during selection)
 * @param teamHasTimeAdvantage Whether the searching team has time advantage
 */
Node* SearchThread::select_and_expand(Board& board, bool teamHasTimeAdvantage) {
    Node* currentNode = root;
    shared_ptr<Node> nextNode;
    int childIdx;

    // Root node has no incoming action, -1 means no child selected yet
    trajectoryBuffer.emplace_back(currentNode, JointActionCandidate(), -1);

    while (true) {
        // If not expanded, this is a leaf node
        if (!currentNode->is_expanded()) {
            break;
        }

        // Grow the large Cartesian joint-action space as the node earns visits.
        if (currentNode->should_expand_new_child(runtimeConfig)) {
            // Expand first to atomically get the action
            JointActionCandidate expandedAction;
            nextNode = currentNode->expand_next_joint_child(
                nullptr, 0, expandedAction, runtimeConfig, &childIdx, true);
            
            if (nextNode) {
                // Make moves with the actual expanded action
                board.make_moves(expandedAction.moveA, expandedAction.moveB);
                
                // MCGS: Compute position hash and register in transposition table
                bool childTeamHasTimeAdvantage = (nextNode->get_team_to_play() == root->get_team_to_play())
                    ? teamHasTimeAdvantage
                    : !teamHasTimeAdvantage;
                uint64_t childHash = board.hash_key(childTeamHasTimeAdvantage);
                nextNode->set_hash(childHash);
                
                // Reuse the canonical node unless doing so would create a cycle in this trajectory.
                bool continueFromTransposition = false;
                if (runtimeConfig.enableMCGS && runtimeConfig.enableTranspositions && transpositionTable) {
                    auto canonicalNode = transpositionTable->insertOrGet(childHash, nextNode);
                    bool isAncestor = std::any_of(
                        trajectoryBuffer.begin(), trajectoryBuffer.end(),
                        [&canonicalNode](const TrajectoryEntry& entry) {
                            return entry.node == canonicalNode.get();
                        });
                    if (canonicalNode != nextNode && !isAncestor) {
                        currentNode->replace_child(childIdx, canonicalNode);
                        nextNode = canonicalNode;
                        continueFromTransposition = canonicalNode->is_expanded();
                    }
                }
                
                // Update the parent trajectory entry with the selected child index
                trajectoryBuffer.back().selectedChildIdx = childIdx;
                
                trajectoryBuffer.emplace_back(nextNode.get(), expandedAction, -1);

                if (continueFromTransposition) {
                    currentNode = nextNode.get();
                    continue;
                }
                
                // Return the newly expanded leaf
                return nextNode.get();
            }
        }

        // Standard PUCT selection among expanded children
        auto [selectedChild, selectedIdx] =
            currentNode->select_child_and_apply_virtual_loss(runtimeConfig);
        if (!selectedChild || selectedIdx < 0) {
            break;  // No children available
        }
        nextNode = selectedChild;
        childIdx = selectedIdx;
        
        // Update the parent trajectory entry with the selected child index
        trajectoryBuffer.back().selectedChildIdx = childIdx;

        JointActionCandidate action = currentNode->get_joint_action(childIdx);
        board.make_moves(action.moveA, action.moveB);
        
        trajectoryBuffer.emplace_back(nextNode.get(), action, -1);
        currentNode = nextNode.get();
    }

    return currentNode;
}

/**
 * @brief Expands a leaf node with joint action candidates using lazy priority queue.
 * Thread-safe: Uses atomic try_init_and_expand to prevent race conditions.
 * 
 * MCGS Enhancement: After initializing the leaf node, we register the first child
 * in the transposition table. Note that during leaf expansion, the first child's
 * position hash is not known yet (would require making moves), so we only register
 * the leaf node itself. Child transpositions are handled during select_and_expand.
 * 
 * @param teamHasTimeAdvantage If true, team is up on time and can sit when on turn
 * @param boardAOnTurn True if it's this team's turn on board A
 * @param boardBOnTurn True if it's this team's turn on board B
 * @param positionHash Zobrist hash of the leaf position for transposition table
 */
void SearchThread::expand_leaf_node(Node* leaf,
                                    const vector<Stockfish::Move>& actionsA,
                                    const vector<Stockfish::Move>& actionsB,
                                    const vector<float>& priorsA,
                                    const vector<float>& priorsB,
                                    bool teamHasTimeAdvantage,
                                    bool boardAOnTurn,
                                    bool boardBOnTurn,
                                    uint64_t positionHash) {
    // Store the position hash in the node for MCGS
    if (positionHash != 0) {
        leaf->set_hash(positionHash);
    }
    
    // Atomically try to initialize and expand if not already done
    // This is safe for concurrent access from multiple threads
    leaf->try_init_and_expand(actionsA, actionsB, priorsA, priorsB,
                              teamHasTimeAdvantage, boardAOnTurn, boardBOnTurn,
                              runtimeConfig);
    
    // Note: The first child created during try_init_and_expand doesn't have its
    // hash computed yet (would require board access). However, when that child
    // is later traversed through select_and_expand, if it's unexpanded, the
    // transposition lookup will happen at that time based on the board state.
    // This is slightly less efficient than computing the hash during expansion,
    // but avoids complexity of passing the board to try_init_and_expand.
}

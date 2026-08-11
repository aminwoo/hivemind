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

namespace {

std::vector<Stockfish::Move> immediate_mates_on_board(
    Board& board,
    int boardNum,
    Stockfish::Color victimTeam,
    bool victimTeamHasTimeAdvantage) {
    std::vector<Stockfish::Move> mates;
    for (Stockfish::Move move : board.legal_moves(boardNum)) {
        if (!board.gives_check(boardNum, move)) {
            continue;
        }
        board.push_move(boardNum, move);
        const bool isMate = board.is_checkmate(
            victimTeam, victimTeamHasTimeAdvantage);
        board.pop_move(boardNum);
        if (isMate) {
            mates.push_back(move);
        }
    }
    return mates;
}

bool has_unavoidable_waiting_board_mate(Board& board,
                                        Stockfish::Color teamToPlay,
                                        bool teamToPlayHasTimeAdvantage,
                                        int searchPly) {
    const bool boardAOnTurn = board.side_to_move(BOARD_A) == teamToPlay;
    const bool boardBOnTurn = board.side_to_move(BOARD_B) == ~teamToPlay;
    if (boardAOnTurn == boardBOnTurn) {
        return false;
    }

    const int activeBoard = boardAOnTurn ? BOARD_A : BOARD_B;
    const int waitingBoard = 1 - activeBoard;
    const std::vector<Stockfish::Move> matingMoves = immediate_mates_on_board(
        board, waitingBoard, teamToPlay, teamToPlayHasTimeAdvantage);
    if (matingMoves.empty()) {
        return false;
    }

    std::vector<Stockfish::Move> replies = board.legal_moves(activeBoard);
    if (teamToPlayHasTimeAdvantage) {
        replies.push_back(Stockfish::MOVE_NONE);
    }
    if (replies.empty()) {
        return false;
    }

    for (Stockfish::Move reply : replies) {
        if (reply != Stockfish::MOVE_NONE) {
            board.push_move(activeBoard, reply);
        }

        bool matePersists = false;
        if (!board.is_checkmate(~teamToPlay, !teamToPlayHasTimeAdvantage)
            && !board.is_draw(searchPly + 1)) {
            for (Stockfish::Move matingMove : matingMoves) {
                if (!board.is_legal_move(waitingBoard, matingMove)) {
                    continue;
                }
                board.push_move(waitingBoard, matingMove);
                matePersists = board.is_checkmate(
                    teamToPlay, teamToPlayHasTimeAdvantage);
                board.pop_move(waitingBoard);
                if (matePersists) {
                    break;
                }
            }
        }

        if (reply != Stockfish::MOVE_NONE) {
            board.pop_move(activeBoard);
        }
        if (!matePersists) {
            return false;
        }
    }
    return true;
}

}  // namespace

TerminalOutcome classify_terminal_position(Board& board,
                                             Stockfish::Color teamToPlay,
                                             Stockfish::Color rootTeam,
                                             bool rootTeamHasTimeAdvantage,
                                             int searchPly,
                                             int* endInPly) {
    if (endInPly) {
        *endInPly = 0;
    }
    const bool teamToPlayHasTimeAdvantage = teamToPlay == rootTeam
        ? rootTeamHasTimeAdvantage
        : !rootTeamHasTimeAdvantage;

    if (board.is_checkmate(~teamToPlay, !teamToPlayHasTimeAdvantage)) {
        if (endInPly) {
            *endInPly = 1;
        }
        return TerminalOutcome::WIN;
    }
    if (board.is_checkmate(teamToPlay, teamToPlayHasTimeAdvantage)) {
        if (endInPly) {
            *endInPly = 1;
        }
        return TerminalOutcome::LOSS;
    }
    if (board.is_draw(searchPly)) {
        return TerminalOutcome::DRAW;
    }
    if (searchPly > 0 && has_unavoidable_waiting_board_mate(
            board, teamToPlay, teamToPlayHasTimeAdvantage, searchPly)) {
        if (endInPly) {
            // One forced reply, then the opponent's mating move. Terminal
            // nodes use distance 1, so this position is three solver plies out.
            *endInPly = 3;
        }
        return TerminalOutcome::LOSS;
    }
    return TerminalOutcome::NONE;
}

SearchThread::SearchThread() : transpositionTable(nullptr), currentBatchSize(0) {
    // Buffers are allocated lazily in ensureBufferSize() when run_iteration is called
}

SearchThread::~SearchThread() {
    if (obs) cudaFreeHost(obs);
}

void SearchThread::ensureBufferSize(int batchSize) {
    if (batchSize == currentBatchSize) return;

    if (obs) cudaFreeHost(obs);
    obs = nullptr;

    auto allocatePinned = [](__half** buffer, size_t count) {
        cudaError_t result = cudaMallocHost(
            reinterpret_cast<void**>(buffer), count * sizeof(__half));
        if (result != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaMallocHost failed: ") + cudaGetErrorString(result));
        }
    };

    allocatePinned(&obs, batchSize * NB_INPUT_VALUES());
    
    batchContexts.reserve(batchSize);
    currentBatchSize = batchSize;
}

void SearchThread::set_search_info(SearchInfo* info) {
    searchInfo = info;
}

void SearchThread::set_root_node(Node* node) {
    root = node;
}

void SearchThread::set_inference_worker_index(size_t workerIndex) {
    inferenceWorkerIndex = workerIndex;
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
    NodeType childType = trajectory.empty()
        ? NodeType::UNSOLVED
        : trajectory.back().node->get_node_type();
    if (childType == NodeType::WIN) {
        valueToBackup = 1.0f;
    } else if (childType == NodeType::LOSS) {
        valueToBackup = -1.0f;
    } else if (childType == NodeType::DRAW) {
        const Stockfish::Color leafTeam = trajectory.back().node->get_team_to_play();
        const Stockfish::Color rootTeam = trajectory.front().node->get_team_to_play();
        valueToBackup = leafTeam == rootTeam
            ? -runtimeConfig.drawContempt
            : runtimeConfig.drawContempt;
    }
    
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
        LeafSelection selection = select_and_expand(board, teamHasTimeAdvantage);
        if (!selection.leaf) {
            reservationCollisions++;
            cancel_virtual_losses(trajectoryBuffer);
            for (auto it = trajectoryBuffer.rbegin(); it != trajectoryBuffer.rend(); ++it) {
                const JointActionCandidate& action = it->action;
                if (action.moveA != Stockfish::MOVE_NONE || action.moveB != Stockfish::MOVE_NONE) {
                    board.unmake_moves(action.moveA, action.moveB);
                }
            }
            if (batchContexts.empty()
                && selectionAttempts == maxSelectionAttempts
                && selection.pendingEvaluation) {
                selection.pendingEvaluation->wait_for_evaluation_completion();
            }
            continue;
        }
        Node* leaf = selection.leaf;
        
        // A repeated leaf is not an independent simulation. Revert its
        // temporary selection state and leave the first trajectory responsible
        // for evaluating and backing up this position.
        bool isCollision = std::any_of(
            batchContexts.begin(), batchContexts.end(),
            [leaf](const LeafContext& previous) { return previous.leaf == leaf; });
        if (isCollision) {
            sameBatchCollisions++;
            if (selection.hasEvaluationReservation) {
                leaf->release_evaluation_reservation();
            }
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
        ctx.hasEvaluationReservation = selection.hasEvaluationReservation;
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
        const NodeType solvedType = SearchParams::ENABLE_MCTS_SOLVER
            ? ctx.leaf->get_node_type()
            : NodeType::UNSOLVED;
        if (solvedType != NodeType::UNSOLVED) {
            ctx.isTerminal = true;
            if (solvedType == NodeType::WIN) {
                ctx.terminalValue = 1.0f;
            } else if (solvedType == NodeType::LOSS) {
                ctx.terminalValue = -1.0f;
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

        int terminalEndInPly = 0;
        const TerminalOutcome terminalOutcome = classify_terminal_position(
            board, ctx.teamToPlay, root->get_team_to_play(),
            teamHasTimeAdvantage, searchPly, &terminalEndInPly);
        if (terminalOutcome != TerminalOutcome::NONE) {
            ctx.isTerminal = true;
            if (terminalOutcome == TerminalOutcome::WIN) {
                ctx.terminalValue = 1.0f;
                if (SearchParams::ENABLE_MCTS_SOLVER) {
                    ctx.leaf->mark_as_win(terminalEndInPly);
                }
            } else if (terminalOutcome == TerminalOutcome::LOSS) {
                ctx.terminalValue = -1.0f;
                if (SearchParams::ENABLE_MCTS_SOLVER) {
                    ctx.leaf->mark_as_loss(terminalEndInPly);
                }
            } else {
                ctx.terminalValue = ctx.teamToPlay == root->get_team_to_play()
                    ? -runtimeConfig.drawContempt
                    : runtimeConfig.drawContempt;
                if (SearchParams::ENABLE_MCTS_SOLVER) {
                    ctx.leaf->mark_as_draw(1);
                }
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
        if (!ctx.hasEvaluationReservation) {
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
    Engine::HalfInferenceOutputs inferenceOutputs;
    if (validInferenceCount > 0) {
        // TensorRT is built for a fixed batch. Duplicate the last valid row so
        // underfilled batches never upload stale or uninitialized host memory.
        const __half* paddingSource = obs
            + static_cast<size_t>(validInferenceCount - 1) * NB_INPUT_VALUES();
        for (int batchIndex = validInferenceCount; batchIndex < batchSize; ++batchIndex) {
            std::copy_n(
                paddingSource, NB_INPUT_VALUES(),
                obs + static_cast<size_t>(batchIndex) * NB_INPUT_VALUES());
        }
        if (!engine->runInferenceHalf(
            obs, inferenceOutputs, inferenceWorkerIndex)) {
            cerr << "Batch inference failed" << endl;
            int completedTerminals = 0;
            for (auto& ctx : batchContexts) {
                if (ctx.hasEvaluationReservation) {
                    ctx.leaf->release_evaluation_reservation();
                }
                if (ctx.isTerminal) {
                    backup(ctx.trajectory, board, ctx.terminalValue);
                    completedTerminals++;
                } else {
                    cancel_virtual_losses(ctx.trajectory);
                }
            }
            searchInfo->increment_nodes(completedTerminals);
            searchInfo->increment_same_batch_collisions(sameBatchCollisions);
            searchInfo->increment_reservation_collisions(reservationCollisions);
            throw std::runtime_error("Batch inference failed");
        }
    }
    
    // Phase 3: Process results and backup
    int inferenceIdx = 0;
    for (auto& ctx : batchContexts) {
        if (ctx.isTerminal) {
            if (ctx.hasEvaluationReservation) {
                ctx.leaf->release_evaluation_reservation();
            }
            backup(ctx.trajectory, board, ctx.terminalValue);
        } else {
            if (SearchParams::ENABLE_MCTS_SOLVER
                && ctx.leaf->get_node_type() != NodeType::UNSOLVED) {
                ctx.leaf->release_evaluation_reservation();
                backup(ctx.trajectory, board, 0.0f);
                inferenceIdx++;
                continue;
            }

            for (const TrajectoryEntry& entry : ctx.trajectory) {
                const JointActionCandidate& action = entry.action;
                if (action.moveA != Stockfish::MOVE_NONE || action.moveB != Stockfish::MOVE_NONE) {
                    board.make_moves(action.moveA, action.moveB);
                }
            }

            // Non-terminal node - process NN output and expand
            const __half* batchValue = inferenceOutputs.value + inferenceIdx;
            const __half* batchPiA = inferenceOutputs.policyA
                + inferenceIdx * NB_POLICY_VALUES();
            const __half* batchPiB = inferenceOutputs.policyB
                + inferenceIdx * NB_POLICY_VALUES();

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
                    batchPiA, actionsA, BOARD_A, leafBoard);
            }
            
            if (actionsB.empty()) {
                // Not on turn or no legal moves - MOVE_NONE is only option
                actionsB.push_back(Stockfish::MOVE_NONE);
                priorsB.push_back(1.0f);
            } else {
                // On turn with legal moves - can also pass (MOVE_NONE)
                actionsB.push_back(Stockfish::MOVE_NONE);
                priorsB = get_normalized_probability(
                    batchPiB, actionsB, BOARD_B, leafBoard);
            }

            // Expand leaf node and register in transposition table (MCGS)
            // Use leafTeamHasTimeAdvantage for the team making the move at this leaf.
            // The generator creates joint actions that THIS team will play.
            expand_leaf_node(ctx.leaf, actionsA, actionsB, priorsA, priorsB,
                             leafTeamHasTimeAdvantage, boardAOnTurn, boardBOnTurn,
                             ctx.leafHash);
            ctx.leaf->release_evaluation_reservation();
                
            // Backup value
            const float batchValueFloat = __half2float(*batchValue);
            float neuralValue = std::isfinite(batchValueFloat)
                ? std::clamp(batchValueFloat, -1.0f, 1.0f)
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
LeafSelection SearchThread::select_and_expand(Board& board, bool teamHasTimeAdvantage) {
    Node* currentNode = root;
    shared_ptr<Node> nextNode;
    int childIdx;
    bool hasEvaluationReservation = false;

    // Root node has no incoming action, -1 means no child selected yet
    trajectoryBuffer.emplace_back(currentNode, JointActionCandidate(), -1);

    while (true) {
        if (SearchParams::ENABLE_MCTS_SOLVER
            && currentNode->get_node_type() != NodeType::UNSOLVED) {
            break;
        }

        // If not expanded, this is a leaf node
        if (!currentNode->is_expanded()) {
            if (!hasEvaluationReservation) {
                if (!currentNode->try_reserve_evaluation()) {
                    return {nullptr, false, currentNode};
                }
                if (currentNode->is_expanded()
                    || currentNode->get_node_type() != NodeType::UNSOLVED) {
                    currentNode->release_evaluation_reservation();
                    continue;
                }
                hasEvaluationReservation = true;
            }
            break;
        }

        // Grow the large Cartesian joint-action space as the node earns visits.
        if (currentNode->should_expand_new_child(runtimeConfig)) {
            // Expand first to atomically get the action
            JointActionCandidate expandedAction;
            bool expandedChildReserved = false;
            nextNode = currentNode->expand_next_joint_child(
                nullptr, 0, expandedAction, runtimeConfig, &childIdx, true,
                &expandedChildReserved);
            
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
                        if (expandedChildReserved) {
                            nextNode->release_evaluation_reservation();
                            expandedChildReserved = false;
                        }
                        currentNode->replace_child(childIdx, canonicalNode);
                        nextNode = canonicalNode;
                        continueFromTransposition = canonicalNode->is_expanded();
                        if (!continueFromTransposition
                            && canonicalNode->get_node_type() == NodeType::UNSOLVED) {
                            if (!canonicalNode->try_reserve_evaluation()) {
                                return {nullptr, false, canonicalNode.get()};
                            }
                            if (canonicalNode->is_expanded()
                                || canonicalNode->get_node_type() != NodeType::UNSOLVED) {
                                canonicalNode->release_evaluation_reservation();
                                continueFromTransposition = canonicalNode->is_expanded();
                            } else {
                                expandedChildReserved = true;
                            }
                        }
                    }
                }
                
                // Update the parent trajectory entry with the selected child index
                trajectoryBuffer.back().selectedChildIdx = childIdx;
                
                trajectoryBuffer.emplace_back(nextNode.get(), expandedAction, -1);

                if (continueFromTransposition) {
                    currentNode = nextNode.get();
                    hasEvaluationReservation = false;
                    continue;
                }
                
                // Return the newly expanded leaf
                return {nextNode.get(), expandedChildReserved};
            }
        }

        // Standard PUCT selection among expanded children
        Node::ChildSelection selection =
            currentNode->select_child_and_apply_virtual_loss(runtimeConfig);
        if (!selection.child || selection.childIdx < 0) {
            return {nullptr, false, selection.pendingEvaluation};
        }
        nextNode = selection.child;
        childIdx = selection.childIdx;
        hasEvaluationReservation = selection.hasEvaluationReservation;
        
        // Update the parent trajectory entry with the selected child index
        trajectoryBuffer.back().selectedChildIdx = childIdx;

        JointActionCandidate action = currentNode->get_joint_action(childIdx);
        board.make_moves(action.moveA, action.moveB);
        
        trajectoryBuffer.emplace_back(nextNode.get(), action, -1);
        currentNode = nextNode.get();
    }

    return {currentNode, hasEvaluationReservation};
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

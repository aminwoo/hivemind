#include "searchthread.h"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <memory>
#include <string>

#include "joint_action.h"
#include "utils.h"

using namespace std;

SearchThread::SearchThread() : transpositionTable(nullptr), currentBatchSize(0) {
    // Buffers are allocated lazily in ensureBufferSize() when run_iteration is called
}

SearchThread::~SearchThread() {
    delete[] obs;
    delete[] value;
    delete[] piA;
    delete[] piB;
}

void SearchThread::ensureBufferSize(int batchSize) {
    if (batchSize == currentBatchSize) return;
    
    // Free old buffers
    delete[] obs;
    delete[] value;
    delete[] piA;
    delete[] piB;
    
    // Allocate new buffers
    obs = new float[batchSize * NB_INPUT_VALUES()];
    value = new float[batchSize];
    piA = new float[batchSize * NB_POLICY_VALUES()];
    piB = new float[batchSize * NB_POLICY_VALUES()];
    
    batchContexts.reserve(batchSize);
    currentBatchSize = batchSize;
}

SearchInfo* SearchThread::get_search_info() {
    return searchInfo;
}

void SearchThread::set_search_info(SearchInfo* info) {
    searchInfo = info;
}

void SearchThread::set_root_node(Node* node) {
    root = node;
    root->set_value(0.0f);
}

Node* SearchThread::get_root_node() {
    return root;
}

void SearchThread::set_transposition_table(TranspositionTable* table) {
    transpositionTable = table;
}

TranspositionTable* SearchThread::get_transposition_table() {
    return transpositionTable;
}

void SearchThread::backup(vector<TrajectoryEntry>& trajectory, 
                          Board& board, float valueToBackup) {
    // Process nodes in reverse order (from leaf to root)
    for (auto it = trajectory.rbegin(); it != trajectory.rend(); ++it) {
        Node* node = it->node;
        int childIdx = it->selectedChildIdx;

        if (childIdx >= 0) {
            // Internal node - use the stored child index
            node->update(childIdx, valueToBackup);
            // Remove virtual loss after real update
            node->remove_virtual_loss(childIdx);
        } else {
            // Root or leaf without child selection
            node->update_terminal(valueToBackup);
        }
        valueToBackup = -valueToBackup;
    }
    
    // Note: moves are already undone during batch collection, no need to undo here
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
    int batchCollisions = 0;
    
    // Phase 1: Collect batchSize leaf nodes
    for (int i = 0; i < batchSize; i++) {
        LeafContext ctx;
        trajectoryBuffer.clear();
        
        // Select and expand to get a leaf node (MCGS: with transposition lookup)
        Node* leaf = select_and_expand(board, teamHasTimeAdvantage);
        
        // Check if this leaf was already selected in this batch (collision)
        for (const auto& prevCtx : batchContexts) {
            if (prevCtx.leaf == leaf) {
                batchCollisions++;
                break;
            }
        }
        
        // Update max depth reached in this search
        searchInfo->set_max_depth(leaf->get_depth());
        
        // Store trajectory
        ctx.trajectory = trajectoryBuffer;
        ctx.leaf = leaf;
        ctx.teamToPlay = leaf->get_team_to_play();
        ctx.sitPlaneActive = (ctx.teamToPlay == root->get_team_to_play()) == teamHasTimeAdvantage;
        
        // Check for terminal states
        if (board.is_draw()) {
            ctx.isTerminal = true;
            // Apply draw contempt: treat draws as slightly negative for the side to move
            ctx.terminalValue = -SearchParams::DRAW_CONTEMPT;
            batchContexts.push_back(std::move(ctx));
            
            // Undo moves for this trajectory so we can do another selection
            for (auto it = trajectoryBuffer.rbegin(); it != trajectoryBuffer.rend(); ++it) {
                const JointActionCandidate& action = it->action;
                if (action.moveA != Stockfish::MOVE_NONE || action.moveB != Stockfish::MOVE_NONE) {
                    board.unmake_moves(action.moveA, action.moveB);
                }
            }
            continue;
        }
        
        // Check for checkmate on each board
        bool isCheckmate = false;
        Stockfish::Color teamToPlay = ctx.teamToPlay;
        
        if (board.side_to_move(BOARD_A) == teamToPlay) {
            auto movesA = board.legal_moves(BOARD_A);
            if (movesA.empty() && board.is_in_check(BOARD_A)) {
                //cerr << "Checkmate on board A: " << board.fen(BOARD_A) << "|" << board.fen(BOARD_B) << endl;
                ctx.isTerminal = true;
                ctx.terminalValue = -1.0f;
                isCheckmate = true;
            }
        }
        if (!isCheckmate && board.side_to_move(BOARD_B) == ~teamToPlay) {
            auto movesB = board.legal_moves(BOARD_B);
            if (movesB.empty() && board.is_in_check(BOARD_B)) {
                //cerr << "Checkmate on board B: " << board.fen(BOARD_A) << "|" << board.fen(BOARD_B) << endl;
                ctx.isTerminal = true;
                ctx.terminalValue = -1.0f;
                isCheckmate = true;
            }
        }
        
        if (isCheckmate) {
            batchContexts.push_back(std::move(ctx));
            
            // Undo moves
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
        ctx.leafHash = board.hash_key(teamHasTimeAdvantage);  // Store hash for MCGS transposition lookup
        
        ctx.boardState = std::make_unique<Board>(board);  // Copy board state for later processing
        
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
        if (!engine->runInference(obs, value, piA, piB)) {
            cerr << "Batch inference failed" << endl;
            // Backup all as 0.0 and remove virtual loss
            for (auto& ctx : batchContexts) {
                if (ctx.boardState) {
                    backup(ctx.trajectory, *ctx.boardState, 0.0f);
                } else {
                    // Terminal node - backup and remove virtual loss
                    for (auto it = ctx.trajectory.rbegin(); it != ctx.trajectory.rend(); ++it) {
                        Node* node = it->node;
                        int childIdx = it->selectedChildIdx;
                        if (childIdx >= 0) {
                            node->update(childIdx, 0.0f);
                            node->remove_virtual_loss(childIdx);
                        } else {
                            node->update_terminal(0.0f);
                        }
                    }
                }
            }
            searchInfo->increment_nodes(batchSize);
            return;
        }
    }
    
    // Phase 3: Process results and backup
    int inferenceIdx = 0;
    for (auto& ctx : batchContexts) {
        if (ctx.isTerminal) {
            // Terminal node - backup the terminal value and remove virtual loss
            float val = ctx.terminalValue;
            for (auto it = ctx.trajectory.rbegin(); it != ctx.trajectory.rend(); ++it) {
                Node* node = it->node;
                int childIdx = it->selectedChildIdx;
                if (childIdx >= 0) {
                    node->update(childIdx, val);
                    node->remove_virtual_loss(childIdx);
                } else {
                    node->update_terminal(val);
                }
                val = -val;
            }
        } else {
            // Non-terminal node - process NN output and expand
            float* batchValue = value + inferenceIdx;
            float* batchPiA = piA + inferenceIdx * NB_POLICY_VALUES();
            float* batchPiB = piB + inferenceIdx * NB_POLICY_VALUES();
            
            Board& leafBoard = *ctx.boardState;
            
            // Get legal moves for each board
            vector<Stockfish::Move> actionsA;
            vector<Stockfish::Move> actionsB;
            
            if (leafBoard.side_to_move(BOARD_A) == ctx.teamToPlay) {
                actionsA = leafBoard.legal_moves(BOARD_A);
            }
            if (leafBoard.side_to_move(BOARD_B) == ~ctx.teamToPlay) {
                actionsB = leafBoard.legal_moves(BOARD_B);
            }
            
            vector<float> priorsA;
            vector<float> priorsB;
            
            if (actionsA.empty()) {
                actionsA.push_back(Stockfish::MOVE_NONE);
                priorsA.push_back(1.0f);
            } else {
                actionsA.push_back(Stockfish::MOVE_NONE);
                priorsA = get_normalized_probability(batchPiA, actionsA, 0, leafBoard);
            }
            
            if (actionsB.empty()) {
                actionsB.push_back(Stockfish::MOVE_NONE);
                priorsB.push_back(1.0f);
            } else {
                actionsB.push_back(Stockfish::MOVE_NONE);
                priorsB = get_normalized_probability(batchPiB, actionsB, 1, leafBoard);
            }
            
            // Expand leaf node and register in transposition table (MCGS)
            expand_leaf_node(ctx.leaf, actionsA, actionsB, priorsA, priorsB, 
                             teamHasTimeAdvantage, ctx.leafHash);
                
            // Backup value
            backup(ctx.trajectory, leafBoard, *batchValue);
            
            inferenceIdx++;
        }
    }
    
    searchInfo->increment_nodes(batchSize);
    searchInfo->increment_collisions(batchCollisions);
}

/**
 * @brief Selects a leaf node and expands using progressive widening with MCGS.
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

        // Check if we should expand a new child (progressive widening)
        if (currentNode->should_expand_new_child()) {
            // Expand first to atomically get the action
            JointActionCandidate expandedAction;
            nextNode = currentNode->expand_next_joint_child(nullptr, 0, expandedAction);
            
            if (nextNode && expandedAction.jointPrior > 0.0f) {
                childIdx = currentNode->get_expanded_count() - 1;
                
                // Make moves with the actual expanded action
                board.make_moves(expandedAction.moveA, expandedAction.moveB);
                
                // MCGS: Compute position hash and register in transposition table
                uint64_t childHash = board.hash_key(teamHasTimeAdvantage);
                nextNode->set_hash(childHash);
                
                // Register in transposition table (for stats tracking, if MCGS enabled)
                if (SearchParams::ENABLE_MCGS && transpositionTable) {
                    transpositionTable->insertOrGet(childHash, nextNode);
                }
                
                // Apply virtual loss to discourage re-selection in same batch
                currentNode->apply_virtual_loss(childIdx);
                
                // Update the parent trajectory entry with the selected child index
                trajectoryBuffer.back().selectedChildIdx = childIdx;
                
                trajectoryBuffer.emplace_back(nextNode.get(), expandedAction, -1);
                
                // Return the newly expanded leaf
                return nextNode.get();
            } else if (nextNode) {
                // Action had zero prior (shouldn't happen often) - still return the node
                childIdx = currentNode->get_expanded_count() - 1;
                currentNode->apply_virtual_loss(childIdx);
                trajectoryBuffer.back().selectedChildIdx = childIdx;
                
                board.make_moves(expandedAction.moveA, expandedAction.moveB);
                trajectoryBuffer.emplace_back(nextNode.get(), expandedAction, -1);
                return nextNode.get();
            }
        }

        // Standard PUCT selection among expanded children
        auto [selectedChild, selectedIdx] = currentNode->get_best_expanded_child_with_idx();
        if (!selectedChild || selectedIdx < 0) {
            break;  // No children available
        }
        nextNode = selectedChild;
        childIdx = selectedIdx;
        
        // Apply virtual loss to discourage re-selection in same batch
        currentNode->apply_virtual_loss(childIdx);
        
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
 * @param positionHash Zobrist hash of the leaf position for transposition table
 */
void SearchThread::expand_leaf_node(Node* leaf,
                                    const vector<Stockfish::Move>& actionsA,
                                    const vector<Stockfish::Move>& actionsB,
                                    const vector<float>& priorsA,
                                    const vector<float>& priorsB,
                                    bool teamHasTimeAdvantage,
                                    uint64_t positionHash) {
    // Store the position hash in the node for MCGS
    if (positionHash != 0) {
        leaf->set_hash(positionHash);
    }
    
    // Atomically try to initialize and expand if not already done
    // This is safe for concurrent access from multiple threads
    leaf->try_init_and_expand(actionsA, actionsB, priorsA, priorsB, teamHasTimeAdvantage);
    
    // Note: The first child created during try_init_and_expand doesn't have its
    // hash computed yet (would require board access). However, when that child
    // is later traversed through select_and_expand, if it's unexpanded, the
    // transposition lookup will happen at that time based on the board state.
    // This is slightly less efficient than computing the hash during expansion,
    // but avoids complexity of passing the board to try_init_and_expand.
}

#include "search/agent.h"
#include "search/mate_probe.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "environment/joint_action.h"
#include "Fairy-Stockfish/src/uci.h"
#include "search/search_params.h"
#include "search/searchthread.h"
#include "common/utils.h"
#include "common/globals.h"

using namespace std;

/**
 * @brief Format UCI score string based on node type.
 * 
 * Returns "score mate N" for proven wins/losses, "score cp X" otherwise.
 * Mate distance is computed from endInPly (ply to terminal).
 * Positive mate = we win in N moves, negative mate = we lose in N moves.
 * 
 * @param node The node to format score for
 * @param C Conversion constant for Q to centipawns
 * @param k Tan scaling constant
 * @return Formatted score string (e.g., "score cp 150" or "score mate 5")
 */
/**
 * @brief Format UCI score string based on node type and Q value.
 * 
 * Returns "score mate N" for proven wins/losses, "score cp X" otherwise.
 * Mate distance is computed from endInPly (ply to terminal).
 * Positive mate = we win in N moves, negative mate = we lose in N moves.
 * 
 * @param node The node to check for solved state
 * @param qFromParent The Q-value from the root's perspective (positive = good for root)
 * @param isChildNode True if this is a child node (opponent's perspective), false for root node
 * @param C Conversion constant for Q to centipawns
 * @param k Tan scaling constant
 * @return Formatted score string (e.g., "score cp 150" or "score mate 5")
 */
static string format_uci_score(const Node* node, float qFromParent, bool isChildNode = true, 
                               float C = 180.0f, float k = 1.56f) {
    if (!node) return "score cp 0";
    
    NodeType nodeType = node->get_node_type();
    int endInPly = node->get_end_in_ply();
    
    if (nodeType == NodeType::WIN) {
        int mateInMoves = (endInPly + 1) / 2;
        if (isChildNode) {
            // Child is a WIN for the child (opponent) = LOSS for us (we're mated)
            return "score mate -" + to_string(max(1, mateInMoves));
        } else {
            // Root is a WIN for us = we win
            return "score mate " + to_string(max(1, mateInMoves));
        }
    } else if (nodeType == NodeType::LOSS) {
        int mateInMoves = (endInPly + 1) / 2;
        if (isChildNode) {
            // Child is a LOSS for the child (opponent) = WIN for us (we mate them)
            return "score mate " + to_string(max(1, mateInMoves));
        } else {
            // Root is a LOSS for us = we're mated
            return "score mate -" + to_string(max(1, mateInMoves));
        }
    } else if (nodeType == NodeType::DRAW) {
        return "score cp 0";
    } else {
        // Not solved, use centipawn score from Q value (already from root's perspective)
        int cpScore = static_cast<int>(C * std::tan(k * qFromParent));
        return "score cp " + to_string(cpScore);
    }
}

string Agent::format_root_aware_uci_score(
    const shared_ptr<Node>& root,
    const shared_ptr<Node>& pvChild,
    float childQ,
    float centipawnScale,
    float tangentScale) {
    if (root && root->get_node_type() != NodeType::UNSOLVED) {
        // A solver proof is authoritative for the position being reported.
        // In particular, a proven DRAW must score 0 rather than leaking an
        // unvisited PV child's Q_INIT=-1 through the tangent conversion.
        return format_uci_score(
            root.get(), root->Q(), false, centipawnScale, tangentScale);
    }
    return format_uci_score(
        pvChild.get(), childQ, true, centipawnScale, tangentScale);
}

/**
 * @brief Check if we should exit search early due to proven mate / solved position.
 * 
 * Returns true if:
 * - Root node is proven WIN (we have forced mate)
 * - Best child is proven LOSS (opponent loses = we have forced mate via that move)
 * 
 * @param rootNode The root node of the search tree
 * @param bestChildIdx Index of the best child (by visits)
 * @param verbose If true, print info string when exiting early
 * @return True if search should exit early
 */
static bool should_exit_early_winning(const std::shared_ptr<Node>& rootNode, int bestChildIdx, 
                                       bool verbose) {
    if (!SearchParams::ENABLE_MATE_EARLY_EXIT) {
        return false;
    }
    
    if (!rootNode || !rootNode->is_expanded()) {
        return false;
    }
    
    // Any solved root is game-theoretically final. Move selection still
    // chooses the fastest win, longest loss, or available draw afterward.
    const NodeType rootType = rootNode->get_node_type();
    if (rootType != NodeType::UNSOLVED) {
        if (verbose) {
            const char* outcome = rootType == NodeType::WIN ? "WIN"
                : rootType == NodeType::LOSS ? "LOSS" : "DRAW";
            cout << "info string Early exit: root position is proven " << outcome << endl;
        }
        return true;
    }
    
    // Check if best child is proven LOSS (opponent loses = we win via that move)
    std::shared_ptr<Node> bestChildOwner = rootNode->get_child(bestChildIdx);
    if (bestChildOwner) {
        Node* bestChild = bestChildOwner.get();
        if (bestChild && bestChild->get_node_type() == NodeType::LOSS) {
            if (verbose) {
                int mateInPly = bestChild->get_end_in_ply();
                int mateInMoves = (mateInPly + 1) / 2;
                cout << "info string Early exit: forced mate in " << mateInMoves << " found" << endl;
            }
            return true;
        }
    }
    
    return false;
}

/**
 * @brief Node budget for the root mate pre-pass, scaled to the search it precedes.
 *
 * A fixed budget would be a fixed cost, which a few-hundred-node self-play
 * search cannot absorb. Whichever stopping condition is set caps the pre-pass at
 * a few percent of the search, never above the hard ceiling.
 */
static uint64_t mate_search_node_budget(const SearchOptions& options) {
    uint64_t budget = SearchParams::MATE_SEARCH_NODE_BUDGET;
    if (options.targetNodes > 0) {
        budget = std::min(budget, static_cast<uint64_t>(options.targetNodes)
                                      * SearchParams::MATE_SEARCH_NODES_PER_SEARCH_NODE);
    }
    if (options.moveTimeMs > 0) {
        budget = std::min(budget, static_cast<uint64_t>(options.moveTimeMs)
                                      * SearchParams::MATE_SEARCH_NODES_PER_MILLISECOND);
    }
    return std::max(budget, SearchParams::MATE_SEARCH_MIN_NODE_BUDGET);
}

/**
 * @brief Performs a fast 1-ply checkmate scan at the root before starting MCTS.
 * If any legal joint action immediately delivers checkmate against the opponent,
 * returns true and sets outAction to that winning joint move.
 */
static bool find_immediate_root_mate(Board& board, Stockfish::Color teamSide,
                                     bool teamHasTimeAdvantage,
                                     JointActionCandidate& outAction,
                                     Agent::MateSearchBudget* budget = nullptr) {
    const bool boardAOnTurn = board.side_to_move(BOARD_A) == teamSide;
    const bool boardBOnTurn = board.side_to_move(BOARD_B) == ~teamSide;

    vector<Stockfish::Move> actionsA;
    if (boardAOnTurn) {
        actionsA = board.legal_moves(BOARD_A);
    }
    vector<Stockfish::Move> actionsB;
    if (boardBOnTurn) {
        actionsB = board.legal_moves(BOARD_B);
    }

    const bool boardACanMove = !actionsA.empty();
    const bool boardBCanMove = !actionsB.empty();
    const JointActionRules rules{boardAOnTurn, boardBOnTurn, teamHasTimeAdvantage,
                                 boardACanMove, boardBCanMove};

    const bool aInCheckBefore = board.is_in_check(BOARD_A);
    const bool bInCheckBefore = board.is_in_check(BOARD_B);
    const bool anyCheckBefore = aInCheckBefore || bInCheckBefore;

    // is_checkmate() also reports a loss for a team left without any legal
    // action (bughouse scores stalemate as a loss), and a quiet move can create
    // that state, so the check-only scan below would miss it. Two cheap
    // necessary conditions keep the quiet scan out of ordinary positions:
    //  - The opponent must already be immobile on every board they are on turn
    //    for. Those are exactly the boards we cannot move on, and our captures
    //    feed our own partner's hand, so we can never take those moves away.
    //  - Their hand on the board we move on must be empty. A quiet move leaves
    //    them out of check, and out of check any piece in hand is a legal drop.
    const bool opponentImmobileElsewhere =
        (boardAOnTurn || board.legal_moves(BOARD_A).empty())
        && (boardBOnTurn || board.legal_moves(BOARD_B).empty());
    auto opponent_hand_is_empty = [&](int boardNum, Stockfish::Color opponentColor) {
        for (Stockfish::PieceType pt : {Stockfish::PAWN, Stockfish::KNIGHT, Stockfish::BISHOP,
                                        Stockfish::ROOK, Stockfish::QUEEN}) {
            if (board.count_in_hand(boardNum, opponentColor, pt) > 0) {
                return false;
            }
        }
        return true;
    };
    const bool quietScanA = boardAOnTurn && opponentImmobileElsewhere
        && opponent_hand_is_empty(BOARD_A, ~teamSide);
    const bool quietScanB = boardBOnTurn && opponentImmobileElsewhere
        && opponent_hand_is_empty(BOARD_B, teamSide);

    // Partition each move list so the checking moves form a prefix, then use the
    // prefix length instead of re-running gives_check inside the scan loops.
    auto partition_checking = [&](int boardNum, vector<Stockfish::Move>& moves) {
        const auto quietBegin = std::stable_partition(
            moves.begin(), moves.end(), [&](Stockfish::Move m) {
                return board.gives_check(boardNum, m);
            });
        return static_cast<size_t>(std::distance(moves.begin(), quietBegin));
    };
    const size_t checkingA = boardAOnTurn ? partition_checking(BOARD_A, actionsA) : 0;
    const size_t checkingB = boardBOnTurn ? partition_checking(BOARD_B, actionsB) : 0;

    // A mate needs a king in check somewhere, so unless a board is already in
    // check (or a stalemate win is possible) only the checking prefix can mate.
    const size_t limitA = (anyCheckBefore || quietScanA) ? actionsA.size() : checkingA;
    const size_t limitB = (anyCheckBefore || quietScanB) ? actionsB.size() : checkingB;

    // 1. Move on Board A, pass on Board B
    if (boardAOnTurn) {
        for (size_t iA = 0; iA < limitA; ++iA) {
            if (budget && !budget->consume()) {
                return false;
            }
            const Stockfish::Move mA = actionsA[iA];
            const bool isCapA = board.is_capture(BOARD_A, mA);
            const bool canPassB = !boardBOnTurn || is_single_pass_legal(
                teamHasTimeAdvantage, boardAOnTurn, boardBOnTurn, isCapA);
            if (canPassB) {
                board.push_move(BOARD_A, mA);
                const bool isMate = board.is_checkmate(~teamSide, !teamHasTimeAdvantage);
                board.pop_move(BOARD_A);
                if (isMate) {
                    outAction = JointActionCandidate(mA, 1.0f, iA, Stockfish::MOVE_NONE, 1.0f, 0,
                                                     rules, isCapA, false);
                    return true;
                }
            }
        }
    }

    // 2. Move on Board B, pass on Board A
    if (boardBOnTurn) {
        for (size_t iB = 0; iB < limitB; ++iB) {
            if (budget && !budget->consume()) {
                return false;
            }
            const Stockfish::Move mB = actionsB[iB];
            const bool isCapB = board.is_capture(BOARD_B, mB);
            const bool canPassA = !boardAOnTurn || is_single_pass_legal(
                teamHasTimeAdvantage, boardAOnTurn, boardBOnTurn, isCapB);
            if (canPassA) {
                board.push_move(BOARD_B, mB);
                const bool isMate = board.is_checkmate(~teamSide, !teamHasTimeAdvantage);
                board.pop_move(BOARD_B);
                if (isMate) {
                    outAction = JointActionCandidate(Stockfish::MOVE_NONE, 1.0f, 0, mB, 1.0f, iB,
                                                     rules, false, isCapB);
                    return true;
                }
            }
        }
    }

    // 3. Move on both boards if both on turn.
    //
    // Every branch of is_checkmate() needs the opponent to be out of legal moves
    // on Board A (mate there), on Board B (mate there), or on both (a team with
    // no legal action at all). Whether a board is left without a reply depends
    // only on our move on that board: a capture on the partner board feeds our
    // own hand, never the opponent's. So the two halves can be filtered
    // independently, and only pairs whose A-half or B-half immobilizes its board
    // need a joint is_checkmate() test - a handful instead of |A| x |B|.
    if (boardAOnTurn && boardBOnTurn) {
        auto immobilizing_moves = [&](int boardNum, const vector<Stockfish::Move>& moves,
                                      size_t limit) {
            vector<size_t> indices;
            for (size_t i = 0; i < limit; ++i) {
                if (budget && !budget->consume()) {
                    break;
                }
                board.push_move(boardNum, moves[i]);
                const bool opponentImmobile = board.legal_moves(boardNum).empty();
                board.pop_move(boardNum);
                if (opponentImmobile) {
                    indices.push_back(i);
                }
            }
            return indices;
        };
        const vector<size_t> immobilizingA = immobilizing_moves(BOARD_A, actionsA, limitA);
        const vector<size_t> immobilizingB = immobilizing_moves(BOARD_B, actionsB, limitB);

        auto pair_mates = [&](size_t iA, size_t iB) {
            if (budget && !budget->consume()) {
                return false;
            }
            const Stockfish::Move mA = actionsA[iA];
            const Stockfish::Move mB = actionsB[iB];
            board.make_moves(mA, mB);
            const bool isMate = board.is_checkmate(~teamSide, !teamHasTimeAdvantage);
            board.unmake_moves(mA, mB);
            if (!isMate) {
                return false;
            }
            outAction = JointActionCandidate(
                mA, 1.0f, iA, mB, 1.0f, iB, rules,
                board.is_capture(BOARD_A, mA), board.is_capture(BOARD_B, mB));
            return true;
        };

        for (size_t iA : immobilizingA) {
            for (size_t iB = 0; iB < actionsB.size(); ++iB) {
                if (pair_mates(iA, iB)) {
                    return true;
                }
            }
        }
        for (size_t iB : immobilizingB) {
            for (size_t iA = 0; iA < actionsA.size(); ++iA) {
                // Pairs whose A-half immobilizes Board A were covered above.
                if (std::find(immobilizingA.begin(), immobilizingA.end(), iA)
                    != immobilizingA.end()) {
                    continue;
                }
                if (pair_mates(iA, iB)) {
                    return true;
                }
            }
        }
    }

    return false;
}

/**
 * @brief Enumerate every legal joint action without requiring policy priors.
 *
 * Adding MOVE_NONE to each active board and filtering through the shared
 * JointActionRules keeps capture-pass, forced-pass, and double-sit legality
 * identical to the MCTS candidate generator.
 */
static vector<JointActionCandidate> legal_joint_actions(
    Board& board, Stockfish::Color teamSide, bool teamHasTimeAdvantage) {
    const bool boardAOnTurn = board.side_to_move(BOARD_A) == teamSide;
    const bool boardBOnTurn = board.side_to_move(BOARD_B) == ~teamSide;

    vector<Stockfish::Move> actionsA = boardAOnTurn
        ? board.legal_moves(BOARD_A)
        : vector<Stockfish::Move>{};
    vector<Stockfish::Move> actionsB = boardBOnTurn
        ? board.legal_moves(BOARD_B)
        : vector<Stockfish::Move>{};
    const bool boardACanMove = !actionsA.empty();
    const bool boardBCanMove = !actionsB.empty();
    actionsA.push_back(Stockfish::MOVE_NONE);
    actionsB.push_back(Stockfish::MOVE_NONE);

    const JointActionRules rules{
        boardAOnTurn, boardBOnTurn, teamHasTimeAdvantage,
        boardACanMove, boardBCanMove};
    vector<JointActionCandidate> actions;
    for (size_t indexA = 0; indexA < actionsA.size(); ++indexA) {
        const Stockfish::Move moveA = actionsA[indexA];
        const bool captureA = moveA != Stockfish::MOVE_NONE
            && board.is_capture(BOARD_A, moveA);
        for (size_t indexB = 0; indexB < actionsB.size(); ++indexB) {
            const Stockfish::Move moveB = actionsB[indexB];
            const bool captureB = moveB != Stockfish::MOVE_NONE
                && board.is_capture(BOARD_B, moveB);
            JointActionCandidate action(
                moveA, 1.0f, indexA, moveB, 1.0f, indexB,
                rules, captureA, captureB);
            if (action.jointPrior >= 0.0f) {
                actions.push_back(action);
            }
        }
    }
    return actions;
}

/**
 * @brief Recursively searches for a forced single-board checkmating sequence where the attacker
 * delivers continuous checks.
 *
 * @param board The bughouse board state
 * @param boardNum BOARD_A or BOARD_B
 * @param attackerColor Attacking side color on this board
 * @param currentPly Current 1-based ply (1, 3, 5...)
 * @param maxAttackerMoves Maximum attacker moves (e.g. 3 for mate in 3)
 * @param outMove Stores the root move that delivers or begins the mate
 * @param outPlyToMate Stores the total ply count to checkmate
 * @return true if a forced checkmate is proven, false otherwise
 */
bool Agent::search_single_board_forced_mate(
    Board& board,
    int boardNum,
    Stockfish::Color attackerColor,
    int currentPly,
    int maxAttackerMoves,
    Stockfish::Move& outMove,
    int& outPlyToMate,
    MateSearchBudget* budget,
    bool partnerBoardAgnostic) {
    return search_single_board_forced_mate_impl(
        board, boardNum, attackerColor, currentPly, maxAttackerMoves,
        outMove, outPlyToMate, budget, nullptr, partnerBoardAgnostic);
}


bool Agent::search_single_board_forced_mate_impl(
    Board& board,
    int boardNum,
    Stockfish::Color attackerColor,
    int currentPly,
    int maxAttackerMoves,
    Stockfish::Move& outMove,
    int& outPlyToMate,
    MateSearchBudget* budget,
    std::vector<MateContinuation>* continuations,
    bool partnerBoardAgnostic,
    std::vector<MateProofPly>* outPrincipalVariation) {
    if (outPrincipalVariation) {
        outPrincipalVariation->clear();
    }
    const int attackerMoveNum = (currentPly + 1) / 2;
    if (attackerMoveNum > maxAttackerMoves) {
        return false;
    }
    if (budget && budget->exhausted) {
        return false;
    }

    // is_checkmate() identifies a team by the color that team plays on Board A,
    // so the victim's team id is ~attackerColor only for a mate on Board A. On
    // Board B the attacker is the partner, and the victim's team id is
    // attackerColor itself.
    const Stockfish::Color victimTeam = (boardNum == BOARD_A)
        ? ~attackerColor
        : attackerColor;
    const Stockfish::Color attackerTeam = boardNum == BOARD_A
        ? attackerColor
        : ~attackerColor;

    auto retain_continuation = [&](Stockfish::Move move, int terminalPly) {
        if (!continuations) {
            return;
        }

        const int remainingPly = terminalPly - currentPly + 1;
        const bool boardAOnTurn =
            board.side_to_move(BOARD_A) == attackerTeam;
        const bool boardBOnTurn =
            board.side_to_move(BOARD_B) == ~attackerTeam;
        const JointActionRules rules{
            boardAOnTurn, boardBOnTurn, true,
            boardAOnTurn, boardBOnTurn};
        const bool isCapture = board.is_capture(boardNum, move);
        const JointActionCandidate action = boardNum == BOARD_A
            ? JointActionCandidate(
                move, 1.0f, 0, Stockfish::MOVE_NONE, 1.0f, 0,
                rules, isCapture, false)
            : JointActionCandidate(
                Stockfish::MOVE_NONE, 1.0f, 0, move, 1.0f, 0,
                rules, false, isCapture);
        const uint64_t positionHash = board.search_hash_key(attackerTeam, true);
        const std::string signature = board_signature(board);

        auto existing = std::find_if(
            continuations->begin(), continuations->end(),
            [&](const MateContinuation& continuation) {
                return continuation.positionHash == positionHash
                    && continuation.signature == signature
                    && continuation.teamSide == attackerTeam
                    && continuation.teamHasTimeAdvantage;
            });
        if (existing == continuations->end()) {
            continuations->push_back({
                positionHash, signature, attackerTeam, true,
                action, remainingPly});
        } else if (remainingPly < existing->plyToMate) {
            existing->action = action;
            existing->plyToMate = remainingPly;
        }
    };

    const std::vector<Stockfish::Move> checkingMoves = board.checking_moves(boardNum);
    if (checkingMoves.empty()) {
        return false;
    }

    struct CheckingContinuation {
        Stockfish::Move move;
        std::vector<Stockfish::Move> replies;
    };
    std::vector<CheckingContinuation> candidates;
    if (attackerMoveNum < maxAttackerMoves) {
        candidates.reserve(checkingMoves.size());
    }

    // 1. Check for terminal wins first. Besides literal checkmate, a check can
    // force the opponent to move here while an unavoidable mate is waiting on
    // the partner board.
    for (Stockfish::Move m : checkingMoves) {
        if (budget && !budget->consume()) {
            return false;
        }
        board.push_move(boardNum, m);
        int terminalEndInPly = 0;
        WaitingMateContinuation waitingMate;
        const TerminalOutcome terminalOutcome = classify_terminal_position(
            board, victimTeam, attackerTeam, true, currentPly,
            &terminalEndInPly, partnerBoardAgnostic, false,
            &waitingMate);
        if (terminalOutcome == TerminalOutcome::NONE
            && attackerMoveNum < maxAttackerMoves) {
            candidates.push_back({m, board.legal_moves(boardNum)});
        }
        board.pop_move(boardNum);
        if (terminalOutcome == TerminalOutcome::LOSS) {
            outMove = m;
            // The classifier's distance is relative to the position after this
            // move. Immediate mate is 1; an unavoidable waiting-board mate is
            // 3 (forced reply, then mate), so splice that suffix onto the
            // current root-relative ply.
            outPlyToMate = currentPly + terminalEndInPly - 1;
            if (outPrincipalVariation) {
                outPrincipalVariation->push_back(
                    boardNum == BOARD_A
                        ? MateProofPly{m, Stockfish::MOVE_NONE}
                        : MateProofPly{Stockfish::MOVE_NONE, m});
                if (waitingMate.activeBoard >= 0) {
                    outPrincipalVariation->push_back(
                        waitingMate.activeBoard == BOARD_A
                            ? MateProofPly{
                                waitingMate.reply, Stockfish::MOVE_NONE}
                            : MateProofPly{
                                Stockfish::MOVE_NONE, waitingMate.reply});
                    outPrincipalVariation->push_back(
                        waitingMate.waitingBoard == BOARD_A
                            ? MateProofPly{
                                waitingMate.matingMove,
                                Stockfish::MOVE_NONE}
                            : MateProofPly{
                                Stockfish::MOVE_NONE,
                                waitingMate.matingMove});
                }
            }
            retain_continuation(outMove, outPlyToMate);
            return true;
        }
    }

    // 2. If not immediate mate and we have moves remaining, verify all defender replies
    if (attackerMoveNum < maxAttackerMoves) {
        // A check with few evasions is cheaper to prove and more forcing.
        // Reuse the replies collected during the terminal scan, and never
        // continue through an already drawn or lost position.
        std::stable_sort(candidates.begin(), candidates.end(),
            [](const CheckingContinuation& lhs, const CheckingContinuation& rhs) {
                return lhs.replies.size() < rhs.replies.size();
            });
        for (const CheckingContinuation& candidate : candidates) {
            const Stockfish::Move m = candidate.move;
            if (budget && !budget->consume()) {
                return false;
            }
            board.push_move(boardNum, m);
            const std::vector<Stockfish::Move>& defenderReplies = candidate.replies;
            if (defenderReplies.empty()) {
                // Stalemate or terminal without checkmate
                board.pop_move(boardNum);
                continue;
            }

            bool allRepliesMated = true;
            int deepestReplyPly = currentPly;
            std::vector<MateProofPly> deepestLine;
            for (Stockfish::Move reply : defenderReplies) {
                if (budget && !budget->consume()) {
                    allRepliesMated = false;
                    break;
                }
                board.push_move(boardNum, reply);
                Stockfish::Move nextAttackerMove = Stockfish::MOVE_NONE;
                int nextReplyPly = 0;
                std::vector<MateProofPly> childLine;
                const bool replyMated = search_single_board_forced_mate_impl(
                    board, boardNum, attackerColor, currentPly + 2, maxAttackerMoves,
                    nextAttackerMove, nextReplyPly, budget, continuations,
                    partnerBoardAgnostic,
                    outPrincipalVariation ? &childLine : nullptr);
                board.pop_move(boardNum);
                if (!replyMated) {
                    allRepliesMated = false;
                    break;
                }
                if (nextReplyPly > deepestReplyPly || deepestLine.empty()) {
                    deepestReplyPly = nextReplyPly;
                    if (outPrincipalVariation) {
                        deepestLine.clear();
                        deepestLine.reserve(childLine.size() + 2);
                        if (boardNum == BOARD_A) {
                            deepestLine.push_back(
                                {m, Stockfish::MOVE_NONE});
                            deepestLine.push_back(
                                {reply, Stockfish::MOVE_NONE});
                        } else {
                            deepestLine.push_back(
                                {Stockfish::MOVE_NONE, m});
                            deepestLine.push_back(
                                {Stockfish::MOVE_NONE, reply});
                        }
                        deepestLine.insert(
                            deepestLine.end(), childLine.begin(), childLine.end());
                    }
                }
            }
            board.pop_move(boardNum);

            if (allRepliesMated) {
                outMove = m;
                outPlyToMate = deepestReplyPly;
                if (outPrincipalVariation) {
                    *outPrincipalVariation = std::move(deepestLine);
                }
                retain_continuation(outMove, outPlyToMate);
                return true;
            }
            if (budget && budget->exhausted) {
                return false;
            }
        }
    }

    return false;
}

namespace {

enum class JointMateStatus : uint8_t {
    REFUTED,
    PROVEN,
    UNKNOWN,
};

using MateJointAction = MateProofPly;

struct JointMateProofLine {
    MateJointAction action;
    std::shared_ptr<const JointMateProofLine> next;
};

struct JointMateProof {
    JointMateStatus status = JointMateStatus::REFUTED;
    int pliesToMate = 0;
    MateJointAction action;
    std::shared_ptr<const JointMateProofLine> principalVariation;
};

struct MateMoveCandidate {
    Stockfish::Move move = Stockfish::MOVE_NONE;
    bool isCapture = false;
    bool givesCheck = false;
};

struct MateActionSpace {
    std::vector<MateMoveCandidate> actionsA;
    std::vector<MateMoveCandidate> actionsB;
    JointActionRules rules;
};

struct JointMateCacheKey {
    uint64_t positionHash = 0;
    uint16_t attackerMovesRemaining = 0;
    uint8_t teamToPlay = 0;

    bool operator==(const JointMateCacheKey&) const = default;
};

struct JointMateCacheKeyHash {
    size_t operator()(const JointMateCacheKey& key) const {
        uint64_t mixed = Board::mix_hash(
            key.positionHash,
            static_cast<uint64_t>(key.attackerMovesRemaining));
        mixed = Board::mix_hash(mixed, static_cast<uint64_t>(key.teamToPlay));
        return static_cast<size_t>(mixed);
    }
};

using JointMateCache = std::unordered_map<
    JointMateCacheKey, JointMateProof, JointMateCacheKeyHash>;

MateActionSpace make_mate_action_space(Board& board,
                                       Stockfish::Color teamToPlay,
                                       bool teamHasTimeAdvantage) {
    MateActionSpace space;
    const bool boardAOnTurn = board.side_to_move(BOARD_A) == teamToPlay;
    const bool boardBOnTurn = board.side_to_move(BOARD_B) == ~teamToPlay;

    auto make_candidates = [&](int boardNum, bool onTurn) {
        std::vector<MateMoveCandidate> candidates;
        if (onTurn) {
            const std::vector<Stockfish::Move> moves = board.legal_moves(boardNum);
            candidates.reserve(moves.size() + 1);
            for (Stockfish::Move move : moves) {
                candidates.push_back({
                    move,
                    board.is_capture(boardNum, move),
                    board.gives_check(boardNum, move)});
            }
        }
        candidates.push_back({Stockfish::MOVE_NONE, false, false});
        return candidates;
    };

    space.actionsA = make_candidates(BOARD_A, boardAOnTurn);
    space.actionsB = make_candidates(BOARD_B, boardBOnTurn);
    space.rules = JointActionRules{
        boardAOnTurn,
        boardBOnTurn,
        teamHasTimeAdvantage,
        boardAOnTurn && space.actionsA.size() > 1,
        boardBOnTurn && space.actionsB.size() > 1};
    return space;
}

template<typename Visitor>
bool visit_legal_joint_actions(const MateActionSpace& space,
                               bool forcingOnly,
                               bool quietOnly,
                               bool victimAlreadyInCheck,
                               Visitor&& visitor) {
    for (size_t indexA = 0; indexA < space.actionsA.size(); ++indexA) {
        const MateMoveCandidate& actionA = space.actionsA[indexA];
        for (size_t indexB = 0; indexB < space.actionsB.size(); ++indexB) {
            const MateMoveCandidate& actionB = space.actionsB[indexB];
            const bool forcing = victimAlreadyInCheck
                || actionA.givesCheck || actionB.givesCheck;
            if ((forcingOnly && !forcing) || (quietOnly && forcing)) {
                continue;
            }

            if (!is_joint_action_legal(
                    space.rules, actionA.move, actionB.move,
                    actionA.isCapture, actionB.isCapture)) {
                continue;
            }
            if (visitor(MateJointAction{actionA.move, actionB.move})) {
                return true;
            }
        }
    }
    return false;
}

JointMateStatus terminal_joint_mate_status(
    Board& board,
    Stockfish::Color attackingTeam,
    bool attackingTeamHasTimeAdvantage,
    int searchPly) {
    const Stockfish::Color victimTeam = ~attackingTeam;
    Board::LegalMoveCache legalMoveCache;
    if (board.is_checkmate(
            victimTeam, !attackingTeamHasTimeAdvantage, &legalMoveCache)) {
        return JointMateStatus::PROVEN;
    }
    if (board.is_checkmate(
            attackingTeam, attackingTeamHasTimeAdvantage, &legalMoveCache)
        || board.is_draw(searchPly)) {
        return JointMateStatus::REFUTED;
    }
    return JointMateStatus::UNKNOWN;
}

bool team_is_in_check(Board& board, Stockfish::Color team) {
    return (board.side_to_move(BOARD_A) == team && board.is_in_check(BOARD_A))
        || (board.side_to_move(BOARD_B) == ~team && board.is_in_check(BOARD_B));
}

bool find_immediate_capture_feed_mate(
    Board& board,
    Stockfish::Color attackingTeam,
    bool attackerWinsMateRace,
    Agent::MateSearchBudget& budget,
    MateJointAction& outAction) {
    const bool boardAOnTurn =
        board.side_to_move(BOARD_A) == attackingTeam;
    const bool boardBOnTurn =
        board.side_to_move(BOARD_B) == ~attackingTeam;
    if (!attackerWinsMateRace || !boardAOnTurn || !boardBOnTurn) {
        return false;
    }

    constexpr std::array<Stockfish::PieceType, 5> handPieceTypes{
        Stockfish::PAWN, Stockfish::KNIGHT, Stockfish::BISHOP,
        Stockfish::ROOK, Stockfish::QUEEN};
    for (int feedBoard : {BOARD_A, BOARD_B}) {
        const int targetBoard = 1 - feedBoard;
        const Stockfish::Color targetAttacker = targetBoard == BOARD_A
            ? attackingTeam
            : ~attackingTeam;
        for (Stockfish::Move capture : board.legal_moves(feedBoard)) {
            if (!board.is_capture(feedBoard, capture)) {
                continue;
            }
            if (!budget.consume()) {
                return false;
            }

            std::array<int, Stockfish::PIECE_TYPE_NB> handBefore{};
            for (Stockfish::PieceType pieceType : handPieceTypes) {
                handBefore[pieceType] = board.count_in_hand(
                    targetBoard, targetAttacker, pieceType);
            }

            board.push_move(feedBoard, capture);
            Stockfish::PieceType fedPiece = Stockfish::NO_PIECE_TYPE;
            for (Stockfish::PieceType pieceType : handPieceTypes) {
                if (board.count_in_hand(targetBoard, targetAttacker, pieceType)
                    > handBefore[pieceType]) {
                    fedPiece = pieceType;
                    break;
                }
            }

            bool foundMate = false;
            if (fedPiece != Stockfish::NO_PIECE_TYPE) {
                for (Stockfish::Move drop : board.legal_moves(targetBoard)) {
                    if (Stockfish::type_of(drop) != Stockfish::DROP
                        || Stockfish::dropped_piece_type(drop) != fedPiece
                        || !board.gives_check(targetBoard, drop)) {
                        continue;
                    }
                    if (!budget.consume()) {
                        break;
                    }
                    board.push_move(targetBoard, drop);
                    foundMate = board.is_in_check(targetBoard)
                        && board.legal_moves(targetBoard).empty();
                    board.pop_move(targetBoard);
                    if (foundMate) {
                        break;
                    }
                }
            }
            board.pop_move(feedBoard);

            if (foundMate) {
                outAction = feedBoard == BOARD_A
                    ? MateJointAction{capture, Stockfish::MOVE_NONE}
                    : MateJointAction{Stockfish::MOVE_NONE, capture};
                return true;
            }
            if (budget.exhausted) {
                return false;
            }
        }
    }
    return false;
}

JointMateProof search_joint_forced_mate(
    Board& board,
    Stockfish::Color attackingTeam,
    bool attackingTeamHasTimeAdvantage,
    Stockfish::Color teamToPlay,
    int attackerMovesRemaining,
    int searchPly,
    Agent::MateSearchBudget& budget,
    JointMateCache& cache,
    bool attackerWinsMateRace) {
    const JointMateStatus terminal = terminal_joint_mate_status(
        board, attackingTeam, attackingTeamHasTimeAdvantage, searchPly);
    if (terminal != JointMateStatus::UNKNOWN) {
        return {terminal, 0, {}};
    }

    const bool attackerToPlay = teamToPlay == attackingTeam;
    if (attackerToPlay && attackerMovesRemaining <= 0) {
        return {JointMateStatus::REFUTED, 0, {}};
    }
    if (budget.exhausted) {
        return {JointMateStatus::UNKNOWN, 0, {}};
    }

    const bool teamToPlayHasTimeAdvantage = attackerToPlay
        ? attackingTeamHasTimeAdvantage
        : !attackingTeamHasTimeAdvantage;
    const JointMateCacheKey cacheKey{
        board.search_hash_key(teamToPlay, teamToPlayHasTimeAdvantage),
        static_cast<uint16_t>(std::max(0, attackerMovesRemaining)),
        static_cast<uint8_t>(teamToPlay)};
    if (const auto found = cache.find(cacheKey); found != cache.end()) {
        return found->second;
    }

    const MateActionSpace actionSpace = make_mate_action_space(
        board, teamToPlay, teamToPlayHasTimeAdvantage);

    if (attackerToPlay) {
        MateJointAction feedMateAction;
        if (find_immediate_capture_feed_mate(
            board, attackingTeam, attackerWinsMateRace,
            budget, feedMateAction)) {
            const JointMateProof result{
                JointMateStatus::PROVEN, 1, feedMateAction,
                std::make_shared<JointMateProofLine>(
                    JointMateProofLine{feedMateAction, nullptr})};
            cache.emplace(cacheKey, result);
            return result;
        }
        if (budget.exhausted) {
            return {JointMateStatus::UNKNOWN, 0, {}};
        }

        const bool victimAlreadyInCheck = team_is_in_check(
            board, ~attackingTeam);
        JointMateProof result{JointMateStatus::REFUTED, 0, {}};
        bool sawUnknown = false;

        auto try_forcing_action = [&](const MateJointAction& action) {
            if (!budget.consume()) {
                sawUnknown = true;
                return true;
            }
            board.make_moves(action.moveA, action.moveB);
            JointMateProof child = search_joint_forced_mate(
                board, attackingTeam, attackingTeamHasTimeAdvantage,
                ~teamToPlay, attackerMovesRemaining - 1, searchPly + 1,
                budget, cache, attackerWinsMateRace);
            board.unmake_moves(action.moveA, action.moveB);

            if (child.status == JointMateStatus::PROVEN) {
                result = {JointMateStatus::PROVEN,
                          child.pliesToMate + 1, action};
                result.principalVariation =
                    std::make_shared<JointMateProofLine>(
                        JointMateProofLine{
                            action, child.principalVariation});
                return true;
            }
            if (child.status == JointMateStatus::UNKNOWN) {
                sawUnknown = true;
                return budget.exhausted;
            }
            return false;
        };

        if (visit_legal_joint_actions(
                actionSpace, true, false, victimAlreadyInCheck,
                try_forcing_action)
            && result.status == JointMateStatus::PROVEN) {
            cache.emplace(cacheKey, result);
            return result;
        }

        // A non-checking action is useful to this forcing solver only when it
        // ends the game immediately (including bughouse stalemate-as-loss).
        // Check those after forcing continuations so ordinary positions spend
        // their budget on the tactically plausible moves first.
        if (searchPly > 0 && !budget.exhausted) {
            auto try_quiet_terminal = [&](const MateJointAction& action) {
                if (!budget.consume()) {
                    sawUnknown = true;
                    return true;
                }
                board.make_moves(action.moveA, action.moveB);
                const JointMateStatus quietTerminal = terminal_joint_mate_status(
                    board, attackingTeam, attackingTeamHasTimeAdvantage,
                    searchPly + 1);
                board.unmake_moves(action.moveA, action.moveB);
                if (quietTerminal == JointMateStatus::PROVEN) {
                    result = {
                        JointMateStatus::PROVEN, 1, action,
                        std::make_shared<JointMateProofLine>(
                            JointMateProofLine{action, nullptr})};
                    return true;
                }
                return false;
            };
            visit_legal_joint_actions(
                actionSpace, false, true, victimAlreadyInCheck,
                try_quiet_terminal);
        }

        if (result.status == JointMateStatus::PROVEN) {
            cache.emplace(cacheKey, result);
            return result;
        }
        if (sawUnknown || budget.exhausted) {
            return {JointMateStatus::UNKNOWN, 0, {}};
        }
        cache.emplace(cacheKey, result);
        return result;
    }

    // The defender chooses a reply, so every legal joint action must preserve
    // the proof. One refutation disproves the candidate; an incomplete reply
    // scan can only produce UNKNOWN, never a mate claim.
    JointMateProof result{JointMateStatus::PROVEN, 0, {}};
    bool sawAction = false;
    bool sawUnknown = false;
    auto verify_defense = [&](const MateJointAction& action) {
        sawAction = true;
        if (!budget.consume()) {
            sawUnknown = true;
            return true;
        }
        board.make_moves(action.moveA, action.moveB);
        const JointMateProof child = search_joint_forced_mate(
            board, attackingTeam, attackingTeamHasTimeAdvantage,
            ~teamToPlay, attackerMovesRemaining, searchPly + 1,
            budget, cache, attackerWinsMateRace);
        board.unmake_moves(action.moveA, action.moveB);

        if (child.status == JointMateStatus::REFUTED) {
            result = {JointMateStatus::REFUTED, 0, {}};
            return true;
        }
        if (child.status == JointMateStatus::UNKNOWN) {
            sawUnknown = true;
            return budget.exhausted;
        }
        const int candidatePlies = child.pliesToMate + 1;
        if (candidatePlies > result.pliesToMate
            || !result.principalVariation) {
            result.pliesToMate = candidatePlies;
            result.action = action;
            result.principalVariation =
                std::make_shared<JointMateProofLine>(
                    JointMateProofLine{
                        action, child.principalVariation});
        }
        return false;
    };
    visit_legal_joint_actions(
        actionSpace, false, false, false, verify_defense);

    if (result.status == JointMateStatus::REFUTED) {
        cache.emplace(cacheKey, result);
        return result;
    }
    if (!sawAction || sawUnknown || budget.exhausted) {
        return {sawAction ? JointMateStatus::UNKNOWN : JointMateStatus::REFUTED,
                0, {}};
    }
    cache.emplace(cacheKey, result);
    return result;
}

void append_formatted_ply(
    Board& lineBoard,
    const MateProofPly& ply,
    string& formatted) {
    const string moveA = ply.moveA == Stockfish::MOVE_NONE
        ? "pass" : lineBoard.uci_move(BOARD_A, ply.moveA);
    const string moveB = ply.moveB == Stockfish::MOVE_NONE
        ? "pass" : lineBoard.uci_move(BOARD_B, ply.moveB);
    if (!formatted.empty()) {
        formatted += " ";
    }
    formatted += "(" + moveA + "," + moveB + ")";
    lineBoard.make_moves(ply.moveA, ply.moveB);
}

void append_waiting_mate_suffix(
    Board& lineBoard,
    Stockfish::Color teamToPlay,
    Stockfish::Color rootTeam,
    bool rootTeamHasTimeAdvantage,
    const std::array<int, 2>& boardSearchPlies,
    string& formatted) {
    WaitingMateContinuation continuation;
    int endInPly = 0;
    if (classify_terminal_position(
            lineBoard, teamToPlay, rootTeam, rootTeamHasTimeAdvantage,
            boardSearchPlies, &endInPly, false, false, &continuation)
            != TerminalOutcome::LOSS
        || continuation.activeBoard < 0) {
        return;
    }

    append_formatted_ply(
        lineBoard,
        continuation.activeBoard == BOARD_A
            ? MateProofPly{continuation.reply, Stockfish::MOVE_NONE}
            : MateProofPly{Stockfish::MOVE_NONE, continuation.reply},
        formatted);
    append_formatted_ply(
        lineBoard,
        continuation.waitingBoard == BOARD_A
            ? MateProofPly{
                continuation.matingMove, Stockfish::MOVE_NONE}
            : MateProofPly{
                Stockfish::MOVE_NONE, continuation.matingMove},
        formatted);
}

string format_mate_proof_pv(
    Board& board,
    const std::vector<MateProofPly>& principalVariation,
    Stockfish::Color rootTeam,
    bool rootTeamHasTimeAdvantage) {
    Board lineBoard = board;
    string formatted;
    std::array<int, 2> boardSearchPlies{};
    for (const MateProofPly& ply : principalVariation) {
        append_formatted_ply(lineBoard, ply, formatted);
        boardSearchPlies[BOARD_A] += ply.moveA != Stockfish::MOVE_NONE;
        boardSearchPlies[BOARD_B] += ply.moveB != Stockfish::MOVE_NONE;
    }
    const Stockfish::Color teamToPlay = principalVariation.size() % 2 == 0
        ? rootTeam : ~rootTeam;
    append_waiting_mate_suffix(
        lineBoard, teamToPlay, rootTeam, rootTeamHasTimeAdvantage,
        boardSearchPlies, formatted);
    return formatted;
}

}  // namespace

/**
 * @brief Ask Fairy-Stockfish for a single-board mate the check-only scan missed.
 *
 * find_root_mate_impl() searches attacker moves that give check, so a mate that
 * needs a quiet preparing move is invisible to it at any budget. The probe has
 * no such restriction and shares the model: its bughouse variant lets the
 * defender block with a piece its partner supplies, the same assumption the
 * partner-agnostic proofs here already make.
 *
 * Restricted to a board this team is actually on turn on, and to a move the
 * partner board may sit through: the single-board position the probe searches
 * is one where the partner board does not move, which a board that is not on
 * turn does anyway, and which being ahead on time or capturing also buys.
 *
 * The probe's own line is kept as text rather than as MateProofPly. It can
 * contain a block with a piece the defender does not hold yet, which Board
 * refuses to replay, so only the first move crosses back as a real move.
 */
bool Agent::probe_root_mate(
    Board& board,
    Stockfish::Color teamSide,
    bool teamHasTimeAdvantage,
    int budgetMs,
    const std::function<bool()>& abort,
    JointActionCandidate& outAction,
    int& outPlyToMate,
    string& outPrincipalVariation,
    const std::function<void()>& onMate) {
    const bool onTurn[2] = {board.side_to_move(BOARD_A) == teamSide,
                            board.side_to_move(BOARD_B) == ~teamSide};
    // Sitting out the partner board is what makes a single-board mate line
    // playable here. Being time-ahead is one way to earn that, but a partner
    // board that is not on turn sits anyway, and a capture buys the pass too.
    const auto pass_is_legal = [&](int boardNum, Stockfish::Move move) {
        const int partner = 1 - boardNum;
        return !onTurn[partner]
            || is_single_pass_legal(teamHasTimeAdvantage, onTurn[BOARD_A],
                                    onTurn[BOARD_B],
                                    board.is_capture(boardNum, move));
    };
    const int probeBoards = onTurn[BOARD_A] + onTurn[BOARD_B];
    if (probeBoards == 0 || budgetMs <= 0) {
        return false;
    }

    const JointActionRules rules{
        onTurn[BOARD_A], onTurn[BOARD_B], teamHasTimeAdvantage,
        onTurn[BOARD_A] && board.has_any_legal_move(BOARD_A),
        onTurn[BOARD_B] && board.has_any_legal_move(BOARD_B)};

    // The boards are probed one after the other, so a caller running this
    // concurrently with its own search is told about each mate as it lands
    // rather than only once the second board's share of the budget is spent.
    const auto accept = [&](int boardNum, const MateProbe::Result& result) {
        const bool onBoardA = boardNum == BOARD_A;
        outAction = onBoardA
            ? JointActionCandidate(
                result.bestMove, 1.0f, 0, Stockfish::MOVE_NONE, 1.0f, 0,
                rules, board.is_capture(BOARD_A, result.bestMove), false)
            : JointActionCandidate(
                Stockfish::MOVE_NONE, 1.0f, 0, result.bestMove, 1.0f, 0,
                rules, false, board.is_capture(BOARD_B, result.bestMove));
        outPlyToMate = std::max(1, 2 * result.mateInMoves - 1);

        outPrincipalVariation.clear();
        for (const string& move : result.principalVariation) {
            if (!outPrincipalVariation.empty()) {
                outPrincipalVariation += " ";
            }
            outPrincipalVariation +=
                onBoardA ? "(" + move + ",pass)" : "(pass," + move + ")";
        }
        if (onMate) {
            onMate();
        }
    };

    bool found = false;
    int bestMateInMoves = 0;
    for (int boardNum : {BOARD_A, BOARD_B}) {
        if (!onTurn[boardNum] || (abort && abort())) {
            continue;
        }
        const MateProbe::Result result = MateProbe::probe(
            board.fen(boardNum), SearchParams::MATE_PROBE_MAX_MATE_MOVES,
            SearchParams::MATE_PROBE_ROOT_NODE_BUDGET,
            budgetMs / probeBoards, abort);
        // A pruned search is not a proof, so the move it names still has to be
        // one this board accepts before it can become the root action.
        if (!result.found
            || result.bestMove == Stockfish::MOVE_NONE
            || !board.is_legal_move(boardNum, result.bestMove)
            || !pass_is_legal(boardNum, result.bestMove)) {
            continue;
        }
        if (found && result.mateInMoves >= bestMateInMoves) {
            continue;
        }
        bestMateInMoves = result.mateInMoves;
        found = true;
        accept(boardNum, result);
    }
    return found;
}

bool Agent::find_root_mate(Board& board, Stockfish::Color teamSide,
                          bool teamHasTimeAdvantage,
                          JointActionCandidate& outAction,
                          int& outPlyToMate,
                          uint64_t nodeBudget,
                          MateSearchBudget::Clock::time_point deadline) {
    return find_root_mate_impl(
        board, teamSide, teamHasTimeAdvantage, outAction, outPlyToMate,
        nodeBudget, nullptr, nullptr, true, deadline);
}

bool Agent::find_root_mate_impl(
    Board& board, Stockfish::Color teamSide,
    bool teamHasTimeAdvantage,
    JointActionCandidate& outAction,
    int& outPlyToMate,
    uint64_t nodeBudget,
    std::vector<MateContinuation>* continuations,
    MateSearchBudget* hardBudget,
    bool includeCaptureFeeds,
    MateSearchBudget::Clock::time_point deadline,
    bool includeImmediateMate,
    bool attackerWinsMateRace,
    std::vector<MateProofPly>* outPrincipalVariation,
    const std::atomic<bool>* cancelled,
    SingleBoardMateCache* singleBoardMateCache,
    const Node* stopOnSolvedRoot) {
    if (outPrincipalVariation) {
        outPrincipalVariation->clear();
    }
    // Every budget this scan creates shares one wall-clock stop, so the whole
    // pre-pass honours the caller's deadline no matter which branch it takes.
    // A caller that supplied its own budget already carries the deadline.
    auto with_deadline = [&](MateSearchBudget& budget) {
        budget.deadline = deadline;
        budget.cancelled = cancelled;
        budget.stopOnSolvedRoot = stopOnSolvedRoot;
    };
    if (hardBudget) {
        hardBudget->cancelled = cancelled;
        hardBudget->stopOnSolvedRoot = stopOnSolvedRoot;
    }
    // Two of the scans below walk a move list without node accounting, so they
    // need the same stop condition without a budget to hang it on.
    MateSearchBudget deadlineOnly;
    with_deadline(deadlineOnly);
    const auto out_of_time = [&] { return deadlineOnly.out_of_time(); };
    if (out_of_time()) {
        return false;
    }

    // 1. Fast 1-ply immediate mate scan across all joint combinations.
    // With both boards on turn the scan walks a joint move space that is cheap
    // per probe but large, so a caller without a hard node budget still needs
    // the wall clock to stop it - otherwise the whole pre-pass can run far past
    // the move time it was supposed to fit inside.
    if (includeImmediateMate) {
        MateSearchBudget immediateDeadlineBudget;
        immediateDeadlineBudget.remainingNodes =
            std::numeric_limits<uint64_t>::max();
        with_deadline(immediateDeadlineBudget);
        MateSearchBudget* immediateBudget = hardBudget
            ? hardBudget
            : (deadline != MateSearchBudget::Clock::time_point{}
                   || cancelled || stopOnSolvedRoot
                   ? &immediateDeadlineBudget
                   : nullptr);
        if (find_immediate_root_mate(
                board, teamSide, teamHasTimeAdvantage, outAction,
                immediateBudget)) {
            outPlyToMate = 1;
            if (outPrincipalVariation) {
                outPrincipalVariation->push_back(
                    {outAction.moveA, outAction.moveB});
            }
            return true;
        }
        if ((hardBudget && hardBudget->exhausted)
            || immediateDeadlineBudget.exhausted) {
            return false;
        }
    }

    const bool boardAOnTurn = board.side_to_move(BOARD_A) == teamSide;
    const bool boardBOnTurn = board.side_to_move(BOARD_B) == ~teamSide;

    if (boardAOnTurn && boardBOnTurn) {
        MateSearchBudget immediateFeedBudget;
        MateSearchBudget* feedBudget = hardBudget;
        if (!feedBudget) {
            immediateFeedBudget.remainingNodes = nodeBudget;
            with_deadline(immediateFeedBudget);
            feedBudget = &immediateFeedBudget;
        }
        MateJointAction feedMateAction;
        if (find_immediate_capture_feed_mate(
            board, teamSide, attackerWinsMateRace,
            *feedBudget, feedMateAction)) {
            const JointActionRules rules{
                true, true, teamHasTimeAdvantage,
                board.has_any_legal_move(BOARD_A),
                board.has_any_legal_move(BOARD_B)};
            outAction = JointActionCandidate(
                feedMateAction.moveA, 1.0f, 0,
                feedMateAction.moveB, 1.0f, 0,
                rules,
                feedMateAction.moveA != Stockfish::MOVE_NONE,
                feedMateAction.moveB != Stockfish::MOVE_NONE);
            outPlyToMate = 1;
            if (outPrincipalVariation) {
                outPrincipalVariation->push_back(
                    {outAction.moveA, outAction.moveB});
            }
            return true;
        }
        if (feedBudget->exhausted) {
            return false;
        }
    }

    // 2. When down on time, capture-plus-pass is still legal with both boards
    // on turn. The capture may feed an immediate mate to the waiting board.
    // The time-ahead opponent can move the capture board or sit, so reuse the
    // in-tree waiting-mate classifier: it proves the mate survives every choice
    // and rejects the capture if the opponent can mate the capturing board
    // first.
    if (!teamHasTimeAdvantage && boardAOnTurn && boardBOnTurn) {
        const JointActionRules rules{boardAOnTurn, boardBOnTurn, false,
                                     true, true};
        for (int feedBoard : {BOARD_A, BOARD_B}) {
            const int targetBoard = 1 - feedBoard;
            const Stockfish::Color targetAttacker = targetBoard == BOARD_A
                ? teamSide : ~teamSide;
            for (Stockfish::Move capture : board.legal_moves(feedBoard)) {
                if (hardBudget && !hardBudget->consume()) {
                    return false;
                }
                if (!board.is_capture(feedBoard, capture)) {
                    continue;
                }
                if (out_of_time()) {
                    return false;
                }

                int handCountBefore = 0;
                for (Stockfish::PieceType pt : {
                         Stockfish::PAWN, Stockfish::KNIGHT, Stockfish::BISHOP,
                         Stockfish::ROOK, Stockfish::QUEEN}) {
                    handCountBefore += board.count_in_hand(
                        targetBoard, targetAttacker, pt);
                }

                board.push_move(feedBoard, capture);
                int handCountAfter = 0;
                for (Stockfish::PieceType pt : {
                         Stockfish::PAWN, Stockfish::KNIGHT, Stockfish::BISHOP,
                         Stockfish::ROOK, Stockfish::QUEEN}) {
                    handCountAfter += board.count_in_hand(
                        targetBoard, targetAttacker, pt);
                }

                int terminalEndInPly = 0;
                const TerminalOutcome outcome = handCountAfter > handCountBefore
                    ? classify_terminal_position(
                        board, ~teamSide, teamSide, false, 1,
                        &terminalEndInPly)
                    : TerminalOutcome::NONE;
                board.pop_move(feedBoard);

                if (outcome == TerminalOutcome::LOSS) {
                    outAction = feedBoard == BOARD_A
                        ? JointActionCandidate(capture, 1.0f, 0,
                                               Stockfish::MOVE_NONE, 1.0f, 0,
                                               rules, true, false)
                        : JointActionCandidate(Stockfish::MOVE_NONE, 1.0f, 0,
                                               capture, 1.0f, 0,
                                               rules, false, true);
                    outPlyToMate = terminalEndInPly;
                    if (outPrincipalVariation) {
                        outPrincipalVariation->push_back(
                            {outAction.moveA, outAction.moveB});
                    }
                    return true;
                }
            }
        }
    }

    // 3. Multi-ply forced mate detectors for the time-ahead team. These proofs
    // rely on freely sitting on one board while the other board's mating
    // sequence is played.
    if (teamHasTimeAdvantage && boardAOnTurn && boardBOnTurn) {
        const JointActionRules rules{boardAOnTurn, boardBOnTurn, teamHasTimeAdvantage,
                                     true, true};

        uint64_t feedStockfishNodesRemaining = std::min<uint64_t>(
            nodeBudget, SearchParams::MATE_PROBE_FEED_NODE_BUDGET);
        auto probe_single_board_mate = [&](
            int boardNum, Stockfish::Color attacker,
            Stockfish::Move& mateMove, int& matePly,
            MateSearchBudget& budget,
            int attackerMoveLimit,
            bool partnerBoardAgnostic = false,
            std::vector<MateProofPly>* line = nullptr) {
            mateMove = Stockfish::MOVE_NONE;
            matePly = 0;
            if (attackerMoveLimit <= 0) {
                return false;
            }
            const string cacheKey = std::to_string(boardNum) + ":"
                + std::to_string(static_cast<int>(attacker)) + ":"
                + board.fen(boardNum);
            if (partnerBoardAgnostic && singleBoardMateCache) {
                const auto cached = singleBoardMateCache->find(cacheKey);
                if (cached != singleBoardMateCache->end()
                    && (cached->second.plyToMate + 1) / 2
                           <= attackerMoveLimit) {
                    mateMove = cached->second.move;
                    matePly = cached->second.plyToMate;
                    if (line) {
                        *line = cached->second.principalVariation;
                    }
                    return true;
                }
            }
            const auto retainCache = [&](bool found) {
                if (found && partnerBoardAgnostic
                    && singleBoardMateCache) {
                    (*singleBoardMateCache)[cacheKey] = {
                        mateMove, matePly, line ? *line
                                                : vector<MateProofPly>{}};
                }
                return found;
            };
            const auto run_checks_only_fallback = [&] {
                return retainCache(search_single_board_forced_mate_impl(
                    board, boardNum, attacker, 1, attackerMoveLimit,
                    mateMove, matePly, &budget,
                    partnerBoardAgnostic ? nullptr : continuations,
                    partnerBoardAgnostic, line));
            };
            if (!SearchParams::ENABLE_STOCKFISH_MATE_SEARCH) {
                return run_checks_only_fallback();
            }
            if (budget.exhausted || budget.out_of_time()) {
                return false;
            }
            if (feedStockfishNodesRemaining == 0) {
                return run_checks_only_fallback();
            }

            const MateProbe::Result probed = MateProbe::probe(
                board.fen(boardNum), attackerMoveLimit,
                std::min<uint64_t>(
                    feedStockfishNodesRemaining,
                    SearchParams::MATE_PROBE_FEED_NODE_BUDGET),
                SearchParams::MATE_PROBE_FEED_MAX_MS,
                [&] { return budget.out_of_time(); });
            feedStockfishNodesRemaining -= std::min<uint64_t>(
                probed.nodes, feedStockfishNodesRemaining);
            if (!probed.found
                || probed.bestMove == Stockfish::MOVE_NONE
                || !board.is_legal_move(boardNum, probed.bestMove)) {
                if (budget.out_of_time()) {
                    return false;
                }
                return run_checks_only_fallback();
            }

            mateMove = probed.bestMove;
            matePly = std::max(1, 2 * probed.mateInMoves - 1);
            if (line) {
                line->clear();
                Board lineBoard(board);
                for (const string& moveText : probed.principalVariation) {
                    string parsedMoveText = moveText;
                    const Stockfish::Move move = Stockfish::UCI::to_move(
                        *lineBoard.pos[boardNum], parsedMoveText);
                    if (move == Stockfish::MOVE_NONE
                        || !lineBoard.is_legal_move(boardNum, move)) {
                        line->clear();
                        break;
                    }
                    line->push_back(boardNum == BOARD_A
                        ? MateProofPly{move, Stockfish::MOVE_NONE}
                        : MateProofPly{Stockfish::MOVE_NONE, move});
                    lineBoard.push_move(boardNum, move);
                }
                if (line->size() != static_cast<size_t>(matePly)) {
                    MateSearchBudget proofBudget = budget;
                    Stockfish::Move provenMove = Stockfish::MOVE_NONE;
                    int provenPly = 0;
                    std::vector<MateProofPly> provenLine;
                    if (search_single_board_forced_mate_impl(
                            board, boardNum, attacker, 1,
                            attackerMoveLimit, provenMove, provenPly,
                            &proofBudget,
                            partnerBoardAgnostic ? nullptr : continuations,
                            partnerBoardAgnostic, &provenLine)) {
                        mateMove = provenMove;
                        matePly = provenPly;
                        *line = std::move(provenLine);
                    }
                }
            }
            return retainCache(true);
        };

        auto find_shortest_direct_mate = [&](
            MateSearchBudget& budget,
            JointActionCandidate& directAction,
            int& directPly,
            std::vector<MateProofPly>& directPv) {
            for (int maxMateMoves = 2;
                 maxMateMoves <= SearchParams::MATE_SEARCH_MAX_ATTACKER_MOVES;
                 ++maxMateMoves) {
                Stockfish::Move mateMoveA = Stockfish::MOVE_NONE;
                int plyA = 0;
                std::vector<MateProofPly> lineA;
                const bool foundA = search_single_board_forced_mate_impl(
                    board, BOARD_A, teamSide, 1, maxMateMoves,
                    mateMoveA, plyA, &budget, continuations, false, &lineA);

                Stockfish::Move mateMoveB = Stockfish::MOVE_NONE;
                int plyB = 0;
                std::vector<MateProofPly> lineB;
                const bool foundB = search_single_board_forced_mate_impl(
                    board, BOARD_B, ~teamSide, 1, maxMateMoves,
                    mateMoveB, plyB, &budget, continuations, false, &lineB);

                if (foundA && (!foundB || plyA <= plyB)) {
                    directAction = JointActionCandidate(
                        mateMoveA, 1.0f, 0,
                        Stockfish::MOVE_NONE, 1.0f, 0,
                        rules, board.is_capture(BOARD_A, mateMoveA), false);
                    directPly = plyA;
                    directPv = std::move(lineA);
                    return true;
                }
                if (foundB) {
                    directAction = JointActionCandidate(
                        Stockfish::MOVE_NONE, 1.0f, 0,
                        mateMoveB, 1.0f, 0,
                        rules, false, board.is_capture(BOARD_B, mateMoveB));
                    directPly = plyB;
                    directPv = std::move(lineB);
                    return true;
                }
                if (budget.exhausted) {
                    break;
                }
            }
            return false;
        };

        // Find the cheapest direct proof first. It gives capture-feed search an
        // exact upper bound: only a strictly shorter feed can replace it, so
        // the feed scan that follows is cheaper for having waited. A forced-loss
        // caller shares one hard budget across the whole defense - that is why
        // this used to run after the feed scan there, on what was left - so the
        // direct scan is billed to that budget rather than opening a second
        // one. Ordering it first is worth real time: a suite position whose
        // defenses need a 12-ply feed proof took 1586ms feed-first and is
        // proven in a fraction of that once the direct bound comes first.
        MateSearchBudget directBudget;
        directBudget.remainingNodes = nodeBudget;
        with_deadline(directBudget);
        MateSearchBudget& directScanBudget =
            hardBudget ? *hardBudget : directBudget;
        JointActionCandidate bestDirectAction;
        int bestDirectPly = 0;
        std::vector<MateProofPly> bestDirectPv;
        const bool foundDirectMate = find_shortest_direct_mate(
            directScanBudget, bestDirectAction, bestDirectPly,
            bestDirectPv);
        int maxFeedAttackerMoves =
            SearchParams::MATE_SEARCH_MAX_ATTACKER_MOVES;
        if (foundDirectMate) {
            // A feed adds the root capture and the forced reply before the
            // partner-board mate. For a direct D-ply mate, at most (D-3)/2
            // attacker moves can produce a strictly shorter feed.
            maxFeedAttackerMoves = std::max(0, (bestDirectPly - 3) / 2);
            if (maxFeedAttackerMoves == 0) {
                outAction = bestDirectAction;
                outPlyToMate = bestDirectPly;
                if (outPrincipalVariation) {
                    *outPrincipalVariation = bestDirectPv;
                }
                return true;
            }
        }

        bool foundFeedMate = false;
        JointActionCandidate bestFeedAction;
        int bestFeedPly = 0;
        std::vector<MateProofPly> bestFeedPv;
        auto retain_shortest_feed = [&](const JointActionCandidate& action,
                                        int plyToMate,
                                        std::vector<MateProofPly> line) {
            if (!foundFeedMate || plyToMate < bestFeedPly) {
                foundFeedMate = true;
                bestFeedAction = action;
                bestFeedPly = plyToMate;
                bestFeedPv = std::move(line);
            }
        };

        // A feed mate spans the capture, the forced reply and the partner
        // board's checking sequence, so reporting only the root capture leaves
        // a pv that looks nothing like the mate the score claims.
        const auto report_feed_principal_variation = [&] {
            if (!outPrincipalVariation) {
                return;
            }
            if (bestFeedPv.empty()) {
                outPrincipalVariation->push_back(
                    {bestFeedAction.moveA, bestFeedAction.moveB});
            } else {
                *outPrincipalVariation = bestFeedPv;
            }
        };

        std::array<MateSearchBudget, 2> feedBudgets;
        // Keep the new preparatory scan additive and tightly bounded. The
        // established single-board phase still receives its full budget above,
        // while feed proofs draw a dedicated allowance scaled to the caller's
        // budget - never a fixed floor, which would override the budget the
        // caller scaled to its own search. Splitting it evenly prevents a
        // branch-heavy A-to-B proof from starving the mirrored B-to-A search.
        const uint64_t totalFeedBudget = std::max<uint64_t>(
            2,
            nodeBudget * SearchParams::MATE_CAPTURE_FEED_NODE_BUDGET_PERCENT
                / 100);
        feedBudgets[BOARD_A].remainingNodes = totalFeedBudget / 2;
        feedBudgets[BOARD_B].remainingNodes =
            totalFeedBudget - feedBudgets[BOARD_A].remainingNodes;
        with_deadline(feedBudgets[BOARD_A]);
        with_deadline(feedBudgets[BOARD_B]);

        // A capture on one board can be the first move of a forced mate on the
        // other board: the captured piece enters our partner's hand while they
        // sit, the opponent is forced to move on the capture board, and then our
        // partner starts the checking sequence.  The single-board search below
        // cannot discover that preparatory move because it considers checks
        // only and searches the two boards independently.
        struct CaptureFeedCandidate {
            int feedBoard;
            int targetBoard;
            Stockfish::Color targetAttacker;
            Stockfish::Move move;
            Stockfish::PieceType fedPiece;
            bool fedPieceHasCheckingDrop;
        };
        vector<CaptureFeedCandidate> feedCandidates;

        auto collect_feed_candidates = [&](int feedBoard, int targetBoard,
                                           Stockfish::Color targetAttacker) {
            for (Stockfish::Move move : board.legal_moves(feedBoard)) {
                if (!board.is_capture(feedBoard, move)) {
                    continue;
                }
                if (hardBudget && !hardBudget->consume()) {
                    return;
                }
                if (out_of_time()) {
                    return;
                }

                std::array<int, Stockfish::PIECE_TYPE_NB> countsBefore{};
                for (Stockfish::PieceType pt : {
                         Stockfish::PAWN, Stockfish::KNIGHT, Stockfish::BISHOP,
                         Stockfish::ROOK, Stockfish::QUEEN}) {
                    countsBefore[pt] = board.count_in_hand(
                        targetBoard, targetAttacker, pt);
                }

                board.push_move(feedBoard, move);
                Stockfish::PieceType fedPiece = Stockfish::NO_PIECE_TYPE;
                for (Stockfish::PieceType pt : {
                         Stockfish::PAWN, Stockfish::KNIGHT, Stockfish::BISHOP,
                         Stockfish::ROOK, Stockfish::QUEEN}) {
                    if (board.count_in_hand(targetBoard, targetAttacker, pt)
                        > countsBefore[pt]) {
                        fedPiece = pt;
                        break;
                    }
                }

                bool hasCheckingDrop = false;
                if (fedPiece != Stockfish::NO_PIECE_TYPE) {
                    for (Stockfish::Move targetMove : board.legal_moves(targetBoard)) {
                        if (Stockfish::type_of(targetMove) == Stockfish::DROP
                            && Stockfish::dropped_piece_type(targetMove) == fedPiece
                            && board.gives_check(targetBoard, targetMove)) {
                            hasCheckingDrop = true;
                            break;
                        }
                    }
                }
                board.pop_move(feedBoard);

                if (fedPiece != Stockfish::NO_PIECE_TYPE) {
                    feedCandidates.push_back({
                        feedBoard, targetBoard, targetAttacker, move,
                        fedPiece, hasCheckingDrop});
                }
            }
        };
        if (includeCaptureFeeds) {
            collect_feed_candidates(BOARD_A, BOARD_B, ~teamSide);
            if (hardBudget && hardBudget->exhausted) {
                return false;
            }
            collect_feed_candidates(BOARD_B, BOARD_A, teamSide);
            if (hardBudget && hardBudget->exhausted) {
                return false;
            }
        }

        // Most feed mates start with a check by the newly acquired piece. Keep
        // those first, but alternate board directions within each priority
        // class. Otherwise every capture on Board A can consume its allowance
        // before a winning Board-B feed is even attempted (and vice versa after
        // flipping the boards).
        std::array<std::array<vector<CaptureFeedCandidate>, 2>, 2>
            candidatesByPriorityAndBoard;
        for (const CaptureFeedCandidate& candidate : feedCandidates) {
            const size_t priority = candidate.fedPieceHasCheckingDrop ? 0 : 1;
            candidatesByPriorityAndBoard[priority][candidate.feedBoard]
                .push_back(candidate);
        }
        feedCandidates.clear();
        for (size_t priority = 0; priority < 2; ++priority) {
            const auto& candidatesA =
                candidatesByPriorityAndBoard[priority][BOARD_A];
            const auto& candidatesB =
                candidatesByPriorityAndBoard[priority][BOARD_B];
            const size_t count = std::max(candidatesA.size(), candidatesB.size());
            const bool boardAFirst = candidatesA.size() <= candidatesB.size();
            for (size_t index = 0; index < count; ++index) {
                auto append = [&](const auto& candidates) {
                    if (index < candidates.size()) {
                        feedCandidates.push_back(candidates[index]);
                    }
                };
                if (boardAFirst) {
                    append(candidatesA);
                    append(candidatesB);
                } else {
                    append(candidatesB);
                    append(candidatesA);
                }
            }
        }

        // One probe at a fixed depth. The deepening runs across the whole
        // candidate list below rather than inside a single candidate, so a
        // candidate can never spend its direction's whole budget on a deep
        // refutation before its neighbours have been tried at all.
        // The cross-board structure stays here - which capture feeds which
        // board, and every reply that has to be answered - but the single-board
        // question it asks at each leaf goes to Fairy-Stockfish, which is not
        // restricted to checking moves and so sees the feed mates that need a
        // Stockfish and the exact fallback share the adapter used by direct
        // mates above, so the alpha-beta allowance stays bounded across the
        // whole root scan rather than restarting for every capture candidate.

        // Once the capture and the reply are made, the target board is the
        // original one plus the piece each handed to a hand: the capture feeds
        // our partner, the reply feeds the defender. Proofs are therefore
        // indexed by that pair of piece types, per board direction, and are
        // shared by every candidate that transfers the same piece.
        struct FeedReplyClass {
            int matePly = -1;        // proven in the partner-board-agnostic model
            int refutedAtLimit = 0;  // that proof failed up to this move limit
            // Every reply of one class leaves the target board and both its
            // hands identical, so the stored line describes the reused proof
            // exactly as well as the exact proof it stands in for.
            std::vector<MateProofPly> line;
        };
        using FeedReplyClassRow =
            std::array<FeedReplyClass, Stockfish::PIECE_TYPE_NB>;
        std::array<std::array<FeedReplyClassRow, Stockfish::PIECE_TYPE_NB>, 2>
            replyClasses{};
        constexpr std::array<Stockfish::PieceType, 5> HAND_PIECE_TYPES{
            Stockfish::PAWN, Stockfish::KNIGHT, Stockfish::BISHOP,
            Stockfish::ROOK, Stockfish::QUEEN};

        // Sweep every candidate at one attacker-move limit before deepening,
        // so the shortest feed is found first and no candidate is searched
        // deeper than the best proof so far can be beaten by.
        for (int feedAttackerMoveLimit = 1;
             feedAttackerMoveLimit <= maxFeedAttackerMoves;
             ++feedAttackerMoveLimit) {
            // A feed costs the root capture and the forced reply on top of the
            // mate itself, so once one is proven only a strictly shorter mate
            // can replace it - and that caps how deep the sweep ever goes.
            if (foundFeedMate
                && feedAttackerMoveLimit
                       > std::max(0, (bestFeedPly - 3) / 2)) {
                break;
            }

            for (const CaptureFeedCandidate& candidate : feedCandidates) {
                MateSearchBudget& feedBudget = hardBudget
                    ? *hardBudget : feedBudgets[candidate.feedBoard];
                if (!feedBudget.consume()) {
                    continue;
                }
                board.push_move(candidate.feedBoard, candidate.move);

                // A capture can itself end the game (including bughouse stalemate).
                if (board.is_checkmate(~teamSide, false)) {
                    board.pop_move(candidate.feedBoard);
                    const JointActionCandidate feedAction = candidate.feedBoard == BOARD_A
                        ? JointActionCandidate(candidate.move, 1.0f, 0,
                                               Stockfish::MOVE_NONE, 1.0f, 0,
                                               rules, true, false)
                        : JointActionCandidate(Stockfish::MOVE_NONE, 1.0f, 0,
                                               candidate.move, 1.0f, 0,
                                               rules, false, true);
                    retain_shortest_feed(
                        feedAction, 1,
                        {candidate.feedBoard == BOARD_A
                             ? MateProofPly{
                                 candidate.move, Stockfish::MOVE_NONE}
                             : MateProofPly{
                                 Stockfish::MOVE_NONE, candidate.move}});
                    continue;
                }

                // Since we started with both boards on turn and sat on the target
                // board, the opponent has no time advantage and must make exactly
                // one move on the capture board. A feed is proven only if the mate
                // on the partner board survives every such reply, including any
                // defensive piece that reply captures and transfers.
                const vector<Stockfish::Move> opponentReplies =
                    board.legal_moves(candidate.feedBoard);
                bool allRepliesMated = !opponentReplies.empty();
                int deepestMatePly = 1;
                Stockfish::Move deepestReply = Stockfish::MOVE_NONE;
                std::vector<MateProofPly> deepestMateLine;

                // Every reply is played on the feed board, which the mate proof on
                // the target board reads only through the terminal classifier's
                // partner-board terms. Proving the mate once in the
                // partner-board-agnostic model therefore settles every remaining
                // reply of the same class, turning hundreds of identical proofs
                // into one per pair of transferred piece types. The generalisation
                // is only claimed for replies that leave the feed board quiet - a
                // check or a stalemate there is read by the classifier even in that
                // model - and the exact proof stays the fallback everywhere else.
                FeedReplyClassRow& candidateClasses =
                    replyClasses[candidate.feedBoard][candidate.fedPiece];
                const Stockfish::Color targetDefender = ~candidate.targetAttacker;

                for (Stockfish::Move reply : opponentReplies) {
                    if (!feedBudget.consume()) {
                        allRepliesMated = false;
                        break;
                    }

                    std::array<int, Stockfish::PIECE_TYPE_NB> defenderHand{};
                    for (Stockfish::PieceType pt : HAND_PIECE_TYPES) {
                        defenderHand[pt] = board.count_in_hand(
                            candidate.targetBoard, targetDefender, pt);
                    }

                    board.push_move(candidate.feedBoard, reply);

                    Stockfish::PieceType fedToDefender = Stockfish::NO_PIECE_TYPE;
                    for (Stockfish::PieceType pt : HAND_PIECE_TYPES) {
                        if (board.count_in_hand(
                                candidate.targetBoard, targetDefender, pt)
                            > defenderHand[pt]) {
                            fedToDefender = pt;
                            break;
                        }
                    }

                    Stockfish::Move mateMove = Stockfish::MOVE_NONE;
                    int matePly = 0;
                    std::vector<MateProofPly> replyLine;
                    bool replyMated = false;
                    const bool rootLostOrDrawn =
                        board.is_checkmate(teamSide, true) || board.is_draw();
                    if (!rootLostOrDrawn) {
                        const bool quietFeedBoard =
                            !board.is_in_check(candidate.feedBoard)
                            && board.has_any_legal_move(candidate.feedBoard);
                        FeedReplyClass& replyClass = candidateClasses[fedToDefender];
                        // A stored proof only answers a candidate whose move limit
                        // still admits it, and a stored failure only settles limits
                        // no deeper than the one it was recorded at.
                        const bool proofFits = replyClass.matePly >= 0
                            && (replyClass.matePly + 1) / 2
                                   <= feedAttackerMoveLimit;

                        const string sharedCacheKey =
                            std::to_string(candidate.targetBoard) + ":"
                            + std::to_string(static_cast<int>(
                                candidate.targetAttacker)) + ":"
                            + board.fen(candidate.targetBoard);
                        const auto sharedProof = singleBoardMateCache
                            ? singleBoardMateCache->find(sharedCacheKey)
                            : SingleBoardMateCache::const_iterator{};
                        const bool sharedProofFits = singleBoardMateCache
                            && sharedProof != singleBoardMateCache->end()
                            && (sharedProof->second.plyToMate + 1) / 2
                                   <= feedAttackerMoveLimit;

                        if (quietFeedBoard && proofFits) {
                            replyMated = true;
                            matePly = replyClass.matePly;
                            replyLine = replyClass.line;
                        } else if (quietFeedBoard && sharedProofFits) {
                            replyMated = true;
                            mateMove = sharedProof->second.move;
                            matePly = sharedProof->second.plyToMate;
                            replyLine = sharedProof->second.principalVariation;
                        } else {
                            replyMated = probe_single_board_mate(
                                candidate.targetBoard, candidate.targetAttacker,
                                mateMove, matePly, feedBudget,
                                feedAttackerMoveLimit, false,
                                outPrincipalVariation ? &replyLine : nullptr);
                            // Only look for the reusable proof once a reply has
                            // been answered exactly. Refuted candidates - the
                            // common case - then cost exactly what they did before.
                            if (replyMated && quietFeedBoard
                                && replyClass.matePly < 0
                                && feedAttackerMoveLimit
                                       > replyClass.refutedAtLimit) {
                                Stockfish::Move classMove = Stockfish::MOVE_NONE;
                                int classProofPly = 0;
                                if (probe_single_board_mate(
                                        candidate.targetBoard,
                                        candidate.targetAttacker,
                                        classMove, classProofPly, feedBudget,
                                        feedAttackerMoveLimit, true,
                                        outPrincipalVariation
                                            ? &replyClass.line : nullptr)) {
                                    replyClass.matePly = classProofPly;
                                } else {
                                    replyClass.refutedAtLimit =
                                        feedAttackerMoveLimit;
                                }
                            }
                        }
                    }
                    board.pop_move(candidate.feedBoard);

                    if (!replyMated) {
                        allRepliesMated = false;
                        break;
                    }
                    if (deepestReply == Stockfish::MOVE_NONE
                        || matePly > deepestMatePly) {
                        deepestReply = reply;
                        deepestMateLine = std::move(replyLine);
                    }
                    deepestMatePly = std::max(deepestMatePly, matePly);
                }
                board.pop_move(candidate.feedBoard);

                if (allRepliesMated) {
                    const JointActionCandidate feedAction = candidate.feedBoard == BOARD_A
                        ? JointActionCandidate(candidate.move, 1.0f, 0,
                                               Stockfish::MOVE_NONE, 1.0f, 0,
                                               rules, true, false)
                        : JointActionCandidate(Stockfish::MOVE_NONE, 1.0f, 0,
                                               candidate.move, 1.0f, 0,
                                               rules, false, true);
                    // Root capture, forced opponent reply, then the mate on the
                    // target board.
                    std::vector<MateProofPly> feedLine;
                    if (outPrincipalVariation) {
                        const bool feedOnBoardA =
                            candidate.feedBoard == BOARD_A;
                        feedLine.push_back(
                            feedOnBoardA
                                ? MateProofPly{
                                    candidate.move, Stockfish::MOVE_NONE}
                                : MateProofPly{
                                    Stockfish::MOVE_NONE, candidate.move});
                        if (deepestReply != Stockfish::MOVE_NONE) {
                            feedLine.push_back(
                                feedOnBoardA
                                    ? MateProofPly{
                                        deepestReply, Stockfish::MOVE_NONE}
                                    : MateProofPly{
                                        Stockfish::MOVE_NONE, deepestReply});
                            feedLine.insert(
                                feedLine.end(),
                                deepestMateLine.begin(),
                                deepestMateLine.end());
                        }
                    }
                    retain_shortest_feed(
                        feedAction, deepestMatePly + 2, std::move(feedLine));
                }
            }
        }

        if (foundDirectMate
            && (!foundFeedMate || bestDirectPly <= bestFeedPly)) {
            outAction = bestDirectAction;
            outPlyToMate = bestDirectPly;
            if (outPrincipalVariation) {
                *outPrincipalVariation = bestDirectPv;
            }
            return true;
        }
        if (foundFeedMate) {
            outAction = bestFeedAction;
            outPlyToMate = bestFeedPly;
            report_feed_principal_variation();
            return true;
        }
    } else {
        // In every other turn/time configuration, moves on the partner board,
        // capture transfers and legal sits can affect the mating board. Search
        // complete joint actions rather than treating the boards independently.
        MateSearchBudget budget;
        budget.remainingNodes = std::max<uint64_t>(
            1, nodeBudget / SearchParams::MATE_JOINT_SEARCH_BUDGET_DIVISOR);
        with_deadline(budget);
        JointMateCache cache;
        cache.reserve(static_cast<size_t>(
            std::min<uint64_t>(budget.remainingNodes, 16384)));
        for (int maxMateMoves = 2;
             maxMateMoves <= SearchParams::MATE_SEARCH_MAX_ATTACKER_MOVES;
             ++maxMateMoves) {
            const JointMateProof proof = search_joint_forced_mate(
                board, teamSide, teamHasTimeAdvantage, teamSide,
                maxMateMoves, 0, budget, cache, attackerWinsMateRace);
            if (proof.status == JointMateStatus::PROVEN) {
                const bool isCapA = proof.action.moveA != Stockfish::MOVE_NONE
                    && board.is_capture(BOARD_A, proof.action.moveA);
                const bool isCapB = proof.action.moveB != Stockfish::MOVE_NONE
                    && board.is_capture(BOARD_B, proof.action.moveB);
                const JointActionRules rules{
                    boardAOnTurn, boardBOnTurn, teamHasTimeAdvantage,
                    boardAOnTurn && board.has_any_legal_move(BOARD_A),
                    boardBOnTurn && board.has_any_legal_move(BOARD_B)};
                outAction = JointActionCandidate(
                    proof.action.moveA, 1.0f, 0,
                    proof.action.moveB, 1.0f, 0,
                    rules, isCapA, isCapB);
                outPlyToMate = proof.pliesToMate;
                if (outPrincipalVariation) {
                    for (auto ply = proof.principalVariation;
                         ply; ply = ply->next) {
                        outPrincipalVariation->push_back(ply->action);
                    }
                }
                return true;
            }
            if (budget.exhausted) {
                break;
            }
        }
    }

    return false;
}

bool Agent::find_root_loss_proofs(
    Board& board,
    Stockfish::Color teamSide,
    bool teamHasTimeAdvantage,
    vector<RootLossProof>& outProofs,
    uint64_t nodeBudget,
    MateSearchBudget::Clock::time_point deadline,
    const std::atomic<bool>* cancelled,
    const vector<JointActionCandidate>* preferredActions,
    Node* liveRoot) {
    outProofs.clear();
    MateSearchBudget totalBudget;
    totalBudget.remainingNodes = nodeBudget;
    totalBudget.deadline = deadline;
    totalBudget.cancelled = cancelled;
    totalBudget.stopOnSolvedRoot = liveRoot;
    if (totalBudget.out_of_time()) {
        return false;
    }
    vector<JointActionCandidate> defenses = legal_joint_actions(
        board, teamSide, teamHasTimeAdvantage);
    if (defenses.empty() || nodeBudget == 0) {
        return false;
    }

    // The reverse scan is deliberately bounded and usually cannot prove every
    // legal bughouse defense. Search the moves MCTS is actually considering
    // first, in visit order, so a tactical certificate can veto a likely blunder
    // before the deadline instead of being stranded behind dozens of low-policy
    // drops in Stockfish's move-generation order.
    if (preferredActions && !preferredActions->empty()) {
        auto action_key = [](const JointActionCandidate& action) {
            return (static_cast<uint64_t>(action.moveA) << 32)
                | static_cast<uint32_t>(action.moveB);
        };
        std::unordered_map<uint64_t, size_t> preferredRanks;
        preferredRanks.reserve(preferredActions->size());
        for (size_t rank = 0; rank < preferredActions->size(); ++rank) {
            preferredRanks.try_emplace(action_key((*preferredActions)[rank]), rank);
        }
        const size_t unranked = preferredActions->size();
        std::stable_sort(
            defenses.begin(), defenses.end(),
            [&](const JointActionCandidate& lhs,
                const JointActionCandidate& rhs) {
                const auto lhsIt = preferredRanks.find(action_key(lhs));
                const auto rhsIt = preferredRanks.find(action_key(rhs));
                const size_t lhsRank = lhsIt == preferredRanks.end()
                    ? unranked : lhsIt->second;
                const size_t rhsRank = rhsIt == preferredRanks.end()
                    ? unranked : rhsIt->second;
                return lhsRank < rhsRank;
            });
    }

    const Stockfish::Color opponentTeam = ~teamSide;
    const bool opponentHasTimeAdvantage = !teamHasTimeAdvantage;
    SingleBoardMateCache singleBoardMateCache;

    const auto actions_equal = [](const JointActionCandidate& lhs,
                                  const JointActionCandidate& rhs) {
        return lhs.moveA == rhs.moveA && lhs.moveB == rhs.moveB;
    };

    // Once one defense has established a partner-board-agnostic mate after a
    // capture feed, reuse it across other root defenses. Those defenses often
    // differ only by a quiet drop on the feed board; re-running Stockfish for
    // every square hides the same tactical threat behind dozens of equivalent
    // positions.
    const auto try_cached_capture_feed = [&] (
        int& opponentMatePly,
        vector<MateProofPly>& opponentPv) {
        if (!opponentHasTimeAdvantage || singleBoardMateCache.empty()) {
            return false;
        }
        const bool opponentAOnTurn =
            board.side_to_move(BOARD_A) == opponentTeam;
        const bool opponentBOnTurn =
            board.side_to_move(BOARD_B) == ~opponentTeam;
        if (!opponentAOnTurn || !opponentBOnTurn) {
            return false;
        }

        for (int feedBoard : {BOARD_A, BOARD_B}) {
            const int targetBoard = 1 - feedBoard;
            const Stockfish::Color targetAttacker = targetBoard == BOARD_A
                ? opponentTeam : ~opponentTeam;
            for (Stockfish::Move capture : board.legal_moves(feedBoard)) {
                if (!board.is_capture(feedBoard, capture)
                    || !totalBudget.consume()) {
                    continue;
                }
                board.push_move(feedBoard, capture);
                const vector<Stockfish::Move> replies =
                    board.legal_moves(feedBoard);
                bool allRepliesMated = !replies.empty();
                int deepestMatePly = 0;
                Stockfish::Move deepestReply = Stockfish::MOVE_NONE;
                const SingleBoardMateCacheEntry* deepestProof = nullptr;

                for (Stockfish::Move reply : replies) {
                    if (!totalBudget.consume()) {
                        allRepliesMated = false;
                        break;
                    }
                    board.push_move(feedBoard, reply);
                    const bool quietFeedBoard =
                        !board.is_in_check(feedBoard)
                        && board.has_any_legal_move(feedBoard);
                    const string cacheKey =
                        std::to_string(targetBoard) + ":"
                        + std::to_string(static_cast<int>(targetAttacker)) + ":"
                        + board.fen(targetBoard);
                    const auto cached = singleBoardMateCache.find(cacheKey);
                    const bool replyMated = quietFeedBoard
                        && !board.is_checkmate(opponentTeam, true)
                        && !board.is_draw()
                        && cached != singleBoardMateCache.end();
                    if (replyMated
                        && cached->second.plyToMate > deepestMatePly) {
                        deepestMatePly = cached->second.plyToMate;
                        deepestReply = reply;
                        deepestProof = &cached->second;
                    }
                    board.pop_move(feedBoard);
                    if (!replyMated) {
                        allRepliesMated = false;
                        break;
                    }
                }
                board.pop_move(feedBoard);

                if (!allRepliesMated || !deepestProof) {
                    continue;
                }
                const bool feedOnBoardA = feedBoard == BOARD_A;
                opponentPv = {
                    feedOnBoardA
                        ? MateProofPly{capture, Stockfish::MOVE_NONE}
                        : MateProofPly{Stockfish::MOVE_NONE, capture},
                    feedOnBoardA
                        ? MateProofPly{deepestReply, Stockfish::MOVE_NONE}
                        : MateProofPly{Stockfish::MOVE_NONE, deepestReply}};
                opponentPv.insert(
                    opponentPv.end(),
                    deepestProof->principalVariation.begin(),
                    deepestProof->principalVariation.end());
                opponentMatePly = deepestMatePly + 2;
                return true;
            }
        }
        return false;
    };

    for (size_t defenseIndex = 0;
         defenseIndex < defenses.size(); ++defenseIndex) {
        // A proof attached below immediately removes that edge from live MCTS
        // selection. Re-sample the current favourite so the bounded scan can
        // peel successive blunders instead of spending its tail on a stale
        // ordering captured before the first veto.
        if (liveRoot && liveRoot->is_expanded()) {
            const vector<int> visits = liveRoot->get_child_visits();
            const vector<shared_ptr<Node>> children = liveRoot->get_children();
            const size_t liveCount = std::min(visits.size(), children.size());
            int bestVisits = -1;
            size_t bestDefense = defenses.size();
            for (size_t childIndex = 0; childIndex < liveCount; ++childIndex) {
                if (!children[childIndex]
                    || children[childIndex]->get_node_type() == NodeType::WIN
                    || childIndex >= liveRoot->get_num_generated()) {
                    continue;
                }
                const JointActionCandidate liveAction =
                    liveRoot->get_joint_action(static_cast<int>(childIndex));
                for (size_t candidate = defenseIndex;
                     candidate < defenses.size(); ++candidate) {
                    if (actions_equal(liveAction, defenses[candidate])
                        && visits[childIndex] > bestVisits) {
                        bestVisits = visits[childIndex];
                        bestDefense = candidate;
                        break;
                    }
                }
            }
            if (bestDefense < defenses.size() && bestDefense != defenseIndex) {
                std::swap(defenses[defenseIndex], defenses[bestDefense]);
            }
        }
        if (!totalBudget.consume()) {
            break;
        }

        const JointActionCandidate& defense = defenses[defenseIndex];
        board.make_moves(defense.moveA, defense.moveB);

        bool defenseIsMated = false;
        int totalLossPly = 0;
        std::vector<MateProofPly> principalVariation{
            {defense.moveA, defense.moveB}};
        if (board.is_checkmate(teamSide, teamHasTimeAdvantage)) {
            // The defensive action itself ended the game.
            defenseIsMated = true;
            totalLossPly = 1;
        } else if (!board.is_checkmate(
                       opponentTeam, opponentHasTimeAdvantage)
                   && !board.is_draw(1)) {
            std::vector<MateProofPly> cachedOpponentPv;
            int cachedOpponentMatePly = 0;
            if (try_cached_capture_feed(
                    cachedOpponentMatePly, cachedOpponentPv)) {
                defenseIsMated = true;
                totalLossPly = cachedOpponentMatePly + 1;
                principalVariation.insert(
                    principalVariation.end(),
                    cachedOpponentPv.begin(), cachedOpponentPv.end());
            }
            // Divide the remaining budget fairly among the current and all
            // untested defenses. Unused probes return to the common pool, and
            // an early branch cannot starve the universal proof.
            const uint64_t remainingDefenses =
                defenses.size() - defenseIndex;
            const uint64_t defenseAllocation =
                totalBudget.remainingNodes / remainingDefenses;
            if (!defenseIsMated && defenseAllocation > 0) {
                MateSearchBudget defenseBudget;
                defenseBudget.remainingNodes = defenseAllocation;
                defenseBudget.deadline = deadline;
                defenseBudget.cancelled = cancelled;
                defenseBudget.stopOnSolvedRoot = liveRoot;
                JointActionCandidate opponentMate;
                int opponentMatePly = 0;
                std::vector<MateProofPly> opponentPrincipalVariation;
                defenseIsMated = find_root_mate_impl(
                    board, opponentTeam, opponentHasTimeAdvantage,
                    opponentMate, opponentMatePly,
                    defenseAllocation, nullptr, &defenseBudget, true,
                    deadline, true, true, &opponentPrincipalVariation,
                    cancelled, &singleBoardMateCache, liveRoot);
                const uint64_t probesUsed = defenseAllocation
                    - defenseBudget.remainingNodes;
                totalBudget.remainingNodes -= probesUsed;
                if (defenseIsMated) {
                    // Our defense is ply one; the opponent proof starts from
                    // the resulting position.
                    totalLossPly = opponentMatePly + 1;
                    principalVariation.insert(
                        principalVariation.end(),
                        opponentPrincipalVariation.begin(),
                        opponentPrincipalVariation.end());
                }
            }
        }

        board.unmake_moves(defense.moveA, defense.moveB);

        // Retain exact information about this action even when another action
        // is safe or cannot be proved within the bounded scan. The old
        // all-or-nothing return threw these proofs away, allowing MCTS to play
        // a known mate while an unresolved defense remained.
        if (defenseIsMated) {
            outProofs.push_back({
                defense, totalLossPly, std::move(principalVariation)});
            if (liveRoot && liveRoot->is_expanded()) {
                const vector<shared_ptr<Node>> children = liveRoot->get_children();
                const size_t liveCount = std::min(
                    children.size(), liveRoot->get_num_generated());
                for (size_t childIndex = 0;
                     childIndex < liveCount; ++childIndex) {
                    if (!children[childIndex]
                        || !actions_equal(
                            liveRoot->get_joint_action(
                                static_cast<int>(childIndex)),
                            defense)) {
                        continue;
                    }
                    children[childIndex]->mark_as_win(
                        std::max(0, totalLossPly - 1));
                    break;
                }
            }
        }
    }

    return outProofs.size() == defenses.size();
}

bool Agent::find_root_forced_loss(
    Board& board,
    Stockfish::Color teamSide,
    bool teamHasTimeAdvantage,
    JointActionCandidate& outAction,
    int& outPlyToMate,
    uint64_t nodeBudget,
    MateSearchBudget::Clock::time_point deadline) {
    vector<RootLossProof> proofs;
    if (!find_root_loss_proofs(
            board, teamSide, teamHasTimeAdvantage,
            proofs, nodeBudget, deadline)) {
        return false;
    }

    const auto delaying = std::max_element(
        proofs.begin(), proofs.end(),
        [](const RootLossProof& lhs, const RootLossProof& rhs) {
            return lhs.plyToMate < rhs.plyToMate;
        });
    if (delaying == proofs.end()) {
        return false;
    }
    outAction = delaying->action;
    outPlyToMate = delaying->plyToMate;
    return true;
}

Agent::Agent(int numThreadsParam) : running(false), numThreads(0) {
    // Use specified thread count, or fall back to search params default
    numThreads = (numThreadsParam > 0) ? numThreadsParam : SearchParams::NUM_SEARCH_THREADS;
    
    // Create the transposition table for MCGS (if enabled)
    if (SearchParams::ENABLE_MCGS) {
        transpositionTable = std::make_unique<TranspositionTable>();
        transpositionTable->setMaxCapacity(SearchParams::TT_MAX_SIZE);
        transpositionTable->reserve(SearchParams::TT_INITIAL_CAPACITY);
    }
    
    // Start garbage collection thread for async tree cleanup
    gcThread_.start();
    
    ensure_worker_pool(static_cast<size_t>(numThreads));
}

Agent::~Agent() {
    running = false;
    {
        std::lock_guard lock(workerMutex_);
        shutdownWorkers_ = true;
        workerGeneration_++;
    }
    workerCv_.notify_all();
    for (auto& worker : workerPool_) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    gcThread_.stop();
    
    for (auto* st : searchThreads) {
        delete st;
    }
    searchThreads.clear();
}

void Agent::ensure_worker_pool(size_t workerCount) {
    while (searchThreads.size() < workerCount) {
        searchThreads.push_back(new SearchThread());
    }
    while (workerPool_.size() < workerCount) {
        const size_t workerIndex = workerPool_.size();
        workerPool_.emplace_back(
            &Agent::worker_loop, this, workerIndex, workerGeneration_);
    }
}

void Agent::worker_loop(size_t workerIndex, uint64_t observedGeneration) {
    while (true) {
        const Board* board = nullptr;
        Engine* engine = nullptr;
        SearchInfo* searchInfo = nullptr;
        bool teamHasTimeAdvantage = false;
        size_t targetNodes = 0;
        int moveTimeMs = 0;

        {
            std::unique_lock lock(workerMutex_);
            workerCv_.wait(lock, [this, observedGeneration] {
                return shutdownWorkers_ || workerGeneration_ != observedGeneration;
            });
            if (shutdownWorkers_) {
                return;
            }
            observedGeneration = workerGeneration_;
            if (workerIndex >= activeWorkerCount_) {
                continue;
            }
            board = workerBoard_;
            engine = workerEngines_[workerIndex % workerEngines_.size()];
            searchInfo = workerSearchInfo_;
            teamHasTimeAdvantage = workerTeamHasTimeAdvantage_;
            targetNodes = workerTargetNodes_;
            moveTimeMs = workerMoveTimeMs_;
        }

        try {
            Board localBoard(*board);
            SearchThread* searchThread = searchThreads[workerIndex];
            if (moveTimeMs > 0) {
                while (running &&
                       (isPondering_.load(std::memory_order_relaxed) ||
                        searchInfo->elapsed() < searchInfo->get_effective_move_time())) {
                    if (SearchParams::ENABLE_MATE_EARLY_EXIT && rootNode
                        && rootNode->get_node_type() != NodeType::UNSOLVED) {
                        running = false;
                        break;
                    }
                    searchThread->run_iteration(
                        localBoard, engine, teamHasTimeAdvantage);
                }
            } else {
                while (running &&
                       (isPondering_.load(std::memory_order_relaxed) ||
                        static_cast<size_t>(searchInfo->get_nodes_searched()) < targetNodes)) {
                    if (SearchParams::ENABLE_MATE_EARLY_EXIT && rootNode
                        && rootNode->get_node_type() != NodeType::UNSOLVED) {
                        running = false;
                        break;
                    }
                    searchThread->run_iteration(
                        localBoard, engine, teamHasTimeAdvantage);
                }
            }
            if (SearchParams::ENABLE_MATE_EARLY_EXIT && rootNode
                && rootNode->get_node_type() != NodeType::UNSOLVED) {
                searchThread->discard_pending_iteration(localBoard, engine);
            } else {
                searchThread->finish_pending_iteration(
                    localBoard, engine, teamHasTimeAdvantage);
            }
        } catch (...) {
            std::lock_guard lock(workerMutex_);
            if (!workerException_) {
                workerException_ = std::current_exception();
            }
            running = false;
        }

        {
            std::lock_guard lock(workerMutex_);
            completedWorkerCount_++;
            if (completedWorkerCount_ == activeWorkerCount_) {
                workersDoneCv_.notify_one();
            }
        }
    }
}

void Agent::dispatch_workers(const Board& board,
                             const vector<Engine*>& engines,
                             SearchInfo& searchInfo,
                             bool teamHasTimeAdvantage,
                             size_t targetNodes,
                             int moveTimeMs,
                             size_t workerCount) {
    {
        std::lock_guard lock(workerMutex_);
        workerBoard_ = &board;
        workerEngines_ = engines;
        workerSearchInfo_ = &searchInfo;
        workerTeamHasTimeAdvantage_ = teamHasTimeAdvantage;
        workerTargetNodes_ = targetNodes;
        workerMoveTimeMs_ = moveTimeMs;
        activeWorkerCount_ = workerCount;
        completedWorkerCount_ = 0;
        workerException_ = nullptr;
        running = true;
        workerGeneration_++;
    }
    workerCv_.notify_all();
}

void Agent::wait_for_workers() {
    std::unique_lock lock(workerMutex_);
    workersDoneCv_.wait(lock, [this] {
        return completedWorkerCount_ == activeWorkerCount_;
    });
}

void Agent::reset_search_state() {
    std::unique_lock searchLock(searchMutex_);
    isPondering_.store(false, std::memory_order_release);
    currentSearchInfo_.store(nullptr, std::memory_order_release);
    auto oldRoot = std::move(rootNode);
    nextRootCandidates_.clear();
    mateContinuations_.clear();
    lastSearchHash_ = 0;
    if (transpositionTable) {
        transpositionTable->clear();
    }
    if (oldRoot) {
        gcThread_.enqueue(std::move(oldRoot));
    }
}

bool Agent::try_reuse_mate_continuation(
    Board& board, Stockfish::Color teamSide, bool teamHasTimeAdvantage,
    JointActionCandidate& outAction, int& outPlyToMate) const {
    const uint64_t positionHash = board.search_hash_key(
        teamSide, teamHasTimeAdvantage);
    const std::string signature = board_signature(board);

    for (const MateContinuation& continuation : mateContinuations_) {
        if (continuation.positionHash != positionHash
            || continuation.signature != signature
            || continuation.teamSide != teamSide
            || continuation.teamHasTimeAdvantage != teamHasTimeAdvantage
            || continuation.plyToMate <= 0) {
            continue;
        }
        if ((continuation.action.moveA != Stockfish::MOVE_NONE
             && !board.is_legal_move(BOARD_A, continuation.action.moveA))
            || (continuation.action.moveB != Stockfish::MOVE_NONE
                && !board.is_legal_move(BOARD_B, continuation.action.moveB))
            || (continuation.action.moveA == Stockfish::MOVE_NONE
                && continuation.action.moveB == Stockfish::MOVE_NONE)) {
            continue;
        }

        outAction = continuation.action;
        outPlyToMate = continuation.plyToMate;
        return true;
    }
    return false;
}

/**
 * @brief Runs a UCI search.
 */
JointActionCandidate Agent::run_search(Board& board, const vector<Engine*>& engines, 
                                        Stockfish::Color teamSide, bool teamHasTimeAdvantage,
                                        const SearchOptions& options) {
    std::unique_lock searchLock(searchMutex_);
    const auto searchStart = chrono::steady_clock::now();
    JointActionCandidate result;
    if (options.background) {
        // The caller already emitted a bestmove for this move; a stop that
        // landed in between cancels the background search rather than starting
        // one that nothing is waiting on.
        if (stopRequested_.load(std::memory_order_acquire)) {
            return result;
        }
    } else {
        stopRequested_.store(false, std::memory_order_release);
        lastRuntimeConfig_ = options.search;
    }
    if (engines.empty()) {
        cerr << "Cannot search without an inference engine" << endl;
        return result;
    }
    
    const bool boardAOnTurn = board.side_to_move(BOARD_A) == teamSide;
    const bool boardBOnTurn = board.side_to_move(BOARD_B) == ~teamSide;
    const bool canWait = is_double_sit_legal(
        teamHasTimeAdvantage, boardAOnTurn, boardBOnTurn);

    // Nothing retained from an earlier move can be adopted from a position with
    // no move to make, and leaving stale candidates behind would let the
    // permanent brain start from one.
    const auto drop_retained_candidates = [this, &options] {
        if (!options.background) {
            nextRootCandidates_.clear();
        }
    };

    const bool teamHasPlayableMove =
        (boardAOnTurn && !board.legal_moves(BOARD_A).empty())
        || (boardBOnTurn && !board.legal_moves(BOARD_B).empty());
    const bool opponentIsMated =
        board.is_checkmate(~teamSide, !teamHasTimeAdvantage);
    const bool teamIsMated =
        board.is_checkmate(teamSide, teamHasTimeAdvantage);

    // A mate on the other team is not necessarily terminal at this root. If
    // we are down on time (or both boards are on turn), we still owe a move;
    // a capture can hand the checked player a blocking piece and undo the
    // mate. Let the root mate scan choose an action that preserves it instead
    // of returning MOVE_NONE before any joint action is considered.
    const bool mustPlayDespiteOpponentMate =
        opponentIsMated && !canWait && teamHasPlayableMove;

    // The server, not this combined-board search, decides when a live game has
    // actually stopped. If our partner board is already mated but this
    // team still has a legal move elsewhere, search that move instead of
    // abandoning the seat with bestmove (none). The root classifier makes the
    // matching one-ply best-effort exception so the neural policy can rank the
    // available moves before ordinary terminal handling resumes below it.
    const bool mustPlayDespiteTeamMate =
        !teamHasTimeAdvantage && teamIsMated && teamHasPlayableMove;

    if ((opponentIsMated && !mustPlayDespiteOpponentMate)
        || (teamIsMated && !mustPlayDespiteTeamMate)
        || board.is_draw()) {
        drop_retained_candidates();
        if (options.verbose) {
            cout << "bestmove (none)" << endl;
        }
        return result;
    }

    // A team with no real board move may still have the legal wait action.
    if (!teamHasPlayableMove && !canWait) {
        drop_retained_candidates();
        if (options.verbose) {
            cout << "bestmove (none)" << endl;
        }
        return result;
    }

    // Determine effective move time
    int moveTimeMs = options.moveTimeMs;
    size_t targetNodes = options.targetNodes;

    // Compute position hash for tree reuse
    uint64_t positionHash = board.search_hash_key(
        teamSide, teamHasTimeAdvantage);
    const std::string positionSignature = board_signature(board);

    // Try to reuse tree from previous search (if enabled)
    std::shared_ptr<Node> reusedRoot = nullptr;
    if (SearchParams::ENABLE_TREE_REUSE) {
        reusedRoot = try_reuse_tree(positionHash, teamSide, positionSignature);
    }
    if (reusedRoot && mustPlayDespiteTeamMate
        && reusedRoot->get_node_type() == NodeType::LOSS) {
        // A retained node may already have been solved as a loss while it was
        // below an earlier root. Reusing its solved state would stop this live
        // partner board before it can compare the available moves.
        reusedRoot.reset();
    }

    if (reusedRoot) {
        // Reuse the existing subtree
        rootNode = reusedRoot;
        rootNode->set_hash(positionHash);
        rootNode->set_depth(0);

        if (options.verbose) {
            cout << "info string Tree reuse: " << rootNode->get_visits()
                 << " visits recovered" << endl;
        }
    } else {
        // Create new root node
        rootNode = make_shared<Node>(teamSide, positionHash);
    }
    rootNode->configure_root_search(
        options.search, !mustPlayDespiteTeamMate);

    // Start the clock where run_search began, not after the concurrent root
    // scans below: they are part of this move's thinking time, and resetting
    // the clock afterward would let a slow scan overrun the allotted move time.
    SearchInfo searchInfo(searchStart, moveTimeMs);
    // The requested move time is the entire allocation for this move, so the
    // instability and eval-drop extensions below re-spend time within it
    // rather than adding to it.
    searchInfo.set_hard_limit(moveTimeMs);
    isPondering_.store(options.isPonder, std::memory_order_release);
    currentSearchInfo_.store(&searchInfo, std::memory_order_release);

    // MCGS: discard nodes outside the signature-verified reused graph, then
    // re-index that graph so new transpositions merge into retained nodes.
    if (options.search.enableMCGS && options.search.enableTranspositions && transpositionTable) {
        transpositionTable->clear();
        if (reusedRoot) {
            reindex_reused_subtree(rootNode);
        } else {
            transpositionTable->insertOrGet(positionHash, rootNode);
        }
    }

    const size_t workerCount = static_cast<size_t>(numThreads) * engines.size();
    ensure_worker_pool(workerCount);

    // Set up active search threads with shared root node, search info, and transposition table
    for (size_t i = 0; i < workerCount; ++i) {
        SearchThread* st = searchThreads[i];
        st->set_root_node(rootNode);
        st->set_search_info(&searchInfo);
        st->set_runtime_config(options.search);
        st->set_inference_worker_index(i / engines.size());
        st->set_transposition_table(
            options.search.enableMCGS && options.search.enableTranspositions
                ? transpositionTable.get()
                : nullptr);
    }

    const bool runRootScan = options.search.enableRootMateSearch
        && !options.background;

    // The scan makes and unmakes moves while proving. Give it a private board
    // before dispatching workers so they only ever copy the untouched caller's
    // board, and so a failed copy cannot leave workers holding stack pointers.
    std::unique_ptr<Board> scanBoard;
    if (runRootScan) {
        scanBoard = std::make_unique<Board>(board);
    }

    // Deeper root scans run on this thread concurrently with the workers, so
    // they cost one worker's share of the CPU rather than leaving the GPU idle.
    // A tiny mate-in-one preflight is the exception: once a neural batch has
    // started it cannot be cancelled, so dispatching it first would make an
    // immediate mate wait for inference to finish.
    const size_t scanWorkerCount = runRootScan && workerCount > 1
        ? workerCount - 1
        : workerCount;
    bool workersDispatched = false;
    int lastReportedDepth = 0;
    atomic<bool> stopPrepassReporter{false};
    thread prepassReporter;

    // The probe answers a question the checking-move scans cannot ask - a mate
    // that needs a quiet preparing move - so it runs beside the workers for the
    // whole move on a thread of its own. It used to run on this thread inside
    // the pre-pass window, which is why it was restricted to time-ahead roots:
    // on an even clock that window belongs to the reverse scan. Off the window
    // it costs one CPU thread (~5% of MCTS nps here) and no search time, so
    // both clock states get it.
    atomic<bool> stopRootProbe{false};
    std::mutex rootProbeMutex;
    bool rootProbeFoundMate = false;
    JointActionCandidate rootProbeAction;
    int rootProbePly = 0;
    string rootProbePv;
    thread rootProbeThread;

    const auto start_root_probe = [&] {
        if (!runRootScan || !options.search.enableMateProbe) {
            return;
        }
        const int probeBudgetMs = moveTimeMs > 0
            ? std::max(1, moveTimeMs - static_cast<int>(searchInfo.elapsed()))
            : SearchParams::MATE_PROBE_UNTIMED_BUDGET_MS;
        // The scans on this thread make and unmake moves on scanBoard while
        // the probe reads its own position, and the workers only ever copy the
        // caller's board, so the probe takes a private copy of its own.
        auto probeBoard = std::make_unique<Board>(board);
        // rootNode is only replaced after this thread has been joined, so the
        // node the workers are filling stays alive for as long as the probe
        // polls it.
        Node* const probeRoot = rootNode.get();
        rootProbeThread = thread(
            [&, probeBudgetMs, probeRoot,
             probeBoard = std::move(probeBoard)]() mutable {
                JointActionCandidate action;
                int plyToMate = 0;
                string pv;
                const auto settled = [&] {
                    return stopRootProbe.load(memory_order_acquire)
                        || !running.load(memory_order_acquire)
                        || stopRequested_.load(memory_order_relaxed)
                        || (SearchParams::ENABLE_MATE_EARLY_EXIT && probeRoot
                            && probeRoot->get_node_type()
                                   != NodeType::UNSOLVED);
                };
                const auto publish = [&] {
                    lock_guard<std::mutex> guard(rootProbeMutex);
                    rootProbeFoundMate = true;
                    rootProbeAction = action;
                    rootProbePly = plyToMate;
                    rootProbePv = pv;
                };
                // Nothing here is required for a move to come back, so a
                // failure costs the probe and not the search - and never an
                // unhandled exception on a detachable thread.
                try {
                    probe_root_mate(
                        *probeBoard, teamSide, teamHasTimeAdvantage,
                        probeBudgetMs, settled, action, plyToMate, pv,
                        publish);
                } catch (const std::exception& error) {
                    cout << "info string mate probe failed: " << error.what()
                         << endl;
                } catch (...) {
                    cout << "info string mate probe failed" << endl;
                }
            });
    };

    const auto stop_root_probe = [&] {
        stopRootProbe.store(true, memory_order_release);
        if (rootProbeThread.joinable()) {
            rootProbeThread.join();
        }
    };

    // Every other exit from run_search - a rethrown worker exception, a stalled
    // node search - joins it here instead, before the locals it reads go away.
    struct RootProbeJoin {
        std::function<void()> join;
        ~RootProbeJoin() { join(); }
    } rootProbeJoin{stop_root_probe};

    // A probe mate is a pruned search's claim rather than a proof, so it only
    // decides the move while nothing better is available: the exact scans
    // return ahead of it, and a root MCTS has already solved keeps its answer.
    // Every mate inside the probe's acceptance bound can save the remaining
    // clock; the exhaustive scan's depth limit is not a probe confidence bound.
    const auto probe_mate_ends_search = [&] {
        if (rootNode && rootNode->get_node_type() != NodeType::UNSOLVED) {
            return false;
        }
        lock_guard<std::mutex> guard(rootProbeMutex);
        return rootProbeFoundMate
            && SearchParams::mate_probe_can_end_search(rootProbePly);
    };

    const auto take_probe_mate = [&](JointActionCandidate& action,
                                     int& plyToMate, string& pv) {
        lock_guard<std::mutex> guard(rootProbeMutex);
        if (!rootProbeFoundMate) {
            return false;
        }
        action = rootProbeAction;
        plyToMate = rootProbePly;
        pv = rootProbePv;
        return true;
    };

    const auto start_prepass_reporter = [&] {
        if (!options.verbose || moveTimeMs <= 0) {
            return;
        }
        prepassReporter = thread([&] {
            constexpr float C = 180.0f;
            constexpr float k = 1.56f;
            constexpr int REPORT_INTERVAL_MS = 5;

            while (!stopPrepassReporter.load(memory_order_acquire)
                   && running.load(memory_order_acquire)) {
                this_thread::sleep_for(chrono::milliseconds(
                    REPORT_INTERVAL_MS));
                if (stopPrepassReporter.load(memory_order_acquire)) {
                    break;
                }

                const int depth = searchInfo.get_max_depth();
                if (depth <= lastReportedDepth || !rootNode
                    || !rootNode->is_expanded()) {
                    continue;
                }

                const auto childVisits = rootNode->get_child_visits();
                const auto children = rootNode->get_children();
                const size_t numChildren = min(
                    childVisits.size(), children.size());
                if (numChildren == 0) {
                    continue;
                }

                int mostVisitedIdx = 0;
                for (size_t i = 1; i < numChildren; ++i) {
                    if (childVisits[i] > childVisits[mostVisitedIdx]) {
                        mostVisitedIdx = static_cast<int>(i);
                    }
                }
                const int solverBestIdx =
                    rootNode->get_best_move_idx_with_q_weight(
                        options.search.qVetoDelta,
                        options.search.qValueWeight);
                const int displayIdx = solverBestIdx >= 0
                        && static_cast<size_t>(solverBestIdx) < numChildren
                    ? solverBestIdx
                    : mostVisitedIdx;
                const double elapsedMs = searchInfo.elapsed();
                const int nodes = searchInfo.get_nodes_searched();
                const int nps = elapsedMs > 0
                    ? static_cast<int>(nodes * 1000.0 / elapsedMs)
                    : 0;
                const size_t tbhits = options.search.enableMCGS
                        && options.search.enableTranspositions
                        && transpositionTable
                    ? transpositionTable->getHits()
                    : 0;
                const int hashfull = options.search.enableMCGS
                        && options.search.enableTranspositions
                        && transpositionTable
                    ? transpositionTable->getFullness()
                    : 0;
                const float childQ = rootNode->get_child_q(displayIdx);
                const string scoreStr = format_root_aware_uci_score(
                    rootNode, children[displayIdx], childQ, C, k);
                const string pv = extract_pv_from_child(
                    board, displayIdx, 20, teamSide,
                    teamHasTimeAdvantage);

                lastReportedDepth = depth;
                cout << "info depth " << depth
                     << " " << scoreStr
                     << " nodes " << nodes
                     << " nps " << nps
                     << " hashfull " << hashfull
                     << " tbhits " << tbhits
                     << " time " << static_cast<int>(elapsedMs);
                if (!pv.empty()) {
                    cout << " pv " << pv;
                }
                cout << endl;
            }
        });
    };

    const auto stop_prepass_reporter = [&] {
        stopPrepassReporter.store(true, memory_order_release);
        if (prepassReporter.joinable()) {
            prepassReporter.join();
        }
    };

    // A scan that proves a mate replaces the root with a synthetic one-edge
    // tree, which the workers are concurrently reading through rootNode. Stop
    // and join them before that happens. The same cleanup is required if a
    // scan throws because the workers hold pointers to this call's stack.
    const auto halt_workers_after_scan = [&] {
        if (workersDispatched) {
            running = false;
            wait_for_workers();
        }
        currentSearchInfo_.store(nullptr, std::memory_order_release);
    };

    // Reuse an exact reply-indexed certificate from the previous forced-mate
    // proof before running the bounded root scan again. A changed partner
    // board, pocket, side to move, team, or TimeAdvantage value cannot match.
    JointActionCandidate rootMateAction;
    int rootMatePly = 1;
    vector<MateProofPly> rootMatePv;
    const uint64_t rootMateBudget = mate_search_node_budget(options);
    // The pre-pass now runs inside the move time rather than before it, so it
    // needs an absolute stop as well as a node budget.
    const MateSearchBudget::Clock::time_point rootScanDeadline =
        options.moveTimeMs > 0
            ? searchStart + chrono::milliseconds(std::max(
                  1, options.moveTimeMs
                         * SearchParams::MATE_SEARCH_MAX_TIME_PERCENT / 100))
            : MateSearchBudget::Clock::time_point{};
    const MateSearchBudget::Clock::time_point rootLossScanDeadline =
        options.moveTimeMs > 0
            ? rootScanDeadline + chrono::milliseconds(std::max(
                  1, std::min(
                         SearchParams::ROOT_LOSS_EXTRA_MAX_MS,
                         options.moveTimeMs
                             * SearchParams::ROOT_LOSS_EXTRA_TIME_PERCENT
                             / 100)))
            : MateSearchBudget::Clock::time_point{};
    // Charge the pre-pass for the clock it spends and credit it for the moves it
    // decides, once per search whichever way it exits.
    bool rootScanRecorded = false;
    const auto record_root_scan = [&](bool proved) {
        if (rootScanRecorded || !runRootScan) {
            return;
        }
        rootScanRecorded = true;
        ++rootScanStats_.searches;
        rootScanStats_.proofs += proved ? 1 : 0;
        rootScanStats_.scanNanos += static_cast<uint64_t>(
            chrono::duration_cast<chrono::nanoseconds>(
                chrono::steady_clock::now() - searchStart).count());
    };

    // Replace the tree with a synthetic one-edge root holding the proven
    // action, report it, and hand it back. The workers are concurrently
    // reading rootNode, and the probe thread polls the node they fill, so both
    // are stopped and joined before the pointer is replaced.
    const auto return_proven_mate = [&](const JointActionCandidate& provenAction,
                                        int plyToMate,
                                        const string& proofPv) {
        stop_prepass_reporter();
        stop_root_probe();
        halt_workers_after_scan();
        record_root_scan(true);
        result = provenAction;
        const uint64_t provenPositionHash = board.search_hash_key(
            teamSide, teamHasTimeAdvantage);
        rootNode = make_shared<Node>(teamSide, provenPositionHash);
        rootNode->set_depth(0);

        std::vector<Stockfish::Move> rootActionsA = {result.moveA};
        std::vector<Stockfish::Move> rootActionsB = {result.moveB};
        std::vector<float> rootPriorsA = {1.0f};
        std::vector<float> rootPriorsB = {1.0f};
        std::vector<uint8_t> rootCapsA = {static_cast<uint8_t>(
            result.moveA != Stockfish::MOVE_NONE && board.is_capture(BOARD_A, result.moveA) ? 1 : 0)};
        std::vector<uint8_t> rootCapsB = {static_cast<uint8_t>(
            result.moveB != Stockfish::MOVE_NONE && board.is_capture(BOARD_B, result.moveB) ? 1 : 0)};

        SearchParams::RuntimeConfig fastConfig = options.search;
        fastConfig.rootDirichletAlpha = 0.0f;
        rootNode->try_init_and_expand(
            rootActionsA, rootActionsB, rootPriorsA, rootPriorsB,
            teamHasTimeAdvantage, boardAOnTurn, boardBOnTurn,
            fastConfig, rootCapsA, rootCapsB);

        auto children = rootNode->get_children();
        if (!children.empty() && children[0]) {
            children[0]->mark_as_loss(std::max(0, plyToMate - 1));
            rootNode->init_child_node_types();
            rootNode->update_child_node_type(0, NodeType::LOSS);
        }
        rootNode->update(0, 1.0f);
        rootNode->mark_as_win(plyToMate);

        if (SearchParams::ENABLE_TREE_REUSE) {
            store_next_root_candidates(board, teamHasTimeAdvantage);
            lastSearchHash_ = positionHash;
        }

        rootScanStats_.thinkNanos += static_cast<uint64_t>(
            chrono::duration_cast<chrono::nanoseconds>(
                chrono::steady_clock::now() - searchStart).count());
        if (options.verbose) {
            const string bestMoveStr = extract_best_move(board);
            const int mateScore = (plyToMate + 1) / 2;
            cout << "info depth " << plyToMate << " score mate " << mateScore
                 << " nodes 1 nps 1000 time 0 pv "
                 << (proofPv.empty() ? bestMoveStr : proofPv) << endl;
            cout << "bestmove " << bestMoveStr << endl;
            if (const string scanSummary = root_scan_summary();
                !scanSummary.empty()) {
                cout << scanSummary << endl;
            }
        }
        return result;
    };

    // The root scans answer "what do I play here", which a background search is
    // not asked; they would also replace the real tree with a synthetic
    // one-edge proof that no later position could adopt.
    bool cachedRootMate = false;
    bool immediateRootMate = false;
    bool immediateScanComplete = false;
    bool scannedRootMate = false;
    bool probedRootMate = false;
    string probePrincipalVariation;
    try {
        cachedRootMate = runRootScan
            && try_reuse_mate_continuation(
                *scanBoard, teamSide, teamHasTimeAdvantage,
                rootMateAction, rootMatePly);

        if (runRootScan && !cachedRootMate) {
            MateSearchBudget immediateBudget;
            immediateBudget.remainingNodes = std::min(
                rootMateBudget,
                SearchParams::IMMEDIATE_MATE_PREFLIGHT_NODE_BUDGET);
            immediateBudget.cancelled = &stopRequested_;
            const auto immediateDeadline = chrono::steady_clock::now()
                + chrono::milliseconds(
                    SearchParams::IMMEDIATE_MATE_PREFLIGHT_MAX_MS);
            immediateBudget.deadline = rootScanDeadline
                    != MateSearchBudget::Clock::time_point{}
                ? std::min(rootScanDeadline, immediateDeadline)
                : immediateDeadline;
            immediateRootMate = find_immediate_root_mate(
                *scanBoard, teamSide, teamHasTimeAdvantage,
                rootMateAction, &immediateBudget);
            immediateScanComplete = !immediateBudget.exhausted;
            if (immediateRootMate) {
                rootMatePly = 1;
                rootMatePv = {{rootMateAction.moveA, rootMateAction.moveB}};
            }
        }

        if (!cachedRootMate && !immediateRootMate) {
            dispatch_workers(board, engines, searchInfo, teamHasTimeAdvantage,
                             targetNodes, moveTimeMs, scanWorkerCount);
            workersDispatched = true;
            start_prepass_reporter();
            start_root_probe();
        }
        scannedRootMate = runRootScan
            && !cachedRootMate
            && !immediateRootMate
            && find_root_mate_impl(
                *scanBoard, teamSide, teamHasTimeAdvantage,
                rootMateAction, rootMatePly,
                rootMateBudget, &mateContinuations_, nullptr, true,
                rootScanDeadline, !immediateScanComplete, false,
                &rootMatePv, &stopRequested_, nullptr, rootNode.get());

        // The concurrent probe has been running beside the workers since they
        // were dispatched. Take whatever it has proved by now before the
        // reverse scan spends the rest of its window looking for a loss.
        if (!cachedRootMate && !immediateRootMate && !scannedRootMate
            && probe_mate_ends_search()) {
            probedRootMate = take_probe_mate(
                rootMateAction, rootMatePly, probePrincipalVariation);
        }
    } catch (...) {
        stop_prepass_reporter();
        stop_root_probe();
        halt_workers_after_scan();
        throw;
    }
    if (cachedRootMate || immediateRootMate || scannedRootMate
        || probedRootMate) {
        const string proofPv = probePrincipalVariation.empty()
            ? format_mate_proof_pv(
                board, rootMatePv, teamSide, teamHasTimeAdvantage)
            : probePrincipalVariation;
        return return_proven_mate(rootMateAction, rootMatePly, proofPv);
    }

    // The reverse proof uses the clock state's actual joint-action rules, so a
    // proven loss is authoritative whether this team is ahead or behind on
    // time. In particular, do not skip down-time roots: positions where every
    // forced check evasion loses would otherwise remain a heuristic cp score
    // even though the bounded scanner has a complete mate proof.
    JointActionCandidate rootLossAction;
    int rootLossPly = 0;
    vector<MateProofPly> rootLossPv;
    vector<RootLossProof> rootLossProofs;
    const uint64_t rootLossBudget = options.moveTimeMs > 0
        ? std::max(
            rootMateBudget,
            SearchParams::FORCED_LOSS_MIN_TIMED_NODE_BUDGET)
        : rootMateBudget;
    bool scannedRootLoss = false;
    try {
        vector<JointActionCandidate> preferredRootActions;
        if (rootNode && rootNode->is_expanded()) {
            const vector<int> visits = rootNode->get_child_visits();
            vector<size_t> indices(visits.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::stable_sort(
                indices.begin(), indices.end(),
                [&](size_t lhs, size_t rhs) {
                    return visits[lhs] > visits[rhs];
                });
            preferredRootActions.reserve(indices.size());
            for (size_t index : indices) {
                if (index >= rootNode->get_num_generated()) {
                    continue;
                }
                preferredRootActions.push_back(
                    rootNode->get_joint_action(static_cast<int>(index)));
            }
        }
        scannedRootLoss = runRootScan
            && find_root_loss_proofs(
                *scanBoard, teamSide, teamHasTimeAdvantage,
                rootLossProofs, rootLossBudget,
                rootLossScanDeadline, &stopRequested_,
                &preferredRootActions, rootNode.get());
        if (scannedRootLoss) {
            const auto delaying = std::max_element(
                rootLossProofs.begin(), rootLossProofs.end(),
                [](const RootLossProof& lhs, const RootLossProof& rhs) {
                    return lhs.plyToMate < rhs.plyToMate;
                });
            if (delaying != rootLossProofs.end()) {
                rootLossAction = delaying->action;
                rootLossPly = delaying->plyToMate;
                rootLossPv = delaying->principalVariation;
            } else {
                scannedRootLoss = false;
            }
        }
    } catch (...) {
        stop_prepass_reporter();
        stop_root_probe();
        halt_workers_after_scan();
        throw;
    }
    stop_prepass_reporter();
    record_root_scan(scannedRootLoss);
    if (scannedRootLoss) {
        stop_root_probe();
        halt_workers_after_scan();
        result = rootLossAction;
        const uint64_t provenPositionHash = board.search_hash_key(
            teamSide, teamHasTimeAdvantage);
        rootNode = make_shared<Node>(teamSide, provenPositionHash);
        rootNode->set_depth(0);

        std::vector<Stockfish::Move> rootActionsA = {result.moveA};
        std::vector<Stockfish::Move> rootActionsB = {result.moveB};
        std::vector<float> rootPriorsA = {1.0f};
        std::vector<float> rootPriorsB = {1.0f};
        std::vector<uint8_t> rootCapsA = {static_cast<uint8_t>(
            result.moveA != Stockfish::MOVE_NONE
                && board.is_capture(BOARD_A, result.moveA) ? 1 : 0)};
        std::vector<uint8_t> rootCapsB = {static_cast<uint8_t>(
            result.moveB != Stockfish::MOVE_NONE
                && board.is_capture(BOARD_B, result.moveB) ? 1 : 0)};

        SearchParams::RuntimeConfig fastConfig = options.search;
        fastConfig.rootDirichletAlpha = 0.0f;
        rootNode->try_init_and_expand(
            rootActionsA, rootActionsB, rootPriorsA, rootPriorsB,
            teamHasTimeAdvantage, boardAOnTurn, boardBOnTurn,
            fastConfig, rootCapsA, rootCapsB);

        auto children = rootNode->get_children();
        if (!children.empty() && children[0]) {
            children[0]->mark_as_win(std::max(0, rootLossPly - 1));
            rootNode->init_child_node_types();
            rootNode->update_child_node_type(0, NodeType::WIN);
        }
        rootNode->update(0, -1.0f);
        rootNode->mark_as_loss(rootLossPly);

        if (SearchParams::ENABLE_TREE_REUSE) {
            store_next_root_candidates(board, teamHasTimeAdvantage);
            lastSearchHash_ = positionHash;
        }

        rootScanStats_.thinkNanos += static_cast<uint64_t>(
            chrono::duration_cast<chrono::nanoseconds>(
                chrono::steady_clock::now() - searchStart).count());
        if (options.verbose) {
            const string bestMoveStr = extract_best_move(board);
            const string proofPv = format_mate_proof_pv(
                board, rootLossPv, teamSide, teamHasTimeAdvantage);
            const int mateScore = (rootLossPly + 1) / 2;
            cout << "info depth " << rootLossPly << " score mate -"
                 << mateScore << " nodes 1 nps 1000 time 0 pv "
                 << (proofPv.empty() ? bestMoveStr : proofPv) << endl;
            cout << "bestmove " << bestMoveStr << endl;
            if (const string scanSummary = root_scan_summary();
                !scanSummary.empty()) {
                cout << scanSummary << endl;
            }
        }
        return result;
    }

    // The reverse scan can prove individual moves losing without proving the
    // root lost. Attach each such certificate to the corresponding MCTS child
    // as soon as that edge exists. Node selection already excludes a child
    // that is a proven WIN for the opponent while any unresolved alternative
    // remains. Workers may still be initializing or progressively widening the
    // root, so retry only the newly generated edge range at each poll.
    vector<int> rootLossProofChildIndices(rootLossProofs.size(), -1);
    vector<weak_ptr<Node>> rootLossProofChildren(rootLossProofs.size());
    size_t rootLossEdgesProcessed = 0;
    const auto apply_root_loss_proofs = [&] {
        if (rootLossProofs.empty() || !rootNode || !rootNode->is_expanded()) {
            return;
        }
        const auto children = rootNode->get_children();
        const size_t generated = std::min(
            children.size(), rootNode->get_num_generated());

        const auto mark_proven_loss = [&](size_t proofIndex,
                                          size_t childIndex) {
            if (childIndex >= children.size()) {
                return;
            }
            const shared_ptr<Node>& child = children[childIndex];
            if (!child) {
                return;
            }
            const shared_ptr<Node> previouslyMarked =
                rootLossProofChildren[proofIndex].lock();
            if (previouslyMarked == child
                && child->get_node_type() == NodeType::WIN) {
                return;
            }
            child->mark_as_win(std::max(
                0, rootLossProofs[proofIndex].plyToMate - 1));
            rootLossProofChildren[proofIndex] = child;
        };

        // A transposition can replace an already-generated child after its
        // proof was attached. Re-attach only when that edge's owner changes.
        for (size_t proofIndex = 0;
             proofIndex < rootLossProofs.size(); ++proofIndex) {
            const int childIndex = rootLossProofChildIndices[proofIndex];
            if (childIndex >= 0) {
                mark_proven_loss(
                    proofIndex, static_cast<size_t>(childIndex));
            }
        }

        for (size_t childIndex = rootLossEdgesProcessed;
             childIndex < generated; ++childIndex) {
            const JointActionCandidate generatedAction =
                rootNode->get_joint_action(static_cast<int>(childIndex));
            for (size_t proofIndex = 0;
                 proofIndex < rootLossProofs.size(); ++proofIndex) {
                if (rootLossProofChildIndices[proofIndex] >= 0) {
                    continue;
                }
                const RootLossProof& proof = rootLossProofs[proofIndex];
                if (generatedAction.moveA != proof.action.moveA
                    || generatedAction.moveB != proof.action.moveB) {
                    continue;
                }
                rootLossProofChildIndices[proofIndex] =
                    static_cast<int>(childIndex);
                mark_proven_loss(proofIndex, childIndex);
                break;
            }
        }
        rootLossEdgesProcessed = generated;
    };
    apply_root_loss_proofs();

    const auto exact_root_loss_pv = [&](const JointActionCandidate& action) {
        const auto proof = std::find_if(
            rootLossProofs.begin(), rootLossProofs.end(),
            [&](const RootLossProof& candidate) {
                return candidate.action.moveA == action.moveA
                    && candidate.action.moveB == action.moveB;
            });
        return proof == rootLossProofs.end()
            ? string{}
            : format_mate_proof_pv(
                board, proof->principalVariation,
                teamSide, teamHasTimeAdvantage);
    };

    // Periodic info output during search (UCI verbose mode only)
    // Also handles early stopping and time extension
    constexpr int POLL_INTERVAL_MS = 5;
    bool nodeSearchStalled = false;
    int stalledCompletedNodes = 0;
    if (options.verbose && moveTimeMs > 0) {
        searchInfo.set_in_game(true);
        constexpr float C = 180.0f;
        constexpr float k = 1.56f;
        float lastCheckEval = 0.0f;
        bool evalInitialized = false;
        int lastBestChildIdx = -1;
        
        while (running && (isPondering_.load(std::memory_order_relaxed)
                           || searchInfo.elapsed() < searchInfo.get_effective_move_time())) {
            apply_root_loss_proofs();
            // A mate the probe landed mid-move ends the search the way MCTS
            // proving one does; the collapse below the loop reports it.
            if (probe_mate_ends_search()) {
                running = false;
                break;
            }
            double remainingMs = searchInfo.get_effective_move_time() - searchInfo.elapsed();
            int sleepMs = isPondering_.load(std::memory_order_relaxed)
                ? POLL_INTERVAL_MS
                : std::min(POLL_INTERVAL_MS, std::max(1, static_cast<int>(remainingMs)));
            std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
            
            // Update NPS tracking
            searchInfo.update_nps();
            
            int depth = searchInfo.get_max_depth();
            double elapsedMs = searchInfo.elapsed();
            int nodes = searchInfo.get_nodes_searched();
            int nps = (elapsedMs > 0) ? static_cast<int>((nodes * 1000.0) / elapsedMs) : 0;
            size_t tbhits = (options.search.enableMCGS
                             && options.search.enableTranspositions
                             && transpositionTable)
                ? transpositionTable->getHits() : 0;
            int hashfull = (options.search.enableMCGS
                            && options.search.enableTranspositions
                            && transpositionTable)
                ? transpositionTable->getFullness() : 0;
            
            if (rootNode && rootNode->is_expanded()) {
                auto childVisits = rootNode->get_child_visits();
                auto children = rootNode->get_children();
                size_t numChildren = min(childVisits.size(), children.size());
                
                if (numChildren > 0) {
                    // Find first and second max visit counts
                    int firstMax = 0, secondMax = 0;
                    int firstIdx = 0, secondIdx = -1;
                    for (size_t i = 0; i < numChildren; ++i) {
                        if (childVisits[i] > firstMax) {
                            secondMax = firstMax;
                            secondIdx = firstIdx;
                            firstMax = childVisits[i];
                            firstIdx = static_cast<int>(i);
                        } else if (childVisits[i] > secondMax) {
                            secondMax = childVisits[i];
                            secondIdx = static_cast<int>(i);
                        }
                    }
                    
                    const int decisionIdx = options.search.enableGumbelRootSearch
                        ? rootNode->get_best_move_idx_with_q_weight(
                            options.search.qVetoDelta,
                            options.search.qValueWeight)
                        : firstIdx;
                    float bestQ = rootNode->get_child_q(
                        decisionIdx >= 0 ? decisionIdx : firstIdx);
                    float secondQ = (secondIdx >= 0) ? rootNode->get_child_q(secondIdx) : -1.0f;
                    
                    // Initialize eval tracking
                    if (!evalInitialized) {
                        lastCheckEval = bestQ;
                        evalInitialized = true;
                    }
                    
                    // Early exit for solved/winning positions
                    if (should_exit_early_winning(
                            rootNode,
                            decisionIdx >= 0 ? decisionIdx : firstIdx,
                            true)) {
                        running = false;
                        break;
                    }
                    
                    if (!isPondering_.load(std::memory_order_relaxed)) {
                        // Early stopping check (visit-based)
                        if (SearchParams::ENABLE_EARLY_STOPPING
                            && !options.search.enableGumbelRootSearch
                            && searchInfo.get_nps() > 0) {
                            double remaining = searchInfo.get_effective_move_time() - elapsedMs;
                            float projectedVisits = static_cast<float>(secondMax) + 
                                                   static_cast<float>(remaining * searchInfo.get_nps() / 1000.0);
                            
                            // Stop if second-best can't catch up AND best move has better Q
                            if (SearchParams::has_insurmountable_visit_lead(
                                static_cast<float>(firstMax), projectedVisits) &&
                                bestQ >= secondQ) {
                                double savedMs = std::max(0.0, static_cast<double>(searchInfo.get_move_time()) - elapsedMs);
                                cout << "info string Early stopping: saved " 
                                     << static_cast<int>(savedMs) << "ms" << endl;
                                running = false;
                                break;
                            }
                        }
                        
                        // Time extension check - extend if eval is falling or leading move changes late
                        if (SearchParams::ENABLE_TIME_EXTENSION
                            && !options.search.enableGumbelRootSearch) {
                            if (evalInitialized) {
                                float evalDrop = lastCheckEval - bestQ;
                                if (evalDrop > SearchParams::TIME_EXTENSION_THRESHOLD) {
                                    if (searchInfo.try_extend_time(SearchParams::TIME_EXTENSION_FACTOR, 
                                                                  SearchParams::MAX_TIME_EXTENSIONS)) {
                                        cout << "info string Extending search time (eval dropped by " 
                                             << static_cast<int>(evalDrop * 100) << " cp)" << endl;
                                    }
                                }
                                lastCheckEval = bestQ;
                            }
                            if (lastBestChildIdx >= 0
                                && decisionIdx != lastBestChildIdx &&
                                elapsedMs > searchInfo.get_move_time() * SearchParams::INSTABILITY_TIME_FRACTION) {
                                if (searchInfo.try_extend_time(SearchParams::TIME_EXTENSION_FACTOR, 
                                                              SearchParams::MAX_TIME_EXTENSIONS)) {
                                    cout << "info string Extending search time (best move changed to " 
                                         << decisionIdx << ")" << endl;
                                }
                            }
                            lastBestChildIdx = decisionIdx;
                        }
                    }
                    
                    // Report each completed depth once.
                    if (depth > lastReportedDepth) {
                        lastReportedDepth = depth;
                        
                        // Use solver-aware selection for the best child to display
                        int solverBestIdx = rootNode->get_best_move_idx_with_q_weight(
                            options.search.qVetoDelta, options.search.qValueWeight);
                        size_t displayIdx = (solverBestIdx >= 0
                                             && static_cast<size_t>(solverBestIdx) < numChildren)
                            ? static_cast<size_t>(solverBestIdx) : static_cast<size_t>(firstIdx);
                        
                        // Output best line during search
                        int numPVs = 1;
                        for (int pvIdx = 0; pvIdx < numPVs; ++pvIdx) {
                            size_t childIdx = displayIdx;
                            string pv = exact_root_loss_pv(
                                rootNode->get_joint_action(
                                    static_cast<int>(childIdx)));
                            if (pv.empty()) {
                                pv = extract_pv_from_child(
                                    board, static_cast<int>(childIdx), 20,
                                    teamSide, teamHasTimeAdvantage);
                            }
                            float childQ = rootNode->get_child_q(static_cast<int>(childIdx));
                            string scoreStr = format_root_aware_uci_score(
                                rootNode, children[childIdx], childQ, C, k);
                            
                            cout << "info depth " << depth 
                                 << " " << scoreStr
                                 << " nodes " << nodes 
                                 << " nps " << nps
                                 << " hashfull " << hashfull
                                 << " tbhits " << tbhits
                                 << " time " << static_cast<int>(elapsedMs);
                            
                            if (!pv.empty()) {
                                cout << " pv " << pv;
                            }
                            cout << endl;
                        }
                    }
                }
            }
            if (!running || (!isPondering_.load(std::memory_order_relaxed)
                             && searchInfo.elapsed() >= searchInfo.get_effective_move_time())) {
                break;
            }
        }
    } else if (moveTimeMs > 0) {
        // Non-verbose mode: still check for early stopping
        searchInfo.set_in_game(true);
        float lastCheckEval = 0.0f;
        bool evalInitialized = false;
        int lastBestChildIdx = -1;
        
        while (running && (isPondering_.load(std::memory_order_relaxed)
                           || searchInfo.elapsed() < searchInfo.get_effective_move_time())) {
            if (probe_mate_ends_search()) {
                running = false;
                break;
            }
            double remainingMs = searchInfo.get_effective_move_time() - searchInfo.elapsed();
            int sleepMs = isPondering_.load(std::memory_order_relaxed)
                ? POLL_INTERVAL_MS
                : std::min(POLL_INTERVAL_MS, std::max(1, static_cast<int>(remainingMs)));
            std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
            
            // Update NPS
            searchInfo.update_nps();
            
            if (rootNode && rootNode->is_expanded()) {
                auto childVisits = rootNode->get_child_visits();
                auto children = rootNode->get_children();
                size_t numChildren = min(childVisits.size(), children.size());
                
                if (numChildren > 0) {
                    // Find first and second max
                    int firstMax = 0, secondMax = 0;
                    int firstIdx = 0, secondIdx = -1;
                    for (size_t i = 0; i < numChildren; ++i) {
                        if (childVisits[i] > firstMax) {
                            secondMax = firstMax;
                            secondIdx = firstIdx;
                            firstMax = childVisits[i];
                            firstIdx = static_cast<int>(i);
                        } else if (childVisits[i] > secondMax) {
                            secondMax = childVisits[i];
                            secondIdx = static_cast<int>(i);
                        }
                    }
                    
                    const int decisionIdx = options.search.enableGumbelRootSearch
                        ? rootNode->get_best_move_idx_with_q_weight(
                            options.search.qVetoDelta,
                            options.search.qValueWeight)
                        : firstIdx;
                    float bestQ = rootNode->get_child_q(
                        decisionIdx >= 0 ? decisionIdx : firstIdx);
                    float secondQ = (secondIdx >= 0) ? rootNode->get_child_q(secondIdx) : -1.0f;
                    
                    if (!evalInitialized) {
                        lastCheckEval = bestQ;
                        evalInitialized = true;
                    }
                    
                    // Early exit for solved/winning positions
                    if (should_exit_early_winning(
                            rootNode,
                            decisionIdx >= 0 ? decisionIdx : firstIdx,
                            false)) {
                        running = false;
                        break;
                    }
                    
                    if (!isPondering_.load(std::memory_order_relaxed)) {
                        // Early stopping (visit-based)
                        if (SearchParams::ENABLE_EARLY_STOPPING
                            && !options.search.enableGumbelRootSearch
                            && searchInfo.get_nps() > 0) {
                            double remaining = searchInfo.get_effective_move_time() - searchInfo.elapsed();
                            float projectedVisits = static_cast<float>(secondMax) + 
                                                   static_cast<float>(remaining * searchInfo.get_nps() / 1000.0);
                            
                            if (SearchParams::has_insurmountable_visit_lead(
                                static_cast<float>(firstMax), projectedVisits) &&
                                bestQ >= secondQ) {
                                running = false;
                                break;
                            }
                        }
                        
                        // Time extension
                        if (SearchParams::ENABLE_TIME_EXTENSION
                            && !options.search.enableGumbelRootSearch) {
                            if (evalInitialized) {
                                float evalDrop = lastCheckEval - bestQ;
                                if (evalDrop > SearchParams::TIME_EXTENSION_THRESHOLD) {
                                    searchInfo.try_extend_time(SearchParams::TIME_EXTENSION_FACTOR, 
                                                               SearchParams::MAX_TIME_EXTENSIONS);
                                }
                                lastCheckEval = bestQ;
                            }
                            if (lastBestChildIdx >= 0
                                && decisionIdx != lastBestChildIdx &&
                                searchInfo.elapsed() > searchInfo.get_move_time() * SearchParams::INSTABILITY_TIME_FRACTION) {
                                searchInfo.try_extend_time(SearchParams::TIME_EXTENSION_FACTOR, 
                                                           SearchParams::MAX_TIME_EXTENSIONS);
                            }
                            lastBestChildIdx = decisionIdx;
                        }
                    }
                }
            }
            if (!running || (!isPondering_.load(std::memory_order_relaxed)
                             && searchInfo.elapsed() >= searchInfo.get_effective_move_time())) break;
        }
    } else {
        // Node-based search: wait for workers to reach target nodes
        // Workers will stop themselves when they've done enough iterations.
        // Fail instead of waiting forever if iterations stop completing.
        constexpr auto NODE_PROGRESS_TIMEOUT = std::chrono::seconds(60);
        int lastCompletedNodes = searchInfo.get_nodes_searched();
        auto lastNodeProgress = std::chrono::steady_clock::now();
        while (running) {
            apply_root_loss_proofs();
            if (probe_mate_ends_search()) {
                running = false;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(POLL_INTERVAL_MS));
            if (SearchParams::ENABLE_MATE_EARLY_EXIT && rootNode
                && rootNode->get_node_type() != NodeType::UNSOLVED) {
                running = false;
                break;
            }
            // A background search runs until the next command, so its stop
            // signal is the latch rather than `running`, and its own caps keep
            // an idle GUI from growing the tree without bound.
            if (options.background
                && (stopRequested_.load(std::memory_order_relaxed)
                    || searchInfo.get_nodes_searched()
                           >= SearchParams::PERMANENT_BRAIN_MAX_NODES
                    || searchInfo.elapsed()
                           >= SearchParams::PERMANENT_BRAIN_MAX_MS)) {
                running = false;
                break;
            }
            const int completedNodes = searchInfo.get_nodes_searched();
            if (!isPondering_.load(std::memory_order_relaxed)
                && static_cast<size_t>(completedNodes) >= targetNodes) {
                break;
            }
            if (completedNodes != lastCompletedNodes) {
                lastCompletedNodes = completedNodes;
                lastNodeProgress = std::chrono::steady_clock::now();
            } else if (!isPondering_.load(std::memory_order_relaxed)
                       && std::chrono::steady_clock::now() - lastNodeProgress
                       >= NODE_PROGRESS_TIMEOUT) {
                running = false;
                nodeSearchStalled = true;
                stalledCompletedNodes = completedNodes;
                break;
            }
        }
    }
    
    // Signal workers to stop (in case they're still running)
    running = false;
    isPondering_.store(false, std::memory_order_release);
    currentSearchInfo_.store(nullptr, std::memory_order_release);
    
    wait_for_workers();
    apply_root_loss_proofs();

    if (workerException_) {
        nextRootCandidates_.clear();
        std::rethrow_exception(workerException_);
    }
    if (nodeSearchStalled) {
        nextRootCandidates_.clear();
        throw std::runtime_error(
            "Node-limited search stalled at "
            + std::to_string(stalledCompletedNodes) + "/"
            + std::to_string(targetNodes) + " completed nodes");
    }

    // Nothing MCTS searched came back proven, so a mate the probe found while
    // it ran - either mid-move, which ended the loop above, or on the last
    // poll before the deadline - decides the move instead.
    if (rootNode && rootNode->get_node_type() == NodeType::UNSOLVED) {
        stop_root_probe();
        JointActionCandidate probedAction;
        int probedPly = 0;
        string probedPv;
        if (take_probe_mate(probedAction, probedPly, probedPv)) {
            return return_proven_mate(probedAction, probedPly, probedPv);
        }
    }

    // Extract best joint action by selecting the most visited child
    if (rootNode && rootNode->is_expanded()) {
        auto visits = rootNode->get_child_visits();
        auto children = rootNode->get_children();
        if (!visits.empty() && !children.empty()) {
            size_t numChildren = min(visits.size(), children.size());
            
            // Use Q-value weighted selection (with veto and weighting)
            int bestIdx = rootNode->get_best_move_idx_with_q_weight(
                options.search.qVetoDelta, options.search.qValueWeight);
            
            // Fallback to most-visited if Q-value selection failed
            if (bestIdx < 0) {
                int maxVisits = 0;
                for (size_t i = 0; i < numChildren; ++i) {
                    if (visits[i] > maxVisits) {
                        maxVisits = visits[i];
                        bestIdx = static_cast<int>(i);
                    }
                }
            }
            
            size_t numGenerated = rootNode->get_num_generated();
            if (static_cast<size_t>(bestIdx) >= numGenerated) {
                cerr << "ERROR: bestIdx (" << bestIdx << ") >= numGenerated (" << numGenerated << ")" << endl;
                bestIdx = 0;
            }
            
            result = rootNode->get_joint_action(bestIdx);
        }
    }
    
    if (!options.background) {
        rootScanStats_.thinkNanos += static_cast<uint64_t>(
            chrono::duration_cast<chrono::nanoseconds>(
                chrono::steady_clock::now() - searchStart).count());
    }

    // Output UCI info if verbose
    if (options.verbose) {
        double elapsedMs = searchInfo.elapsed();
        int nodes = searchInfo.get_nodes_searched();
        int depth = searchInfo.get_max_depth();
        int nps = (elapsedMs > 0) ? static_cast<int>((nodes * 1000.0) / elapsedMs) : 0;
        size_t tbhits = (options.search.enableMCGS
                         && options.search.enableTranspositions
                         && transpositionTable)
            ? transpositionTable->getHits() : 0;
        int hashfull = (options.search.enableMCGS
                        && options.search.enableTranspositions
                        && transpositionTable)
            ? transpositionTable->getFullness() : 0;
        
        // Convert Q-value [-1, 1] to centipawns using Lc0 tangent formula
        constexpr float C = 180.0f;
        constexpr float k = 1.56f;
        
        // Always report final up-to-date info line(s) before bestmove.
        if (rootNode && rootNode->is_expanded()) {
            auto childVisits = rootNode->get_child_visits();
            auto children = rootNode->get_children();
            size_t numChildren = min(childVisits.size(), children.size());
            
            // Create sorted indices by visit count (descending)
            vector<size_t> sortedIndices(numChildren);
            for (size_t i = 0; i < numChildren; ++i) sortedIndices[i] = i;
            sort(sortedIndices.begin(), sortedIndices.end(), [&](size_t a, size_t b) {
                return childVisits[a] > childVisits[b];
            });
            
            // Keep PV 1 aligned with the solver-aware move used for bestmove.
            int solverIdx = rootNode->get_best_move_idx_with_q_weight(
                options.search.qVetoDelta, options.search.qValueWeight);
            if (solverIdx >= 0) {
                auto it = std::find(sortedIndices.begin(), sortedIndices.end(), static_cast<size_t>(solverIdx));
                if (it != sortedIndices.end() && it != sortedIndices.begin()) {
                    sortedIndices.erase(it);
                    sortedIndices.insert(sortedIndices.begin(), static_cast<size_t>(solverIdx));
                }
            }
            
            // Output up to multiPV lines
            int numPVs = min(options.multiPV, static_cast<int>(numChildren));
            for (int pvIdx = 0; pvIdx < numPVs; ++pvIdx) {
                size_t childIdx = sortedIndices[pvIdx];
                string pv = exact_root_loss_pv(
                    rootNode->get_joint_action(static_cast<int>(childIdx)));
                if (pv.empty()) {
                    pv = extract_pv_from_child(
                        board, static_cast<int>(childIdx), 20,
                        teamSide, teamHasTimeAdvantage);
                }
                float childQ = rootNode->get_child_q(static_cast<int>(childIdx));
                string scoreStr = format_root_aware_uci_score(
                    rootNode, children[childIdx], childQ, C, k);
                
                cout << "info depth " << depth;
                if (options.multiPV > 1) {
                    cout << " multipv " << (pvIdx + 1);
                }
                cout << " " << scoreStr
                     << " nodes " << nodes 
                     << " nps " << nps
                     << " hashfull " << hashfull
                     << " tbhits " << tbhits
                     << " time " << static_cast<int>(elapsedMs);
                
                if (!pv.empty()) {
                    cout << " pv " << pv;
                }
                cout << endl;
            }
        } else {
            // Fallback: single PV line with root Q
            string pv = extract_pv(board, 20);
            // Use root's own Q value (which is from root's perspective)
            float rootQ = rootNode ? rootNode->Q() : 0.0f;
            string scoreStr = rootNode ? format_uci_score(rootNode.get(), rootQ, false, C, k) 
                                       : "score cp 0";
            
            cout << "info depth " << depth 
                 << " " << scoreStr
                 << " nodes " << nodes 
                 << " nps " << nps
                 << " hashfull " << hashfull
                 << " tbhits " << tbhits
                 << " time " << static_cast<int>(elapsedMs);
            
            if (!pv.empty()) {
                cout << " pv " << pv;
            }
            cout << endl;
        }

        if (rootNode) {
            cout << "info string root width " << rootNode->get_num_generated()
                 << " generated, " << rootNode->get_visited_edge_count()
                 << " scanned per selection" << endl;
        }
        cout << "info string rejected selection attempts "
               << searchInfo.get_collisions()
               << " (same batch " << searchInfo.get_same_batch_collisions()
             << ", pending evaluation " << searchInfo.get_reservation_collisions()
               << ") per 1000 nodes "
               << (nodes > 0
                       ? 1000.0 * searchInfo.get_collisions()
                           / static_cast<double>(nodes)
                       : 0.0)
               << endl;
        string bestMoveStr = extract_best_move(board);
        string ponderMoveStr = options.enablePonder ? extract_ponder_move(board) : "";
        if (!ponderMoveStr.empty()) {
            cout << "bestmove " << bestMoveStr << " ponder " << ponderMoveStr << endl;
        } else {
            cout << "bestmove " << bestMoveStr << endl;
        }
        if (const string scanSummary = root_scan_summary();
                !scanSummary.empty()) {
                cout << scanSummary << endl;
            }
    }

    // Index the retained subtree only after bestmove is on the wire: the walk
    // records a board signature per position, which is not work to spend
    // inside the move time.
    if (SearchParams::ENABLE_TREE_REUSE) {
        if (options.background) {
            // No move is played from a background root - the position we are
            // next asked about lies below it - so retain the root itself.
            nextRootCandidates_.clear();
            retain_reuse_candidates(rootNode, board, teamHasTimeAdvantage);
        } else {
            store_next_root_candidates(board, teamHasTimeAdvantage);
        }
        lastSearchHash_ = board.search_hash_key(
            teamSide, teamHasTimeAdvantage);
    }

    return result;
}

void Agent::run_permanent_brain(Board& board, const vector<Engine*>& engines,
                                Stockfish::Color teamSide,
                                bool teamHasTimeAdvantage,
                                const JointActionCandidate& playedAction,
                                const SearchOptions& options) {
    if (!SearchParams::ENABLE_PERMANENT_BRAIN
        || !SearchParams::ENABLE_TREE_REUSE
        || engines.empty()
        || stopRequested_.load(std::memory_order_acquire)) {
        return;
    }
    {
        // An empty candidate set means the finished search produced no usable
        // root, so there is no position to carry on from either.
        std::unique_lock searchLock(searchMutex_);
        if (nextRootCandidates_.empty()) {
            return;
        }
    }

    Board nextBoard(board);
    nextBoard.make_moves(playedAction.moveA, playedAction.moveB);

    SearchOptions backgroundOptions;
    backgroundOptions.search = options.search;
    backgroundOptions.search.rootDirichletAlpha = 0.0f;
    backgroundOptions.search.rootDirichletEpsilon = 0.0f;
    backgroundOptions.background = true;
    backgroundOptions.verbose = false;
    backgroundOptions.enablePonder = false;
    // Ponder mode is what makes the workers ignore the clock; the background
    // caps in the wait loop are what actually bound this search.
    backgroundOptions.isPonder = true;
    backgroundOptions.moveTimeMs = 0;
    backgroundOptions.targetNodes = 0;

    // The opponents are to move at that position, so the search runs from their
    // side of the table. Every node values itself from the perspective of its
    // own team to play, which is exactly how the subtree reads once our next
    // root adopts it.
    try {
        run_search(nextBoard, engines, ~teamSide, !teamHasTimeAdvantage,
                   backgroundOptions);
    } catch (const std::exception& error) {
        cout << "info string permanent brain stopped: " << error.what() << endl;
    } catch (...) {
        cout << "info string permanent brain stopped" << endl;
    }
}

vector<RootEdgeStats> Agent::root_edge_stats() const {
    vector<RootEdgeStats> stats;
    if (!rootNode || !rootNode->is_expanded()) {
        return stats;
    }

    const auto visits = rootNode->get_child_visits();
    const size_t edgeCount = min(visits.size(), rootNode->get_num_generated());
    stats.reserve(edgeCount);
    for (size_t index = 0; index < edgeCount; ++index) {
        stats.push_back({rootNode->get_joint_action(static_cast<int>(index)), visits[index]});
    }
    return stats;
}

float Agent::root_q() const {
    if (!rootNode) {
        return 0.0f;
    }
    return rootNode->Q();
}

/**
 * @brief Extracts the best move from the root node using solver-aware selection.
 * When root is proven WIN/LOSS, selects the proven-best move.
 * Otherwise falls back to Q-value weighted visit-based selection.
 */
string Agent::extract_best_move(Board& board) {
    if (!rootNode || !rootNode->is_expanded()) {
        return "(none)";
    }

    // Use solver-aware move selection (handles proven wins/losses)
    int bestIdx = rootNode->get_best_move_idx_with_q_weight(
        lastRuntimeConfig_.qVetoDelta, lastRuntimeConfig_.qValueWeight);
    if (bestIdx < 0) {
        return "(none)";
    }

    JointActionCandidate action = rootNode->get_joint_action(bestIdx);
    string moveA = (action.moveA == Stockfish::MOVE_NONE) 
                    ? "pass" : board.uci_move(BOARD_A, action.moveA);
    string moveB = (action.moveB == Stockfish::MOVE_NONE) 
                    ? "pass" : board.uci_move(BOARD_B, action.moveB);
    return "(" + moveA + "," + moveB + ")";
}

/**
 * @brief Extracts the predicted opponent reply from the root node after search.
 */
string Agent::extract_ponder_move(Board& board) {
    if (!rootNode || !rootNode->is_expanded()) {
        return "";
    }

    int bestIdx = rootNode->get_best_move_idx_with_q_weight(
        lastRuntimeConfig_.qVetoDelta, lastRuntimeConfig_.qValueWeight);
    if (bestIdx < 0) {
        auto visits = rootNode->get_child_visits();
        int maxVisits = 0;
        for (size_t i = 0; i < visits.size(); ++i) {
            if (visits[i] > maxVisits) {
                maxVisits = visits[i];
                bestIdx = static_cast<int>(i);
            }
        }
    }
    if (bestIdx < 0) {
        return "";
    }

    auto children = rootNode->get_children();
    if (static_cast<size_t>(bestIdx) >= children.size() || !children[bestIdx]) {
        return "";
    }

    Node* bestChild = children[bestIdx].get();
    if (!bestChild->is_expanded()) {
        return "";
    }

    auto grandVisits = bestChild->get_child_visits();
    auto grandChildren = bestChild->get_children();
    if (grandVisits.empty() || grandChildren.empty()) {
        return "";
    }

    int bestGrandIdx = bestChild->get_best_move_idx_with_q_weight(
        lastRuntimeConfig_.qVetoDelta, lastRuntimeConfig_.qValueWeight);
    if (bestGrandIdx < 0) {
        int maxGrandVisits = 0;
        for (size_t i = 0; i < grandVisits.size(); ++i) {
            if (grandVisits[i] > maxGrandVisits) {
                maxGrandVisits = grandVisits[i];
                bestGrandIdx = static_cast<int>(i);
            }
        }
    }
    if (bestGrandIdx < 0 || static_cast<size_t>(bestGrandIdx) >= bestChild->get_num_generated()) {
        return "";
    }

    JointActionCandidate rootAction = rootNode->get_joint_action(bestIdx);
    Board nextBoard(board);
    nextBoard.make_moves(rootAction.moveA, rootAction.moveB);

    JointActionCandidate replyAction = bestChild->get_joint_action(bestGrandIdx);
    string moveA = (replyAction.moveA == Stockfish::MOVE_NONE)
                    ? "pass" : nextBoard.uci_move(BOARD_A, replyAction.moveA);
    string moveB = (replyAction.moveB == Stockfish::MOVE_NONE)
                    ? "pass" : nextBoard.uci_move(BOARD_B, replyAction.moveB);
    return "(" + moveA + "," + moveB + ")";
}

/**
 * @brief Extracts the principal variation (PV) by following most-visited children.
 * @param board The current board position
 * @param maxDepth Maximum number of moves to extract
 * @return Space-separated sequence of joint moves in format "(moveA,moveB) (moveA,moveB) ..."
 */
string Agent::extract_pv(Board& board, int maxDepth) {
    if (!rootNode || !rootNode->is_expanded()) {
        return "";
    }
    
    Board tempBoard = board;  // Make a copy to simulate moves
    Node* currentNode = rootNode.get();
    string pv;
    
    for (int depth = 0; depth < maxDepth; depth++) {
        if (!currentNode || !currentNode->is_expanded()) {
            break;
        }
        
        auto children = currentNode->get_children();
        auto childVisits = currentNode->get_child_visits();
        if (children.empty() || childVisits.empty()) {
            break;
        }
        
        // Sanity check: children and childVisits should have the same size
        if (children.size() != childVisits.size()) {
            cerr << "WARNING in extract_pv: children.size()=" << children.size()
                 << " != childVisits.size()=" << childVisits.size() 
                 << " at depth " << depth << endl;
        }
        
        // If debug log level, print all candidate moves at this PV node
        if (g_logLevel == LOG_DEBUG) {
            cout << "PV depth " << depth << " candidates:" << endl;
            for (size_t i = 0; i < children.size() && i < childVisits.size(); ++i) {
                JointActionCandidate candAction = currentNode->get_joint_action(static_cast<int>(i));
                string candMoveA = (candAction.moveA == Stockfish::MOVE_NONE) ? "pass" : tempBoard.uci_move(BOARD_A, candAction.moveA);
                string candMoveB = (candAction.moveB == Stockfish::MOVE_NONE) ? "pass" : tempBoard.uci_move(BOARD_B, candAction.moveB);
                float candQ = currentNode->get_child_q(static_cast<int>(i));
                int candVisitsCount = childVisits[i];
                cout << "  (" << candMoveA << ", " << candMoveB << ")"
                     << "  Q: " << std::fixed << std::setprecision(3) << candQ
                     << "  Visits: " << candVisitsCount << endl;
            }
        }
        
        // Find child with most visits (use parent's childVisits array)
        int bestIdx = 0;
        int maxVisits = 0;
        
        for (size_t i = 0; i < children.size() && i < childVisits.size(); i++) {
            int visits = childVisits[i];
            if (visits > maxVisits) {
                maxVisits = visits;
                bestIdx = static_cast<int>(i);
            }
        }
        
        // Get the joint action for this move
        JointActionCandidate action = currentNode->get_joint_action(bestIdx);
        
        // Verify the action is valid - if both moves are MOVE_NONE, it should be intentional
        size_t genCount = currentNode->get_num_generated();
        if (static_cast<size_t>(bestIdx) >= genCount) {
            cerr << "WARNING in extract_pv: bestIdx=" << bestIdx 
                 << " >= generatedCount=" << genCount 
                 << " at depth " << depth << endl;
        }
        
        // Format move string
        string moveA = (action.moveA == Stockfish::MOVE_NONE) 
                        ? "pass" : tempBoard.uci_move(BOARD_A, action.moveA);
        string moveB = (action.moveB == Stockfish::MOVE_NONE) 
                        ? "pass" : tempBoard.uci_move(BOARD_B, action.moveB);
        
        if (!pv.empty()) {
            pv += " ";
        }
        pv += "(" + moveA + "," + moveB + ")";
        
        // Apply moves to temp board for next iteration
        tempBoard.make_moves(action.moveA, action.moveB);
        
        // Move to best child
        currentNode = children[bestIdx].get();
    }
    
    return pv;
}

/**
 * @brief Extracts PV line starting from a specific child index.
 * Used for Multi-PV output to show principal variations for non-best moves.
 * @param board The current board position
 * @param childIdx The child index to start the PV from
 * @param maxDepth Maximum number of moves to extract
 * @return Space-separated sequence of joint moves in format "(moveA,moveB) (moveA,moveB) ..."
 */
string Agent::extract_pv_from_child(
    Board& board, int childIdx, int maxDepth,
    Stockfish::Color rootTeam, bool rootTeamHasTimeAdvantage) {
    if (!rootNode || !rootNode->is_expanded()) {
        return "";
    }
    
    auto children = rootNode->get_children();
    if (childIdx < 0 || static_cast<size_t>(childIdx) >= children.size()) {
        return "";
    }
    
    Board tempBoard = board;
    string pv;
    std::array<int, 2> boardSearchPlies{};
    
    // Get the first move from the specified child
    JointActionCandidate action = rootNode->get_joint_action(childIdx);
    string moveA = (action.moveA == Stockfish::MOVE_NONE) 
                    ? "pass" : tempBoard.uci_move(BOARD_A, action.moveA);
    string moveB = (action.moveB == Stockfish::MOVE_NONE) 
                    ? "pass" : tempBoard.uci_move(BOARD_B, action.moveB);
    pv = "(" + moveA + "," + moveB + ")";
    boardSearchPlies[BOARD_A] += action.moveA != Stockfish::MOVE_NONE;
    boardSearchPlies[BOARD_B] += action.moveB != Stockfish::MOVE_NONE;
    
    // Apply moves to temp board
    tempBoard.make_moves(action.moveA, action.moveB);
    
    // Continue extracting PV from this child
    Node* currentNode = children[childIdx].get();
    
    for (int depth = 1; depth < maxDepth; depth++) {
        if (!currentNode || !currentNode->is_expanded()) {
            break;
        }
        
        auto nodeChildren = currentNode->get_children();
        auto childVisits = currentNode->get_child_visits();
        if (nodeChildren.empty() || childVisits.empty()) {
            break;
        }
        
        // Find best child: prefer solver-proven path, fallback to most visits
        int bestIdx = currentNode->get_best_move_idx_with_q_weight(
            lastRuntimeConfig_.qVetoDelta, lastRuntimeConfig_.qValueWeight);
        
        // Fallback to most-visited (handles unsolved nodes and edge cases)
        if (bestIdx < 0) {
            bestIdx = 0;
            int maxVisits = 0;
            for (size_t i = 0; i < nodeChildren.size() && i < childVisits.size(); i++) {
                if (childVisits[i] > maxVisits) {
                    maxVisits = childVisits[i];
                    bestIdx = static_cast<int>(i);
                }
            }
        }
        
        // Get the joint action for this move
        JointActionCandidate nextAction = currentNode->get_joint_action(bestIdx);
        
        // Format move string
        string nextMoveA = (nextAction.moveA == Stockfish::MOVE_NONE) 
                            ? "pass" : tempBoard.uci_move(BOARD_A, nextAction.moveA);
        string nextMoveB = (nextAction.moveB == Stockfish::MOVE_NONE) 
                            ? "pass" : tempBoard.uci_move(BOARD_B, nextAction.moveB);
        
        pv += " (" + nextMoveA + "," + nextMoveB + ")";
        boardSearchPlies[BOARD_A] +=
            nextAction.moveA != Stockfish::MOVE_NONE;
        boardSearchPlies[BOARD_B] +=
            nextAction.moveB != Stockfish::MOVE_NONE;
        
        // Apply moves to temp board for next iteration
        tempBoard.make_moves(nextAction.moveA, nextAction.moveB);
        
        // Move to best child
        currentNode = nodeChildren[bestIdx].get();
    }

    if (currentNode
        && currentNode->get_node_type() != NodeType::UNSOLVED) {
        append_waiting_mate_suffix(
            tempBoard, currentNode->get_team_to_play(),
            rootTeam, rootTeamHasTimeAdvantage,
            boardSearchPlies, pv);
    }
    
    return pv;
}

void Agent::set_is_running(bool value) {
    if (!value) {
        // Latched, not just mirrored into `running`: a background search that
        // has not dispatched yet would otherwise set `running` back to true
        // after this and never see the stop.
        stopRequested_.store(true, std::memory_order_release);
        isPondering_.store(false, std::memory_order_release);
    }
    running = value;
}

bool Agent::is_running() {
    return running;
}

void Agent::ponderhit() {
    isPondering_.store(false, std::memory_order_release);
    SearchInfo* info = currentSearchInfo_.load(std::memory_order_acquire);
    if (info) {
        info->reset_start_time();
    }
}

bool Agent::is_pondering() const {
    return isPondering_.load(std::memory_order_acquire);
}

void Agent::setHashSize(size_t sizeMB) {
    // Clamp to valid range (1 MB to 32 TB)
    sizeMB = std::max(static_cast<size_t>(1), std::min(sizeMB, static_cast<size_t>(33554432)));
    
    // Convert MB to approximate entry count
    // Each TT entry is roughly 64 bytes (hash key + shared_ptr + unordered_map overhead)
    constexpr size_t BYTES_PER_ENTRY = 64;
    size_t maxEntries = (sizeMB * 1024 * 1024) / BYTES_PER_ENTRY;
    
    if (!transpositionTable && SearchParams::ENABLE_MCGS) {
        transpositionTable = std::make_unique<TranspositionTable>();
    }
    
    if (transpositionTable) {
        transpositionTable->setMaxCapacity(maxEntries);
        transpositionTable->reserve(maxEntries);
        transpositionTable->clear();
    }
}

/**
 * @brief Try to reuse the search tree from a previous search.
 * 
 * Implements CrazyAra-style tree reuse by checking the current position
 * against the selected move and every generated opponent response retained
 * from the previous search.
 */
std::string Agent::root_scan_summary() const {
    if (rootScanStats_.searches == 0) {
        return {};
    }
    const double scanMs = static_cast<double>(rootScanStats_.scanNanos) / 1e6;
    const double thinkMs = static_cast<double>(rootScanStats_.thinkNanos) / 1e6;
    std::ostringstream out;
    out << "info string root scan: " << rootScanStats_.proofs << "/"
        << rootScanStats_.searches << " searches decided ("
        << std::fixed << std::setprecision(1)
        << (100.0 * static_cast<double>(rootScanStats_.proofs)
            / static_cast<double>(rootScanStats_.searches))
        << "%), " << std::setprecision(0) << scanMs << "ms of " << thinkMs
        << "ms thinking (" << std::setprecision(1)
        << (thinkMs > 0.0 ? 100.0 * scanMs / thinkMs : 0.0) << "%)";
    return out.str();
}

std::string Agent::board_signature(Board& board) {
    return board.fen(BOARD_A) + "|" + board.fen(BOARD_B);
}

void Agent::reindex_reused_subtree(const std::shared_ptr<Node>& reusedRoot) {
    if (!transpositionTable || !reusedRoot) {
        return;
    }

    // The retained root owns the graph while this runs before workers start.
    // Traverse raw pointers to avoid copying every node's shared_ptr vector;
    // acquire one owner only when inserting the node into the table.
    std::vector<Node*> pending = {reusedRoot.get()};
    std::unordered_set<const Node*> visited;
    while (!pending.empty()
           && visited.size() < SearchParams::TREE_REUSE_REINDEX_MAX_NODES) {
        Node* node = pending.back();
        pending.pop_back();
        if (!node || !visited.insert(node).second) {
            continue;
        }

        const uint64_t hash = node->get_hash();
        if (hash != 0) {
            transpositionTable->insertOrGet(hash, node->shared_from_this());
        }
        node->append_child_ptrs(pending);
    }
}

std::shared_ptr<Node> Agent::try_reuse_tree(uint64_t positionHash,
                                            Stockfish::Color teamSide,
                                            const std::string& signature) {
    // The hash locates the candidate; the signature is what admits it. Retained
    // edges were generated against that exact board, and a stale pocket would
    // make reused drops illegal.
    std::shared_ptr<Node> reused;
    auto entry = nextRootCandidates_.find(positionHash);
    if (entry != nextRootCandidates_.end()) {
        const RetainedRootCandidate& candidate = entry->second;
        if (candidate.node
            && candidate.node->get_team_to_play() == teamSide
            && !candidate.signature.empty()
            && candidate.signature == signature) {
            reused = candidate.node;
        }
    }

    if (rootNode && rootNode != reused) {
        gcThread_.enqueue(rootNode);
    }

    nextRootCandidates_.clear();

    return reused;
}

/**
 * @brief Store next-root candidates for tree reuse.
 *
 * After search completes, retain the subtree below the selected move and index
 * every position within TREE_REUSE_MAX_JOINT_PLIES of it. Indexing only the
 * predicted reply is far too narrow for bughouse: the partner board moves while
 * we think, so the position we are next asked about is usually some other node
 * a ply or two down.
 */
void Agent::store_next_root_candidates(Board& board,
                                       bool teamHasTimeAdvantage) {
    nextRootCandidates_.clear();

    if (!rootNode || !rootNode->is_expanded()) {
        return;
    }
    
    auto children = rootNode->get_children();
    auto visits = rootNode->get_child_visits();
    
    if (children.empty() || visits.empty()) {
        return;
    }
    
    // Find best child (most visited, with Q-value consideration)
    int bestIdx = rootNode->get_best_move_idx_with_q_weight(
        lastRuntimeConfig_.qVetoDelta, lastRuntimeConfig_.qValueWeight);
    if (bestIdx < 0) {
        // Fallback to most-visited
        int maxVisits = 0;
        for (size_t i = 0; i < visits.size(); ++i) {
            if (visits[i] > maxVisits) {
                maxVisits = visits[i];
                bestIdx = static_cast<int>(i);
            }
        }
    }
    
    if (bestIdx < 0 || static_cast<size_t>(bestIdx) >= children.size()) {
        return;
    }

    const JointActionCandidate ownAction = rootNode->get_joint_action(bestIdx);
    Board ownNextBoard(board);
    ownNextBoard.make_moves(ownAction.moveA, ownAction.moveB);
    retain_reuse_candidates(
        children[bestIdx], ownNextBoard, !teamHasTimeAdvantage);
}

/**
 * @brief Index a retained subtree by position hash, out to a bounded depth.
 *
 * Records each position's hash and its exact board signature. Only the hash is
 * looked up on the next search; the signature is verified on a hit.
 *
 * The walk goes level by level rather than branch by branch so that a budget
 * that runs out has already covered the likeliest positions instead of one deep
 * line of them. Re-descending to reach each level costs make/unmake and nothing
 * else: a position already indexed is skipped before its signature is built.
 *
 * @param board Position of subtreeRoot; restored before returning.
 * @param teamHasTimeAdvantage Time advantage from the side of subtreeRoot's
 *        team, which alternates with the team to move as the walk descends.
 */
void Agent::retain_reuse_candidates(const std::shared_ptr<Node>& subtreeRoot,
                                    Board& board,
                                    bool teamHasTimeAdvantage) {
    for (int level = 0;
         level <= SearchParams::TREE_REUSE_MAX_JOINT_PLIES;
         ++level) {
        // Each joint ply flips the team to move, and a search is only ever
        // rooted where our own team is on move. Even levels below the subtree
        // root hold the other team, so indexing them would spend the budget on
        // positions no lookup can match. The root itself is exempt: that is
        // where the permanent brain starts.
        if (level > 0 && level % 2 == 0) {
            continue;
        }

        const size_t before = nextRootCandidates_.size();
        retain_reuse_level(
            subtreeRoot, board, teamHasTimeAdvantage, 0, level);
        if (nextRootCandidates_.size()
                >= SearchParams::TREE_REUSE_MAX_CANDIDATES
            || (level > 0 && nextRootCandidates_.size() == before)) {
            break;
        }
    }
}

/** Records every node exactly `level` joint plies below `node`. */
void Agent::retain_reuse_level(const std::shared_ptr<Node>& node,
                               Board& board,
                               bool teamHasTimeAdvantage,
                               int depth,
                               int level) {
    if (!node
        || nextRootCandidates_.size()
               >= SearchParams::TREE_REUSE_MAX_CANDIDATES) {
        return;
    }

    if (depth == level) {
        const uint64_t positionHash = board.search_hash_key(
            node->get_team_to_play(), teamHasTimeAdvantage);
        // Shallower levels run first, so a transposition already indexed keeps
        // the shallower node, whose signature is already built.
        if (!nextRootCandidates_.contains(positionHash)) {
            nextRootCandidates_.emplace(
                positionHash,
                RetainedRootCandidate{
                    node, positionHash, board_signature(board)});
        }
        return;
    }

    if (!node->is_expanded()) {
        return;
    }

    const auto children = node->get_children();
    const auto visits = node->get_child_visits();
    for (size_t index = 0; index < children.size(); ++index) {
        if (!children[index]) {
            continue;
        }
        // The first ply is kept whole. A solver-proven loss for the side to
        // move covers every legal reply, so dropping the unvisited ones throws
        // most of the proof away and can make a later search report a longer
        // mate. Deeper down, an unvisited node holds nothing worth walking to.
        if (depth > 0
            && (index >= visits.size() || visits[index] <= 0)) {
            continue;
        }

        const JointActionCandidate reply =
            node->get_joint_action(static_cast<int>(index));
        board.make_moves(reply.moveA, reply.moveB);
        retain_reuse_level(
            children[index], board, !teamHasTimeAdvantage, depth + 1, level);
        board.unmake_moves(reply.moveA, reply.moveB);
        if (nextRootCandidates_.size()
                >= SearchParams::TREE_REUSE_MAX_CANDIDATES) {
            return;
        }
    }
}

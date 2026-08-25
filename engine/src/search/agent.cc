#include "search/agent.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <thread>
#include <unordered_set>
#include <vector>

#include "environment/joint_action.h"
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
    } else {
        // Not solved, use centipawn score from Q value (already from root's perspective)
        int cpScore = static_cast<int>(C * std::tan(k * qFromParent));
        return "score cp " + to_string(cpScore);
    }
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
    auto children = rootNode->get_children();
    if (bestChildIdx >= 0 && static_cast<size_t>(bestChildIdx) < children.size()) {
        Node* bestChild = children[bestChildIdx].get();
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
    MateSearchBudget* budget) {
    return search_single_board_forced_mate_impl(
        board, boardNum, attackerColor, currentPly, maxAttackerMoves,
        outMove, outPlyToMate, budget, nullptr);
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
    std::vector<MateContinuation>* continuations) {
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
        const uint64_t positionHash = board.hash_key(true);
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

    std::vector<Stockfish::Move> legalMoves = board.legal_moves(boardNum);
    std::vector<Stockfish::Move> checkingMoves;
    checkingMoves.reserve(legalMoves.size());
    for (Stockfish::Move m : legalMoves) {
        if (board.gives_check(boardNum, m)) {
            checkingMoves.push_back(m);
        }
    }
    if (checkingMoves.empty()) {
        return false;
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
        const TerminalOutcome terminalOutcome = classify_terminal_position(
            board, victimTeam, attackerTeam, true, currentPly,
            &terminalEndInPly);
        board.pop_move(boardNum);
        if (terminalOutcome == TerminalOutcome::LOSS) {
            outMove = m;
            // The classifier's distance is relative to the position after this
            // move. Immediate mate is 1; an unavoidable waiting-board mate is
            // 3 (forced reply, then mate), so splice that suffix onto the
            // current root-relative ply.
            outPlyToMate = currentPly + terminalEndInPly - 1;
            retain_continuation(outMove, outPlyToMate);
            return true;
        }
    }

    // 2. If not immediate mate and we have moves remaining, verify all defender replies
    if (attackerMoveNum < maxAttackerMoves) {
        for (Stockfish::Move m : checkingMoves) {
            if (budget && !budget->consume()) {
                return false;
            }
            board.push_move(boardNum, m);
            std::vector<Stockfish::Move> defenderReplies = board.legal_moves(boardNum);
            if (defenderReplies.empty()) {
                // Stalemate or terminal without checkmate
                board.pop_move(boardNum);
                continue;
            }

            bool allRepliesMated = true;
            int deepestReplyPly = currentPly;
            for (Stockfish::Move reply : defenderReplies) {
                if (budget && !budget->consume()) {
                    allRepliesMated = false;
                    break;
                }
                board.push_move(boardNum, reply);
                Stockfish::Move nextAttackerMove = Stockfish::MOVE_NONE;
                int nextReplyPly = 0;
                const bool replyMated = search_single_board_forced_mate_impl(
                    board, boardNum, attackerColor, currentPly + 2, maxAttackerMoves,
                    nextAttackerMove, nextReplyPly, budget, continuations);
                board.pop_move(boardNum);
                if (!replyMated) {
                    allRepliesMated = false;
                    break;
                }
                deepestReplyPly = std::max(deepestReplyPly, nextReplyPly);
            }
            board.pop_move(boardNum);

            if (allRepliesMated) {
                outMove = m;
                outPlyToMate = deepestReplyPly;
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

bool Agent::find_root_mate(Board& board, Stockfish::Color teamSide,
                          bool teamHasTimeAdvantage,
                          JointActionCandidate& outAction,
                          int& outPlyToMate,
                          uint64_t nodeBudget) {
    return find_root_mate_impl(
        board, teamSide, teamHasTimeAdvantage, outAction, outPlyToMate,
        nodeBudget, nullptr, nullptr, true);
}

bool Agent::find_root_mate_impl(
    Board& board, Stockfish::Color teamSide,
    bool teamHasTimeAdvantage,
    JointActionCandidate& outAction,
    int& outPlyToMate,
    uint64_t nodeBudget,
    std::vector<MateContinuation>* continuations,
    MateSearchBudget* hardBudget,
    bool includeCaptureFeeds) {
    // 1. Fast 1-ply immediate mate scan across all joint combinations
    if (find_immediate_root_mate(
            board, teamSide, teamHasTimeAdvantage, outAction, hardBudget)) {
        outPlyToMate = 1;
        return true;
    }
    if (hardBudget && hardBudget->exhausted) {
        return false;
    }

    const bool boardAOnTurn = board.side_to_move(BOARD_A) == teamSide;
    const bool boardBOnTurn = board.side_to_move(BOARD_B) == ~teamSide;

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

        bool foundFeedMate = false;
        JointActionCandidate bestFeedAction;
        int bestFeedPly = 0;
        auto retain_shortest_feed = [&](const JointActionCandidate& action,
                                        int plyToMate) {
            if (!foundFeedMate || plyToMate < bestFeedPly) {
                foundFeedMate = true;
                bestFeedAction = action;
                bestFeedPly = plyToMate;
            }
        };

        std::array<MateSearchBudget, 2> feedBudgets;
        // Keep the new preparatory scan additive and tightly bounded. The
        // established single-board phase still receives its full budget below,
        // while feed proofs use at least 20k cheap make/unmake probes. The 20k
        // floor keeps capture feeds available even for the minimum/very-short
        // search allocation; split it evenly so each direction receives at
        // least 10k and neither A-to-B nor B-to-A can starve the other.
        const uint64_t totalFeedBudget =
            std::max<uint64_t>(20000, nodeBudget / 10);
        feedBudgets[BOARD_A].remainingNodes = totalFeedBudget / 2;
        feedBudgets[BOARD_B].remainingNodes =
            totalFeedBudget - feedBudgets[BOARD_A].remainingNodes;

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
                        hasCheckingDrop});
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

        // Most feed mates start with a check by the newly acquired piece. Try
        // those first, but retain every capture so longer feed combinations are
        // still considered if the shared budget permits.
        std::stable_sort(
            feedCandidates.begin(), feedCandidates.end(),
            [](const CaptureFeedCandidate& left,
               const CaptureFeedCandidate& right) {
                return left.fedPieceHasCheckingDrop
                    && !right.fedPieceHasCheckingDrop;
            });

        auto find_shortest_single_board_mate = [&](
            int boardNum, Stockfish::Color attacker,
            Stockfish::Move& mateMove, int& matePly,
            MateSearchBudget& feedBudget) {
            for (int maxMateMoves = 1;
                 maxMateMoves <= SearchParams::MATE_SEARCH_MAX_ATTACKER_MOVES;
                 ++maxMateMoves) {
                mateMove = Stockfish::MOVE_NONE;
                matePly = 0;
                if (search_single_board_forced_mate_impl(
                        board, boardNum, attacker, 1, maxMateMoves,
                        mateMove, matePly, &feedBudget, continuations)) {
                    return true;
                }
                if (feedBudget.exhausted) {
                    return false;
                }
            }
            return false;
        };

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
                retain_shortest_feed(feedAction, 1);
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
            for (Stockfish::Move reply : opponentReplies) {
                if (!feedBudget.consume()) {
                    allRepliesMated = false;
                    break;
                }
                board.push_move(candidate.feedBoard, reply);

                Stockfish::Move mateMove = Stockfish::MOVE_NONE;
                int matePly = 0;
                const bool rootLostOrDrawn =
                    board.is_checkmate(teamSide, true) || board.is_draw();
                const bool replyMated = !rootLostOrDrawn
                    && find_shortest_single_board_mate(
                        candidate.targetBoard, candidate.targetAttacker,
                        mateMove, matePly, feedBudget);
                board.pop_move(candidate.feedBoard);

                if (!replyMated) {
                    allRepliesMated = false;
                    break;
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
                retain_shortest_feed(feedAction, deepestMatePly + 2);
            }
        }

        // Iterative deepening: mate in 1 is already covered by the joint scan
        // above, so start at two attacker moves. Deepening returns the shortest
        // forced mate instead of whichever proof the depth-first order reaches
        // first, and keeps the wasted work bounded when no mate exists. The
        // budget is shared by every iteration and by both boards.
        MateSearchBudget localBudget;
        localBudget.remainingNodes = nodeBudget;
        MateSearchBudget& budget = hardBudget ? *hardBudget : localBudget;
        for (int maxMateMoves = 2;
             maxMateMoves <= SearchParams::MATE_SEARCH_MAX_ATTACKER_MOVES;
             ++maxMateMoves) {
            Stockfish::Move mateMoveA = Stockfish::MOVE_NONE;
            int plyA = 0;
            const bool foundA = search_single_board_forced_mate_impl(
                board, BOARD_A, teamSide, 1, maxMateMoves, mateMoveA, plyA,
                &budget, continuations);

            Stockfish::Move mateMoveB = Stockfish::MOVE_NONE;
            int plyB = 0;
            const bool foundB = search_single_board_forced_mate_impl(
                board, BOARD_B, ~teamSide, 1, maxMateMoves, mateMoveB, plyB,
                &budget, continuations);

            if (foundA && (!foundB || plyA <= plyB)) {
                const bool isCapA = board.is_capture(BOARD_A, mateMoveA);
                const JointActionCandidate directAction(
                    mateMoveA, 1.0f, 0, Stockfish::MOVE_NONE, 1.0f, 0,
                    rules, isCapA, false);
                if (!foundFeedMate || plyA <= bestFeedPly) {
                    outAction = directAction;
                    outPlyToMate = plyA;
                } else {
                    outAction = bestFeedAction;
                    outPlyToMate = bestFeedPly;
                }
                return true;
            }
            if (foundB) {
                const bool isCapB = board.is_capture(BOARD_B, mateMoveB);
                const JointActionCandidate directAction(
                    Stockfish::MOVE_NONE, 1.0f, 0, mateMoveB, 1.0f, 0,
                    rules, false, isCapB);
                if (!foundFeedMate || plyB <= bestFeedPly) {
                    outAction = directAction;
                    outPlyToMate = plyB;
                } else {
                    outAction = bestFeedAction;
                    outPlyToMate = bestFeedPly;
                }
                return true;
            }
            if (budget.exhausted) {
                break;
            }
        }

        if (foundFeedMate) {
            outAction = bestFeedAction;
            outPlyToMate = bestFeedPly;
            return true;
        }
    }

    return false;
}

bool Agent::find_root_forced_loss(
    Board& board,
    Stockfish::Color teamSide,
    bool teamHasTimeAdvantage,
    JointActionCandidate& outAction,
    int& outPlyToMate,
    uint64_t nodeBudget) {
    const vector<JointActionCandidate> defenses = legal_joint_actions(
        board, teamSide, teamHasTimeAdvantage);
    if (defenses.empty() || nodeBudget == 0) {
        return false;
    }

    MateSearchBudget totalBudget;
    totalBudget.remainingNodes = nodeBudget;
    const Stockfish::Color opponentTeam = ~teamSide;
    const bool opponentHasTimeAdvantage = !teamHasTimeAdvantage;
    bool foundDelayingAction = false;
    int longestLossPly = 0;

    for (size_t defenseIndex = 0;
         defenseIndex < defenses.size(); ++defenseIndex) {
        if (!totalBudget.consume()) {
            return false;
        }

        const JointActionCandidate& defense = defenses[defenseIndex];
        board.make_moves(defense.moveA, defense.moveB);

        bool defenseIsMated = false;
        int totalLossPly = 0;
        if (board.is_checkmate(teamSide, teamHasTimeAdvantage)) {
            // The defensive action itself ended the game.
            defenseIsMated = true;
            totalLossPly = 1;
        } else if (!board.is_checkmate(
                       opponentTeam, opponentHasTimeAdvantage)
                   && !board.is_draw(1)) {
            // Divide the remaining budget fairly among the current and all
            // untested defenses. Unused probes return to the common pool, and
            // an early branch cannot starve the universal proof.
            const uint64_t remainingDefenses =
                defenses.size() - defenseIndex;
            const uint64_t defenseAllocation =
                totalBudget.remainingNodes / remainingDefenses;
            if (defenseAllocation > 0) {
                MateSearchBudget defenseBudget;
                defenseBudget.remainingNodes = defenseAllocation;
                JointActionCandidate opponentMate;
                int opponentMatePly = 0;
                defenseIsMated = find_root_mate_impl(
                    board, opponentTeam, opponentHasTimeAdvantage,
                    opponentMate, opponentMatePly,
                    defenseAllocation, nullptr, &defenseBudget, true);
                const uint64_t probesUsed = defenseAllocation
                    - defenseBudget.remainingNodes;
                totalBudget.remainingNodes -= probesUsed;
                if (defenseIsMated) {
                    // Our defense is ply one; the opponent proof starts from
                    // the resulting position.
                    totalLossPly = opponentMatePly + 1;
                }
            }
        }

        board.unmake_moves(defense.moveA, defense.moveB);

        // A single drawing, winning, unproven, or over-budget defense is enough
        // to refute an exact forced-loss claim.
        if (!defenseIsMated) {
            return false;
        }
        if (!foundDelayingAction || totalLossPly > longestLossPly) {
            foundDelayingAction = true;
            longestLossPly = totalLossPly;
            outAction = defense;
        }
    }

    if (!foundDelayingAction) {
        return false;
    }
    outPlyToMate = longestLossPly;
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
    const uint64_t positionHash = board.hash_key(teamHasTimeAdvantage);
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
    JointActionCandidate result;
    lastRuntimeConfig_ = options.search;
    if (engines.empty()) {
        cerr << "Cannot search without an inference engine" << endl;
        return result;
    }
    
    const bool boardAOnTurn = board.side_to_move(BOARD_A) == teamSide;
    const bool boardBOnTurn = board.side_to_move(BOARD_B) == ~teamSide;
    const bool canWait = is_double_sit_legal(
        teamHasTimeAdvantage, boardAOnTurn, boardBOnTurn);

    if (board.is_checkmate(~teamSide, !teamHasTimeAdvantage)
        || board.is_checkmate(teamSide, teamHasTimeAdvantage)
        || board.is_draw()) {
        if (options.verbose) {
            cout << "bestmove (none)" << endl;
        }
        return result;
    }

    // A team with no real board move may still have the legal wait action.
    if (board.legal_moves(teamSide, teamHasTimeAdvantage).empty() && !canWait) {
        if (options.verbose) {
            cout << "bestmove (none)" << endl;
        }
        return result;
    }

    // Reuse an exact reply-indexed certificate from the previous forced-mate
    // proof before running the bounded root scan again. A changed partner
    // board, pocket, side to move, team, or TimeAdvantage value cannot match.
    JointActionCandidate rootMateAction;
    int rootMatePly = 1;
    const uint64_t rootMateBudget = mate_search_node_budget(options);
    const bool cachedRootMate = SearchParams::ENABLE_MATE_EARLY_EXIT
        && try_reuse_mate_continuation(
            board, teamSide, teamHasTimeAdvantage,
            rootMateAction, rootMatePly);
    const bool scannedRootMate = SearchParams::ENABLE_MATE_EARLY_EXIT
        && !cachedRootMate
        && find_root_mate_impl(
            board, teamSide, teamHasTimeAdvantage,
            rootMateAction, rootMatePly,
            rootMateBudget, &mateContinuations_);
    if (cachedRootMate || scannedRootMate) {
        result = rootMateAction;
        uint64_t positionHash = board.hash_key(teamHasTimeAdvantage);
        rootNode = make_shared<Node>(teamSide, positionHash);
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
            children[0]->mark_as_loss(std::max(0, rootMatePly - 1));
            rootNode->init_child_node_types();
            rootNode->update_child_node_type(0, NodeType::LOSS);
        }
        rootNode->update(0, 1.0f);
        rootNode->mark_as_win(rootMatePly);

        if (SearchParams::ENABLE_TREE_REUSE) {
            store_next_root_candidates(board, teamHasTimeAdvantage);
            lastSearchHash_ = positionHash;
        }

        if (options.verbose) {
            string bestMoveStr = extract_best_move(board);
            const int mateScore = (rootMatePly + 1) / 2;
            cout << "info depth " << rootMatePly << " score mate " << mateScore
                 << " nodes 1 nps 1000 time 0 pv " << bestMoveStr << endl;
            cout << "bestmove " << bestMoveStr << endl;
        }
        return result;
    }

    // Winning proofs need one successful action; losing proofs must quantify
    // every legal defense. Keep that reverse proof separately bounded and
    // return "not proven" on exhaustion so MCTS remains the fallback.
    JointActionCandidate rootLossAction;
    int rootLossPly = 0;
    const uint64_t rootLossBudget = options.moveTimeMs > 0
        ? std::max(
            rootMateBudget,
            SearchParams::FORCED_LOSS_MIN_TIMED_NODE_BUDGET)
        : rootMateBudget;
    const bool scannedRootLoss = SearchParams::ENABLE_MATE_EARLY_EXIT
        && find_root_forced_loss(
            board, teamSide, teamHasTimeAdvantage,
            rootLossAction, rootLossPly, rootLossBudget);
    if (scannedRootLoss) {
        result = rootLossAction;
        const uint64_t positionHash = board.hash_key(teamHasTimeAdvantage);
        rootNode = make_shared<Node>(teamSide, positionHash);
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

        if (options.verbose) {
            const string bestMoveStr = extract_best_move(board);
            const int mateScore = (rootLossPly + 1) / 2;
            cout << "info depth " << rootLossPly << " score mate -"
                 << mateScore << " nodes 1 nps 1000 time 0 pv "
                 << bestMoveStr << endl;
            cout << "bestmove " << bestMoveStr << endl;
        }
        return result;
    }

    // Determine effective move time
    int moveTimeMs = options.moveTimeMs;
    size_t targetNodes = options.targetNodes;

    // Compute position hash for tree reuse
    uint64_t positionHash = board.hash_key(teamHasTimeAdvantage);
    const std::string positionSignature = board_signature(board);
    
    // Try to reuse tree from previous search (if enabled)
    std::shared_ptr<Node> reusedRoot = nullptr;
    if (SearchParams::ENABLE_TREE_REUSE) {
        reusedRoot = try_reuse_tree(positionHash, teamSide, positionSignature);
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
    rootNode->configure_root_search(options.search);
    
    SearchInfo searchInfo(chrono::steady_clock::now(), moveTimeMs);
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
    
    dispatch_workers(board, engines, searchInfo, teamHasTimeAdvantage,
                     targetNodes, moveTimeMs, workerCount);

    // Periodic info output during search (UCI verbose mode only)
    // Also handles early stopping and time extension
    constexpr int POLL_INTERVAL_MS = 5;
    bool nodeSearchStalled = false;
    int stalledCompletedNodes = 0;
    int lastReportedDepth = 0;
    if (options.verbose && moveTimeMs > 0) {
        searchInfo.set_in_game(true);
        constexpr float C = 180.0f;
        constexpr float k = 1.56f;
        float lastCheckEval = 0.0f;
        bool evalInitialized = false;
        int lastBestChildIdx = -1;
        
        while (running && (isPondering_.load(std::memory_order_relaxed)
                           || searchInfo.elapsed() < searchInfo.get_effective_move_time())) {
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
                            string pv = extract_pv_from_child(board, static_cast<int>(childIdx), 20);
                            float childQ = rootNode->get_child_q(static_cast<int>(childIdx));
                            string scoreStr = format_uci_score(children[childIdx].get(), childQ, true, C, k);
                            
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
            std::this_thread::sleep_for(std::chrono::milliseconds(POLL_INTERVAL_MS));
            if (SearchParams::ENABLE_MATE_EARLY_EXIT && rootNode
                && rootNode->get_node_type() != NodeType::UNSOLVED) {
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
    
    // Store next-root candidates for tree reuse
    if (SearchParams::ENABLE_TREE_REUSE) {
        store_next_root_candidates(board, teamHasTimeAdvantage);
        lastSearchHash_ = board.hash_key(teamHasTimeAdvantage);
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
                string pv = extract_pv_from_child(board, static_cast<int>(childIdx), 20);
                float childQ = rootNode->get_child_q(static_cast<int>(childIdx));
                string scoreStr = format_uci_score(children[childIdx].get(), childQ, true, C, k);
                
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

        cout << "info string rejected selection attempts "
               << searchInfo.get_collisions()
               << " (same batch " << searchInfo.get_same_batch_collisions()
             << ", pending evaluation " << searchInfo.get_reservation_collisions()
               << ")" << endl;
        string bestMoveStr = extract_best_move(board);
        string ponderMoveStr = options.enablePonder ? extract_ponder_move(board) : "";
        if (!ponderMoveStr.empty()) {
            cout << "bestmove " << bestMoveStr << " ponder " << ponderMoveStr << endl;
        } else {
            cout << "bestmove " << bestMoveStr << endl;
        }
    }
    
    return result;
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
string Agent::extract_pv_from_child(Board& board, int childIdx, int maxDepth) {
    if (!rootNode || !rootNode->is_expanded()) {
        return "";
    }
    
    auto children = rootNode->get_children();
    if (childIdx < 0 || static_cast<size_t>(childIdx) >= children.size()) {
        return "";
    }
    
    Board tempBoard = board;
    string pv;
    
    // Get the first move from the specified child
    JointActionCandidate action = rootNode->get_joint_action(childIdx);
    string moveA = (action.moveA == Stockfish::MOVE_NONE) 
                    ? "pass" : tempBoard.uci_move(BOARD_A, action.moveA);
    string moveB = (action.moveB == Stockfish::MOVE_NONE) 
                    ? "pass" : tempBoard.uci_move(BOARD_B, action.moveB);
    pv = "(" + moveA + "," + moveB + ")";
    
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
        
        // Apply moves to temp board for next iteration
        tempBoard.make_moves(nextAction.moveA, nextAction.moveB);
        
        // Move to best child
        currentNode = nodeChildren[bestIdx].get();
    }
    
    return pv;
}

void Agent::set_is_running(bool value) {
    if (!value) {
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
std::string Agent::board_signature(Board& board) {
    return board.fen(BOARD_A) + "|" + board.fen(BOARD_B);
}

void Agent::reindex_reused_subtree(const std::shared_ptr<Node>& reusedRoot) {
    if (!transpositionTable || !reusedRoot) {
        return;
    }

    std::vector<std::shared_ptr<Node>> pending = {reusedRoot};
    std::unordered_set<const Node*> visited;
    while (!pending.empty()) {
        std::shared_ptr<Node> node = std::move(pending.back());
        pending.pop_back();
        if (!node || !visited.insert(node.get()).second) {
            continue;
        }

        const uint64_t hash = node->get_hash();
        if (hash != 0) {
            transpositionTable->insertOrGet(hash, node);
        }

        for (const std::shared_ptr<Node>& child : node->get_children()) {
            if (child) {
                pending.push_back(child);
            }
        }
    }
}

std::shared_ptr<Node> Agent::try_reuse_tree(uint64_t positionHash,
                                            Stockfish::Color teamSide,
                                            const std::string& signature) {
    // The signature must match exactly: retained edges were generated against
    // that board, and a stale pocket would make reused drops illegal.
    std::shared_ptr<Node> reused;
    for (const RetainedRootCandidate& candidate : nextRootCandidates_) {
        if (candidate.node
            && candidate.positionHash == positionHash
            && candidate.node->get_team_to_play() == teamSide
            && !candidate.signature.empty()
            && candidate.signature == signature) {
            reused = candidate.node;
            break;
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
 * After search completes, retain the selected child and every generated
 * opponent response beneath it. A solver-proven loss for the opponent covers
 * all legal replies, so retaining only the principal reply throws away most of
 * the proof and can make a later search report a longer mate.
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
    const std::shared_ptr<Node>& ownNextRoot = children[bestIdx];
    nextRootCandidates_.push_back({
        ownNextRoot,
        ownNextBoard.hash_key(!teamHasTimeAdvantage),
        board_signature(ownNextBoard)});

    if (!ownNextRoot->is_expanded()) {
        return;
    }

    auto grandchildren = ownNextRoot->get_children();
    if (grandchildren.empty()) {
        return;
    }

    nextRootCandidates_.reserve(1 + grandchildren.size());
    Board opponentsNextBoard(ownNextBoard);
    for (size_t index = 0; index < grandchildren.size(); ++index) {
        if (!grandchildren[index]) {
            continue;
        }

        const JointActionCandidate reply =
            ownNextRoot->get_joint_action(static_cast<int>(index));
        opponentsNextBoard.make_moves(reply.moveA, reply.moveB);
        const uint64_t replyHash =
            opponentsNextBoard.hash_key(teamHasTimeAdvantage);
        const std::string replySignature =
            board_signature(opponentsNextBoard);

        nextRootCandidates_.push_back({
            grandchildren[index], replyHash, replySignature});
        opponentsNextBoard.unmake_moves(reply.moveA, reply.moveB);
    }
}

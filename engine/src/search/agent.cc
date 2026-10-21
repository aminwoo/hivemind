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
#include <optional>
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
#include "common/thread_priority.h"

using namespace std;

namespace {

/**
 * @brief Whether a ponder has spent its node or time ceiling.
 *
 * A ponder ignores both the clock and the node target - that is what makes it
 * a ponder - so nothing but a GUI stop ends one. These ceilings bound the tree
 * it grows in the meantime. Applies to `go ponder` and to the permanent brain
 * alike; the caller checks it only while `isPondering_` holds.
 */
bool ponder_budget_exhausted(const SearchInfo& searchInfo) {
    return searchInfo.get_nodes_searched() >= SearchParams::PONDER_MAX_NODES
        || searchInfo.elapsed() >= SearchParams::PONDER_MAX_MS;
}

struct InternalMateProbeTarget {
    std::unique_ptr<Board> board;
    std::weak_ptr<Node> node;
    std::weak_ptr<Node> rootParent;
    std::weak_ptr<Node> replyParent;
    int rootEdge = -1;
    int replyEdge = -1;
    int treeDepth = 0;
    int visitsAtSubmit = 0;
    float rootQAtSubmit = 0.0f;
    uint64_t positionHash = 0;
    // The clock the target's path reached: who may sit there, exactly.
    TimeState time;
};

std::vector<size_t> ranked_visited_edges(const std::shared_ptr<Node>& node,
                                         int minimumVisits) {
    if (!node || !node->is_expanded()) {
        return {};
    }
    const std::vector<int> visits = node->get_child_visits();
    const size_t generated = std::min(visits.size(), node->get_num_generated());
    std::vector<size_t> indices;
    indices.reserve(generated);
    for (size_t index = 0; index < generated; ++index) {
        if (visits[index] >= minimumVisits) {
            indices.push_back(index);
        }
    }
    std::stable_sort(indices.begin(), indices.end(), [&](size_t lhs, size_t rhs) {
        return visits[lhs] > visits[rhs];
    });
    return indices;
}

bool edge_is_top_k(const std::weak_ptr<Node>& parent, int edge, int topK) {
    const std::shared_ptr<Node> node = parent.lock();
    if (!node || edge < 0 || topK <= 0) {
        return false;
    }
    const std::vector<int> visits = node->get_child_visits();
    if (static_cast<size_t>(edge) >= visits.size()) {
        return false;
    }
    int rank = 1;
    for (size_t index = 0; index < visits.size(); ++index) {
        if (visits[index] > visits[static_cast<size_t>(edge)]) {
            ++rank;
        }
    }
    return rank <= topK;
}

bool node_generated_action(const std::shared_ptr<Node>& node,
                           const JointActionCandidate& action) {
    if (!node || !node->is_expanded()) {
        return false;
    }
    const size_t generated = node->get_num_generated();
    for (size_t index = 0; index < generated; ++index) {
        const JointActionCandidate existing = node->get_joint_action(
            static_cast<int>(index));
        if (existing.moveA == action.moveA && existing.moveB == action.moveB) {
            return true;
        }
    }
    return false;
}

std::vector<InternalMateProbeTarget> collect_internal_probe_targets(
    const Board& rootBoard,
    const std::shared_ptr<Node>& root,
    Stockfish::Color rootTeam,
    const TimeState& rootTime,
    int topK,
    int minimumVisits) {
    const bool rootTeamHasTimeAdvantage = rootTime.can_sit(rootTeam);
    std::vector<InternalMateProbeTarget> targets;
    if (!root || topK <= 0) {
        return targets;
    }

    const std::vector<size_t> rootEdges = ranked_visited_edges(
        root, minimumVisits);
    const std::vector<std::shared_ptr<Node>> rootChildren = root->get_children();
    std::unordered_set<uint64_t> seen;
    std::unordered_set<const Node*> seenNodes;

    auto add_target = [&](size_t rootEdge, int replyEdge,
                          const std::shared_ptr<Node>& targetNode,
                          int visitsAtSubmit) {
        if (!targetNode || targetNode->get_node_type() != NodeType::UNSOLVED
            || rootEdge >= rootChildren.size()
            || seenNodes.contains(targetNode.get())) {
            return;
        }

        std::weak_ptr<Node> replyParent;
        int treeDepth = 1;
        JointActionCandidate reply;
        if (replyEdge >= 0) {
            const std::shared_ptr<Node>& child = rootChildren[rootEdge];
            if (!child || static_cast<size_t>(replyEdge)
                    >= child->get_num_generated()) {
                return;
            }
            reply = child->get_joint_action(replyEdge);
            replyParent = child;
            treeDepth = 2;
        }
        seenNodes.insert(targetNode.get());

        auto targetBoard = std::make_unique<Board>(rootBoard);
        const JointActionCandidate rootAction = root->get_joint_action(
            static_cast<int>(rootEdge));
        if (!targetBoard->is_legal_move(BOARD_A, rootAction.moveA)
            || !targetBoard->is_legal_move(BOARD_B, rootAction.moveB)) {
            return;
        }
        // Each action is charged as the search charged it, so the target's
        // hash is the one its tree node carries.
        TimeState targetTime = rootTime;
        targetTime.advance(rootTeam, targetBoard->on_turn_on_both_boards(rootTeam));
        targetBoard->make_moves(rootAction.moveA, rootAction.moveB);

        if (replyEdge >= 0) {
            if (!targetBoard->is_legal_move(BOARD_A, reply.moveA)
                || !targetBoard->is_legal_move(BOARD_B, reply.moveB)) {
                return;
            }
            targetTime.advance(~rootTeam, targetBoard->on_turn_on_both_boards(~rootTeam));
            targetBoard->make_moves(reply.moveA, reply.moveB);
        }

        const Stockfish::Color targetTeam = treeDepth == 1
            ? ~rootTeam : rootTeam;
        const uint64_t positionHash = targetBoard->search_hash_key(
            targetTeam, targetTime);
        if (!seen.insert(positionHash).second) {
            return;
        }
        targets.push_back({
            std::move(targetBoard), targetNode, root,
            replyParent, static_cast<int>(rootEdge), replyEdge,
            treeDepth, visitsAtSubmit,
            root->get_child_q(static_cast<int>(rootEdge)), positionHash,
            targetTime});
    };

    if (!rootTeamHasTimeAdvantage) {
        const std::vector<int> visits = root->get_child_visits();
        for (size_t rootEdge : rootEdges) {
            if (targets.size() >= static_cast<size_t>(topK)
                || rootEdge >= rootChildren.size()
                || rootEdge >= visits.size()) {
                break;
            }
            add_target(rootEdge, -1, rootChildren[rootEdge], visits[rootEdge]);
        }
        return targets;
    }

    struct GrandchildPath {
        size_t rootEdge = 0;
        size_t replyEdge = 0;
        int visits = 0;
    };
    std::vector<GrandchildPath> paths;
    for (size_t rootEdge : rootEdges) {
        if (rootEdge >= rootChildren.size()) {
            continue;
        }
        const std::shared_ptr<Node>& child = rootChildren[rootEdge];
        const std::vector<size_t> replyEdges = ranked_visited_edges(
            child, minimumVisits);
        const std::vector<int> replyVisits = child
            ? child->get_child_visits() : std::vector<int>{};
        for (size_t replyEdge : replyEdges) {
            if (replyEdge < replyVisits.size()) {
                paths.push_back({rootEdge, replyEdge, replyVisits[replyEdge]});
            }
        }
    }
    std::stable_sort(paths.begin(), paths.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.visits > rhs.visits;
    });
    for (const GrandchildPath& path : paths) {
        if (targets.size() >= static_cast<size_t>(topK)) {
            break;
        }
        const std::shared_ptr<Node>& child = rootChildren[path.rootEdge];
        const std::vector<std::shared_ptr<Node>> grandchildren =
            child->get_children();
        if (path.replyEdge < grandchildren.size()) {
            add_target(path.rootEdge, static_cast<int>(path.replyEdge),
                       grandchildren[path.replyEdge], path.visits);
        }
    }
    return targets;
}

}  // namespace

/**
 * Moves the root's team plays in the mate a solved node holds: its own mating
 * move, or every move it makes before it is mated.
 *
 * The solver counts the mated position as one ply, so a mate in N from the
 * root is 2N plies and a loss in N is 2N+1. The root scans and probes mark
 * their proofs counting it as zero, one ply less. Both give the same moves.
 *
 * @param isChildNode True if the node follows one of the root team's actions,
 *        which is one more ply of the line.
 */
static int mate_moves(const Node& node, bool isChildNode) {
    const int plies = node.get_end_in_ply() + (isChildNode ? 1 : 0);
    const bool rootTeamWins = (node.get_node_type() == NodeType::WIN)
        != isChildNode;
    return max(1, rootTeamWins ? (plies + 1) / 2 : plies / 2);
}

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
    
    if (nodeType == NodeType::WIN) {
        int mateInMoves = mate_moves(*node, isChildNode);
        if (isChildNode) {
            // Child is a WIN for the child (opponent) = LOSS for us (we're mated)
            return "score mate -" + to_string(max(1, mateInMoves));
        } else {
            // Root is a WIN for us = we win
            return "score mate " + to_string(max(1, mateInMoves));
        }
    } else if (nodeType == NodeType::LOSS) {
        int mateInMoves = mate_moves(*node, isChildNode);
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

static void mark_immediate_root_repetitions(
    const shared_ptr<Node>& rootNode, Board& board) {
    if (!rootNode || !rootNode->is_expanded()) {
        return;
    }
    const auto children = rootNode->get_children();
    const size_t generated = min(
        children.size(), rootNode->get_num_generated());
    for (size_t index = 0; index < generated; ++index) {
        const shared_ptr<Node>& child = children[index];
        if (!child || child->get_node_type() != NodeType::UNSOLVED) {
            continue;
        }
        const JointActionCandidate action =
            rootNode->get_joint_action(static_cast<int>(index));
        board.make_moves(action.moveA, action.moveB);
        const bool repeats = board.is_repetition_draw({0, 0});
        board.unmake_moves(action.moveA, action.moveB);
        if (repeats) {
            child->mark_as_draw(1);
            rootNode->update_child_node_type(index, NodeType::DRAW);
        }
    }
}

string Agent::format_root_aware_uci_score(
    const shared_ptr<Node>& root,
    const shared_ptr<Node>& pvChild,
    float childQ,
    int childVisits,
    float centipawnScale,
    float tangentScale) {
    if (root && root->get_node_type() != NodeType::UNSOLVED) {
        // A solver proof is authoritative for the position being reported.
        // In particular, a proven DRAW must score 0 rather than leaking an
        // unvisited PV child's Q_INIT=-1 through the tangent conversion.
        return format_uci_score(
            root.get(), root->Q(), false, centipawnScale, tangentScale);
    }
    const bool childSolved = pvChild
        && pvChild->get_node_type() != NodeType::UNSOLVED;
    if (root && !childSolved && childVisits <= 0) {
        // The search ended before visiting the move it reports, as when a
        // deadline passes during the first evaluations. The edge still holds
        // Q_INIT=-1, a certain loss; the root's own value is what is known.
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
                int mateInMoves = mate_moves(*bestChild, true);
                cout << "info string Early exit: forced mate in " << mateInMoves << " found" << endl;
            }
            return true;
        }
    }
    
    return false;
}

/**
 * Longest line, in attacker moves, of any proof the opponents may be asked
 * for against a candidate move: Fairy's probe and the joint solver. While a
 * team can sit through that many actions of its own, the opponents never gain
 * the right to sit within such a line.
 */
constexpr int OPPONENT_PROOF_HORIZON_MOVES = std::max(
    SearchParams::MATE_PROBE_MAX_MATE_MOVES,
    SearchParams::SELECTED_MOVE_JOINT_MAX_ATTACKER_MOVES);

/**
 * Actions the attacker may play in the root scan's single-board proofs, which
 * sit the partner board for the whole line: a capture feed, up to
 * MATE_SEARCH_MAX_ATTACKER_MOVES checks, and the mating move on the waiting
 * board that a single-board proof may end with.
 */
constexpr int SAT_PARTNER_PROOF_ACTIONS =
    SearchParams::MATE_SEARCH_MAX_ATTACKER_MOVES + 2;

/**
 * Longest line, in attacker moves on the mating board, that the root scan's
 * sat-partner proofs may claim while @p team's lead pays for every action of
 * it: the line itself plus the capture feed and the waiting-board mating move
 * that SAT_PARTNER_PROOF_ACTIONS allows for. Zero when the lead covers none.
 *
 * A lead short of the full horizon still covers the short lines, and those
 * are exactly the ones a clock search should not hand to the joint search,
 * whose budget the joint action product exhausts first. A legacy state
 * answers all or nothing, as before.
 */
static int sat_partner_attacker_moves(const TimeState& time, Stockfish::Color team) {
    constexpr int OVERHEAD =
        SAT_PARTNER_PROOF_ACTIONS - SearchParams::MATE_SEARCH_MAX_ATTACKER_MOVES;
    for (int moves = SearchParams::MATE_SEARCH_MAX_ATTACKER_MOVES; moves >= 1;
         --moves) {
        if (time.can_sit_through(team, moves + OVERHEAD)) {
            return moves;
        }
    }
    return 0;
}

/**
 * @brief Performs a fast 1-ply checkmate scan at the root before starting MCTS.
 * If any legal joint action immediately delivers checkmate against the opponent,
 * returns true and sets outAction to that winning joint move.
 */
static bool find_immediate_root_mate(Board& board, Stockfish::Color teamSide,
                                     const TimeState& time,
                                     JointActionCandidate& outAction,
                                     Agent::MateSearchBudget* budget = nullptr,
                                     bool requireUnblockable = false) {
    const bool boardAOnTurn = board.side_to_move(BOARD_A) == teamSide;
    const bool boardBOnTurn = board.side_to_move(BOARD_B) == ~teamSide;
    const bool teamHasTimeAdvantage = time.can_sit(teamSide);
    // Every action from here costs the same: whether this team is on turn on
    // both boards decides it. The opponents are mated, or not, under the
    // clock that action leaves them.
    const bool opponentHasTimeAdvantage = time.after(
        teamSide, boardAOnTurn && boardBOnTurn).can_sit(~teamSide);

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

    // is_checkmate() deliberately includes bughouse stalemate. Distinguish a
    // literal checkmate so a stalemate can be retained as a fallback while the
    // other board is checked for an equally immediate checkmate.
    const auto hasLiteralCheckmate = [&](Stockfish::Color victimTeam,
                                         bool victimHasTimeAdvantage) {
        for (int boardNum : {BOARD_A, BOARD_B}) {
            const Stockfish::Color victim = boardNum == BOARD_A
                ? victimTeam : ~victimTeam;
            if (board.side_to_move(boardNum) == victim
                && board.is_in_check(boardNum)
                && !board.has_any_legal_move(boardNum)
                && !board.can_partner_provide_blocking_piece(
                    boardNum, victim, victimHasTimeAdvantage)) {
                return true;
            }
        }
        return false;
    };
    const auto hasUnblockableCheckmate = [&](Stockfish::Color victimTeam) {
        for (int boardNum : {BOARD_A, BOARD_B}) {
            const Stockfish::Color victim = boardNum == BOARD_A
                ? victimTeam : ~victimTeam;
            if (board.side_to_move(boardNum) == victim
                && board.is_in_check(boardNum)
                && !board.has_any_legal_move(boardNum)
                && !board.can_partner_provide_blocking_piece(
                    boardNum, victim, false, true)) {
                return true;
            }
        }
        return false;
    };
    // Mates rank by how surely they end the game, best first:
    //  3: no dropped piece could block the check, so the game ends on the spot;
    //  2: a drop would block it, but the defenders cannot get a piece in time;
    //  1: any other loss is_checkmate() reports (bughouse stalemate).
    // Call it with the candidate action made. Under requireUnblockable only a
    // rank-3 mate counts.
    constexpr int UNBLOCKABLE_MATE = 3;
    const auto mate_rank = [&] {
        if (!board.is_checkmate(~teamSide, opponentHasTimeAdvantage)) {
            return 0;
        }
        if (hasUnblockableCheckmate(~teamSide)) {
            return UNBLOCKABLE_MATE;
        }
        if (requireUnblockable) {
            return 0;
        }
        return hasLiteralCheckmate(~teamSide, opponentHasTimeAdvantage) ? 2 : 1;
    };
    // The best playable mate so far. A rank-3 mate ends the scan; anything
    // less is kept while the rest of the scan looks for a better one.
    std::optional<JointActionCandidate> bestMate;
    int bestRank = 0;
    const auto offer = [&](const JointActionCandidate& action, int rank,
                           bool winsTies = false) {
        if (rank > bestRank || (winsTies && rank > 0 && rank == bestRank)) {
            bestMate = action;
            bestRank = rank;
        }
    };
    const auto finish = [&] {
        if (!bestMate) {
            return false;
        }
        outAction = *bestMate;
        return true;
    };
    // Each board's best checkmate, playable alone or not, for finishing both
    // boards in one action.
    struct BoardMate {
        size_t index = 0;
        int rank = 0;
    };
    BoardMate matingA;
    BoardMate matingB;

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
                return finish();
            }
            const Stockfish::Move mA = actionsA[iA];
            const bool isCapA = board.is_capture(BOARD_A, mA);
            const bool canPassB = !boardBOnTurn || is_single_pass_legal(
                teamHasTimeAdvantage, boardAOnTurn, boardBOnTurn, isCapA);
            board.push_move(BOARD_A, mA);
            const int rank = mate_rank();
            board.pop_move(BOARD_A);
            if (rank >= 2 && rank > matingA.rank) {
                matingA = {iA, rank};
            }
            if (canPassB) {
                offer(JointActionCandidate(
                          mA, 1.0f, iA, Stockfish::MOVE_NONE, 1.0f, 0,
                          rules, isCapA, false),
                      rank);
            }
            if (rank == UNBLOCKABLE_MATE) {
                break;
            }
        }
    }

    // 2. Move on Board B, pass on Board A
    if (boardBOnTurn) {
        for (size_t iB = 0; iB < limitB; ++iB) {
            if (budget && !budget->consume()) {
                return finish();
            }
            const Stockfish::Move mB = actionsB[iB];
            const bool isCapB = board.is_capture(BOARD_B, mB);
            const bool canPassA = !boardAOnTurn || is_single_pass_legal(
                teamHasTimeAdvantage, boardAOnTurn, boardBOnTurn, isCapB);
            board.push_move(BOARD_B, mB);
            const int rank = mate_rank();
            board.pop_move(BOARD_B);
            if (rank >= 2 && rank > matingB.rank) {
                matingB = {iB, rank};
            }
            if (canPassA) {
                offer(JointActionCandidate(
                          Stockfish::MOVE_NONE, 1.0f, 0, mB, 1.0f, iB,
                          rules, false, isCapB),
                      rank);
            }
            if (rank == UNBLOCKABLE_MATE) {
                break;
            }
        }
    }

    // Both boards may already have mate in one. Finish both when possible,
    // rather than reporting a pass or an arbitrary partner move beside a mate;
    // the pair wins a tie with either half alone.
    if (matingA.rank > 0 && matingB.rank > 0) {
        const Stockfish::Move mA = actionsA[matingA.index];
        const Stockfish::Move mB = actionsB[matingB.index];
        if (!budget || budget->consume()) {
            board.make_moves(mA, mB);
            const int rank = mate_rank();
            board.unmake_moves(mA, mB);
            offer(JointActionCandidate(
                      mA, 1.0f, matingA.index, mB, 1.0f, matingB.index, rules,
                      board.is_capture(BOARD_A, mA),
                      board.is_capture(BOARD_B, mB)),
                  rank, true);
        }
    }
    if (bestRank == UNBLOCKABLE_MATE) {
        return finish();
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
            const int rank = mate_rank();
            board.unmake_moves(mA, mB);
            offer(JointActionCandidate(
                      mA, 1.0f, iA, mB, 1.0f, iB, rules,
                      board.is_capture(BOARD_A, mA),
                      board.is_capture(BOARD_B, mB)),
                  rank);
            return rank == UNBLOCKABLE_MATE;
        };

        for (size_t iA : immobilizingA) {
            for (size_t iB = 0; iB < actionsB.size(); ++iB) {
                if (pair_mates(iA, iB)) {
                    return finish();
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
                    return finish();
                }
            }
        }
    }

    return finish();
}

/** Legacy form: @p teamHasTimeAdvantage is a permanent flag. */
static bool find_immediate_root_mate(Board& board, Stockfish::Color teamSide,
                                     bool teamHasTimeAdvantage,
                                     JointActionCandidate& outAction,
                                     Agent::MateSearchBudget* budget = nullptr,
                                     bool requireUnblockable = false) {
    return find_immediate_root_mate(
        board, teamSide, TimeState::legacy(teamSide, teamHasTimeAdvantage),
        outAction, budget, requireUnblockable);
}

/**
 * @brief Whether a capture the defending team can play now hands their partner
 *        an immediate mate.
 *
 * A capture lands in the partner's hand the instant it is made, and that
 * partner plays on their own clock: there is no turn of ours between the
 * capture and the drop. So a team that holds no mate right now, but can
 * capture into one, is as good as mating already - the move we planned to
 * follow up with is never made. @p defenders is the team to test, on the
 * boards where they are on turn. Leaves @p board as it found it.
 */
static bool defenders_capture_into_immediate_mate(
    Board& board, Stockfish::Color defenders, const TimeState& time,
    Agent::MateSearchBudget* budget) {
    // The capture is the defenders' action, charged as one; the mate they
    // then hold is judged under the clock it leaves.
    const TimeState afterCapture = time.after(
        defenders, board.on_turn_on_both_boards(defenders));
    for (int boardNum : {BOARD_A, BOARD_B}) {
        const Stockfish::Color mover = boardNum == BOARD_A
            ? defenders : ~defenders;
        if (board.side_to_move(boardNum) != mover) {
            continue;
        }
        for (Stockfish::Move reply : board.legal_moves(boardNum)) {
            if (!board.is_capture(boardNum, reply)) {
                continue;
            }
            if (budget && !budget->consume()) {
                return false;
            }
            board.push_move(boardNum, reply);
            JointActionCandidate mate;
            const bool mates = find_immediate_root_mate(
                board, defenders, afterCapture, mate, budget);
            board.pop_move(boardNum);
            if (mates) {
                return true;
            }
        }
    }
    return false;
}

bool Agent::has_immediate_mate(Board& board, Stockfish::Color teamSide,
                               const TimeState& time) {
    JointActionCandidate mate;
    return find_immediate_root_mate(board, teamSide, time, mate);
}

bool Agent::action_loses_mate_race(Board& board,
                                   const JointActionCandidate& action,
                                   Stockfish::Color teamSide,
                                   bool teamHasTimeAdvantage,
                                   MateSearchBudget* budget) {
    return action_loses_mate_race(
        board, action, teamSide,
        TimeState::legacy(teamSide, teamHasTimeAdvantage), budget);
}

bool Agent::action_loses_mate_race(Board& board,
                                   const JointActionCandidate& action,
                                   Stockfish::Color teamSide,
                                   const TimeState& time,
                                   MateSearchBudget* budget) {
    if (!board.is_legal_move(BOARD_A, action.moveA)
        || !board.is_legal_move(BOARD_B, action.moveB)) {
        return false;
    }
    if (budget && budget->exhausted) {
        return false;
    }

    JointActionCandidate instantMate;
    if (find_immediate_root_mate(
            board, ~teamSide, time, instantMate, budget, true)) {
        return true;
    }

    const TimeState afterAction = time.after(
        teamSide, board.on_turn_on_both_boards(teamSide));
    board.make_moves(action.moveA, action.moveB);
    bool lost = false;
    // Mating ends the game where it stands, so no reply of theirs is played.
    // Anything short of that leaves them free to play a mate they already
    // hold, and the follow-up this action was chosen for never arrives. The
    // same goes for a mate they are one capture away from: the piece is in
    // their partner's hand the moment it is taken, so a forced recapture of
    // our checking piece is a mate for them, not a tempo for us.
    if (!board.is_checkmate(~teamSide, afterAction.can_sit(~teamSide))) {
        JointActionCandidate reply;
        lost = find_immediate_root_mate(
                   board, ~teamSide, afterAction, reply, budget)
            || defenders_capture_into_immediate_mate(
                   board, ~teamSide, afterAction, budget);
    }
    board.unmake_moves(action.moveA, action.moveB);
    return lost;
}

std::optional<JointActionCandidate> Agent::claimed_win_action(
    const Node& node, float qVetoDelta, float qValueWeight,
    bool avoidSolvedDraw) {
    if (!node.is_expanded()) {
        return std::nullopt;
    }
    const int index = node.get_best_move_idx_with_q_weight(
        qVetoDelta, qValueWeight, avoidSolvedDraw);
    if (index < 0 || static_cast<size_t>(index) >= node.get_num_generated()) {
        return std::nullopt;
    }
    const shared_ptr<Node> child = node.get_child(index);
    // The root proves a win by holding a child the opponent loses from. Read
    // both: the root is only marked once every branch below it agrees, and the
    // certificate can be on the edge before it reaches the node.
    if (node.get_node_type() != NodeType::WIN
        && (!child || child->get_node_type() != NodeType::LOSS)) {
        return std::nullopt;
    }
    return node.get_joint_action(index);
}

std::optional<JointActionCandidate> Agent::race_safe_alternative(
    Board& board, const Node& node, Stockfish::Color teamSide,
    bool teamHasTimeAdvantage, float qVetoDelta, float qValueWeight,
    bool avoidSolvedDraw, const std::atomic<bool>* cancelled) {
    return race_safe_alternative(
        board, node, teamSide,
        TimeState::legacy(teamSide, teamHasTimeAdvantage),
        qVetoDelta, qValueWeight, avoidSolvedDraw, cancelled);
}

std::optional<JointActionCandidate> Agent::race_safe_alternative(
    Board& board, const Node& node, Stockfish::Color teamSide,
    const TimeState& time, float qVetoDelta, float qValueWeight,
    bool avoidSolvedDraw, const std::atomic<bool>* cancelled) {
    const std::optional<JointActionCandidate> claimed = claimed_win_action(
        node, qVetoDelta, qValueWeight, avoidSolvedDraw);
    if (!claimed) {
        return std::nullopt;
    }
    const auto race_budget = [cancelled] {
        MateSearchBudget budget;
        budget.remainingNodes = SearchParams::MATE_RACE_VETO_NODE_BUDGET;
        budget.deadline = MateSearchBudget::Clock::now()
            + chrono::milliseconds(SearchParams::MATE_RACE_VETO_MAX_MS);
        budget.cancelled = cancelled;
        return budget;
    };
    MateSearchBudget claimBudget = race_budget();
    if (!action_loses_mate_race(
            board, *claimed, teamSide, time, &claimBudget)) {
        return std::nullopt;
    }

    const vector<int> visits = node.get_child_visits();
    const vector<shared_ptr<Node>> children = node.get_children();
    const size_t generated = std::min(
        {visits.size(), children.size(), node.get_num_generated()});
    vector<size_t> byVisits;
    byVisits.reserve(generated);
    for (size_t index = 0; index < generated; ++index) {
        const JointActionCandidate action =
            node.get_joint_action(static_cast<int>(index));
        if (action.moveA == claimed->moveA && action.moveB == claimed->moveB) {
            continue;
        }
        byVisits.push_back(index);
    }
    // Keep the replacement out of a mate of its own: a child the solver proved
    // a WIN is a proven loss for this team.
    const auto isProvenLoss = [&](size_t index) {
        return children[index]
            && children[index]->get_node_type() == NodeType::WIN;
    };
    std::stable_sort(
        byVisits.begin(), byVisits.end(), [&](size_t lhs, size_t rhs) {
            if (isProvenLoss(lhs) != isProvenLoss(rhs)) {
                return isProvenLoss(rhs);
            }
            return visits[lhs] > visits[rhs];
        });
    if (byVisits.size() > static_cast<size_t>(
            SearchParams::MATE_RACE_VETO_MAX_ALTERNATIVES)) {
        byVisits.resize(SearchParams::MATE_RACE_VETO_MAX_ALTERNATIVES);
    }
    for (size_t index : byVisits) {
        const JointActionCandidate candidate =
            node.get_joint_action(static_cast<int>(index));
        MateSearchBudget budget = race_budget();
        if (!action_loses_mate_race(
                board, candidate, teamSide, time, &budget)) {
            return candidate;
        }
    }
    return std::nullopt;
}

bool Agent::action_walks_into_certified_mate(
    Board& board, const JointActionCandidate& action,
    Stockfish::Color teamSide, bool teamHasTimeAdvantage,
    MateSearchBudget::Clock::time_point deadline,
    const std::atomic<bool>* cancelled,
    int& outPlyToMate, SelectedMoveCertStats* stats) {
    return action_walks_into_certified_mate(
        board, action, teamSide,
        TimeState::legacy(teamSide, teamHasTimeAdvantage),
        deadline, cancelled, outPlyToMate, stats);
}

bool Agent::action_walks_into_certified_mate(
    Board& board, const JointActionCandidate& action,
    Stockfish::Color teamSide, const TimeState& time,
    MateSearchBudget::Clock::time_point deadline,
    const std::atomic<bool>* cancelled,
    int& outPlyToMate, SelectedMoveCertStats* stats) {
    outPlyToMate = 0;
    // Ahead on time for longer than any line the opponents could be asked to
    // prove, this team may sit the board a line is played on, so no
    // single-board mate is forced against it: the probe would have nothing
    // to say and the certifier declines an attacker behind on the clock.
    if (time.can_sit_through(teamSide, OPPONENT_PROOF_HORIZON_MOVES)) {
        return false;
    }
    if (!board.is_legal_move(BOARD_A, action.moveA)
        || !board.is_legal_move(BOARD_B, action.moveB)) {
        return false;
    }
    const auto remaining_ms = [&] {
        return static_cast<int>(chrono::duration_cast<chrono::milliseconds>(
            deadline - MateSearchBudget::Clock::now()).count());
    };
    const auto abort = [&] {
        return (cancelled && cancelled->load(memory_order_relaxed))
            || remaining_ms() <= 0;
    };
    if (abort()) {
        return false;
    }
    if (stats) {
        ++stats->candidates;
    }

    const TimeState afterAction = time.after(
        teamSide, board.on_turn_on_both_boards(teamSide));
    board.make_moves(action.moveA, action.moveB);
    bool certified = false;
    // Mating them ends the game where it stands.
    if (!board.is_checkmate(~teamSide, afterAction.can_sit(~teamSide))) {
        JointActionCandidate reply;
        int replyPly = 0;
        string replyPv;
        uint64_t probeNodes = 0;
        const int probeMs = std::max(
            1, std::min(SearchParams::SELECTED_MOVE_PROBE_MAX_MS,
                        remaining_ms()));
        // Their proofs run under the clock this action leaves them: the
        // probe and the certifier only claim a line their lead can pay for.
        const bool found = probe_position_mate(
            board, ~teamSide, afterAction,
            SearchParams::SELECTED_MOVE_PROBE_NODE_BUDGET, probeMs, abort,
            reply, replyPly, replyPv, {}, false, &probeNodes);
        if (stats) {
            stats->nodes += probeNodes;
        }
        if (found && !abort()) {
            if (stats) {
                ++stats->probeHits;
            }
            MateSearchBudget budget;
            budget.remainingNodes = SearchParams::SELECTED_MOVE_CERT_NODE_BUDGET;
            budget.cancelled = cancelled;
            budget.deadline = std::min(
                deadline,
                MateSearchBudget::Clock::now() + chrono::milliseconds(
                    SearchParams::SELECTED_MOVE_CERT_PER_CANDIDATE_MAX_MS));
            int certifiedPly = 0;
            MateCertificateTier tier = MateCertificateTier::NONE;
            certified = certify_mate_candidate(
                board, ~teamSide, afterAction, reply, replyPly, budget,
                certifiedPly, tier);
            if (stats) {
                stats->certificateNodes +=
                    SearchParams::SELECTED_MOVE_CERT_NODE_BUDGET
                    - budget.remainingNodes;
                stats->certificates += certified ? 1 : 0;
            }
            if (certified) {
                outPlyToMate = certifiedPly;
            }
        }
        // Fairy searches one board with the hand it has, so a mate that
        // needs a piece captured on one board and dropped on the other is
        // invisible to it. The joint solver follows the pieces across; a
        // proof of its own is already the certificate.
        if (!certified && !abort()) {
            MateSearchBudget jointBudget;
            jointBudget.remainingNodes =
                SearchParams::SELECTED_MOVE_JOINT_NODE_BUDGET;
            jointBudget.cancelled = cancelled;
            jointBudget.deadline = deadline;
            JointActionCandidate jointAction;
            int jointPly = 0;
            certified = prove_joint_forced_mate(
                board, ~teamSide, afterAction,
                SearchParams::SELECTED_MOVE_JOINT_MAX_ATTACKER_MOVES,
                jointBudget, jointAction, jointPly, nullptr, false);
            if (stats) {
                stats->certificateNodes +=
                    SearchParams::SELECTED_MOVE_JOINT_NODE_BUDGET
                    - jointBudget.remainingNodes;
                stats->certificates += certified ? 1 : 0;
            }
            if (certified) {
                outPlyToMate = jointPly;
            }
        }
    }
    board.unmake_moves(action.moveA, action.moveB);
    return certified;
}

std::optional<JointActionCandidate> Agent::certified_mate_free_alternative(
    Board& board, const Node& node, const JointActionCandidate& chosen,
    Stockfish::Color teamSide, bool teamHasTimeAdvantage,
    MateSearchBudget::Clock::time_point deadline,
    const std::atomic<bool>* cancelled, SelectedMoveCertStats* stats) {
    return certified_mate_free_alternative(
        board, node, chosen, teamSide,
        TimeState::legacy(teamSide, teamHasTimeAdvantage),
        deadline, cancelled, stats);
}

std::optional<JointActionCandidate> Agent::certified_mate_free_alternative(
    Board& board, const Node& node, const JointActionCandidate& chosen,
    Stockfish::Color teamSide, const TimeState& time,
    MateSearchBudget::Clock::time_point deadline,
    const std::atomic<bool>* cancelled, SelectedMoveCertStats* stats) {
    if (time.can_sit_through(teamSide, OPPONENT_PROOF_HORIZON_MOVES)
        || !node.is_expanded()) {
        return std::nullopt;
    }
    const vector<int> visits = node.get_child_visits();
    const vector<shared_ptr<Node>> children = node.get_children();
    const size_t generated = std::min(
        {visits.size(), children.size(), node.get_num_generated()});
    const auto same_action = [](const JointActionCandidate& lhs,
                                const JointActionCandidate& rhs) {
        return lhs.moveA == rhs.moveA && lhs.moveB == rhs.moveB;
    };
    // The child is the opponents' node, so their proven mate is its WIN. The
    // solver carries that up to the root edge, and tree reuse keeps it.
    const auto isProvenLoss = [&](size_t index) {
        return children[index]
            && children[index]->get_node_type() == NodeType::WIN;
    };
    const auto record_proven_loss = [&](const JointActionCandidate& action,
                                        int ply) {
        for (size_t index = 0; index < generated; ++index) {
            if (children[index]
                && children[index]->get_node_type() == NodeType::UNSOLVED
                && same_action(
                    node.get_joint_action(static_cast<int>(index)), action)) {
                children[index]->mark_as_win(ply);
                return;
            }
        }
    };

    int ply = 0;
    if (!action_walks_into_certified_mate(
            board, chosen, teamSide, time, deadline,
            cancelled, ply, stats)) {
        return std::nullopt;
    }
    if (stats) {
        stats->vetoed = true;
    }
    record_proven_loss(chosen, ply);

    vector<size_t> byVisits;
    byVisits.reserve(generated);
    for (size_t index = 0; index < generated; ++index) {
        if (isProvenLoss(index)
            || same_action(
                node.get_joint_action(static_cast<int>(index)), chosen)) {
            continue;
        }
        byVisits.push_back(index);
    }
    std::stable_sort(
        byVisits.begin(), byVisits.end(), [&](size_t lhs, size_t rhs) {
            return visits[lhs] > visits[rhs];
        });
    if (byVisits.size() > static_cast<size_t>(
            SearchParams::SELECTED_MOVE_CERT_MAX_ALTERNATIVES)) {
        byVisits.resize(SearchParams::SELECTED_MOVE_CERT_MAX_ALTERNATIVES);
    }
    // An alternative the deadline cuts short is unproven rather than safe,
    // but unproven beats the certified loss it replaces.
    for (size_t index : byVisits) {
        const JointActionCandidate candidate =
            node.get_joint_action(static_cast<int>(index));
        int candidatePly = 0;
        if (!action_walks_into_certified_mate(
                board, candidate, teamSide, time, deadline,
                cancelled, candidatePly, stats)) {
            if (stats) {
                stats->replaced = true;
            }
            return candidate;
        }
        record_proven_loss(candidate, candidatePly);
    }
    return std::nullopt;
}

/**
 * @brief Enumerate every legal joint action without requiring policy priors.
 *
 * Adding MOVE_NONE to each active board and filtering through the shared
 * JointActionRules keeps capture-pass, forced-pass, and double-sit legality
 * identical to the MCTS candidate generator.
 */
static vector<JointActionCandidate> legal_joint_actions(
    Board& board, Stockfish::Color teamSide, const TimeState& time) {
    const bool teamHasTimeAdvantage = time.can_sit(teamSide);
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
                const bool feedsPartner = board.is_capture(boardNum, reply);
                board.push_move(boardNum, reply);
                Stockfish::Move nextAttackerMove = Stockfish::MOVE_NONE;
                int nextReplyPly = 0;
                std::vector<MateProofPly> childLine;
                // A capturing evasion puts the piece in the victim's
                // partner's hand at once. If that partner mates with it on
                // the waiting board, our next check is never played - so the
                // evasion refutes the line, however forced it looked here.
                // The proof assumes the attacking team is time-ahead, so
                // the victims are not.
                JointActionCandidate raceMate;
                const bool replyMated = feedsPartner && !partnerBoardAgnostic
                        && find_immediate_root_mate(
                            board, victimTeam, false, raceMate, budget)
                    ? false
                    : search_single_board_forced_mate_impl(
                        board, boardNum, attackerColor, currentPly + 2,
                        maxAttackerMoves, nextAttackerMove, nextReplyPly,
                        budget, continuations, partnerBoardAgnostic,
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

namespace joint_mate {

// Board the proof was made on with the partner board sat, or NO_REDUCED_BOARD
// for a full joint proof. The two answer different questions of the same
// position, so they never share an entry.
constexpr uint8_t NO_REDUCED_BOARD = 0xFF;

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
    // A refutation that somewhere ran out of attacker moves: a deeper
    // horizon could still prove it. Otherwise every line it refuted was
    // answered by the defence within the forcing moves searched.
    bool depthLimited = false;
};

struct JointMateCacheKey {
    uint64_t positionHash = 0;
    uint16_t attackerMovesRemaining = 0;
    uint8_t teamToPlay = 0;
    uint8_t reducedBoard = NO_REDUCED_BOARD;

    bool operator==(const JointMateCacheKey&) const = default;
};

struct JointMateCacheKeyHash {
    size_t operator()(const JointMateCacheKey& key) const {
        uint64_t mixed = Board::mix_hash(
            key.positionHash,
            static_cast<uint64_t>(key.attackerMovesRemaining));
        mixed = Board::mix_hash(mixed, static_cast<uint64_t>(key.teamToPlay));
        mixed = Board::mix_hash(mixed, static_cast<uint64_t>(key.reducedBoard));
        return static_cast<size_t>(mixed);
    }
};

using JointMateProofCache = std::unordered_map<
    JointMateCacheKey, JointMateProof, JointMateCacheKeyHash>;

/**
 * What a sat-partner-board attempt on one board state has already covered.
 * The board is identified by its own key, hand included, so it changes only
 * when a move is played there or a capture elsewhere feeds it. Until then a
 * retry with no more depth and no more nodes can only repeat the answer.
 */
struct ReducedAttemptRecord {
    int attackerMoves = 0;
    uint64_t nodes = 0;
    bool refuted = false;
};

struct JointMateCache {
    JointMateProofCache proofs;
    std::unordered_map<uint64_t, ReducedAttemptRecord> reducedAttempts;

    void reserve(size_t count) { proofs.reserve(count); }
};

}  // namespace joint_mate

namespace {

using namespace joint_mate;

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

/// Key of a sat-partner-board attempt on @p activeBoard: that board with its
/// hands, and the partner board while the defender can move there or
/// @p partnerBoardMatters (see defenders_may_wait_within below).
uint64_t reduced_partner_hash(Board& board, int activeBoard,
                              Stockfish::Color attackingTeam,
                              bool partnerBoardMatters) {
    const int partnerBoard = 1 - activeBoard;
    const bool defenderOnPartnerBoard = board.side_to_move(partnerBoard)
        == (partnerBoard == BOARD_A ? ~attackingTeam : attackingTeam);
    uint64_t hash = Board::mix_hash(
        board.pos[activeBoard]->key(),
        static_cast<uint64_t>(board.pos[activeBoard]->rule50_count()));
    if (defenderOnPartnerBoard || partnerBoardMatters) {
        hash = Board::mix_hash(hash, board.pos[partnerBoard]->key());
    }
    // The two modes search different defenses (one quiet partner move for
    // all, or every one), so they never share an answer.
    hash = Board::mix_hash(hash, partnerBoardMatters ? 1 : 0);
    return Board::mix_hash(hash, static_cast<uint64_t>(activeBoard));
}

/**
 * Whether the defenders could gain the right to sit within the attackers'
 * next @p attackerMoves actions. While they cannot, a sat partner board only
 * reaches a sat-partner proof through the defenders' moves on it, and any
 * quiet one stands for the rest. Once they can, a mate they may wait out is
 * no mate if their partner can capture them a blocker, so the frozen partner
 * board's position decides the terminal verdict and every quiet move there is
 * a different defense.
 */
bool defenders_may_wait_within(const TimeState& time,
                               Stockfish::Color attackingTeam,
                               int attackerMoves) {
    return !time.can_sit_through(attackingTeam, std::max(1, attackerMoves));
}

/**
 * @param treeNode The search tree's node for this position, when it has
 *        one. Its network priors are the only ordering the proof uses: each
 *        board's moves are listed by prior, highest first, so a depth-first
 *        proof meets the line the network expects before the rest. Without
 *        a node the moves stay in generation order.
 */
MateActionSpace make_mate_action_space(Board& board,
                                       Stockfish::Color teamToPlay,
                                       bool teamHasTimeAdvantage,
                                       const Node* treeNode = nullptr) {
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
            if (treeNode) {
                std::vector<std::pair<float, size_t>> byPrior;
                byPrior.reserve(candidates.size());
                for (size_t index = 0; index < candidates.size(); ++index) {
                    byPrior.emplace_back(
                        -treeNode->move_prior(boardNum, candidates[index].move),
                        index);
                }
                std::stable_sort(byPrior.begin(), byPrior.end());
                std::vector<MateMoveCandidate> ordered;
                ordered.reserve(candidates.size());
                for (const auto& [negPrior, index] : byPrior) {
                    ordered.push_back(candidates[index]);
                }
                candidates = std::move(ordered);
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

/**
 * @param keepQuietPairs Visit a forcing action paired with a quiet move on
 *        the other board too. The default leaves those out on the strength of
 *        the sit pair, which asks the defenders the same question; but a team
 *        on turn on both boards without the lead may not sit, so where that
 *        sit is illegal the pairing is the only way to play the check.
 */
template<typename Visitor>
bool visit_legal_joint_actions(const MateActionSpace& space,
                               bool forcingOnly,
                               bool quietOnly,
                               bool victimAlreadyInCheck,
                               Visitor&& visitor,
                               bool keepQuietPairs = false) {
    // A forcing line is nearly always one check with the other board sat:
    // a check paired with any of the partner board's moves reaches the same
    // defender replies plus whatever the extra move gave away. Visit the
    // check-and-sit pairs first so a proof is found before the product of
    // every check with every partner move is walked. Order is all this
    // changes; every legal action is still visited.
    const auto visit = [&](bool sitPairsOnly) {
        for (size_t indexA = 0; indexA < space.actionsA.size(); ++indexA) {
            const MateMoveCandidate& actionA = space.actionsA[indexA];
            for (size_t indexB = 0; indexB < space.actionsB.size(); ++indexB) {
                const MateMoveCandidate& actionB = space.actionsB[indexB];
                const bool sitPair = actionA.move == Stockfish::MOVE_NONE
                    || actionB.move == Stockfish::MOVE_NONE;
                if (sitPair != sitPairsOnly) {
                    continue;
                }
                const bool forcing = victimAlreadyInCheck
                    || actionA.givesCheck || actionB.givesCheck;
                if ((forcingOnly && !forcing) || (quietOnly && forcing)) {
                    continue;
                }
                // A check paired with a quiet move on the other board asks
                // the defender the same question as the check alone, which
                // the sit pair above already asked. What the pairing can add
                // is a capture that feeds a hand, or a second check. The
                // rest is left out: fewer proofs are reachable, never a
                // false one.
                if (forcingOnly && !sitPair && !keepQuietPairs
                    && !(actionA.givesCheck || actionA.isCapture)
                    && !(actionB.givesCheck || actionB.isCapture)) {
                    continue;
                }
                if (forcingOnly && !sitPair && !keepQuietPairs
                    && ((actionA.givesCheck && !actionB.givesCheck
                         && !actionB.isCapture)
                        || (actionB.givesCheck && !actionA.givesCheck
                            && !actionA.isCapture))) {
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
    };
    return visit(true) || visit(false);
}

JointMateStatus terminal_joint_mate_status(
    Board& board,
    Stockfish::Color attackingTeam,
    const TimeState& time,
    int searchPly) {
    const Stockfish::Color victimTeam = ~attackingTeam;
    Board::LegalMoveCache legalMoveCache;
    if (board.is_checkmate(
            victimTeam, time.can_sit(victimTeam), &legalMoveCache)) {
        return JointMateStatus::PROVEN;
    }
    if (board.is_checkmate(
            attackingTeam, time.can_sit(attackingTeam), &legalMoveCache)
        || board.is_draw(searchPly)) {
        return JointMateStatus::REFUTED;
    }
    return JointMateStatus::UNKNOWN;
}

bool team_is_in_check(Board& board, Stockfish::Color team) {
    return (board.side_to_move(BOARD_A) == team && board.is_in_check(BOARD_A))
        || (board.side_to_move(BOARD_B) == ~team && board.is_in_check(BOARD_B));
}

/**
 * A capture whose piece the partner drops at once for mate. The drop is
 * judged as a bughouse mate under the clock the capture leaves: the victims
 * may be saved by a blocker their own partner can capture, now or, if they
 * may wait, later.
 */
bool find_immediate_capture_feed_mate(
    Board& board,
    Stockfish::Color attackingTeam,
    const TimeState& time,
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
    // On turn on both boards, the capture is charged; the drop that follows
    // is the partner's move on the other board within the same action.
    const bool victimsMaySit = time.after(attackingTeam, true).can_sit(~attackingTeam);

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
                        && board.is_checkmate(~attackingTeam, victimsMaySit);
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

JointMateProof search_reduced_partner_mate(
    Board& board,
    Stockfish::Color attackingTeam,
    int activeBoard,
    Stockfish::Color attackerColor,
    Stockfish::Color teamToPlay,
    const TimeState& time,
    int attackerMovesRemaining,
    int searchPly,
    Agent::MateSearchBudget& budget,
    JointMateCache* cache = nullptr,
    const Node* treeNode = nullptr);

/// The tree's child for the joint action played from @p treeNode, or null.
/// Frames of the recursion hold the pointer they were given alive, so the
/// child stays valid for as long as the frame that looked it up.
std::shared_ptr<Node> tree_child_for(const Node* treeNode,
                                     Stockfish::Move moveA,
                                     Stockfish::Move moveB) {
    return treeNode ? treeNode->child_for_action(moveA, moveB) : nullptr;
}

/**
 * Checking moves on @p boardNum by the tree node's priors, highest first,
 * when the tree has the position; generation order otherwise.
 */
std::vector<Stockfish::Move> checks_by_prior(Board& board, int boardNum,
                                             const Node* treeNode) {
    std::vector<Stockfish::Move> checks = board.checking_moves(boardNum);
    if (!treeNode) {
        return checks;
    }
    std::vector<std::pair<float, Stockfish::Move>> ranked;
    ranked.reserve(checks.size());
    for (Stockfish::Move move : checks) {
        ranked.emplace_back(-treeNode->move_prior(boardNum, move), move);
    }
    std::stable_sort(ranked.begin(), ranked.end(),
                     [](const auto& lhs, const auto& rhs) {
                         return lhs.first < rhs.first;
                     });
    for (size_t index = 0; index < ranked.size(); ++index) {
        checks[index] = ranked[index].second;
    }
    return checks;
}

/**
 * @param time The clock at this position. Every action the solver plays is
 *        charged to the team playing it before the recursion, so each node's
 *        sit rights, terminal status and cache entry follow its own path.
 * @param waitingBoard The board the line does not mate on, when the caller
 *        knows it (BOARD_A or BOARD_B), or -1. Attackers who may not sit and
 *        are on turn there alone - the defenders have sat the mating board -
 *        must play some move there, check or not, so every legal one is
 *        theirs to try. Without it the attackers play forcing actions only.
 */
JointMateProof search_joint_forced_mate(
    Board& board,
    Stockfish::Color attackingTeam,
    const TimeState& time,
    Stockfish::Color teamToPlay,
    int attackerMovesRemaining,
    int searchPly,
    Agent::MateSearchBudget& budget,
    JointMateCache& cache,
    bool attackerWinsMateRace,
    const Node* treeNode = nullptr,
    int waitingBoard = -1) {
    const JointMateStatus terminal = terminal_joint_mate_status(
        board, attackingTeam, time, searchPly);
    if (terminal != JointMateStatus::UNKNOWN) {
        return {terminal, 0, {}};
    }

    const bool attackerToPlay = teamToPlay == attackingTeam;
    if (attackerToPlay && attackerMovesRemaining <= 0) {
        JointMateProof outOfMoves{JointMateStatus::REFUTED, 0, {}};
        outOfMoves.depthLimited = true;
        return outOfMoves;
    }
    if (budget.exhausted || budget.out_of_time()) {
        budget.exhausted = true;
        return {JointMateStatus::UNKNOWN, 0, {}};
    }

    const bool teamToPlayHasTimeAdvantage = time.can_sit(teamToPlay);
    // The search hash carries the exact clock, so a proof is only shared by
    // positions with the same sit rights for the rest of the line.
    const JointMateCacheKey cacheKey{
        board.search_hash_key(teamToPlay, time),
        static_cast<uint16_t>(std::max(0, attackerMovesRemaining)),
        static_cast<uint8_t>(teamToPlay)};
    // Whatever the action, it costs the same: it depends only on whether
    // this team is on turn on both boards here.
    const TimeState childTime = time.after(
        teamToPlay, board.on_turn_on_both_boards(teamToPlay));
    if (const auto found = cache.proofs.find(cacheKey);
        found != cache.proofs.end()) {
        return found->second;
    }

    const MateActionSpace actionSpace = make_mate_action_space(
        board, teamToPlay, teamToPlayHasTimeAdvantage, treeNode);

    if (attackerToPlay) {
        MateJointAction feedMateAction;
        if (find_immediate_capture_feed_mate(
            board, attackingTeam, time, attackerWinsMateRace,
            budget, feedMateAction)) {
            const JointMateProof result{
                JointMateStatus::PROVEN, 1, feedMateAction,
                std::make_shared<JointMateProofLine>(
                    JointMateProofLine{feedMateAction, nullptr})};
            cache.proofs.emplace(cacheKey, result);
            return result;
        }
        if (budget.exhausted) {
            return {JointMateStatus::UNKNOWN, 0, {}};
        }

        // Ahead on time, the attacker may sit one board for the rest of the
        // line and mate on the other. That proof is a restriction of this
        // one - the attacker gives up its moves on the sat board - so it is
        // valid here, and it is far cheaper: the joint product below pairs
        // every check with every partner-board move and answers each with
        // every defender reply. Try it first on each board the attacker is
        // on turn, and fall through to the joint product when it fails.
        // A mate that is there is found in a few hundred nodes with the
        // checks ordered by evasions; one that is not can burn hundreds of
        // thousands refuting every check line. The attempt is an
        // accelerator, so it gets a small slice of the budget and hands an
        // unfinished search back to the joint product rather than a claim.
        // The sat-board search charges every action to the real clock, so
        // any line it proves is legal here; it only needs the attacker to be
        // ahead now, or there is no sitting to begin with.
        if (attackerMovesRemaining >= 1 && time.can_sit(attackingTeam)) {
            for (int activeBoard : {BOARD_A, BOARD_B}) {
                const Stockfish::Color attackerColor = activeBoard == BOARD_A
                    ? attackingTeam : ~attackingTeam;
                if (board.side_to_move(activeBoard) != attackerColor) {
                    continue;
                }
                // The board this attempt searches changes only when a move
                // lands there or a capture elsewhere feeds its hand. While
                // it stands still, an attempt with no more depth and no
                // more nodes than the last one repeats that answer, so it
                // is skipped; a refutation at more depth covers less.
                const uint64_t attemptKey = Board::mix_hash(
                    reduced_partner_hash(
                        board, activeBoard, attackingTeam,
                        defenders_may_wait_within(
                            time, attackingTeam, attackerMovesRemaining)),
                    time.exact_key());
                const uint64_t attemptNodes = std::min(
                    budget.remainingNodes,
                    SearchParams::JOINT_MATE_REDUCED_ATTEMPT_NODES);
                if (const auto record = cache.reducedAttempts.find(attemptKey);
                    record != cache.reducedAttempts.end()
                    && record->second.attackerMoves >= attackerMovesRemaining
                    && (record->second.refuted
                        || record->second.nodes >= attemptNodes)) {
                    continue;
                }
                Agent::MateSearchBudget slice = budget;
                slice.remainingNodes = attemptNodes;
                const JointMateProof reduced = search_reduced_partner_mate(
                    board, attackingTeam, activeBoard, attackerColor,
                    attackingTeam, time, attackerMovesRemaining, searchPly,
                    slice, &cache, treeNode);
                const uint64_t spent = attemptNodes - slice.remainingNodes;
                budget.remainingNodes -= std::min(budget.remainingNodes, spent);
                if (reduced.status != JointMateStatus::PROVEN
                    && !slice.out_of_time()) {
                    ReducedAttemptRecord& record =
                        cache.reducedAttempts[attemptKey];
                    if (attackerMovesRemaining >= record.attackerMoves) {
                        record.attackerMoves = attackerMovesRemaining;
                        record.nodes = std::max(record.nodes, attemptNodes);
                        record.refuted =
                            reduced.status == JointMateStatus::REFUTED;
                    }
                }
                if (reduced.status == JointMateStatus::PROVEN) {
                    JointMateProof result = reduced;
                    result.principalVariation =
                        std::make_shared<JointMateProofLine>(
                            JointMateProofLine{reduced.action, nullptr});
                    cache.proofs.emplace(cacheKey, result);
                    return result;
                }
                if (budget.remainingNodes == 0 || slice.out_of_time()) {
                    budget.exhausted = true;
                    return {JointMateStatus::UNKNOWN, 0, {}};
                }
            }
        }

        const bool victimAlreadyInCheck = team_is_in_check(
            board, ~attackingTeam);
        JointMateProof result{JointMateStatus::REFUTED, 0, {}};
        bool sawUnknown = false;

        // Widening rather than one pass: each forcing action gets a small
        // slice of the budget, those still unknown are kept, and the slice
        // grows each round. A depth-first pass has to refute every wrong
        // check completely before the next is tried, and with a full hand
        // there is always another check, so that refutation is itself a
        // search of the whole depth. Here a check that does not prove
        // quickly is set aside while the others get their turn, and the
        // shared cache keeps whatever its slice settled for the next round.
        // A caller that names the waiting board is certifying a line under
        // the real clock, where attackers without the lead may have to play
        // their check beside a move on the other board.
        const bool keepQuietPairs =
            waitingBoard >= 0 && !teamToPlayHasTimeAdvantage;
        std::vector<MateJointAction> forcing;
        visit_legal_joint_actions(
            actionSpace, true, false, victimAlreadyInCheck,
            [&](const MateJointAction& action) {
                forcing.push_back(action);
                return false;
            },
            keepQuietPairs);
        // Sat on the mating board by defenders with the lead, attackers who
        // may not sit have to move on the waiting board. A check there is no
        // more theirs than any other move, so the quiet ones join the list.
        const bool onWaitingBoardAlone = waitingBoard == BOARD_A
            ? actionSpace.rules.boardAOnTurn && !actionSpace.rules.boardBOnTurn
            : waitingBoard == BOARD_B
                && actionSpace.rules.boardBOnTurn
                && !actionSpace.rules.boardAOnTurn;
        if (onWaitingBoardAlone && !teamToPlayHasTimeAdvantage
            && !victimAlreadyInCheck) {
            visit_legal_joint_actions(
                actionSpace, false, true, false,
                [&](const MateJointAction& action) {
                    forcing.push_back(action);
                    return false;
                });
        }
        // Where the tree has this position, its network priors decide how
        // the round's nodes are shared out: an action's slice is the round's
        // per-action slice scaled by its joint prior relative to the rest,
        // never below a floor, so the line the network expects is searched
        // deep first while every other action still gets its turn. Without
        // a node the shares are equal.
        std::vector<float> share(forcing.size(), 1.0f);
        if (treeNode && !forcing.empty()) {
            float total = 0.0f;
            for (size_t index = 0; index < forcing.size(); ++index) {
                share[index] = treeNode->move_prior(BOARD_A, forcing[index].moveA)
                    * treeNode->move_prior(BOARD_B, forcing[index].moveB);
                total += share[index];
            }
            for (float& value : share) {
                value = total > 0.0f
                    ? std::max(
                          SearchParams::JOINT_MATE_WIDENING_SHARE_FLOOR,
                          value / total * static_cast<float>(forcing.size()))
                    : 1.0f;
            }
            std::vector<size_t> byShare(forcing.size());
            std::iota(byShare.begin(), byShare.end(), 0);
            std::stable_sort(byShare.begin(), byShare.end(),
                             [&](size_t lhs, size_t rhs) {
                                 return share[lhs] > share[rhs];
                             });
            std::vector<MateJointAction> ordered;
            std::vector<float> orderedShare;
            ordered.reserve(forcing.size());
            orderedShare.reserve(forcing.size());
            for (size_t index : byShare) {
                ordered.push_back(forcing[index]);
                orderedShare.push_back(share[index]);
            }
            forcing = std::move(ordered);
            share = std::move(orderedShare);
        }
        // With the victims already in check, waiting is how the attacker
        // makes them answer it: they have to play the evasion while the
        // attacker commits nothing, and their other board cannot help them
        // once they have to move. It is also the one action no network prior
        // speaks for, and it comes last in move order, so give it the lead
        // and as much of each round as every other action together.
        if (victimAlreadyInCheck) {
            const auto wait = std::find_if(
                forcing.begin(), forcing.end(),
                [](const MateJointAction& action) {
                    return action.moveA == Stockfish::MOVE_NONE
                        && action.moveB == Stockfish::MOVE_NONE;
                });
            if (wait != forcing.end()) {
                const size_t index = static_cast<size_t>(wait - forcing.begin());
                std::rotate(forcing.begin(), wait, wait + 1);
                std::rotate(share.begin(), share.begin() + index,
                            share.begin() + index + 1);
                float others = 0.0f;
                for (size_t other = 1; other < share.size(); ++other) {
                    others += share[other];
                }
                share.front() = std::max(share.front(), others);
            }
        }
        std::vector<uint8_t> open(forcing.size(), 1);
        uint64_t slice = SearchParams::JOINT_MATE_WIDENING_BASE_NODES;
        bool widened = !forcing.empty();
        while (widened) {
            widened = false;
            bool anyUnknown = false;
            bool anyOpen = false;
            for (size_t index = 0; index < forcing.size(); ++index) {
                if (!open[index]) {
                    continue;
                }
                anyOpen = true;
                if (!budget.consume()) {
                    sawUnknown = true;
                    break;
                }
                Agent::MateSearchBudget childBudget = budget;
                childBudget.remainingNodes = std::min(
                    budget.remainingNodes,
                    std::max<uint64_t>(
                        1, static_cast<uint64_t>(
                            static_cast<double>(slice) * share[index])));
                const MateJointAction& action = forcing[index];
                const std::shared_ptr<Node> childNode = tree_child_for(
                    treeNode, action.moveA, action.moveB);
                board.make_moves(action.moveA, action.moveB);
                JointMateProof child = search_joint_forced_mate(
                    board, attackingTeam, childTime,
                    ~teamToPlay, attackerMovesRemaining - 1, searchPly + 1,
                    childBudget, cache, attackerWinsMateRace, childNode.get(),
                    waitingBoard);
                board.unmake_moves(action.moveA, action.moveB);
                const uint64_t spent = std::min(
                    budget.remainingNodes,
                    std::max<uint64_t>(
                        1, static_cast<uint64_t>(
                            static_cast<double>(slice) * share[index])))
                    - childBudget.remainingNodes;
                budget.remainingNodes -= std::min(budget.remainingNodes, spent);
                if (budget.remainingNodes == 0) {
                    budget.exhausted = true;
                }
                if (child.status == JointMateStatus::PROVEN) {
                    result = {JointMateStatus::PROVEN,
                              child.pliesToMate + 1, action};
                    result.principalVariation =
                        std::make_shared<JointMateProofLine>(
                            JointMateProofLine{
                                action, child.principalVariation});
                    break;
                }
                if (child.status == JointMateStatus::REFUTED) {
                    open[index] = 0;
                    result.depthLimited =
                        result.depthLimited || child.depthLimited;
                    continue;
                }
                anyUnknown = true;
                if (budget.exhausted || budget.out_of_time()) {
                    sawUnknown = true;
                    break;
                }
            }
            if (result.status == JointMateStatus::PROVEN || sawUnknown
                || !anyOpen || !anyUnknown) {
                break;
            }
            // Nothing left to widen into: the last round already offered
            // every open action all that remained.
            if (slice >= budget.remainingNodes) {
                sawUnknown = true;
                break;
            }
            slice *= SearchParams::JOINT_MATE_WIDENING_FACTOR;
            widened = true;
        }
        if (result.status == JointMateStatus::PROVEN) {
            cache.proofs.emplace(cacheKey, result);
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
                    board, attackingTeam, childTime, searchPly + 1);
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
            cache.proofs.emplace(cacheKey, result);
            return result;
        }
        if (sawUnknown || budget.exhausted) {
            return {JointMateStatus::UNKNOWN, 0, {}};
        }
        cache.proofs.emplace(cacheKey, result);
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
        const bool feedsPartner =
            (action.moveA != Stockfish::MOVE_NONE
             && board.is_capture(BOARD_A, action.moveA))
            || (action.moveB != Stockfish::MOVE_NONE
                && board.is_capture(BOARD_B, action.moveB));
        board.make_moves(action.moveA, action.moveB);
        // A defender's capture is in their partner's hand at once, and that
        // partner does not wait for our next action to drop it. A reply that
        // captures into an immediate mate therefore ends the line before the
        // attacker moves again, whatever the alternating model says.
        JointActionCandidate raceMate;
        const std::shared_ptr<Node> childNode = tree_child_for(
            treeNode, action.moveA, action.moveB);
        const JointMateProof child = feedsPartner && !attackerWinsMateRace
                && find_immediate_root_mate(
                    board, teamToPlay, childTime, raceMate, &budget)
            ? JointMateProof{JointMateStatus::REFUTED, 0, {}}
            : search_joint_forced_mate(
                board, attackingTeam, childTime,
                ~teamToPlay, attackerMovesRemaining, searchPly + 1,
                budget, cache, attackerWinsMateRace, childNode.get(),
                waitingBoard);
        board.unmake_moves(action.moveA, action.moveB);

        if (child.status == JointMateStatus::REFUTED) {
            result = {JointMateStatus::REFUTED, 0, {}};
            result.depthLimited = child.depthLimited;
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
        cache.proofs.emplace(cacheKey, result);
        return result;
    }
    if (!sawAction || sawUnknown || budget.exhausted) {
        return {sawAction ? JointMateStatus::UNKNOWN : JointMateStatus::REFUTED,
                0, {}};
    }
    cache.proofs.emplace(cacheKey, result);
    return result;
}

/**
 * The attackers' win in a race on one board while the other stands still:
 * @p raceBoard is frozen against the defenders, or about to be, and the
 * defenders can only hope to mate on the other board before their lead runs
 * out.
 *
 * Every defender action is searched - the full legal set, the capture-feed
 * escape included - so a proof here is a proof of the joint game. The
 * attackers are restricted, which only ever costs proofs: on @p raceBoard
 * they pass or give check (the freezing move, the mate), and on the other
 * board every legal move is theirs, the quiet defensive ones above all, which
 * the forcing prover never plays. The clock is charged after every action.
 */
JointMateProof search_frozen_race(
    Board& board,
    Stockfish::Color attackingTeam,
    const TimeState& time,
    Stockfish::Color teamToPlay,
    int attackerMovesRemaining,
    int searchPly,
    Agent::MateSearchBudget& budget,
    JointMateCache& cache,
    int raceBoard) {
    // Terminals as the tree judges them - the team to play's opponents
    // mated first, then the team itself, then the game's own draws - so a
    // certificate never claims a position the tree would score otherwise.
    // The attackers count as the root team, so a waiting-board mate against
    // the defenders is held to its race and one against the attackers is not.
    switch (classify_terminal_position(
                board, teamToPlay, attackingTeam, time,
                std::array<int, 2>{1, 1})) {
    case TerminalOutcome::WIN:
        return {teamToPlay == attackingTeam ? JointMateStatus::PROVEN
                                            : JointMateStatus::REFUTED, 0, {}};
    case TerminalOutcome::LOSS:
        return {teamToPlay == attackingTeam ? JointMateStatus::REFUTED
                                            : JointMateStatus::PROVEN, 0, {}};
    case TerminalOutcome::DRAW:
        return {JointMateStatus::REFUTED, 0, {}};
    case TerminalOutcome::NONE:
        break;
    }
    const bool attackerToPlay = teamToPlay == attackingTeam;
    if (attackerToPlay && attackerMovesRemaining <= 0) {
        return {JointMateStatus::REFUTED, 0, {}};
    }
    if (budget.exhausted || budget.out_of_time()) {
        budget.exhausted = true;
        return {JointMateStatus::UNKNOWN, 0, {}};
    }
    const JointMateCacheKey cacheKey{
        board.search_hash_key(teamToPlay, time),
        static_cast<uint16_t>(std::max(0, attackerMovesRemaining)),
        static_cast<uint8_t>(teamToPlay),
        static_cast<uint8_t>(0xF0 | raceBoard)};
    if (const auto found = cache.proofs.find(cacheKey);
        found != cache.proofs.end()) {
        return found->second;
    }
    const TimeState childTime = time.after(
        teamToPlay, board.on_turn_on_both_boards(teamToPlay));
    const MateActionSpace actionSpace = make_mate_action_space(
        board, teamToPlay, time.can_sit(teamToPlay));
    const int otherBoard = 1 - raceBoard;

    if (attackerToPlay) {
        // Order only: a sit first, since it is what makes the defenders move
        // on the frozen board; then checks on the race board; then the other
        // board's captures and checks; then the rest.
        std::vector<std::pair<int, MateJointAction>> actions;
        bool enumerationCut = false;
        visit_legal_joint_actions(
            actionSpace, false, false, false,
            [&](const MateJointAction& action) {
                if (!budget.consume()) {
                    enumerationCut = true;
                    return true;
                }
                const Stockfish::Move raceMove = raceBoard == BOARD_A
                    ? action.moveA : action.moveB;
                const Stockfish::Move otherMove = raceBoard == BOARD_A
                    ? action.moveB : action.moveA;
                int rank = 5;
                if (raceMove != Stockfish::MOVE_NONE) {
                    if (!board.gives_check(raceBoard, raceMove)) {
                        return false;
                    }
                    rank = 1;
                } else if (otherMove == Stockfish::MOVE_NONE) {
                    rank = 0;
                } else if (board.is_capture(otherBoard, otherMove)) {
                    rank = 2;
                } else if (board.gives_check(otherBoard, otherMove)) {
                    rank = 3;
                }
                actions.emplace_back(rank, action);
                return false;
            });
        if (enumerationCut) {
            return {JointMateStatus::UNKNOWN, 0, {}};
        }
        std::stable_sort(actions.begin(), actions.end(),
                         [](const auto& lhs, const auto& rhs) {
                             return lhs.first < rhs.first;
                         });
        bool sawUnknown = false;
        for (const auto& [rank, action] : actions) {
            if (!budget.consume()) {
                sawUnknown = true;
                break;
            }
            board.make_moves(action.moveA, action.moveB);
            const JointMateProof child = search_frozen_race(
                board, attackingTeam, childTime, ~teamToPlay,
                attackerMovesRemaining - 1, searchPly + 1, budget, cache,
                raceBoard);
            board.unmake_moves(action.moveA, action.moveB);
            if (child.status == JointMateStatus::PROVEN) {
                const JointMateProof result{
                    JointMateStatus::PROVEN, child.pliesToMate + 1, action,
                    std::make_shared<JointMateProofLine>(
                        JointMateProofLine{action, child.principalVariation})};
                cache.proofs.emplace(cacheKey, result);
                return result;
            }
            if (child.status == JointMateStatus::UNKNOWN) {
                sawUnknown = true;
                if (budget.exhausted) {
                    break;
                }
            }
        }
        if (sawUnknown) {
            return {JointMateStatus::UNKNOWN, 0, {}};
        }
        const JointMateProof refuted{JointMateStatus::REFUTED, 0, {}};
        cache.proofs.emplace(cacheKey, refuted);
        return refuted;
    }

    // The defenders: every legal action must lose. A capture whose piece
    // hands their partner an immediate mate ends the line in their favour
    // before the attackers act, as in search_joint_forced_mate.
    JointMateProof result{JointMateStatus::PROVEN, 0, {}};
    bool sawAction = false;
    bool sawUnknown = false;
    visit_legal_joint_actions(
        actionSpace, false, false, false,
        [&](const MateJointAction& action) {
            sawAction = true;
            if (!budget.consume()) {
                sawUnknown = true;
                return true;
            }
            const bool feedsPartner =
                (action.moveA != Stockfish::MOVE_NONE
                 && board.is_capture(BOARD_A, action.moveA))
                || (action.moveB != Stockfish::MOVE_NONE
                    && board.is_capture(BOARD_B, action.moveB));
            board.make_moves(action.moveA, action.moveB);
            JointActionCandidate raceMate;
            const JointMateProof child = feedsPartner
                    && find_immediate_root_mate(
                        board, teamToPlay, childTime, raceMate, &budget)
                ? JointMateProof{JointMateStatus::REFUTED, 0, {}}
                : search_frozen_race(
                    board, attackingTeam, childTime, ~teamToPlay,
                    attackerMovesRemaining, searchPly + 1, budget, cache,
                    raceBoard);
            board.unmake_moves(action.moveA, action.moveB);
            if (child.status == JointMateStatus::REFUTED) {
                result = {JointMateStatus::REFUTED, 0, {}};
                return true;
            }
            if (child.status == JointMateStatus::UNKNOWN) {
                sawUnknown = true;
                return budget.exhausted;
            }
            if (child.pliesToMate + 1 > result.pliesToMate
                || !result.principalVariation) {
                result.pliesToMate = child.pliesToMate + 1;
                result.action = action;
                result.principalVariation =
                    std::make_shared<JointMateProofLine>(
                        JointMateProofLine{action, child.principalVariation});
            }
            return false;
        });
    if (result.status == JointMateStatus::REFUTED) {
        cache.proofs.emplace(cacheKey, result);
        return result;
    }
    if (!sawAction || sawUnknown || budget.exhausted) {
        return {sawAction ? JointMateStatus::UNKNOWN : JointMateStatus::REFUTED,
                0, {}};
    }
    cache.proofs.emplace(cacheKey, result);
    return result;
}

/**
 * A mate on @p activeBoard with the attackers sitting the partner board for
 * the whole line: a restriction of the joint game, so a proof here is a proof
 * there. @p time is the clock at this position and every action is charged
 * to the team playing it, so each sit, each defender wait and each terminal
 * verdict follows the real lead. Under a legacy state that is the old model:
 * the attackers may always sit and the defenders never may.
 */
JointMateProof search_reduced_partner_mate(
    Board& board,
    Stockfish::Color attackingTeam,
    int activeBoard,
    Stockfish::Color attackerColor,
    Stockfish::Color teamToPlay,
    const TimeState& time,
    int attackerMovesRemaining,
    int searchPly,
    Agent::MateSearchBudget& budget,
    JointMateCache* cache,
    const Node* treeNode) {
    const JointMateStatus terminal = terminal_joint_mate_status(
        board, attackingTeam, time, searchPly);
    if (terminal != JointMateStatus::UNKNOWN) {
        return {terminal, 0, {}};
    }
    if (budget.exhausted || budget.out_of_time()) {
        return {JointMateStatus::UNKNOWN, 0, {}};
    }

    // The partner board is sat, so it reaches this proof only through the
    // defender's moves there, and those exist only while the defender is on
    // turn there - or through the terminal verdict, once the defenders may
    // wait for a blocker from it. Otherwise the proof depends on the active
    // board alone, hands included, and the clock, and is shared by every line
    // that reaches the same pair.
    const int partnerBoard = 1 - activeBoard;
    const bool defendersMayWait = defenders_may_wait_within(
        time, attackingTeam, attackerMovesRemaining);
    const JointMateCacheKey cacheKey{
        Board::mix_hash(
            reduced_partner_hash(
                board, activeBoard, attackingTeam, defendersMayWait),
            time.exact_key()),
        static_cast<uint16_t>(std::max(0, attackerMovesRemaining)),
        static_cast<uint8_t>(teamToPlay),
        static_cast<uint8_t>(activeBoard)};
    if (cache) {
        if (const auto found = cache->proofs.find(cacheKey);
            found != cache->proofs.end()) {
            return found->second;
        }
    }
    const auto remember = [&](const JointMateProof& proof) {
        if (cache && proof.status != JointMateStatus::UNKNOWN) {
            cache->proofs.emplace(cacheKey, proof);
        }
        return proof;
    };

    // Whatever the action, it costs the same: it depends only on whether
    // this team is on turn on both boards here.
    const bool onTurnOnBothBoards = board.on_turn_on_both_boards(teamToPlay);
    const TimeState childTime = time.after(teamToPlay, onTurnOnBothBoards);

    if (teamToPlay == attackingTeam) {
        // Sitting the partner board while on turn there needs the lead.
        if (attackerMovesRemaining <= 0
            || board.side_to_move(activeBoard) != attackerColor
            || (onTurnOnBothBoards && !time.can_sit(attackingTeam))) {
            return remember({JointMateStatus::REFUTED, 0, {}});
        }
        // A check whose line came back unknown (the intervention cutoff,
        // say) is not refuted, so neither is this position.
        bool sawUnknownCheck = false;
        for (Stockfish::Move move
             : checks_by_prior(board, activeBoard, treeNode)) {
            if (!budget.consume()) {
                return {JointMateStatus::UNKNOWN, 0, {}};
            }
            const MateJointAction action = activeBoard == BOARD_A
                ? MateJointAction{move, Stockfish::MOVE_NONE}
                : MateJointAction{Stockfish::MOVE_NONE, move};
            const std::shared_ptr<Node> childNode = tree_child_for(
                treeNode, action.moveA, action.moveB);
            board.make_moves(action.moveA, action.moveB);
            JointMateProof child = search_reduced_partner_mate(
                board, attackingTeam, activeBoard, attackerColor,
                ~teamToPlay, childTime, attackerMovesRemaining - 1,
                searchPly + 1, budget, cache, childNode.get());
            board.unmake_moves(action.moveA, action.moveB);
            if (child.status == JointMateStatus::PROVEN) {
                return remember({JointMateStatus::PROVEN,
                                 child.pliesToMate + 1, action,
                                 std::make_shared<JointMateProofLine>(
                                     JointMateProofLine{
                                         action, child.principalVariation})});
            }
            if (child.status == JointMateStatus::UNKNOWN
                && budget.exhausted) {
                return child;
            }
            sawUnknownCheck = sawUnknownCheck
                || child.status == JointMateStatus::UNKNOWN;
        }
        return budget.exhausted || sawUnknownCheck
            ? JointMateProof{JointMateStatus::UNKNOWN, 0, {}}
            : remember({JointMateStatus::REFUTED, 0, {}});
    }

    MateActionSpace actionSpace = make_mate_action_space(
        board, teamToPlay, time.can_sit(teamToPlay), treeNode);
    std::vector<MateMoveCandidate>& partnerActions = partnerBoard == BOARD_A
        ? actionSpace.actionsA : actionSpace.actionsB;
    const bool partnerCanIntervene = partnerActions.size() > 1;
    // The attacker sits on the partner board for the rest of this proof, so
    // after one defender move that board stays frozen. A defender capture that
    // would force the attacker to move there is refuted below, not searched.
    // One quiet move stands for the rest only while the defenders cannot come
    // to wait on that frozen board's captures.
    if (partnerCanIntervene && !defendersMayWait) {
        std::vector<MateMoveCandidate> reducedPartnerActions;
        reducedPartnerActions.reserve(partnerActions.size());
        std::optional<MateMoveCandidate> quietRepresentative;
        MateMoveCandidate pass;
        for (const MateMoveCandidate& action : partnerActions) {
            if (action.move == Stockfish::MOVE_NONE) {
                pass = action;
                continue;
            }

            bool changesTerminalOrDraw = false;
            if (!action.isCapture && !action.givesCheck) {
                board.push_move(partnerBoard, action.move);
                changesTerminalOrDraw =
                    board.legal_moves(partnerBoard).empty()
                    || board.is_draw_on_board(partnerBoard, 1);
                board.pop_move(partnerBoard);
            }
            if (action.isCapture || action.givesCheck
                || changesTerminalOrDraw) {
                reducedPartnerActions.push_back(action);
            } else if (!quietRepresentative) {
                quietRepresentative = action;
            }
        }
        if (quietRepresentative) {
            reducedPartnerActions.push_back(*quietRepresentative);
        }
        // MOVE_NONE preserves both capture-and-pass directions after the
        // ordinary joint-action legality filter is applied.
        reducedPartnerActions.push_back(pass);
        partnerActions = std::move(reducedPartnerActions);
    }
    if (partnerCanIntervene
        && partnerActions.size() - 1
            > static_cast<size_t>(
                SearchParams::INTERNAL_MATE_CERT_MAX_PARTNER_INTERVENTIONS)) {
        return {JointMateStatus::UNKNOWN, 0, {}};
    }

    JointMateProof result{JointMateStatus::PROVEN, 0, {}};
    bool sawAction = false;
    bool sawUnknown = false;
    const auto verifyDefense = [&](const MateJointAction& action) {
        sawAction = true;
        if (!budget.consume()) {
            sawUnknown = true;
            return true;
        }
        const bool feedsPartner =
            (action.moveA != Stockfish::MOVE_NONE
             && board.is_capture(BOARD_A, action.moveA))
            || (action.moveB != Stockfish::MOVE_NONE
                && board.is_capture(BOARD_B, action.moveB));
        board.make_moves(action.moveA, action.moveB);
        JointActionCandidate raceMate;
        const std::shared_ptr<Node> childNode = tree_child_for(
            treeNode, action.moveA, action.moveB);
        const JointMateProof child = feedsPartner
                && find_immediate_root_mate(
                    board, teamToPlay, childTime, raceMate, &budget)
            ? JointMateProof{JointMateStatus::REFUTED, 0, {}}
            : search_reduced_partner_mate(
                board, attackingTeam, activeBoard, attackerColor,
                ~teamToPlay, childTime, attackerMovesRemaining,
                searchPly + 1, budget, cache, childNode.get());
        board.unmake_moves(action.moveA, action.moveB);
        if (child.status == JointMateStatus::REFUTED) {
            result = {JointMateStatus::REFUTED, 0, {}};
            return true;
        }
        if (child.status == JointMateStatus::UNKNOWN) {
            sawUnknown = true;
            return true;
        }
        if (child.pliesToMate + 1 > result.pliesToMate
            || !result.principalVariation) {
            result.pliesToMate = child.pliesToMate + 1;
            result.action = action;
            result.principalVariation =
                std::make_shared<JointMateProofLine>(
                    JointMateProofLine{action, child.principalVariation});
        }
        return false;
    };
    visit_legal_joint_actions(
        actionSpace, false, false, false, verifyDefense);
    if (result.status == JointMateStatus::REFUTED) {
        return remember(result);
    }
    if (!sawAction) {
        return remember({JointMateStatus::REFUTED, 0, {}});
    }
    if (sawUnknown || budget.exhausted) {
        return {JointMateStatus::UNKNOWN, 0, {}};
    }
    return remember(result);
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
    const TimeState& lineTime,
    const std::array<int, 2>& boardSearchPlies,
    string& formatted) {
    WaitingMateContinuation continuation;
    int endInPly = 0;
    if (classify_terminal_position(
            lineBoard, teamToPlay, rootTeam, lineTime,
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
    const TimeState& rootTime) {
    Board lineBoard = board;
    string formatted;
    std::array<int, 2> boardSearchPlies{};
    TimeState lineTime = rootTime;
    Stockfish::Color mover = rootTeam;
    for (const MateProofPly& ply : principalVariation) {
        lineTime.advance(mover, lineBoard.on_turn_on_both_boards(mover));
        append_formatted_ply(lineBoard, ply, formatted);
        boardSearchPlies[BOARD_A] += ply.moveA != Stockfish::MOVE_NONE;
        boardSearchPlies[BOARD_B] += ply.moveB != Stockfish::MOVE_NONE;
        mover = ~mover;
    }
    append_waiting_mate_suffix(
        lineBoard, mover, rootTeam, lineTime, boardSearchPlies, formatted);
    return formatted;
}

/**
 * Moves a conditional mate's first action to the front of @p root's
 * expansion order. The probe's action sits the other board; where the real
 * clock forbids that sit, the mating move goes with the legal move on the
 * other board the network rates highest, so the hint still reaches the tree.
 */
bool promote_conditional_mate(Node& root, Board& board, Stockfish::Color team,
                              const JointActionCandidate& conditional) {
    if (root.promote_joint_action(conditional)) {
        return true;
    }
    const bool movesA = conditional.moveA != Stockfish::MOVE_NONE;
    const bool movesB = conditional.moveB != Stockfish::MOVE_NONE;
    if (movesA == movesB) {
        return false;
    }
    const int otherBoard = movesA ? BOARD_B : BOARD_A;
    const Stockfish::Color otherColor = otherBoard == BOARD_A ? team : ~team;
    if (board.side_to_move(otherBoard) != otherColor) {
        return false;
    }
    Stockfish::Move best = Stockfish::MOVE_NONE;
    float bestPrior = -1.0f;
    for (Stockfish::Move move : board.legal_moves(otherBoard)) {
        const float prior = root.move_prior(otherBoard, move);
        if (prior > bestPrior) {
            bestPrior = prior;
            best = move;
        }
    }
    if (best == Stockfish::MOVE_NONE) {
        return false;
    }
    JointActionCandidate paired = conditional;
    (movesA ? paired.moveB : paired.moveA) = best;
    return root.promote_joint_action(paired);
}

}  // namespace

Agent::RaceVerdict Agent::prove_frozen_race(
    Board& board, Stockfish::Color attackingTeam, const TimeState& time,
    Stockfish::Color teamToPlay, int raceBoard, int maxAttackerMoves,
    MateSearchBudget& budget, int* outPlies, std::string* outLine,
    JointActionCandidate* outAction) {
    JointMateCache cache;
    // Shallow first: a short win is found before the deep iterations, and
    // the cache keeps what each iteration settled for the next.
    for (int moves = 1; moves <= maxAttackerMoves; ++moves) {
        const JointMateProof proof = search_frozen_race(
            board, attackingTeam, time, teamToPlay, moves, 0, budget, cache,
            raceBoard);
        if (proof.status == JointMateStatus::PROVEN) {
            if (outPlies) {
                *outPlies = proof.pliesToMate;
            }
            if (outAction) {
                *outAction = JointActionCandidate();
                outAction->moveA = proof.action.moveA;
                outAction->moveB = proof.action.moveB;
            }
            if (outLine) {
                outLine->clear();
                Board line(board);
                for (const JointMateProofLine* ply = proof.principalVariation.get();
                     ply; ply = ply->next.get()) {
                    if (!outLine->empty()) {
                        *outLine += " ";
                    }
                    *outLine += "(" + (ply->action.moveA == Stockfish::MOVE_NONE
                            ? std::string("pass")
                            : line.uci_move(BOARD_A, ply->action.moveA))
                        + "," + (ply->action.moveB == Stockfish::MOVE_NONE
                            ? std::string("pass")
                            : line.uci_move(BOARD_B, ply->action.moveB)) + ")";
                    line.make_moves(ply->action.moveA, ply->action.moveB);
                }
            }
            return RaceVerdict::PROVEN;
        }
        if (proof.status == JointMateStatus::UNKNOWN) {
            return RaceVerdict::UNKNOWN;
        }
    }
    return RaceVerdict::REFUTED;
}

void Agent::run_frozen_race_stage(
    Board& rootBoard, const std::shared_ptr<Node>& root,
    Stockfish::Color rootTeam, const TimeState& rootTime, int budgetMs,
    const std::function<bool()>& settled,
    const std::atomic<bool>* cancelled, FrozenRaceStats& stats) {
    const auto start = MateSearchBudget::Clock::now();
    const auto deadline = start + std::chrono::milliseconds(std::max(1, budgetMs));
    const auto out_of_time = [&] {
        return settled() || MateSearchBudget::Clock::now() >= deadline;
    };
    // A target is a node and the path to it from the root.
    struct Target {
        std::shared_ptr<Node> node;
        std::vector<JointActionCandidate> path;
    };
    // A won node is marked through the child its proof plays: marked lost,
    // that child makes the node won by the solver's own propagation, and
    // the node keeps a move to play. A child the tree has not generated yet
    // waits here until it has.
    struct Witness {
        std::shared_ptr<Node> node;
        JointActionCandidate action;
        int endInPly = 0;
    };
    std::vector<Witness> witnesses;
    const auto settle_witnesses = [&] {
        std::erase_if(witnesses, [&](const Witness& witness) {
            if (witness.node->get_node_type() != NodeType::UNSOLVED) {
                return true;
            }
            const std::shared_ptr<Node> child = witness.node->child_for_action(
                witness.action.moveA, witness.action.moveB);
            if (!child) {
                return false;
            }
            if (child->get_node_type() == NodeType::UNSOLVED) {
                child->mark_as_loss(witness.endInPly);
            }
            return true;
        });
    };
    std::unordered_set<const Node*> tried;
    const auto next_target = [&]() -> std::optional<Target> {
        // Every visited root action first, leading ones first: a root is
        // proven lost only once each of its actions is. Then the leading
        // replies under the leading actions.
        const std::vector<size_t> edges = ranked_visited_edges(
            root, SearchParams::FROZEN_RACE_MIN_VISITS);
        for (size_t edge : edges) {
            const std::shared_ptr<Node> child = root->get_child(edge);
            if (child && !tried.contains(child.get())) {
                return Target{child, {root->get_joint_action(edge)}};
            }
        }
        for (size_t rank = 0; rank < edges.size()
                 && rank < static_cast<size_t>(SearchParams::FROZEN_RACE_TOP_K);
             ++rank) {
            const std::shared_ptr<Node> child = root->get_child(edges[rank]);
            if (!child || child->get_node_type() != NodeType::UNSOLVED) {
                continue;
            }
            const std::vector<size_t> replies = ranked_visited_edges(
                child, SearchParams::FROZEN_RACE_MIN_VISITS);
            for (size_t reply = 0; reply < replies.size()
                     && reply < static_cast<size_t>(
                         SearchParams::FROZEN_RACE_REPLY_TOP_K);
                 ++reply) {
                const std::shared_ptr<Node> grandchild =
                    child->get_child(replies[reply]);
                if (grandchild && !tried.contains(grandchild.get())) {
                    return Target{grandchild,
                                  {root->get_joint_action(edges[rank]),
                                   child->get_joint_action(replies[reply])}};
                }
            }
        }
        return std::nullopt;
    };

    int idleMs = 0;
    while (!out_of_time()
           && stats.nodes < SearchParams::FROZEN_RACE_NODE_BUDGET) {
        settle_witnesses();
        const std::optional<Target> target = next_target();
        if (!target) {
            // Nothing new to look at. Early on the tree has simply not
            // visited enough yet; later, a proof still waiting for its child
            // is worth a short wait. Otherwise the thread goes on to its next
            // job rather than hold the verifier's time.
            const bool treeYoung = root->get_visits()
                < SearchParams::FROZEN_RACE_MIN_VISITS
                    * SearchParams::FROZEN_RACE_TOP_K * 8;
            if (!treeYoung && (witnesses.empty()
                               || ++idleMs > SearchParams::FROZEN_RACE_IDLE_MS)) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        tried.insert(target->node.get());
        ++stats.targets;
        if (target->node->get_node_type() != NodeType::UNSOLVED) {
            continue;
        }
        Board board(rootBoard);
        TimeState time = rootTime;
        Stockfish::Color team = rootTeam;
        for (const JointActionCandidate& action : target->path) {
            time = time.after(team, board.on_turn_on_both_boards(team));
            board.make_moves(action.moveA, action.moveB);
            team = ~team;
        }
        Stockfish::Color attackers = Stockfish::WHITE;
        const int raceBoard = frozen_race_board(board, attackers, out_of_time);
        if (raceBoard < 0) {
            continue;
        }
        ++stats.races;
        MateSearchBudget budget;
        budget.remainingNodes =
            SearchParams::FROZEN_RACE_NODE_BUDGET - stats.nodes;
        budget.deadline = deadline;
        budget.cancelled = cancelled;
        budget.stopOnSolvedRoot = root.get();
        const uint64_t startNodes = budget.remainingNodes;
        int plies = 0;
        JointActionCandidate action;
        const RaceVerdict verdict = prove_frozen_race(
            board, attackers, time, team, raceBoard,
            SearchParams::FROZEN_RACE_MAX_ATTACKER_MOVES, budget, &plies,
            nullptr, &action);
        stats.nodes += startNodes - budget.remainingNodes;
        if (verdict != RaceVerdict::PROVEN
            || target->node->get_node_type() != NodeType::UNSOLVED) {
            continue;
        }
        ++stats.proven;
        // The proof counts actions to the mate, the mated position as zero;
        // the tree counts that position as one.
        if (attackers != team) {
            target->node->mark_as_loss(plies + 1);
        } else if (team != rootTeam) {
            // The opponents win here: the root's team only has to avoid this
            // node, and a node with the opponents to play never becomes this
            // team's root, so it needs no move of its own to stand behind.
            target->node->mark_as_win(plies + 1);
        } else {
            // A win for this team: tree reuse can make this node the root,
            // so it is marked through the move that wins.
            witnesses.push_back({target->node, action, plies});
        }
    }
    settle_witnesses();
    stats.unplayedWitnesses += static_cast<int>(witnesses.size());
}

/**
 * A bounded search keeps its mate score across a move that returns the board
 * to a position already played: the mate is still there, just one move further
 * away. So the probe will name a shuffle as the "mating" move, the caller
 * plays it, and the next search names the shuffle back. From outside that is a
 * mate distance which grows move by move, until the third occurrence draws a
 * won game. A move that completes the repetition is not a mating move, and no
 * mate score attached to it makes it one.
 */
bool Agent::move_completes_repetition(
    Board& board, int boardNum, Stockfish::Move move) {
    if (move == Stockfish::MOVE_NONE) {
        return false;
    }
    board.push_move(boardNum, move);
    const bool repeats = board.repetition_count(boardNum) >= 3;
    board.pop_move(boardNum);
    return repeats;
}

/**
 * @brief Ask Fairy-Stockfish for a single-board mate the check-only scan missed.
 *
 * find_root_mate_impl() searches attacker moves that give check, so a mate that
 * needs a quiet preparing move is invisible to it at any budget. The probe has
 * no such restriction.
 *
 * The probe searches one board as if the other stood still, and that is only
 * the game when this team is ahead on time. Ahead, the partner board may sit
 * through every move of the line. The defender, behind, still has a reply
 * the probe never searches whenever its partner is on turn on the other board
 * with a capture: the capture buys a pass here and the captured piece arrives
 * in the defender's hand. Such a root accepts only a mate in one. The replay
 * below catches the defenders' immediate mates and captures into one, not
 * this.
 * Behind, it is the defender who may otherwise sit after the first check and
 * play the other board, while this team is made to keep moving there. A line
 * that holds only until the defender declines to follow it is not a mate,
 * whatever its length, so a root without the time advantage is left to the
 * joint prover, which searches the defender's sits.
 *
 * The probe's line is kept as text; only the first move crosses back as a
 * real move, the rest is replayed once to check it against the second board.
 */
bool Agent::probe_position_mate(
    Board& board,
    Stockfish::Color teamSide,
    bool teamHasTimeAdvantage,
    uint64_t nodeBudget,
    int budgetMs,
    const std::function<bool()>& abort,
    JointActionCandidate& outAction,
    int& outPlyToMate,
    string& outPrincipalVariation,
    const std::function<void()>& onMate,
    bool avoidRepetition,
    uint64_t* outNodes) {
    return probe_position_mate(
        board, teamSide, TimeState::legacy(teamSide, teamHasTimeAdvantage),
        nodeBudget, budgetMs, abort, outAction, outPlyToMate,
        outPrincipalVariation, onMate, avoidRepetition, outNodes);
}

bool Agent::probe_position_mate(
    Board& board,
    Stockfish::Color teamSide,
    const TimeState& time,
    uint64_t nodeBudget,
    int budgetMs,
    const std::function<bool()>& abort,
    JointActionCandidate& outAction,
    int& outPlyToMate,
    string& outPrincipalVariation,
    const std::function<void()>& onMate,
    bool avoidRepetition,
    uint64_t* outNodes) {
    if (outNodes) {
        *outNodes = 0;
    }
    // The probe sits the partner board for the whole line. A mate it finds
    // is accepted below only when this team's lead pays for every move of it.
    if (!time.can_sit(teamSide)) {
        return false;
    }
    const bool teamHasTimeAdvantage = true;
    // The line is replayed under the model the probe assumed: this team may
    // sit throughout and the defenders never may, which the lead check below
    // makes true of the real clock as well.
    const TimeState lineTime = TimeState::legacy(teamSide, true);
    const bool onTurn[2] = {board.side_to_move(BOARD_A) == teamSide,
                            board.side_to_move(BOARD_B) == ~teamSide};
    const int probeBoards = onTurn[BOARD_A] + onTurn[BOARD_B];
    if (probeBoards == 0 || nodeBudget == 0 || budgetMs < 0) {
        return false;
    }

    const JointActionRules rules{
        onTurn[BOARD_A], onTurn[BOARD_B], teamHasTimeAdvantage,
        onTurn[BOARD_A] && board.has_any_legal_move(BOARD_A),
        onTurn[BOARD_B] && board.has_any_legal_move(BOARD_B)};

    // The boards are probed one after the other. The caller may be notified as
    // soon as a checkmate lands, but a stalemate waits until both boards have
    // been searched so it cannot stop MCTS ahead of a later checkmate.
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
    };

    // The probe sees one board and never the other, so it cannot know that
    // the recapture of our checking piece hands the partner board a mate, or
    // that the partner can capture a blocker. Replay its line on the two-board
    // model before believing it. The replay catches an immediate race mate and
    // a capture that changes the final terminal result. A blocker supplied in
    // the middle of a longer line remains for the joint prover; rejecting every
    // possible partner capture here would discard valid probes such as benchmark
    // position 4, where the available capture cannot stop the mate.
    const auto line_holds_on_both_boards = [&](
        int boardNum, const MateProbe::Result& result,
        bool& outEndsInCheckmate) {
        outEndsInCheckmate = false;
        Board line(board);
        MateSearchBudget budget;
        budget.remainingNodes = SearchParams::MATE_RACE_VETO_NODE_BUDGET;
        budget.deadline = MateSearchBudget::Clock::now()
            + chrono::milliseconds(SearchParams::MATE_PROBE_VERIFY_MAX_MS);

        // The probe carries its line on from the table, but an overwritten
        // entry still ends it short of the mate. The line is then carried on
        // with a probe from where it stopped, attacker to move, so a long
        // mate can still be seen to end in checkmate.
        vector<string> principalVariation = result.principalVariation;
        vector<Stockfish::Move> played;
        int extensions = 0;
        size_t plies = 0;
        const auto extend_line = [&] {
            if (extensions >= SearchParams::MATE_PROBE_LINE_EXTENSIONS
                || principalVariation.empty() || (abort && abort())) {
                return false;
            }
            // The probe asks with the attacker to move, so a line that
            // stopped on the defender's turn is taken back one move and the
            // attacker's move is asked again.
            const bool takeBack = plies % 2 != 0;
            const size_t tailPlies = plies - (takeBack ? 1 : 0);
            // A copied board cannot take back moves made before the copy,
            // so the tail's position is replayed from the root.
            Board tailBoard(board);
            for (size_t ply = 0; ply < tailPlies; ++ply) {
                tailBoard.push_move(boardNum, played[ply]);
            }
            const int movesLeft =
                result.mateInMoves - static_cast<int>(tailPlies / 2);
            if (movesLeft < 1) {
                return false;
            }
            ++extensions;
            const MateProbe::Result tail = MateProbe::probe(
                tailBoard.fen(boardNum), movesLeft,
                SearchParams::MATE_PROBE_LINE_EXTENSION_NODES,
                SearchParams::MATE_PROBE_LINE_EXTENSION_MS, abort);
            if (!tail.found || tail.mateInMoves > movesLeft
                || tail.principalVariation.empty()) {
                return false;
            }
            if (takeBack) {
                line.pop_move(boardNum);
                principalVariation.pop_back();
                played.pop_back();
                --plies;
            }
            principalVariation.insert(
                principalVariation.end(), tail.principalVariation.begin(),
                tail.principalVariation.end());
            return true;
        };
        // The verdict on the probe's own line, before any extension: the
        // extended tail can confirm a checkmate or expose a race the
        // defenders win, but where it merely runs out it changes nothing.
        bool extended = false;
        bool probedVerdict = false;
        const auto inconclusive = [&] { return extended && probedVerdict; };
        while (true) {
            while (plies < principalVariation.size()) {
                string parsedMoveText = principalVariation[plies];
                const Stockfish::Move move = Stockfish::UCI::to_move(
                    *line.pos[boardNum], parsedMoveText);
                if (move == Stockfish::MOVE_NONE
                    || !line.is_legal_move(boardNum, move)) {
                    return inconclusive();
                }
                line.push_move(boardNum, move);
                played.push_back(move);
                ++plies;
                if (plies % 2 != 0
                    && !line.is_checkmate(
                        ~teamSide, lineTime.can_sit(~teamSide))) {
                    JointActionCandidate raceMate;
                    if (find_immediate_root_mate(
                            line, ~teamSide, lineTime, raceMate, &budget)
                        || defenders_capture_into_immediate_mate(
                            line, ~teamSide, lineTime, &budget)) {
                        return false;
                    }
                    if (budget.exhausted) {
                        return inconclusive();
                    }
                } else if (plies % 2 == 0) {
                    JointActionCandidate raceMate;
                    if (find_immediate_root_mate(
                            line, ~teamSide, lineTime, raceMate, &budget)) {
                        return false;
                    }
                    if (budget.exhausted) {
                        return inconclusive();
                    }
                }
            }
            if (line.is_checkmate(~teamSide, lineTime.can_sit(~teamSide))) {
                outEndsInCheckmate = line.is_in_check(boardNum);
                return true;
            }
            // A line as long as the mate it scored has to end in one.
            if (plies + 1 >= 2 * static_cast<size_t>(result.mateInMoves)) {
                return inconclusive();
            }
            if (extended) {
                if (!extend_line()) {
                    return probedVerdict;
                }
                continue;
            }
            // Otherwise the line stopped short, which happens at a table hit.
            // A mate that is one move away can be settled exactly; a longer
            // tail is the probe's word against nothing, since the mates it
            // exists to find are the ones a checks-only prover cannot reach.
            probedVerdict = true;
            if (plies % 2 == 0) {
                const int attackerMovesLeft =
                    result.mateInMoves - static_cast<int>(plies / 2);
                if (attackerMovesLeft <= 1) {
                    JointActionCandidate finish;
                    const bool foundFinish = find_immediate_root_mate(
                        line, teamSide, lineTime, finish, &budget);
                    probedVerdict = foundFinish && !budget.exhausted;
                }
            }
            if (!probedVerdict || !extend_line()) {
                return probedVerdict;
            }
            extended = true;
        }
    };

    bool found = false;
    bool bestEndsInCheckmate = false;
    int bestMateInMoves = 0;
    uint64_t probeIndex = 0;
    // The longest mate the lead pays for. The probe stops at the first mate
    // within its bound, so asking for longer only finds mates the lead check
    // below would throw away, in place of the shorter one it would keep.
    int clockMateMoves = 0;
    for (int moves = SearchParams::MATE_PROBE_MAX_MATE_MOVES; moves >= 1;
         --moves) {
        if (time.can_sit_through(teamSide, moves)) {
            clockMateMoves = moves;
            break;
        }
    }
    // What each board may still spend, and whether it is finished with: a
    // board is probed in slices that alternate with the other board's and
    // double each round, so a mate on the second board is not kept waiting
    // behind a first board with none. Each board's total stays its half.
    struct BoardProbe {
        bool active = false;
        uint64_t nodesLeft = 0;
        int msLeft = 0;
        int maxMateMoves = 0;
    };
    std::array<BoardProbe, 2> boards;
    for (int boardNum : {BOARD_A, BOARD_B}) {
        if (!onTurn[boardNum]) {
            continue;
        }
        BoardProbe& probe = boards[boardNum];
        probe.nodesLeft = nodeBudget / probeBoards
            + (probeIndex < nodeBudget % probeBoards ? 1 : 0);
        ++probeIndex;
        probe.msLeft = budgetMs > 0 ? std::max(1, budgetMs / probeBoards) : 0;
        // The probe sees one board, and the partner board is sat. If the
        // defenders are on turn there with a capture to play, every one of
        // their turns in the line has a reply the probe never searched: they
        // may leave this board waiting - a capture buys that pass even when
        // they are behind - and their partner gets the captured piece to
        // defend with. Only a mate in one leaves them no turn to do it in.
        const int partnerBoard = 1 - boardNum;
        const Stockfish::Color partnerDefender = partnerBoard == BOARD_A
            ? ~teamSide : teamSide;
        bool partnerCaptureAvailable = false;
        if (board.side_to_move(partnerBoard) == partnerDefender) {
            for (Stockfish::Move move : board.legal_moves(partnerBoard)) {
                if (board.is_capture(partnerBoard, move)) {
                    partnerCaptureAvailable = true;
                    break;
                }
            }
        }
        probe.maxMateMoves = partnerCaptureAvailable
            ? std::min(1, clockMateMoves)
            : clockMateMoves;
        probe.active = probe.nodesLeft > 0 && probe.maxMateMoves >= 1;
    }
    // Only a probe with both boards to search needs slicing.
    const bool sliced = boards[BOARD_A].active && boards[BOARD_B].active;
    for (int round = 0; boards[BOARD_A].active || boards[BOARD_B].active;
         ++round) {
        for (int boardNum : {BOARD_A, BOARD_B}) {
            BoardProbe& probe = boards[boardNum];
            if (!probe.active) {
                continue;
            }
            if (abort && abort()) {
                probe.active = false;
                continue;
            }
            const int growth = std::min(round, 20);
            const uint64_t sliceNodes = sliced
                ? std::min(probe.nodesLeft,
                           SearchParams::MATE_PROBE_FIRST_SLICE_NODES << growth)
                : probe.nodesLeft;
            const int sliceMs = sliced && probe.msLeft > 0
                ? std::min<int64_t>(
                      probe.msLeft,
                      static_cast<int64_t>(SearchParams::MATE_PROBE_FIRST_SLICE_MS)
                          << growth)
                : probe.msLeft;
            const auto sliceStart = MateSearchBudget::Clock::now();
            const MateProbe::Result result = MateProbe::probe(
                board.fen(boardNum), probe.maxMateMoves,
                sliceNodes, sliceMs, abort);
            const int spentMs = static_cast<int>(
                chrono::duration_cast<chrono::milliseconds>(
                    MateSearchBudget::Clock::now() - sliceStart).count());
            if (outNodes) {
                *outNodes += result.nodes;
            }
            probe.nodesLeft -= std::min(probe.nodesLeft, result.nodes);
            if (probe.msLeft > 0) {
                probe.msLeft = sliceMs >= probe.msLeft
                    ? 0 : std::max(0, probe.msLeft - std::max(spentMs, 1));
            }
            // A board stops at its first mate, kept or not: asking again
            // would find the same one. A probe that searched nothing could
            // not search more next round either.
            if (result.found || result.nodes == 0 || probe.nodesLeft == 0
                || (budgetMs > 0 && probe.msLeft == 0)) {
                probe.active = false;
            }
            bool endsInCheckmate = false;
            // A pruned search is not a proof, so the move it names still has
            // to be one this board accepts before it can become the root
            // action.
            if (!result.found
                || result.bestMove == Stockfish::MOVE_NONE
                || !board.is_legal_move(boardNum, result.bestMove)
                || !time.can_sit_through(teamSide, result.mateInMoves)
                || (avoidRepetition
                    && move_completes_repetition(
                        board, boardNum, result.bestMove))
                || !line_holds_on_both_boards(
                    boardNum, result, endsInCheckmate)) {
                continue;
            }
            if (found) {
                if (bestEndsInCheckmate != endsInCheckmate) {
                    if (!endsInCheckmate) {
                        continue;
                    }
                } else if (result.mateInMoves >= bestMateInMoves) {
                    continue;
                }
            }
            bestEndsInCheckmate = endsInCheckmate;
            bestMateInMoves = result.mateInMoves;
            found = true;
            accept(boardNum, result);
            if (endsInCheckmate && onMate) {
                onMate();
            }
        }
    }
    if (found && !bestEndsInCheckmate && onMate) {
        onMate();
    }
    return found;
}

bool Agent::probe_root_mate(
    Board& board,
    Stockfish::Color teamSide,
    bool teamHasTimeAdvantage,
    uint64_t nodeBudget,
    int budgetMs,
    const std::function<bool()>& abort,
    JointActionCandidate& outAction,
    int& outPlyToMate,
    string& outPrincipalVariation,
    const std::function<void()>& onMate,
    bool avoidRepetition) {
    return probe_position_mate(
        board, teamSide, TimeState::legacy(teamSide, teamHasTimeAdvantage),
        nodeBudget, budgetMs, abort,
        outAction, outPlyToMate, outPrincipalVariation,
        onMate, avoidRepetition, nullptr);
}

bool Agent::certify_mate_candidate(
    Board& board,
    Stockfish::Color teamSide,
    bool teamHasTimeAdvantage,
    const JointActionCandidate& candidate,
    int candidatePlyToMate,
    MateSearchBudget& budget,
    int& outPlyToMate,
    MateCertificateTier& outTier,
    const Node* treeNode) {
    return certify_mate_candidate(
        board, teamSide, TimeState::legacy(teamSide, teamHasTimeAdvantage),
        candidate, candidatePlyToMate, budget, outPlyToMate, outTier,
        treeNode);
}

bool Agent::certify_mate_candidate(
    Board& board,
    Stockfish::Color teamSide,
    const TimeState& time,
    const JointActionCandidate& proposed,
    int candidatePlyToMate,
    MateSearchBudget& budget,
    int& outPlyToMate,
    MateCertificateTier& outTier,
    const Node* treeNode,
    JointActionCandidate* outAction,
    MateCertificateFailure* outFailure) {
    outPlyToMate = 0;
    outTier = MateCertificateTier::NONE;
    MateCertificateFailure failure = MateCertificateFailure::ILLEGAL;
    const auto note_failure = [&](MateCertificateFailure reason) {
        failure = std::max(failure, reason);
    };
    // Reports the most hopeful reason on every way out that certifies nothing.
    struct FailureReport {
        MateCertificateFailure* out;
        const MateCertificateFailure& failure;
        const MateCertificateTier& tier;
        ~FailureReport() {
            if (out) {
                *out = tier == MateCertificateTier::NONE
                    ? failure : MateCertificateFailure::NONE;
            }
        }
    } failureReport{outFailure, failure, outTier};
    // No lead check here: a team that may not sit has no sat-board tier
    // (sitsThroughLine below), but the joint tier plays the real clock -
    // every sit, wait and capture of the defenders - so it may certify a
    // team that is level or behind.
    if (candidatePlyToMate <= 0
        || budget.exhausted || budget.remainingNodes == 0
        || budget.out_of_time()) {
        note_failure(MateCertificateFailure::BUDGET);
        return false;
    }
    const bool proposedMovesA = proposed.moveA != Stockfish::MOVE_NONE;
    const bool proposedMovesB = proposed.moveB != Stockfish::MOVE_NONE;
    if ((!proposedMovesA && !proposedMovesB)
        || (proposedMovesA && !board.is_legal_move(BOARD_A, proposed.moveA))
        || (proposedMovesB && !board.is_legal_move(BOARD_B, proposed.moveB))) {
        return false;
    }

    // Certifies one joint action that is legal under this clock.
    const auto certify_legal = [&](const JointActionCandidate& candidate) {
    const bool movesA = candidate.moveA != Stockfish::MOVE_NONE;
    const bool movesB = candidate.moveB != Stockfish::MOVE_NONE;
    const bool losesRace = action_loses_mate_race(
        board, candidate, teamSide, time, &budget);
    if (losesRace || budget.exhausted || budget.out_of_time()) {
        note_failure(losesRace ? MateCertificateFailure::RACE
                               : MateCertificateFailure::BUDGET);
        return false;
    }

    const int candidateAttackerMoves = std::clamp(
        (candidatePlyToMate + 1) / 2, 1,
        SearchParams::MATE_SEARCH_MAX_ATTACKER_MOVES);
    const std::shared_ptr<Node> candidateNode = tree_child_for(
        treeNode, candidate.moveA, candidate.moveB);
    // The single-board and sat-board tiers let this team sit the partner
    // board at every move of the line and never the defenders: true of the
    // clock only while the lead covers every move, the mating move a
    // single-board proof may end with on the waiting board included. The
    // joint tier follows the clock itself.
    const bool sitsThroughLine = time.can_sit_through(
        teamSide, candidateAttackerMoves + 1);
    const TimeState afterCandidate = time.after(
        teamSide, board.on_turn_on_both_boards(teamSide));
    board.make_moves(candidate.moveA, candidate.moveB);
    const auto restore = [&] {
        board.unmake_moves(candidate.moveA, candidate.moveB);
    };

    const JointMateStatus terminal = terminal_joint_mate_status(
        board, teamSide, afterCandidate, 1);
    if (terminal == JointMateStatus::PROVEN) {
        restore();
        outPlyToMate = 1;
        outTier = MateCertificateTier::CHECKS_ONLY;
        return true;
    }
    if (terminal == JointMateStatus::REFUTED || budget.exhausted) {
        note_failure(budget.exhausted ? MateCertificateFailure::BUDGET
                                      : MateCertificateFailure::REFUTED);
        restore();
        return false;
    }

    const auto prove_tail = [&](int activeBoard,
                                Stockfish::Color attackerColor,
                                bool partnerBoardAgnostic,
                                int& tailPly) {
        Stockfish::Move continuation = Stockfish::MOVE_NONE;
        int continuationPly = 0;
        const bool proven = !board.is_draw(2)
            && search_single_board_forced_mate_impl(
                board, activeBoard, attackerColor, 3,
                candidateAttackerMoves, continuation,
                continuationPly, &budget, nullptr,
                partnerBoardAgnostic, nullptr);
        if (proven) {
            tailPly = std::max(tailPly, continuationPly);
        }
        return proven;
    };

    // A single-board proof is exact while the time-behind defender is not on
    // turn on the partner board: every legal reply is then an evasion on the
    // mating board, and the time-ahead attacker may keep its partner sitting.
    if (movesA != movesB && sitsThroughLine) {
        const int activeBoard = movesA ? BOARD_A : BOARD_B;
        const int partnerBoard = 1 - activeBoard;
        const Stockfish::Color defenderPartnerColor = partnerBoard == BOARD_A
            ? ~teamSide : teamSide;
        if (board.side_to_move(partnerBoard) != defenderPartnerColor) {
            const Stockfish::Color attackerColor = activeBoard == BOARD_A
                ? teamSide : ~teamSide;
            const std::vector<Stockfish::Move> replies =
                board.legal_moves(activeBoard);
            bool allRepliesMated = !replies.empty();
            int deepestPly = 1;
            for (Stockfish::Move reply : replies) {
                if (!budget.consume()) {
                    allRepliesMated = false;
                    break;
                }
                board.push_move(activeBoard, reply);
                const bool replyMated = prove_tail(
                    activeBoard, attackerColor, false, deepestPly);
                board.pop_move(activeBoard);
                if (!replyMated) {
                    allRepliesMated = false;
                    break;
                }
            }
            if (allRepliesMated) {
                restore();
                outPlyToMate = deepestPly;
                outTier = MateCertificateTier::CHECKS_ONLY;
                return true;
            }
            if (budget.exhausted) {
                note_failure(MateCertificateFailure::BUDGET);
                restore();
                return false;
            }
        } else {
            const Stockfish::Color attackerColor = activeBoard == BOARD_A
                ? teamSide : ~teamSide;
            const JointMateProof reducedProof = search_reduced_partner_mate(
                board, teamSide, activeBoard, attackerColor, ~teamSide,
                TimeState::legacy(teamSide, true),
                std::max(0, candidateAttackerMoves - 1), 1, budget, nullptr,
                candidateNode.get());
            if (reducedProof.status == JointMateStatus::PROVEN) {
                restore();
                outPlyToMate = reducedProof.pliesToMate + 1;
                outTier = MateCertificateTier::REDUCED_PARTNER;
                return true;
            }
            if (budget.exhausted) {
                note_failure(MateCertificateFailure::BUDGET);
                restore();
                return false;
            }
        }
    }

    // Defenders with the lead may sit the mating board and make the
    // attackers spend actions elsewhere; a check does not stop them, the
    // clock does. Their lead can reach at most what it is now plus our move
    // time for each attacker action of the line. Where that never turns
    // positive the line is as long as the candidate said. Where it does,
    // every sit lengthens the line by an attacker action - free ones too,
    // while they are on turn on the mating board alone - so the proof gets
    // the solver's full depth and the budget decides how far it looks.
    const int lineAttackerMoves = std::max(0, candidateAttackerMoves - 1);
    const int64_t defenderLeadCeiling =
        static_cast<int64_t>(afterCandidate.lead_ms(~teamSide))
        + static_cast<int64_t>(lineAttackerMoves)
            * afterCandidate.move_time_ms(teamSide);
    const int jointAttackerMoves = defenderLeadCeiling > 0
        ? SearchParams::MATE_SEARCH_MAX_ATTACKER_MOVES
        : lineAttackerMoves;

    JointMateCache cache;
    cache.reserve(static_cast<size_t>(
        std::min<uint64_t>(budget.remainingNodes, 16384)));
    // The proposed action names the mating board; the other one is where
    // defenders who sit it can make the attackers wait.
    const int waitingBoard = proposedMovesA == proposedMovesB ? -1
        : proposedMovesA ? BOARD_B : BOARD_A;
    const JointMateProof proof = search_joint_forced_mate(
        board, teamSide, afterCandidate, ~teamSide,
        jointAttackerMoves, 1,
        budget, cache, false, candidateNode.get(), waitingBoard);
    restore();
    if (proof.status != JointMateStatus::PROVEN) {
        note_failure(proof.status == JointMateStatus::UNKNOWN
                || budget.exhausted
            ? MateCertificateFailure::BUDGET
            : proof.depthLimited ? MateCertificateFailure::DEPTH
            : MateCertificateFailure::REFUTED);
        return false;
    }
    outPlyToMate = proof.pliesToMate + 1;
    outTier = MateCertificateTier::FULL_JOINT;
    return true;
    };

    const bool onTurnA = board.side_to_move(BOARD_A) == teamSide;
    const bool onTurnB = board.side_to_move(BOARD_B) == ~teamSide;
    const JointActionRules rules{
        onTurnA, onTurnB, time.can_sit(teamSide),
        onTurnA && board.has_any_legal_move(BOARD_A),
        onTurnB && board.has_any_legal_move(BOARD_B)};
    const auto legal_here = [&](Stockfish::Move moveA, Stockfish::Move moveB) {
        return is_joint_action_legal(
            rules, moveA, moveB,
            moveA != Stockfish::MOVE_NONE && board.is_capture(BOARD_A, moveA),
            moveB != Stockfish::MOVE_NONE && board.is_capture(BOARD_B, moveB));
    };
    if (legal_here(proposed.moveA, proposed.moveB)) {
        const bool certified = certify_legal(proposed);
        if (certified && outAction) {
            *outAction = proposed;
        }
        return certified;
    }

    // A probe's candidate moves one board and sits the other. On turn on both
    // boards without the lead, that sit is legal only beside a capture, so
    // the attackers have to move on the other board as well. Any legal move
    // there is theirs to choose, so each is tried, in the tree's prior order
    // where it has the position, until one is certified.
    if (proposedMovesA == proposedMovesB) {
        return false;
    }
    const int otherBoard = proposedMovesA ? BOARD_B : BOARD_A;
    std::vector<Stockfish::Move> otherMoves = board.legal_moves(otherBoard);
    if (treeNode) {
        std::stable_sort(otherMoves.begin(), otherMoves.end(),
                         [&](Stockfish::Move lhs, Stockfish::Move rhs) {
                             return treeNode->move_prior(otherBoard, lhs)
                                 > treeNode->move_prior(otherBoard, rhs);
                         });
    }
    for (Stockfish::Move otherMove : otherMoves) {
        const Stockfish::Move moveA = proposedMovesA ? proposed.moveA : otherMove;
        const Stockfish::Move moveB = proposedMovesA ? otherMove : proposed.moveB;
        if (!legal_here(moveA, moveB)) {
            continue;
        }
        const JointActionCandidate paired(
            moveA, 1.0f, 0, moveB, 1.0f, 0, rules,
            board.is_capture(BOARD_A, moveA), board.is_capture(BOARD_B, moveB));
        if (certify_legal(paired)) {
            if (outAction) {
                *outAction = paired;
            }
            return true;
        }
        if (budget.exhausted || budget.remainingNodes == 0
            || budget.out_of_time()) {
            return false;
        }
    }
    return false;
}

void Agent::verify_root_action_slice(
    Board& board, const JointActionCandidate& action,
    Stockfish::Color teamSide, RootActionVerdict& verdict,
    uint64_t sliceNodes, int probeMs,
    MateSearchBudget::Clock::time_point deadline,
    const std::atomic<bool>* cancelled, const Node* stopOnSolvedRoot,
    uint64_t& nodesSpent, ConcurrentVerifierStats* stats,
    const Node* actionNode, int maxAttackerMoves) {
    verify_root_action_slice(
        board, action, teamSide, TimeState::legacy(teamSide, false), verdict,
        sliceNodes, probeMs, deadline, cancelled, stopOnSolvedRoot,
        nodesSpent, stats, actionNode, maxAttackerMoves);
}

void Agent::verify_root_action_slice(
    Board& board, const JointActionCandidate& action,
    Stockfish::Color teamSide, const TimeState& time,
    RootActionVerdict& verdict,
    uint64_t sliceNodes, int probeMs,
    MateSearchBudget::Clock::time_point deadline,
    const std::atomic<bool>* cancelled, const Node* stopOnSolvedRoot,
    uint64_t& nodesSpent, ConcurrentVerifierStats* stats,
    const Node* actionNode, int maxAttackerMoves) {
    if (verdict.state != RootActionVerdict::State::UNKNOWN) {
        return;
    }
    if (!board.is_legal_move(BOARD_A, action.moveA)
        || !board.is_legal_move(BOARD_B, action.moveB)) {
        verdict.state = RootActionVerdict::State::REFUTED_AT_BOUND;
        return;
    }
    const auto remaining_ms = [&] {
        return static_cast<int>(chrono::duration_cast<chrono::milliseconds>(
            deadline - MateSearchBudget::Clock::now()).count());
    };
    const auto abort = [&] {
        return (cancelled && cancelled->load(memory_order_relaxed))
            || (SearchParams::ENABLE_MATE_EARLY_EXIT && stopOnSolvedRoot
                && stopOnSolvedRoot->get_node_type() != NodeType::UNSOLVED)
            || remaining_ms() <= 0;
    };
    if (abort()) {
        return;
    }
    ++verdict.slices;
    if (stats) {
        ++stats->slices;
    }

    // The opponents' proofs run under the clock this action leaves them.
    const TimeState afterAction = time.after(
        teamSide, board.on_turn_on_both_boards(teamSide));
    board.make_moves(action.moveA, action.moveB);
    bool proven = false;
    int provenPly = 0;
    bool refuted = false;
    // Mating them ends the game where it stands.
    if (board.is_checkmate(~teamSide, afterAction.can_sit(~teamSide))) {
        refuted = true;
    } else if (!verdict.fairyProbed) {
        // Fairy first: a single-board mate is found in a few milliseconds
        // and certified exactly, when the opponents' lead lets them sit the
        // other board for the whole line.
        verdict.fairyProbed = true;
        if (stats) {
            ++stats->probes;
        }
        JointActionCandidate reply;
        int replyPly = 0;
        string replyPv;
        uint64_t probeNodes = 0;
        const bool found = probe_position_mate(
            board, ~teamSide, afterAction,
            SearchParams::SELECTED_MOVE_PROBE_NODE_BUDGET,
            std::max(1, std::min(probeMs, remaining_ms())), abort,
            reply, replyPly, replyPv, {}, false, &probeNodes);
        if (found && !abort()) {
            if (stats) {
                ++stats->probeHits;
            }
            MateSearchBudget certBudget;
            certBudget.remainingNodes =
                SearchParams::SELECTED_MOVE_CERT_NODE_BUDGET;
            certBudget.cancelled = cancelled;
            certBudget.stopOnSolvedRoot = stopOnSolvedRoot;
            certBudget.deadline = std::min(
                deadline,
                MateSearchBudget::Clock::now() + chrono::milliseconds(
                    SearchParams::SELECTED_MOVE_CERT_PER_CANDIDATE_MAX_MS));
            MateCertificateTier tier = MateCertificateTier::NONE;
            int certifiedPly = 0;
            proven = certify_mate_candidate(
                board, ~teamSide, afterAction, reply, replyPly, certBudget,
                certifiedPly, tier, actionNode);
            provenPly = certifiedPly;
            const uint64_t spent = SearchParams::SELECTED_MOVE_CERT_NODE_BUDGET
                - certBudget.remainingNodes;
            nodesSpent += spent;
            verdict.nodes += spent;
            if (stats) {
                stats->nodes += spent;
            }
        }
    } else {
        if (!verdict.cache) {
            verdict.cache = std::make_shared<joint_mate::JointMateCache>();
        }
        MateSearchBudget slice;
        slice.remainingNodes = std::max<uint64_t>(1, sliceNodes);
        slice.cancelled = cancelled;
        slice.stopOnSolvedRoot = stopOnSolvedRoot;
        slice.deadline = deadline;
        // The bound in one pass rather than deepening: a shallow pass that
        // finds nothing has to refute every forcing action, which with a
        // full hand costs more than the deep proof, while the sat-board
        // attempt at each attacker node still finds a short mate first.
        JointActionCandidate mateAction;
        int matePly = 0;
        bool refutedAtBound = false;
        proven = prove_joint_forced_mate(
            board, ~teamSide, afterAction, maxAttackerMoves, slice,
            mateAction, matePly, nullptr, false, verdict.cache.get(),
            &refutedAtBound, actionNode);
        provenPly = matePly;
        refuted = !proven && refutedAtBound;
        const uint64_t spent = std::max<uint64_t>(1, sliceNodes)
            - slice.remainingNodes;
        nodesSpent += spent;
        verdict.nodes += spent;
        if (stats) {
            stats->nodes += spent;
        }
    }
    board.unmake_moves(action.moveA, action.moveB);

    if (proven) {
        verdict.state = RootActionVerdict::State::PROVEN_LOSS;
        verdict.plyToMate = provenPly;
        if (stats) {
            ++stats->proven;
            stats->provenNodes += verdict.nodes;
        }
    } else if (refuted) {
        verdict.state = RootActionVerdict::State::REFUTED_AT_BOUND;
        if (stats) {
            ++stats->refuted;
        }
    }
}

bool Agent::prove_joint_forced_mate(
    Board& board, Stockfish::Color attackingTeam,
    bool attackingTeamHasTimeAdvantage, int maxAttackerMoves,
    MateSearchBudget& budget, JointActionCandidate& outAction,
    int& outPlyToMate, std::vector<MateProofPly>* outLine,
    bool shortestFirst, joint_mate::JointMateCache* cache,
    bool* outRefutedAtBound, const Node* treeNode) {
    return prove_joint_forced_mate(
        board, attackingTeam,
        TimeState::legacy(attackingTeam, attackingTeamHasTimeAdvantage),
        maxAttackerMoves, budget, outAction, outPlyToMate, outLine,
        shortestFirst, cache, outRefutedAtBound, treeNode);
}

bool Agent::prove_joint_forced_mate(
    Board& board, Stockfish::Color attackingTeam,
    const TimeState& time, int maxAttackerMoves,
    MateSearchBudget& budget, JointActionCandidate& outAction,
    int& outPlyToMate, std::vector<MateProofPly>* outLine,
    bool shortestFirst, joint_mate::JointMateCache* cache,
    bool* outRefutedAtBound, const Node* treeNode) {
    outPlyToMate = 0;
    if (outRefutedAtBound) {
        *outRefutedAtBound = false;
    }
    const bool boardAOnTurn = board.side_to_move(BOARD_A) == attackingTeam;
    const bool boardBOnTurn = board.side_to_move(BOARD_B) == ~attackingTeam;
    if (!boardAOnTurn && !boardBOnTurn || maxAttackerMoves < 1) {
        return false;
    }
    JointMateCache localCache;
    if (!cache) {
        cache = &localCache;
        cache->reserve(static_cast<size_t>(
            std::min<uint64_t>(budget.remainingNodes, 16384)));
    }
    bool refutedThroughout = true;
    for (int attackerMoves = shortestFirst
             ? std::min(2, maxAttackerMoves) : maxAttackerMoves;
         attackerMoves <= maxAttackerMoves; ++attackerMoves) {
        const JointMateProof proof = search_joint_forced_mate(
            board, attackingTeam, time,
            attackingTeam, attackerMoves, 0, budget, *cache, false, treeNode);
        refutedThroughout = refutedThroughout
            && proof.status == JointMateStatus::REFUTED;
        if (proof.status == JointMateStatus::PROVEN) {
            const bool isCapA = proof.action.moveA != Stockfish::MOVE_NONE
                && board.is_capture(BOARD_A, proof.action.moveA);
            const bool isCapB = proof.action.moveB != Stockfish::MOVE_NONE
                && board.is_capture(BOARD_B, proof.action.moveB);
            const JointActionRules rules{
                boardAOnTurn, boardBOnTurn, time.can_sit(attackingTeam),
                boardAOnTurn && board.has_any_legal_move(BOARD_A),
                boardBOnTurn && board.has_any_legal_move(BOARD_B)};
            outAction = JointActionCandidate(
                proof.action.moveA, 1.0f, 0, proof.action.moveB, 1.0f, 0,
                rules, isCapA, isCapB);
            outPlyToMate = proof.pliesToMate;
            if (outLine) {
                outLine->clear();
                for (auto ply = proof.principalVariation; ply; ply = ply->next) {
                    outLine->push_back(
                        MateProofPly{ply->action.moveA, ply->action.moveB});
                }
            }
            return true;
        }
        if (budget.exhausted || budget.out_of_time()) {
            return false;
        }
    }
    if (outRefutedAtBound) {
        *outRefutedAtBound = refutedThroughout
            && !budget.exhausted && !budget.out_of_time();
    }
    return false;
}

bool Agent::find_root_mate(Board& board, Stockfish::Color teamSide,
                          bool teamHasTimeAdvantage,
                          JointActionCandidate& outAction,
                          int& outPlyToMate,
                          uint64_t nodeBudget,
                          MateSearchBudget::Clock::time_point deadline) {
    return find_root_mate(
        board, teamSide, TimeState::legacy(teamSide, teamHasTimeAdvantage),
        outAction, outPlyToMate, nodeBudget, deadline);
}

bool Agent::find_root_mate(Board& board, Stockfish::Color teamSide,
                          const TimeState& time,
                          JointActionCandidate& outAction,
                          int& outPlyToMate,
                          uint64_t nodeBudget,
                          MateSearchBudget::Clock::time_point deadline) {
    return find_root_mate_impl(
        board, teamSide, time, outAction, outPlyToMate,
        nodeBudget, nullptr, nullptr, true, deadline);
}

bool Agent::find_root_mate_impl(
    Board& board, Stockfish::Color teamSide,
    const TimeState& time,
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
    const Node* stopOnSolvedRoot,
    bool stopAtFirstProof,
    bool* outComplete,
    bool* outDepthLimited) {
    if (outPrincipalVariation) {
        outPrincipalVariation->clear();
    }
    if (outComplete) {
        *outComplete = false;
    }
    if (outDepthLimited) {
        *outDepthLimited = false;
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
    const bool teamHasTimeAdvantage = time.can_sit(teamSide);
    // The single-board and capture-feed proofs below sit the partner board
    // for the whole line and never let the defenders sit. The clock allows
    // that only while this team's lead covers every action of the line, so
    // they search no deeper than the lead pays for.
    const int satAttackerMoves = sat_partner_attacker_moves(time, teamSide);
    const bool sitsThroughLine =
        satAttackerMoves == SearchParams::MATE_SEARCH_MAX_ATTACKER_MOVES;

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
                board, teamSide, time, outAction, immediateBudget)) {
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
            board, teamSide, time, attackerWinsMateRace,
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

    // 2. Without the lead for the sat-board proofs below, capture-plus-pass
    // is still legal with both boards on turn. The capture may feed an
    // immediate mate to the waiting board. The opponent may be able to move
    // the capture board or sit, so reuse the in-tree waiting-mate classifier,
    // under the clock the capture leaves: it proves the mate survives every
    // choice and rejects the capture if the opponent can mate the capturing
    // board first.
    if (!sitsThroughLine && boardAOnTurn && boardBOnTurn) {
        const JointActionRules rules{boardAOnTurn, boardBOnTurn,
                                     teamHasTimeAdvantage, true, true};
        const TimeState afterCapture = time.after(teamSide, true);
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
                        board, ~teamSide, teamSide, afterCapture, {1, 1},
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
    // sequence is played, so every line is capped at satAttackerMoves. A
    // clock lead short of the full horizon still proves the short lines here
    // and leaves the rest to the joint search below.
    if (satAttackerMoves > 0 && boardAOnTurn && boardBOnTurn) {
        // With a lead short of the full horizon the joint search below still
        // runs after this section, so this section gets half the time and
        // half the hard-budget nodes left, and its running out is not the
        // caller's: an exhausted slice falls through to the joint search
        // instead of ending the scan. The nodes it spends are billed to the
        // caller's budget all the same. (The joint search keeps its own
        // allowance, unbilled, as it always has.)
        const bool jointSearchFollows = !sitsThroughLine;
        const auto halfway = [&](MateSearchBudget::Clock::time_point stop) {
            const auto now = MateSearchBudget::Clock::now();
            if (!jointSearchFollows
                || stop == MateSearchBudget::Clock::time_point{}
                || stop <= now) {
                return stop;
            }
            return now + (stop - now) / 2;
        };
        const MateSearchBudget::Clock::time_point satDeadline = halfway(deadline);
        auto with_sat_deadline = [&](MateSearchBudget& budget) {
            with_deadline(budget);
            budget.deadline = satDeadline;
        };
        MateSearchBudget satDeadlineOnly;
        with_sat_deadline(satDeadlineOnly);
        const auto sat_out_of_time = [&] {
            return satDeadlineOnly.out_of_time();
        };
        MateSearchBudget satHardBudgetSlice;
        MateSearchBudget* satHardBudget = hardBudget;
        if (hardBudget && jointSearchFollows) {
            satHardBudgetSlice = *hardBudget;
            satHardBudgetSlice.remainingNodes = hardBudget->remainingNodes / 2;
            satHardBudgetSlice.deadline = halfway(hardBudget->deadline);
            satHardBudget = &satHardBudgetSlice;
        }
        struct SliceBilling {
            const MateSearchBudget& slice;
            uint64_t allowance;
            MateSearchBudget* caller;
            ~SliceBilling() {
                if (caller) {
                    caller->consume_bulk(allowance - slice.remainingNodes);
                }
            }
        } sliceBilling{satHardBudgetSlice, satHardBudgetSlice.remainingNodes,
                       satHardBudget == &satHardBudgetSlice ? hardBudget
                                                            : nullptr};

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
            std::vector<MateProofPly>& directPv,
            int firstDepth = 2,
            int lastDepth = SearchParams::MATE_SEARCH_MAX_ATTACKER_MOVES) {
            for (int maxMateMoves = firstDepth;
                 maxMateMoves <= lastDepth;
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
                const bool foundB = !(foundA && stopAtFirstProof)
                    && search_single_board_forced_mate_impl(
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
        with_sat_deadline(directBudget);
        MateSearchBudget& directScanBudget =
            satHardBudget ? *satHardBudget : directBudget;
        // Reverse proofs must also leave room for cheap capture feeds. Try
        // short direct lines first, then deeper ones if the feed scan fails.
        const int directDepth = satHardBudget
            ? std::min(SearchParams::ROOT_LOSS_DIRECT_SCOUT_MAX_ATTACKER_MOVES,
                       satAttackerMoves)
            : satAttackerMoves;
        JointActionCandidate bestDirectAction;
        int bestDirectPly = 0;
        std::vector<MateProofPly> bestDirectPv;
        bool foundDirectMate = find_shortest_direct_mate(
            directScanBudget, bestDirectAction, bestDirectPly,
            bestDirectPv, 2, directDepth);
        // A feed's root capture is charged on top of the mate, which is what
        // the overhead in satAttackerMoves allows for.
        int maxFeedAttackerMoves = satAttackerMoves;
        if (foundDirectMate) {
            // A feed adds the root capture and the forced reply before the
            // partner-board mate. For a direct D-ply mate, at most (D-3)/2
            // attacker moves can produce a strictly shorter feed.
            maxFeedAttackerMoves = std::min(
                maxFeedAttackerMoves, std::max(0, (bestDirectPly - 3) / 2));
            if (stopAtFirstProof || maxFeedAttackerMoves == 0) {
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
        with_sat_deadline(feedBudgets[BOARD_A]);
        with_sat_deadline(feedBudgets[BOARD_B]);

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
                if (satHardBudget && !satHardBudget->consume()) {
                    return;
                }
                if (sat_out_of_time()) {
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
            if (satHardBudget && satHardBudget->exhausted
                && !jointSearchFollows) {
                return false;
            }
            collect_feed_candidates(BOARD_B, BOARD_A, teamSide);
            if (satHardBudget && satHardBudget->exhausted
                && !jointSearchFollows) {
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
                MateSearchBudget& feedBudget = satHardBudget
                    ? *satHardBudget : feedBudgets[candidate.feedBoard];
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
                    if (stopAtFirstProof) {
                        outAction = bestFeedAction;
                        outPlyToMate = bestFeedPly;
                        report_feed_principal_variation();
                        return true;
                    }
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
                    if (stopAtFirstProof) {
                        outAction = bestFeedAction;
                        outPlyToMate = bestFeedPly;
                        report_feed_principal_variation();
                        return true;
                    }
                }
            }
        }

        // A deep direct proof that did not fit the scout still gets the
        // remaining reverse-proof budget once capture feeds have been tried.
        if (satHardBudget && !foundDirectMate && !foundFeedMate
            && !satHardBudget->exhausted && !satHardBudget->out_of_time()) {
            foundDirectMate = find_shortest_direct_mate(
                *satHardBudget, bestDirectAction, bestDirectPly, bestDirectPv,
                directDepth + 1, satAttackerMoves);
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
    }
    if (!(sitsThroughLine && boardAOnTurn && boardBOnTurn)) {
        // In every other turn/time configuration, moves on the partner board,
        // capture transfers and legal sits can affect the mating board. Search
        // complete joint actions rather than treating the boards independently.
        // That includes a lead too short for every sat-partner line: those
        // proofs above stopped where the lead ran out.
        MateSearchBudget budget;
        budget.remainingNodes = std::max<uint64_t>(
            1, nodeBudget / SearchParams::MATE_JOINT_SEARCH_BUDGET_DIVISOR);
        with_deadline(budget);
        JointMateCache cache;
        cache.reserve(static_cast<size_t>(
            std::min<uint64_t>(budget.remainingNodes, 16384)));
        const auto publish_proof = [&](const JointMateProof& proof) {
            const JointActionRules rules{
                boardAOnTurn, boardBOnTurn, teamHasTimeAdvantage,
                boardAOnTurn && board.has_any_legal_move(BOARD_A),
                boardBOnTurn && board.has_any_legal_move(BOARD_B)};
            outAction = JointActionCandidate(
                proof.action.moveA, 1.0f, 0,
                proof.action.moveB, 1.0f, 0, rules,
                proof.action.moveA != Stockfish::MOVE_NONE
                    && board.is_capture(BOARD_A, proof.action.moveA),
                proof.action.moveB != Stockfish::MOVE_NONE
                    && board.is_capture(BOARD_B, proof.action.moveB));
            outPlyToMate = proof.pliesToMate;
            if (outPrincipalVariation) {
                for (auto ply = proof.principalVariation;
                     ply; ply = ply->next) {
                    outPrincipalVariation->push_back(ply->action);
                }
            }
            return true;
        };

        // Excluding every shorter joint mate can consume the entire budget
        // before a simple longer line is reached. Keep a verified upper bound
        // first, then spend what remains trying to shorten it. When only one
        // board is ours to move, the other board's defender can still capture,
        // check or buy a pass, so a local scout alone is never a certificate.
        JointMateProof scoutedProof{JointMateStatus::UNKNOWN, 0, {}};
        // The scout's candidate and its reduced verification both sit the
        // partner board, so they search no deeper than the lead pays for.
        if (satAttackerMoves >= 2 && boardAOnTurn != boardBOnTurn) {
            const int activeBoard = boardAOnTurn ? BOARD_A : BOARD_B;
            const Stockfish::Color attackerColor = boardAOnTurn
                ? teamSide : ~teamSide;
            MateSearchBudget scoutBudget = budget;
            const uint64_t scoutNodes = std::min(
                budget.remainingNodes / 2,
                SearchParams::ROOT_MATE_SCOUT_NODES);
            scoutBudget.remainingNodes = scoutNodes;
            Stockfish::Move candidate = Stockfish::MOVE_NONE;
            int candidatePly = 0;
            int candidateMoveLimit = 0;
            bool foundCandidate = false;
            for (int moves = 2;
                 scoutNodes > 0 && moves <= satAttackerMoves;
                 ++moves) {
                foundCandidate = search_single_board_forced_mate_impl(
                    board, activeBoard, attackerColor, 1, moves,
                    candidate, candidatePly, &scoutBudget, nullptr, true);
                if (foundCandidate) {
                    candidateMoveLimit = moves;
                }
                if (foundCandidate || scoutBudget.exhausted) {
                    break;
                }
            }
            budget.remainingNodes -= scoutNodes - scoutBudget.remainingNodes;
            if (foundCandidate && !budget.out_of_time()) {
                MateSearchBudget verifyBudget = budget;
                const uint64_t verifyNodes = std::min(
                    budget.remainingNodes,
                    SearchParams::ROOT_MATE_SCOUT_VERIFY_NODES);
                verifyBudget.remainingNodes = verifyNodes;
                const MateJointAction action = boardAOnTurn
                    ? MateJointAction{candidate, Stockfish::MOVE_NONE}
                    : MateJointAction{Stockfish::MOVE_NONE, candidate};
                board.make_moves(action.moveA, action.moveB);
                // Verify at the depth that found the candidate. Its reported
                // distance can be shortened by a waiting mate, while verifying
                // at the global maximum wastes the bounded proof budget.
                const JointMateProof tail = search_reduced_partner_mate(
                    board, teamSide, activeBoard, attackerColor, ~teamSide,
                    TimeState::legacy(teamSide, true),
                    candidateMoveLimit - 1,
                    1, verifyBudget, &cache);
                board.unmake_moves(action.moveA, action.moveB);
                budget.remainingNodes -= verifyNodes - verifyBudget.remainingNodes;
                if (tail.status == JointMateStatus::PROVEN) {
                    scoutedProof = {
                        JointMateStatus::PROVEN, tail.pliesToMate + 1, action,
                        std::make_shared<JointMateProofLine>(
                            JointMateProofLine{action, tail.principalVariation})};
                    if (stopAtFirstProof) {
                        return publish_proof(scoutedProof);
                    }
                }
            }
        }
        // Every depth refuted within the budget: no forcing mate within
        // the scan's horizon, whatever a larger budget would add. The
        // sat-partner proofs above can also find quiet mates, so a scan
        // that ran them only counts as complete once they had no lead to
        // run on.
        bool jointComplete = satAttackerMoves == 0
            || !(boardAOnTurn && boardBOnTurn);
        for (int maxMateMoves = 2;
             maxMateMoves <= SearchParams::MATE_SEARCH_MAX_ATTACKER_MOVES;
             ++maxMateMoves) {
            if (scoutedProof.status == JointMateStatus::PROVEN
                && 2 * maxMateMoves - 1 >= scoutedProof.pliesToMate) {
                return publish_proof(scoutedProof);
            }
            const JointMateProof proof = search_joint_forced_mate(
                board, teamSide, time, teamSide,
                maxMateMoves, 0, budget, cache, attackerWinsMateRace,
                stopOnSolvedRoot);
            if (proof.status == JointMateStatus::PROVEN) {
                return publish_proof(proof);
            }
            if (outDepthLimited && proof.status == JointMateStatus::REFUTED) {
                *outDepthLimited = proof.depthLimited;
            }
            if (proof.status != JointMateStatus::REFUTED || budget.exhausted) {
                jointComplete = false;
            }
            if (budget.exhausted) {
                break;
            }
        }
        if (scoutedProof.status == JointMateStatus::PROVEN) {
            return publish_proof(scoutedProof);
        }
        if (outComplete) {
            *outComplete = jointComplete;
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
    Node* liveRoot,
    bool scanWaitingChecks) {
    return find_root_loss_proofs(
        board, teamSide, TimeState::legacy(teamSide, teamHasTimeAdvantage),
        outProofs, nodeBudget, deadline, cancelled, preferredActions,
        liveRoot, scanWaitingChecks);
}

bool Agent::find_root_loss_proofs(
    Board& board,
    Stockfish::Color teamSide,
    const TimeState& time,
    vector<RootLossProof>& outProofs,
    uint64_t nodeBudget,
    MateSearchBudget::Clock::time_point deadline,
    const std::atomic<bool>* cancelled,
    const vector<JointActionCandidate>* preferredActions,
    Node* liveRoot,
    bool scanWaitingChecks,
    bool* outComplete,
    const std::function<void(const RootLossProof&)>& onProof,
    const vector<RootLossProof>* knownProofs,
    LossScanShare* share) {
    outProofs.clear();
    if (outComplete) {
        *outComplete = false;
    }
    // Every defense proven lost or settled safe within its share.
    bool everyDefenseSettled = true;
    MateSearchBudget totalBudget;
    totalBudget.remainingNodes = nodeBudget;
    totalBudget.deadline = deadline;
    totalBudget.cancelled = cancelled;
    totalBudget.stopOnSolvedRoot = liveRoot;
    if (totalBudget.out_of_time()) {
        return false;
    }
    vector<JointActionCandidate> defenses = legal_joint_actions(
        board, teamSide, time);
    if (defenses.empty() || nodeBudget == 0) {
        return false;
    }
    const size_t legalDefenseCount = defenses.size();
    // Defenses already proven lost keep their proofs; the budget goes to the
    // rest.
    if (knownProofs && !knownProofs->empty()) {
        defenses.erase(
            std::remove_if(
                defenses.begin(), defenses.end(),
                [&](const JointActionCandidate& defense) {
                    const auto known = std::find_if(
                        knownProofs->begin(), knownProofs->end(),
                        [&](const RootLossProof& proof) {
                            return proof.action.moveA == defense.moveA
                                && proof.action.moveB == defense.moveB;
                        });
                    if (known == knownProofs->end()) {
                        return false;
                    }
                    outProofs.push_back(*known);
                    return true;
                }),
            defenses.end());
    }

    const bool hasFocusedCandidates =
        (preferredActions && !preferredActions->empty())
        || (liveRoot && liveRoot->is_expanded());
    const size_t focusedDefenseCount = hasFocusedCandidates
        ? std::min(
            SearchParams::ROOT_LOSS_FOCUSED_ACTIONS, defenses.size())
        : 0;
    const uint64_t focusedAllocation = focusedDefenseCount > 0
        ? (nodeBudget * SearchParams::ROOT_LOSS_FOCUSED_BUDGET_PERCENT / 100)
            / focusedDefenseCount
        : 0;

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
    if (defenses.empty()) {
        if (outComplete) {
            *outComplete = true;
        }
        return true;
    }

    const Stockfish::Color opponentTeam = ~teamSide;
    // Every defense costs the same, so the opponents answer all of them under
    // one clock: the one the defense leaves.
    const TimeState defenseTime = time.after(
        teamSide, board.on_turn_on_both_boards(teamSide));
    const bool teamHasTimeAdvantage = defenseTime.can_sit(teamSide);
    const bool opponentHasTimeAdvantage = defenseTime.can_sit(opponentTeam);
    // The cached proofs are the opponents' sat-partner-board proofs, valid
    // only while their lead covers the whole line.
    const bool opponentSitsThroughLine = defenseTime.can_sit_through(
        opponentTeam, SAT_PARTNER_PROOF_ACTIONS);
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
        if (!opponentSitsThroughLine || singleBoardMateCache.empty()) {
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

    // A pass has no neural policy entry, so the joint prior of (pass, check)
    // can remain too small for MCTS to expand even when the check forces a
    // queen feed and mate. When the opponent owns the clock, enumerate those
    // waiting checks explicitly and ask the existing reverse prover whether
    // every reply loses. The recursive proof disables this wrapper: it is
    // proving the check evasions, not opening another unbounded threat tree.
    std::optional<MateProofPly> preferredWaitingCheck;
    // Set when the wrapper below searched a defense: it bounds each check's
    // proof, so a defense it leaves unproven is not settled.
    bool waitingChecksSearched = false;
    const auto try_time_ahead_waiting_check = [&] (
        uint64_t& defenseAllowance,
        int& opponentMatePly,
        vector<MateProofPly>& opponentPv) {
        if (!scanWaitingChecks || !opponentHasTimeAdvantage
            || defenseAllowance == 0) {
            return false;
        }
        const bool opponentAOnTurn =
            board.side_to_move(BOARD_A) == opponentTeam;
        const bool opponentBOnTurn =
            board.side_to_move(BOARD_B) == ~opponentTeam;
        if (!opponentAOnTurn || !opponentBOnTurn) {
            return false;
        }
        waitingChecksSearched = true;

        // This optional wrapper must not consume the candidate's entire clock
        // before the ordinary direct and capture-feed proofs get a turn.
        const auto waitingStart = MateSearchBudget::Clock::now();
        const auto waitingDeadline = deadline == MateSearchBudget::Clock::time_point{}
            ? deadline
            : waitingStart + (deadline - waitingStart)
                  / SearchParams::ROOT_LOSS_WAITING_CHECK_TIME_DIVISOR;

        struct WaitingCheck {
            MateProofPly action;
            size_t replyCount;
            int captureThreat;
        };
        vector<WaitingCheck> checks;
        for (int boardNum : {BOARD_A, BOARD_B}) {
            for (Stockfish::Move move : board.checking_moves(boardNum)) {
                MateProofPly action = boardNum == BOARD_A
                    ? MateProofPly{move, Stockfish::MOVE_NONE}
                    : MateProofPly{Stockfish::MOVE_NONE, move};
                board.make_moves(action.moveA, action.moveB);
                const size_t replyCount = board.legal_moves(boardNum).size();
                const auto& checkedPosition = *board.pos[boardNum];
                const auto checker = checkedPosition.piece_on(Stockfish::to_sq(move));
                int captureThreat = 0;
                if (checker != Stockfish::NO_PIECE) {
                    const auto attacked = checkedPosition.attacks_from(
                        Stockfish::color_of(checker), Stockfish::type_of(checker),
                        Stockfish::to_sq(move));
                    for (auto pt : {Stockfish::PAWN, Stockfish::KNIGHT,
                                    Stockfish::BISHOP, Stockfish::ROOK,
                                    Stockfish::QUEEN}) {
                        if (attacked & checkedPosition.pieces(
                                checkedPosition.side_to_move(), pt)) {
                            captureThreat = std::max(
                                captureThreat, static_cast<int>(pt));
                        }
                    }
                }
                board.unmake_moves(action.moveA, action.moveB);
                checks.push_back({action, replyCount, captureThreat});
            }
        }
        std::stable_sort(
            checks.begin(), checks.end(),
            [&](const WaitingCheck& lhs, const WaitingCheck& rhs) {
                const auto isPreferred = [&](const WaitingCheck& candidate) {
                    return preferredWaitingCheck
                        && candidate.action.moveA
                               == preferredWaitingCheck->moveA
                        && candidate.action.moveB
                               == preferredWaitingCheck->moveB;
                };
                if (isPreferred(lhs) != isPreferred(rhs)) {
                    return isPreferred(lhs);
                }
                // Forks and skewers can force the capture that supplies the
                // partner's mate. Try those before checks with no feed threat.
                if (lhs.captureThreat != rhs.captureThreat) {
                    return lhs.captureThreat > rhs.captureThreat;
                }
                return lhs.replyCount < rhs.replyCount;
            });

        for (const WaitingCheck& check : checks) {
            if (defenseAllowance == 0 || totalBudget.out_of_time()
                || (waitingDeadline != MateSearchBudget::Clock::time_point{}
                    && MateSearchBudget::Clock::now() >= waitingDeadline)) {
                break;
            }
            const uint64_t allocation = std::min(
                {defenseAllowance, totalBudget.remainingNodes,
                 SearchParams::ROOT_LOSS_WAITING_CHECK_NODE_BUDGET});
            if (allocation == 0) {
                break;
            }

            // The waiting check is the opponents' action on both boards,
            // charged as such; this team defends under what it leaves.
            const TimeState checkTime = defenseTime.after(
                opponentTeam, board.on_turn_on_both_boards(opponentTeam));
            board.make_moves(check.action.moveA, check.action.moveB);
            bool forcedMate = board.is_checkmate(
                teamSide, checkTime.can_sit(teamSide));
            int matePly = forcedMate ? 1 : 0;
            vector<MateProofPly> line{check.action};
            if (!forcedMate
                && !board.is_checkmate(
                    opponentTeam, checkTime.can_sit(opponentTeam))
                && !board.is_draw(1)) {
                vector<RootLossProof> replyProofs;
                forcedMate = find_root_loss_proofs(
                    board, teamSide, checkTime,
                    replyProofs, allocation, waitingDeadline, cancelled,
                    nullptr, nullptr, false);
                if (forcedMate) {
                    const auto delaying = std::max_element(
                        replyProofs.begin(), replyProofs.end(),
                        [](const RootLossProof& lhs,
                           const RootLossProof& rhs) {
                            return lhs.plyToMate < rhs.plyToMate;
                        });
                    if (delaying == replyProofs.end()) {
                        forcedMate = false;
                    } else {
                        matePly = delaying->plyToMate + 1;
                        line.insert(
                            line.end(),
                            delaying->principalVariation.begin(),
                            delaying->principalVariation.end());
                    }
                }
            }
            board.unmake_moves(check.action.moveA, check.action.moveB);

            // The nested budget is intentionally charged in full. Its API is
            // conservative and does not expose unused nodes; treating them as
            // spent keeps this safety pass inside the candidate's allocation.
            defenseAllowance -= allocation;
            totalBudget.remainingNodes -= allocation;
            if (forcedMate) {
                preferredWaitingCheck = check.action;
                opponentMatePly = matePly;
                opponentPv = std::move(line);
                return true;
            }
        }
        return false;
    };

    if (share) {
        totalBudget.remainingNodes = 0;
    }
    for (size_t nextDefense = 0;;) {
        const size_t defenseIndex = share
            ? share->nextDefense.fetch_add(1, std::memory_order_relaxed)
            : nextDefense++;
        if (defenseIndex >= defenses.size()) {
            break;
        }
        // A shared scan sizes this defense from what every thread has left
        // between them and charges the pool what it spends.
        if (share) {
            totalBudget.remainingNodes =
                share->remainingNodes.load(std::memory_order_relaxed);
        }
        const uint64_t defenseStartNodes = totalBudget.remainingNodes;
        const auto charge_shared_nodes = [&] {
            if (!share) {
                return;
            }
            const uint64_t used =
                defenseStartNodes - totalBudget.remainingNodes;
            uint64_t left = share->remainingNodes.load(std::memory_order_relaxed);
            while (!share->remainingNodes.compare_exchange_weak(
                left, left - std::min(left, used),
                std::memory_order_relaxed)) {
            }
        };
        // A proof attached below immediately removes that edge from live MCTS
        // selection. Re-sample the current favourite so the bounded scan can
        // peel successive blunders instead of spending its tail on a stale
        // ordering captured before the first veto. Threads sharing a scan
        // share its order, so they keep it.
        if (!share && liveRoot && liveRoot->is_expanded()) {
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
            everyDefenseSettled = false;
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
            const uint64_t remainingDefenses =
                defenses.size() - defenseIndex;
            const uint64_t fairAllocation =
                totalBudget.remainingNodes / remainingDefenses;
            uint64_t defenseAllocation = std::min(
                totalBudget.remainingNodes,
                defenseIndex < focusedDefenseCount
                    ? std::max(fairAllocation, focusedAllocation)
                    : fairAllocation);

            std::vector<MateProofPly> waitingCheckPv;
            int waitingCheckMatePly = 0;
            waitingChecksSearched = false;
            if (defenseIndex < focusedDefenseCount
                && try_time_ahead_waiting_check(
                    defenseAllocation,
                    waitingCheckMatePly, waitingCheckPv)) {
                defenseIsMated = true;
                totalLossPly = waitingCheckMatePly + 1;
                principalVariation.insert(
                    principalVariation.end(),
                    waitingCheckPv.begin(), waitingCheckPv.end());
            }
            std::vector<MateProofPly> cachedOpponentPv;
            int cachedOpponentMatePly = 0;
            if (!defenseIsMated && try_cached_capture_feed(
                    cachedOpponentMatePly, cachedOpponentPv)) {
                defenseIsMated = true;
                totalLossPly = cachedOpponentMatePly + 1;
                principalVariation.insert(
                    principalVariation.end(),
                    cachedOpponentPv.begin(), cachedOpponentPv.end());
            }
            // The first few MCTS favourites receive enough concentrated work
            // to serve as a useful move-safety veto at a very wide root. After
            // that, divide the remaining budget fairly among every untested
            // defense. Unused probes always return to the common pool.
            defenseAllocation = std::min(
                defenseAllocation, totalBudget.remainingNodes);
            if (!defenseIsMated && defenseAllocation == 0) {
                everyDefenseSettled = false;
            }
            if (!defenseIsMated && defenseAllocation > 0) {
                MateSearchBudget defenseBudget;
                defenseBudget.remainingNodes = defenseAllocation;
                defenseBudget.deadline = deadline;
                defenseBudget.cancelled = cancelled;
                defenseBudget.stopOnSolvedRoot = liveRoot;
                JointActionCandidate opponentMate;
                int opponentMatePly = 0;
                std::vector<MateProofPly> opponentPrincipalVariation;
                bool opponentScanComplete = false;
                defenseIsMated = find_root_mate_impl(
                    board, opponentTeam, defenseTime,
                    opponentMate, opponentMatePly,
                    defenseAllocation, nullptr, &defenseBudget, true,
                    deadline, true, true, &opponentPrincipalVariation,
                    cancelled, &singleBoardMateCache, liveRoot, false,
                    &opponentScanComplete);
                if (!defenseIsMated
                    && (!opponentScanComplete || waitingChecksSearched)) {
                    everyDefenseSettled = false;
                }
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
        charge_shared_nodes();

        // Retain exact information about this action even when another action
        // is safe or cannot be proved within the bounded scan. The old
        // all-or-nothing return threw these proofs away, allowing MCTS to play
        // a known mate while an unresolved defense remained.
        if (defenseIsMated) {
            outProofs.push_back({
                defense, totalLossPly, std::move(principalVariation)});
            if (onProof) {
                onProof(outProofs.back());
            }
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

    if (outComplete) {
        *outComplete = everyDefenseSettled;
    }
    return outProofs.size() == legalDefenseCount;
}

bool Agent::find_root_forced_loss(
    Board& board,
    Stockfish::Color teamSide,
    bool teamHasTimeAdvantage,
    JointActionCandidate& outAction,
    int& outPlyToMate,
    uint64_t nodeBudget,
    MateSearchBudget::Clock::time_point deadline) {
    return find_root_forced_loss(
        board, teamSide, TimeState::legacy(teamSide, teamHasTimeAdvantage),
        outAction, outPlyToMate, nodeBudget, deadline);
}

bool Agent::find_root_forced_loss(
    Board& board,
    Stockfish::Color teamSide,
    const TimeState& time,
    JointActionCandidate& outAction,
    int& outPlyToMate,
    uint64_t nodeBudget,
    MateSearchBudget::Clock::time_point deadline) {
    vector<RootLossProof> proofs;
    if (!find_root_loss_proofs(
            board, teamSide, time,
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
        TimeState rootTime;
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
            rootTime = workerTime_;
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
                        localBoard, engine, rootTime, &running);
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
                        localBoard, engine, rootTime, &running);
                }
            }
            if (SearchParams::ENABLE_MATE_EARLY_EXIT && rootNode
                && rootNode->get_node_type() != NodeType::UNSOLVED) {
                searchThread->discard_pending_iteration(localBoard, engine);
            } else {
                searchThread->finish_pending_iteration(
                    localBoard, engine, rootTime);
            }
            // The agent still owns the tree here, so this only drops
            // reference counts; the GC thread then holds the last one.
            searchThread->release_tree_references();
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
                // Neither proof scan should hold bestmove after MCTS ends.
                rootWinScanStop_.store(true, std::memory_order_release);
                rootLossScanStop_.store(true, std::memory_order_release);
                workersDoneCv_.notify_one();
            }
        }
    }
}

void Agent::dispatch_workers(const Board& board,
                             const vector<Engine*>& engines,
                             SearchInfo& searchInfo,
                             size_t targetNodes,
                             int moveTimeMs,
                             size_t workerCount) {
    {
        std::lock_guard lock(workerMutex_);
        workerBoard_ = &board;
        workerEngines_ = engines;
        workerSearchInfo_ = &searchInfo;
        workerTime_ = searchTime_;
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
        discard(transpositionTable->detach(), false);
    }
    if (oldRoot) {
        discard(std::move(oldRoot), false);
    }
}

bool Agent::try_reuse_mate_continuation(
    Board& board, Stockfish::Color teamSide, bool teamHasTimeAdvantage,
    JointActionCandidate& outAction, int& outPlyToMate,
    bool avoidRepetition) const {
    return try_reuse_mate_continuation(
        board, teamSide, TimeState::legacy(teamSide, teamHasTimeAdvantage),
        outAction, outPlyToMate, avoidRepetition);
}

bool Agent::try_reuse_mate_continuation(
    Board& board, Stockfish::Color teamSide, const TimeState& time,
    JointActionCandidate& outAction, int& outPlyToMate,
    bool avoidRepetition) const {
    // Continuations come from the single-board prover, which sits the
    // partner board for the whole line: keyed by the flag that model takes,
    // and replayed only while the lead still covers every move left.
    const bool teamHasTimeAdvantage = time.can_sit(teamSide);
    const uint64_t positionHash = board.search_hash_key(
        teamSide, teamHasTimeAdvantage);
    const std::string signature = board_signature(board);

    for (const MateContinuation& continuation : mateContinuations_) {
        if (continuation.positionHash != positionHash
            || continuation.signature != signature
            || continuation.teamSide != teamSide
            || continuation.teamHasTimeAdvantage != teamHasTimeAdvantage
            || continuation.plyToMate <= 0
            || !time.can_sit_through(
                teamSide, (continuation.plyToMate + 1) / 2)) {
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
        // The stored line was proven in one history; the position it is being
        // replayed into carries another. A move that completes a threefold
        // ends the game as a draw before the mate arrives, so it is no longer
        // this position's continuation whatever the retained distance says.
        if (avoidRepetition) {
            board.make_moves(continuation.action.moveA,
                             continuation.action.moveB);
            const bool repeats = board.is_repetition_draw({0, 0});
            board.unmake_moves(continuation.action.moveA,
                               continuation.action.moveB);
            if (repeats) {
                continue;
            }
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
    // Freed memory goes back to the OS only once no search has run for a
    // while: the trim locks the arenas these workers allocate from.
    const GCThread::BusyScope gcBusy(gcThread_);
    {
        std::lock_guard statsLock(internalMateProbeStatsMutex_);
        lastInternalMateProbeStats_ = {};
    }
    mateCandidateHints_.clear();
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
        rootWinScanStop_.store(false, std::memory_order_release);
        rootLossScanStop_.store(false, std::memory_order_release);
        lastRuntimeConfig_ = options.search;
    }
    
    certifiedChoice_ = false;
    // Without a clock the flag stands for a permanent lead or deficit. With
    // one, the flag is whatever the clock implies for this team now, and the
    // workers charge every action they search to it.
    searchTime_ = options.clock
        ? *options.clock
        : TimeState::legacy(teamSide, teamHasTimeAdvantage);
    teamHasTimeAdvantage = searchTime_.can_sit(teamSide);
    searchStatus_ = adjudicate_game(board, teamSide, searchTime_);
    if (searchStatus_ != GameStatus::ONGOING) {
        if (!options.background) {
            nextRootCandidates_.clear();
        }
        if (options.verbose) {
            cout << "bestmove (none)" << endl;
        }
        return result;
    }

    if (engines.empty()) {
        cerr << "Cannot search without an inference engine" << endl;
        return result;
    }

    const bool boardAOnTurn = board.side_to_move(BOARD_A) == teamSide;
    const bool boardBOnTurn = board.side_to_move(BOARD_B) == ~teamSide;

    const bool mustPlayDespiteTeamMate = !teamHasTimeAdvantage
        && board.is_checkmate(teamSide, teamHasTimeAdvantage);
    const auto drop_retained_candidates = [this, &options] {
        if (!options.background) {
            nextRootCandidates_.clear();
        }
    };

    // Determine effective move time
    int moveTimeMs = options.moveTimeMs;
    size_t targetNodes = options.targetNodes;

    // Compute position hash for tree reuse
    uint64_t positionHash = board.search_hash_key(teamSide, searchTime_);
    const std::string positionSignature = board_signature(board);

    // Every claimed forced win passes this before it is acted on, whichever
    // scan, cache or subtree produced it. A win that does not mate at once
    // while the opponents can is not a win at all: bughouse clocks run in
    // parallel, so their mate is played before the rest of ours arrives.
    const auto loses_mate_race = [&](Board& target,
                                     const JointActionCandidate& action) {
        MateSearchBudget budget;
        budget.remainingNodes = SearchParams::MATE_RACE_VETO_NODE_BUDGET;
        budget.deadline = MateSearchBudget::Clock::now()
            + chrono::milliseconds(SearchParams::MATE_RACE_VETO_MAX_MS);
        budget.cancelled = &stopRequested_;
        return action_loses_mate_race(
            target, action, teamSide, searchTime_, &budget);
    };

    // Try to reuse tree from previous search (if enabled)
    std::shared_ptr<Node> reusedRoot = nullptr;
    if (SearchParams::ENABLE_TREE_REUSE) {
        reusedRoot = try_reuse_tree(
            positionHash, teamSide, positionSignature, options.background);
    }
    if (reusedRoot && mustPlayDespiteTeamMate
        && reusedRoot->get_node_type() == NodeType::LOSS) {
        // A retained node may already have been solved as a loss while it was
        // below an earlier root. Reusing its solved state would stop this live
        // partner board before it can compare the available moves.
        reusedRoot.reset();
    }
    if (reusedRoot && !options.background) {
        // A win solved under an earlier root is adopted whole, so a race the
        // solver could not express travels with it and no scan below this
        // point ever revisits it. Drop the subtree rather than the proof: a
        // stale certificate is reason enough to search the position again.
        const std::optional<JointActionCandidate> claimed = claimed_win_action(
            *reusedRoot, options.search.qVetoDelta, options.search.qValueWeight,
            options.search.drawContempt > 0.0f);
        if (claimed && loses_mate_race(board, *claimed)) {
            if (options.verbose) {
                cout << "info string Tree reuse: discarding a proven win that "
                        "loses the mate race" << endl;
            }
            reusedRoot.reset();
            drop_retained_candidates();
            lastSearchHash_ = 0;
        }
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

    // Certifying the chosen move runs once the tree has stopped growing, so
    // its time comes off the front of the allocation rather than after it.
    // It only ever vetoes when the opponents could sit through a line of
    // theirs, so with a lead longer than any such line the search keeps the
    // whole move.
    const int selectedMoveCertReserveMs =
        options.search.certifySelectedMove
            && !searchTime_.can_sit_through(teamSide, OPPONENT_PROOF_HORIZON_MOVES)
            && !options.background && options.search.enableMateProbe
            ? SearchParams::selected_move_cert_reserve_ms(moveTimeMs)
            : 0;
    const int treeMoveTimeMs = moveTimeMs > 0
        ? std::max(1, moveTimeMs - selectedMoveCertReserveMs)
        : moveTimeMs;

    // Start the clock where run_search began, not after the concurrent root
    // scans below: they are part of this move's thinking time, and resetting
    // the clock afterward would let a slow scan overrun the allotted move time.
    SearchInfo searchInfo(searchStart, treeMoveTimeMs);
    // The requested move time is the entire allocation for this move unless
    // the front end grants a ceiling above it: then the eval-drop and
    // instability extensions below may spend up to that ceiling, less the
    // same certification reserve.
    searchInfo.set_hard_limit(
        moveTimeMs > 0 && options.maxMoveTimeMs > moveTimeMs && !options.isPonder
            ? std::max(treeMoveTimeMs, options.maxMoveTimeMs - selectedMoveCertReserveMs)
            : treeMoveTimeMs);
    isPondering_.store(options.isPonder, std::memory_order_release);
    currentSearchInfo_.store(&searchInfo, std::memory_order_release);

    // MCGS: discard nodes outside the signature-verified reused graph, then
    // re-index that graph so new transpositions merge into retained nodes.
    if (options.search.enableMCGS && options.search.enableTranspositions && transpositionTable) {
        // Released on the GC thread: clearing in place costs tens of
        // milliseconds before the first mate check can run.
        discard(transpositionTable->detach(), options.background);
        if (reusedRoot) {
            reindex_reused_subtree(rootNode, options.background);
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
        st->set_mate_candidate_table(
            SearchParams::internal_mate_probe_bias_enabled(
                options.search.internalMateProbeMode)
                ? &mateCandidateHints_
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

    // Reserve one worker's CPU share for the independent root mate thread.
    // A tiny mate-in-one preflight is the exception: once a neural batch has
    // started it cannot be cancelled, so dispatching it first would make an
    // immediate mate wait for inference to finish.
    const size_t scanWorkerCount = runRootScan && workerCount > 1
        ? workerCount - 1
        : workerCount;
    bool workersDispatched = false;

    // The probe answers a question the checking-move scans cannot ask - a mate
    // that needs a quiet preparing move - so it runs beside the workers for the
    // whole move on a thread of its own, at the cost of one CPU thread (~5% of
    // MCTS nps here) and no search time. It searches one board with the other
    // sitting, which is only this team's to do when it is ahead on time; a
    // root behind on time is the joint prover's, where the defender's sits
    // are part of the search.
    atomic<bool> stopRootProbe{false};
    std::mutex rootProbeMutex;
    bool rootProbeFoundMate = false;
    JointActionCandidate rootProbeAction;
    int rootProbePly = 0;
    string rootProbePv;
    InternalMateProbeStats internalProbeStats;
    bool internalProbeStatsStored = false;
    // Mates the root probe found only with the other board sat, beyond what
    // the clock grants, and what became of them. Written by the probe
    // thread, read after it is joined.
    struct {
        int found = 0;
        int certified = 0;
        int promoted = 0;
    } conditionalRootMates;
    FrozenRaceStats frozenRaceStats;
    // Where this search's proofs came from, and how each mate finder that
    // found none ended: one line per verbose search, to tell the finders a
    // team behind on time lacks from the mates it genuinely does not have.
    // The scan and probe threads write it; the line is printed once every
    // one of them has been joined.
    struct MateTelemetry {
        atomic<int> scanRounds{0};
        atomic<uint64_t> scanNodes{0};
        atomic<const char*> scanEnd{"none"};
        atomic<bool> scanDepthLimited{false};
        atomic<int64_t> scanWinMs{-1};
        atomic<int64_t> lossScanMs{-1};
        atomic<int64_t> probeWinMs{-1};
        atomic<int64_t> verifierStartMs{-1};
        int conditionalTried = 0;
        MateCertificateFailure conditionalFailure =
            MateCertificateFailure::NONE;
        int64_t treeWinMs = -1;
        int64_t treeLossMs = -1;
    } mateTelemetry;
    const auto search_ms = [&] {
        return static_cast<int64_t>(
            chrono::duration_cast<chrono::milliseconds>(
                chrono::steady_clock::now() - searchStart).count());
    };
    const auto failure_name = [](MateCertificateFailure failure) {
        switch (failure) {
        case MateCertificateFailure::NONE: return "none";
        case MateCertificateFailure::ILLEGAL: return "illegal";
        case MateCertificateFailure::RACE: return "race";
        case MateCertificateFailure::REFUTED: return "refuted";
        case MateCertificateFailure::DEPTH: return "depth";
        case MateCertificateFailure::BUDGET: return "budget";
        }
        return "none";
    };
    struct MateTelemetryReport {
        std::function<void()> report;
        ~MateTelemetryReport() { report(); }
    } mateTelemetryReport{[&] {
        if (!options.verbose) {
            return;
        }
        const MateTelemetry& t = mateTelemetry;
        cout << "info string mate telemetry lead="
             << (searchTime_.is_clocked()
                     ? std::to_string(searchTime_.lead_ms(teamSide))
                     : searchTime_.can_sit(teamSide) ? "flag+" : "flag-")
             << " scan_rounds=" << t.scanRounds.load()
             << " scan_nodes=" << t.scanNodes.load()
             << " scan_end=" << t.scanEnd.load()
             << " scan_depth=" << (t.scanDepthLimited.load() ? 1 : 0)
             << " scan_win_ms=" << t.scanWinMs.load()
             << " probe=" << (searchTime_.can_sit(teamSide) ? 1 : 0)
             << " probe_win_ms=" << t.probeWinMs.load()
             << " cond_tried=" << t.conditionalTried
             << " cond_found=" << conditionalRootMates.found
             << " cond_cert=" << conditionalRootMates.certified
             << " cond_fail=" << failure_name(t.conditionalFailure)
             << " tree_win_ms=" << t.treeWinMs
             << " tree_loss_ms=" << t.treeLossMs
             << " loss_scan_ms=" << t.lossScanMs.load()
             << " verifier_start_ms=" << t.verifierStartMs.load() << endl;
    }};
    thread rootProbeThread;

    // The concurrent verifier's verdict on each root action, by action. A
    // proof or an exhaustive refutation within the bound is final for this
    // search; anything cut short stays eligible for another slice, and for
    // the serial tail if it is the action finally chosen. The proof cache
    // stays with the action so a later slice resumes from settled subtrees.
    using RootActionKey = std::pair<Stockfish::Move, Stockfish::Move>;
    struct RootActionKeyHash {
        size_t operator()(const RootActionKey& key) const {
            return static_cast<size_t>(Board::mix_hash(
                static_cast<uint64_t>(key.first) + 1,
                static_cast<uint64_t>(key.second) + 1));
        }
    };
    const auto action_key = [](const JointActionCandidate& action) {
        return RootActionKey{action.moveA, action.moveB};
    };
    std::unordered_map<RootActionKey, RootActionVerdict, RootActionKeyHash>
        verifierVerdicts;
    ConcurrentVerifierStats verifierStats;
    const bool runConcurrentVerifier = selectedMoveCertReserveMs > 0
        && runRootScan && moveTimeMs > 0;

    const auto start_root_probe = [&] {
        const bool runInternalProbe =
            SearchParams::internal_mate_probe_enabled(
                options.search.internalMateProbeMode);
        const bool certifyOnly = options.search.internalMateProbeMode
            == SearchParams::InternalMateProbeMode::CERTIFY_ONLY;
        // A root that is not ahead still probes: what it finds there is a
        // conditional mate, held to the joint certifier below.
        if (!runRootScan || !options.search.enableMateProbe) {
            return;
        }
        // The probe's node allowance is a schedule rather than a cap: this
        // thread searches board A, then board B, then the concurrent
        // verifier and the internal probes, one after the other, and the
        // allowance is what moves it on to the next. Every job still ends
        // with the tree (rootSettled, internalSettled).
        const uint64_t totalProbeNodeBudget = options.mateProbeNodes > 0
            ? options.mateProbeNodes
            : SearchParams::MATE_PROBE_ROOT_NODE_BUDGET;
        // The root keeps its share whatever the clock: behind, it still
        // runs the conditional probe below.
        const uint64_t internalProbeNodeBudget = runInternalProbe
            ? std::min<uint64_t>(
                  SearchParams::INTERNAL_MATE_PROBE_NODE_BUDGET,
                  totalProbeNodeBudget
                      * (100 - SearchParams::
                          INTERNAL_MATE_PROBE_ROOT_NODE_SHARE_PERCENT)
                      / 100)
            : 0;
        const uint64_t rootProbeNodeBudget =
            totalProbeNodeBudget - internalProbeNodeBudget;
        // No time limit (0) when the search has none: the probe stops with
        // the tree, through rootSettled below.
        const int totalProbeBudgetMs = options.completeMateProbe
            ? 0
            : moveTimeMs > 0
            ? std::max(1, moveTimeMs - static_cast<int>(searchInfo.elapsed()))
            : 0;
        const int rootProbeBudgetMs = runInternalProbe
                && totalProbeBudgetMs > 0
            ? std::max(
                  1, totalProbeBudgetMs
                      * SearchParams::internal_mate_probe_root_time_percent(
                          options.search.internalMateProbeMode)
                      / 100)
            : totalProbeBudgetMs;
        // Every proof scan owns its board; workers copy the caller's board.
        auto probeBoard = std::make_unique<Board>(board);
        // rootNode is only replaced after this thread has been joined, so the
        // node the workers are filling stays alive for as long as the probe
        // polls it.
        const std::shared_ptr<Node> probeRoot = rootNode;
        rootProbeThread = thread(
            [&, rootProbeNodeBudget, internalProbeNodeBudget,
             rootProbeBudgetMs, runInternalProbe, certifyOnly, probeRoot,
             probeBoard = std::move(probeBoard)]() mutable {
                JointActionCandidate action;
                int plyToMate = 0;
                string pv;
                const auto rootSettled = [&] {
                    return stopRootProbe.load(memory_order_acquire)
                        || (!options.completeMateProbe
                            && !running.load(memory_order_acquire))
                        || stopRequested_.load(memory_order_relaxed)
                        || (SearchParams::ENABLE_MATE_EARLY_EXIT && probeRoot
                            && probeRoot->get_node_type()
                                   != NodeType::UNSOLVED);
                };
                const auto publish = [&] {
                    lock_guard<std::mutex> guard(rootProbeMutex);
                    int64_t unset = -1;
                    mateTelemetry.probeWinMs.compare_exchange_strong(
                        unset, search_ms());
                    rootProbeFoundMate = true;
                    rootProbeAction = action;
                    rootProbePly = plyToMate;
                    rootProbePv = pv;
                    if (SearchParams::mate_probe_can_end_search(plyToMate)
                        && probeRoot
                        && probeRoot->get_node_type() == NodeType::UNSOLVED) {
                        rootLossScanStop_.store(true, memory_order_release);
                        running.store(false, memory_order_release);
                    }
                };
                // Nothing here is required for a move to come back, so a
                // failure costs the probe and not the search - and never an
                // unhandled exception on a detachable thread.
                try {
                    // A lead short of the probe's full depth leaves a
                    // conditional probe to follow this one. The two share
                    // the allowance, and the conditional one - probe and
                    // certificate together - takes only its fixed share: this
                    // thread runs the concurrent verifier next, and behind on
                    // time it is the verifier's proofs of the opponents'
                    // mates the move needs most. Without a lead the probe
                    // under the clock returns at once.
                    const bool conditionalFollows =
                        options.search.conditionalMateProbe
                        && !searchTime_.can_sit_through(
                            teamSide, SearchParams::MATE_PROBE_MAX_MATE_MOVES);
                    const auto share_of = [](auto amount) {
                        return amount * SearchParams::
                            ROOT_CONDITIONAL_MATE_SHARE_PERCENT / 100;
                    };
                    const uint64_t conditionalNodes = conditionalFollows
                        ? share_of(rootProbeNodeBudget) : 0;
                    const int conditionalMs = conditionalFollows
                            && rootProbeBudgetMs > 0
                        ? std::max(1, share_of(rootProbeBudgetMs)) : 0;
                    const bool clockProbeShares = conditionalFollows
                        && searchTime_.can_sit(teamSide);
                    uint64_t clockProbeNodes = 0;
                    bool rootFound = rootProbeNodeBudget > 0
                        && probe_position_mate(
                            *probeBoard, teamSide, searchTime_,
                            clockProbeShares
                                ? std::max<uint64_t>(
                                      1, rootProbeNodeBudget - conditionalNodes)
                                : rootProbeNodeBudget,
                            clockProbeShares && rootProbeBudgetMs > 0
                                ? std::max(1, rootProbeBudgetMs - conditionalMs)
                                : rootProbeBudgetMs,
                            rootSettled, action, plyToMate, pv,
                            publish, options.search.drawContempt > 0.0f,
                            &clockProbeNodes);
                    // The probe above looks only as deep as the lead pays to
                    // sit the other board, and not at all when level or
                    // behind. Past that, ask again with the other board sat
                    // regardless: a hit is a conditional mate, and only the
                    // joint certifier, playing the real clock - the
                    // defenders' waits, the moves it forces on the other
                    // board, every capture that feeds a hand - makes it a
                    // proven one. Unproven, it only orders the tree.
                    if (!rootFound && conditionalNodes > 0
                        && !rootSettled()) {
                        ++mateTelemetry.conditionalTried;
                        const auto conditionalStart =
                            MateSearchBudget::Clock::now();
                        JointActionCandidate conditional;
                        int conditionalPly = 0;
                        string conditionalPv;
                        uint64_t conditionalProbeNodes = 0;
                        if (probe_position_mate(
                                *probeBoard, teamSide,
                                TimeState::legacy(teamSide, true),
                                conditionalNodes, conditionalMs, rootSettled,
                                conditional, conditionalPly, conditionalPv,
                                {}, options.search.drawContempt > 0.0f,
                                &conditionalProbeNodes)) {
                            ++conditionalRootMates.found;
                            MateSearchBudget certificateBudget;
                            certificateBudget.remainingNodes = std::min(
                                SearchParams::
                                    ROOT_CONDITIONAL_MATE_CERT_NODE_BUDGET,
                                std::max<uint64_t>(
                                    1, conditionalNodes
                                        - std::min(conditionalNodes,
                                                   conditionalProbeNodes)));
                            certificateBudget.cancelled = &stopRootProbe;
                            certificateBudget.stopOnSolvedRoot = probeRoot.get();
                            if (conditionalMs > 0) {
                                certificateBudget.deadline = conditionalStart
                                    + chrono::milliseconds(conditionalMs);
                            }
                            int certifiedPly = 0;
                            MateCertificateTier certificateTier =
                                MateCertificateTier::NONE;
                            JointActionCandidate certified;
                            if (certify_mate_candidate(
                                    *probeBoard, teamSide, searchTime_,
                                    conditional, conditionalPly,
                                    certificateBudget, certifiedPly,
                                    certificateTier, probeRoot.get(),
                                    &certified,
                                    &mateTelemetry.conditionalFailure)) {
                                ++conditionalRootMates.certified;
                                action = certified;
                                plyToMate = certifiedPly;
                                pv = format_mate_proof_pv(
                                    *probeBoard,
                                    {MateProofPly{certified.moveA,
                                                  certified.moveB}},
                                    teamSide, searchTime_);
                                publish();
                                rootFound = true;
                            } else if (probeRoot
                                       && promote_conditional_mate(
                                           *probeRoot, *probeBoard, teamSide,
                                           conditional)) {
                                ++conditionalRootMates.promoted;
                            }
                        }
                    }
                    // Short frozen races under the leading root actions and
                    // their leading replies, proven exactly. Each proof marks
                    // the node it is about; nothing else is touched.
                    if (options.search.frozenRace && !rootFound && probeRoot) {
                        run_frozen_race_stage(
                            *probeBoard, probeRoot, teamSide, searchTime_,
                            moveTimeMs > 0
                                ? std::max(1, std::min(
                                      moveTimeMs
                                          * SearchParams::FROZEN_RACE_TIME_PERCENT
                                          / 100,
                                      (moveTimeMs - static_cast<int>(
                                          searchInfo.elapsed())) / 2))
                                : SearchParams::FROZEN_RACE_MAX_MS,
                            [&] {
                                return stopRootProbe.load(memory_order_acquire)
                                    || stopRequested_.load(memory_order_relaxed)
                                    || (!options.completeMateProbe
                                        && !running.load(memory_order_acquire))
                                    || probeRoot->get_node_type()
                                        != NodeType::UNSOLVED;
                            },
                            &stopRootProbe, frozenRaceStats);
                    }
                    const bool runInternal = runInternalProbe && !rootFound
                        && internalProbeNodeBudget > 0;
                    if (!runInternal && !(runConcurrentVerifier && !rootFound)) {
                        return;
                    }

                    const auto internalSettled = [&] {
                        return stopRootProbe.load(memory_order_acquire)
                            || (!options.completeMateProbe
                                && !running.load(memory_order_acquire))
                            || stopRequested_.load(memory_order_relaxed)
                            || (probeRoot
                                && probeRoot->get_node_type()
                                    != NodeType::UNSOLVED);
                    };

                    std::vector<InternalMateProbeTarget> targets;
                    while (runInternal && !internalSettled()) {
                        targets = collect_internal_probe_targets(
                            *probeBoard, probeRoot, teamSide,
                            searchTime_,
                            SearchParams::INTERNAL_MATE_PROBE_TOP_K,
                            SearchParams::INTERNAL_MATE_PROBE_MIN_VISITS);
                        if (!targets.empty()
                            || !running.load(memory_order_acquire)) {
                            break;
                        }
                        this_thread::sleep_for(chrono::milliseconds(1));
                    }

                    uint64_t remainingNodes = internalProbeNodeBudget;
                    for (size_t index = 0;
                         index < targets.size() && remainingNodes > 0;
                         ++index) {
                        if (internalSettled()) {
                            break;
                        }
                        InternalMateProbeTarget& target = targets[index];
                        const std::shared_ptr<Node> targetNode = target.node.lock();
                        if (!targetNode
                            || (targetNode->get_hash() != 0
                                && targetNode->get_hash()
                                    != target.positionHash)) {
                            continue;
                        }
                        if (target.treeDepth == 1) {
                            ++internalProbeStats.childProbes;
                        } else {
                            ++internalProbeStats.grandchildProbes;
                        }

                        const size_t jobsLeft = targets.size() - index;
                        const uint64_t jobNodes = std::max<uint64_t>(
                            1, remainingNodes / jobsLeft);
                        const int jobBudgetMs = options.completeMateProbe
                            ? 0
                            : moveTimeMs > 0
                            ? std::max(
                                  1, std::min(
                                         SearchParams::INTERNAL_MATE_PROBE_JOB_MAX_MS,
                                         moveTimeMs - static_cast<int>(
                                             searchInfo.elapsed())))
                            : SearchParams::INTERNAL_MATE_PROBE_JOB_MAX_MS;
                        JointActionCandidate candidate;
                        int candidatePly = 0;
                        string candidatePv;
                        uint64_t nodesUsed = 0;
                        // A single-board candidate assumes this target team
                        // can sit on the partner board for the whole line. In
                        // the hint modes the probe asks under that assumption
                        // and the result stays a hint; the proof-only mode has
                        // no hint to keep, so it asks with the clock the
                        // target's path reached. Either way only the
                        // certifier, held to that same clock, settles a node.
                        const Stockfish::Color targetTeam =
                            targetNode->get_team_to_play();
                        const TimeState probeTime = certifyOnly
                            ? target.time
                            : TimeState::legacy(targetTeam, true);
                        const bool found = probe_position_mate(
                            *target.board, targetTeam, probeTime,
                            jobNodes, jobBudgetMs, internalSettled,
                            candidate, candidatePly, candidatePv, {},
                            options.search.drawContempt > 0.0f, &nodesUsed);
                        internalProbeStats.nodes += nodesUsed;
                        remainingNodes -= std::min(remainingNodes, nodesUsed);
                        if (!found) {
                            continue;
                        }

                        if (target.treeDepth == 1) {
                            ++internalProbeStats.childHits;
                        } else {
                            ++internalProbeStats.grandchildHits;
                        }
                        const bool stillTop = edge_is_top_k(
                                target.rootParent, target.rootEdge,
                                SearchParams::INTERNAL_MATE_PROBE_TOP_K)
                            && (target.replyEdge < 0
                                || edge_is_top_k(
                                    target.replyParent, target.replyEdge,
                                    SearchParams::INTERNAL_MATE_PROBE_TOP_K));
                        const std::shared_ptr<Node> rootParent =
                            target.rootParent.lock();
                        const float currentRootQ = rootParent
                            ? rootParent->get_child_q(target.rootEdge)
                            : target.rootQAtSubmit;
                        const bool alreadySolved =
                            targetNode->get_node_type() != NodeType::UNSOLVED;
                        const bool alreadyGenerated = node_generated_action(
                            targetNode, candidate);
                        if (!stillTop) {
                            ++internalProbeStats.staleHits;
                        }
                        if (alreadySolved) {
                            ++internalProbeStats.alreadySolvedHits;
                        }
                        if (alreadyGenerated) {
                            ++internalProbeStats.alreadyGeneratedHits;
                        }
                        const bool opponentMate = target.treeDepth == 1;
                        if (stillTop && !alreadySolved
                            && SearchParams::internal_mate_probe_hit_is_actionable(
                                options.search.internalMateProbeMode,
                                opponentMate, currentRootQ)) {
                            if (target.treeDepth == 1) {
                                ++internalProbeStats.childActionableHits;
                            } else {
                                ++internalProbeStats.grandchildActionableHits;
                            }

                            if (!SearchParams::internal_mate_probe_affects_play(
                                    options.search.internalMateProbeMode)) {
                                continue;
                            }
                            if (SearchParams::internal_mate_probe_bias_enabled(
                                    options.search.internalMateProbeMode)) {
                                // Promotion mutates only candidate ordering.
                                // Do it once here so workers never take the
                                // target's exclusive lock merely because a
                                // hint exists.
                                if (!targetNode->promote_joint_action(
                                        candidate)) {
                                    ++internalProbeStats.promotionsSkipped;
                                }
                                mateCandidateHints_.publish(
                                    target.positionHash, candidate,
                                    candidatePly);
                            }

                            if (!SearchParams::
                                    internal_mate_probe_certification_enabled(
                                        options.search.internalMateProbeMode)) {
                                continue;
                            }

                            // Certification failure does not retract a hint
                            // the bias mode published: solver state still
                            // requires an exact result. In the proof-only
                            // mode there is no hint, so a failure leaves no
                            // trace at all.
                            const uint64_t certificateNodeBudget = std::min(
                                remainingNodes,
                                SearchParams::INTERNAL_MATE_CERT_NODE_BUDGET);
                            const int certificateBudgetMs = options.completeMateProbe
                                ? SearchParams::INTERNAL_MATE_CERT_MAX_MS
                                : moveTimeMs > 0
                                ? std::max(
                                    0, std::min(
                                        SearchParams::INTERNAL_MATE_CERT_MAX_MS,
                                        moveTimeMs - static_cast<int>(
                                            searchInfo.elapsed())))
                                : SearchParams::INTERNAL_MATE_CERT_MAX_MS;
                            if (certificateNodeBudget == 0
                                || certificateBudgetMs == 0) {
                                continue;
                            }
                            MateSearchBudget certificateBudget;
                            certificateBudget.remainingNodes =
                                certificateNodeBudget;
                            certificateBudget.cancelled = &stopRootProbe;
                            certificateBudget.stopOnSolvedRoot = targetNode.get();
                            certificateBudget.deadline =
                                MateSearchBudget::Clock::now()
                                + chrono::milliseconds(
                                    certificateBudgetMs);
                            int certifiedPly = 0;
                            MateCertificateTier certificateTier =
                                MateCertificateTier::NONE;
                            const bool certified = certify_mate_candidate(
                                *target.board, targetTeam, target.time,
                                candidate, candidatePly, certificateBudget,
                                certifiedPly, certificateTier,
                                targetNode.get());
                            internalProbeStats.certificateNodes +=
                                certificateNodeBudget
                                - certificateBudget.remainingNodes;
                            remainingNodes -= std::min(
                                remainingNodes,
                                certificateNodeBudget
                                    - certificateBudget.remainingNodes);
                            if (certified) {
                                if (certificateTier
                                    == MateCertificateTier::FULL_JOINT) {
                                    ++internalProbeStats.jointCertificates;
                                } else if (certificateTier
                                    == MateCertificateTier::REDUCED_PARTNER) {
                                    ++internalProbeStats
                                        .reducedPartnerCertificates;
                                } else {
                                    ++internalProbeStats.checksOnlyCertificates;
                                }
                                targetNode->mark_as_win(certifiedPly);
                            }
                        }
                    }

                    if (!runConcurrentVerifier || rootFound) {
                        return;
                    }
                    mateTelemetry.verifierStartMs.store(search_ms());
                    // The verifier: the leading root actions, each played
                    // on the private board and asked whether the opponents
                    // then have a proven mate. Fairy first for the
                    // single-board ones, then the joint solver in slices, so
                    // an action that stops leading stops being searched and
                    // the rest of the move goes to whichever leads now.
                    const auto remaining_move_ms = [&] {
                        return std::max(
                            0, moveTimeMs - static_cast<int>(searchInfo.elapsed()));
                    };
                    const auto pick_target = [&]() -> int {
                        const std::vector<size_t> edges = ranked_visited_edges(
                            probeRoot, SearchParams::CONCURRENT_VERIFIER_MIN_VISITS);
                        int chosen = -1;
                        int fewestWeighted = std::numeric_limits<int>::max();
                        for (size_t rank = 0; rank < edges.size()
                                 && rank < static_cast<size_t>(
                                     SearchParams::CONCURRENT_VERIFIER_TOP_K);
                             ++rank) {
                            const int edge = static_cast<int>(edges[rank]);
                            const std::shared_ptr<Node> child =
                                probeRoot->get_child(edge);
                            if (!child
                                || child->get_node_type() != NodeType::UNSOLVED) {
                                continue;
                            }
                            const auto verdict = verifierVerdicts.find(
                                action_key(probeRoot->get_joint_action(edge)));
                            const int slices = verdict == verifierVerdicts.end()
                                ? 0 : verdict->second.slices;
                            if (verdict != verifierVerdicts.end()
                                && verdict->second.state
                                    != RootActionVerdict::State::UNKNOWN) {
                                continue;
                            }
                            // The leader is the move most likely to be
                            // played, so it gets the larger share; the rest
                            // still get theirs, so a long job on the leader
                            // never leaves them unexamined.
                            const int weighted = rank == 0
                                ? slices
                                : slices * SearchParams::
                                    CONCURRENT_VERIFIER_LEADER_SHARE;
                            if (weighted < fewestWeighted) {
                                fewestWeighted = weighted;
                                chosen = edge;
                            }
                        }
                        return chosen;
                    };

                    uint64_t nodesSpent = 0;
                    while (!internalSettled()
                           && nodesSpent
                               < SearchParams::CONCURRENT_VERIFIER_NODE_CEILING
                           && remaining_move_ms() > 0) {
                        const int edge = pick_target();
                        if (edge < 0) {
                            this_thread::sleep_for(chrono::milliseconds(1));
                            continue;
                        }
                        const JointActionCandidate action =
                            probeRoot->get_joint_action(edge);
                        const std::shared_ptr<Node> child =
                            probeRoot->get_child(edge);
                        RootActionVerdict& verdict =
                            verifierVerdicts[action_key(action)];
                        if (verdict.slices == 0) {
                            ++verifierStats.actions;
                        }
                        verify_root_action_slice(
                            *probeBoard, action, teamSide, searchTime_, verdict,
                            std::min(
                                SearchParams::CONCURRENT_VERIFIER_SLICE_NODES,
                                SearchParams::CONCURRENT_VERIFIER_NODE_CEILING
                                    - nodesSpent),
                            SearchParams::CONCURRENT_VERIFIER_PROBE_MAX_MS,
                            MateSearchBudget::Clock::now()
                                + chrono::milliseconds(
                                    std::max(1, remaining_move_ms())),
                            &stopRootProbe, probeRoot.get(), nodesSpent,
                            &verifierStats, child.get());
                        if (verdict.state == RootActionVerdict::State::PROVEN_LOSS
                            && child
                            && child->get_node_type() == NodeType::UNSOLVED) {
                            child->mark_as_win(verdict.plyToMate);
                        }
                    }
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
        if (!internalProbeStatsStored) {
            internalProbeStatsStored = true;
            {
                std::lock_guard statsLock(internalMateProbeStatsMutex_);
                lastInternalMateProbeStats_ = internalProbeStats;
            }
            if (frozenRaceStats.races > 0) {
                cout << "info string frozen race: " << frozenRaceStats.targets
                     << " targets, " << frozenRaceStats.races << " races, "
                     << frozenRaceStats.proven << " proven ("
                     << frozenRaceStats.unplayedWitnesses
                     << " without a generated move), "
                     << frozenRaceStats.nodes << " nodes" << endl;
            }
            if (options.verbose && conditionalRootMates.found > 0) {
                cout << "info string conditional root mate: "
                     << conditionalRootMates.found << " found, "
                     << conditionalRootMates.certified << " certified, "
                     << conditionalRootMates.promoted << " promoted" << endl;
            }
            if (options.verbose
                && SearchParams::internal_mate_probe_enabled(
                    options.search.internalMateProbeMode)) {
                cout << "info string internal mate probe: child "
                     << internalProbeStats.childHits << "/"
                     << internalProbeStats.childProbes
                     << " actionable "
                     << internalProbeStats.childActionableHits
                     << ", grandchild "
                     << internalProbeStats.grandchildHits << "/"
                     << internalProbeStats.grandchildProbes
                     << " actionable "
                     << internalProbeStats.grandchildActionableHits
                     << ", stale " << internalProbeStats.staleHits
                     << ", solved " << internalProbeStats.alreadySolvedHits
                     << ", generated "
                     << internalProbeStats.alreadyGeneratedHits
                     << ", promotion skipped "
                     << internalProbeStats.promotionsSkipped
                     << ", fairy nodes " << internalProbeStats.nodes
                     << ", certificates "
                     << internalProbeStats.checksOnlyCertificates << "+"
                     << internalProbeStats.reducedPartnerCertificates << "+"
                     << internalProbeStats.jointCertificates
                     << ", certificate nodes "
                     << internalProbeStats.certificateNodes
                     << endl;
            }
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
    // proof after checking for a fresh mate in one. A changed partner
    // board, pocket, side to move, team, or TimeAdvantage value cannot match.
    JointActionCandidate rootMateAction;
    int rootMatePly = 1;
    vector<MateProofPly> rootMatePv;
    // The winning and losing root scans run beside the tree for exactly as
    // long as it searches. Neither has a budget or deadline of its own: each
    // runs complete scans in rounds of growing size (run_scan_rounds below)
    // until the tree stops - its deadline, the stop command, a solved root or
    // its node target all set the scans' stop flags - or a round settles
    // everything the scan can search.
    const auto scan_stopped = [&](const std::atomic<bool>& stop,
                                  const Node* root) {
        return stop.load(memory_order_acquire)
            || (SearchParams::ENABLE_MATE_EARLY_EXIT && root
                && root->get_node_type() != NodeType::UNSOLVED);
    };
    // Runs @p round with node budgets growing by ROOT_SCAN_ROUND_GROWTH until
    // it returns true (a decision), the scan is stopped, or a round reports
    // it searched everything it could.
    const auto run_scan_rounds = [&](const std::atomic<bool>& stop,
                                     const Node* root, uint64_t firstRoundNodes,
                                     const auto& round) {
        for (uint64_t roundNodes = firstRoundNodes;;
             roundNodes *= SearchParams::ROOT_SCAN_ROUND_GROWTH) {
            bool complete = false;
            if (round(roundNodes, complete) || complete
                || scan_stopped(stop, root)
                || roundNodes > SearchParams::MATE_SEARCH_UNBOUNDED_NODES
                       / SearchParams::ROOT_SCAN_ROUND_GROWTH) {
                return;
            }
        }
    };
    thread rootMateThread;
    atomic<bool> rootMateFound{false};
    JointActionCandidate backgroundMateAction;
    int backgroundMatePly = 0;
    vector<MateProofPly> backgroundMatePv;
    uint64_t rootMateScanNanos = 0;
    uint64_t rootLossScanNanos = 0;
    const auto stop_root_mate_scan = [&] {
        rootWinScanStop_.store(true, memory_order_release);
        if (rootMateThread.joinable()) {
            rootMateThread.join();
        }
    };
    RootProbeJoin rootMateJoin{stop_root_mate_scan};
    const auto start_root_mate_scan = [&](bool includeImmediateMate) {
        if (!runRootScan) {
            return;
        }
        auto mateBoard = std::make_unique<Board>(board);
        const std::shared_ptr<Node> mateRoot = rootNode;
        rootMateThread = thread(
            [&, includeImmediateMate, mateRoot,
             mateBoard = std::move(mateBoard)] {
                lower_mate_search_thread_priority();
                try {
                    bool accepted = false;
                    bool firstRound = true;
                    run_scan_rounds(
                        rootWinScanStop_, mateRoot.get(),
                        SearchParams::ROOT_SCAN_FIRST_ROUND_NODES,
                        [&](uint64_t roundNodes, bool& complete) {
                            bool depthLimited = false;
                            const bool found = find_root_mate_impl(
                                *mateBoard, teamSide, searchTime_,
                                backgroundMateAction, backgroundMatePly,
                                roundNodes, &mateContinuations_, nullptr,
                                true, {}, includeImmediateMate && firstRound,
                                false, &backgroundMatePv, &rootWinScanStop_,
                                nullptr, mateRoot.get(), true, &complete,
                                &depthLimited);
                            firstRound = false;
                            ++mateTelemetry.scanRounds;
                            mateTelemetry.scanNodes += roundNodes;
                            mateTelemetry.scanDepthLimited.store(depthLimited);
                            mateTelemetry.scanEnd.store(
                                found ? "proof" : complete ? "complete"
                                                           : "cut");
                            // A larger round would find the same first proof,
                            // so one that loses the race ends the scan too.
                            accepted = found
                                && !loses_mate_race(
                                    *mateBoard, backgroundMateAction);
                            return found;
                        });
                    rootMateScanNanos = static_cast<uint64_t>(
                        chrono::duration_cast<chrono::nanoseconds>(
                            chrono::steady_clock::now() - searchStart).count());
                    if (accepted) {
                        mateTelemetry.scanWinMs.store(search_ms());
                        // Publish the complete proof before stopping the tree.
                        // Interrupt a reverse scan too: the mate answers it.
                        rootMateFound.store(true, memory_order_release);
                        rootLossScanStop_.store(true, memory_order_release);
                        stopRootProbe.store(true, memory_order_release);
                        running.store(false, memory_order_release);
                    }
                } catch (const std::exception& error) {
                    cout << "info string root mate scan failed: "
                         << error.what() << endl;
                } catch (...) {
                    cout << "info string root mate scan failed" << endl;
                }
            });
    };
    // The reverse scan proves root actions losing, under the clock state's
    // actual joint-action rules, so a proven loss is authoritative whether
    // this team is ahead or behind on time. Each proof is published here as
    // its round finds it and attached to the tree by the polling loop below;
    // a proof that every action loses ends the search the way a mate does.
    std::mutex rootLossMutex;
    vector<RootLossProof> rootLossProofs;  // guarded by rootLossMutex
    atomic<bool> rootLossProven{false};
    // The delaying defense, written before rootLossProven is set.
    JointActionCandidate rootLossAction;
    int rootLossPly = 0;
    vector<MateProofPly> rootLossPv;
    thread rootLossThread;
    const auto stop_root_loss_scan = [&] {
        rootLossScanStop_.store(true, memory_order_release);
        if (rootLossThread.joinable()) {
            rootLossThread.join();
        }
    };
    RootProbeJoin rootLossJoin{stop_root_loss_scan};
    const auto start_root_loss_scan = [&] {
        if (!runRootScan) {
            return;
        }
        // The moves MCTS is actually considering go first, so a certificate
        // can veto a likely blunder early; the scan re-reads the live
        // favourites as it goes.
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
        auto lossBoard = std::make_unique<Board>(board);
        const std::shared_ptr<Node> lossRoot = rootNode;
        rootLossThread = thread(
            [&, lossRoot, lossBoard = std::move(lossBoard),
             preferredRootActions = std::move(preferredRootActions)] {
                lower_mate_search_thread_priority();
                try {
                    bool lost = false;
                    // Each proof is published the moment its defense is
                    // settled, so the polling loop can attach it to an edge
                    // generated later in the round.
                    const auto publish = [&](const RootLossProof& proof) {
                        lock_guard<std::mutex> guard(rootLossMutex);
                        const bool known = std::any_of(
                            rootLossProofs.begin(), rootLossProofs.end(),
                            [&](const RootLossProof& seen) {
                                return seen.action.moveA == proof.action.moveA
                                    && seen.action.moveB == proof.action.moveB;
                            });
                        if (!known) {
                            rootLossProofs.push_back(proof);
                        }
                    };
                    // Defenses a round proves stay proven: later rounds
                    // spend their larger budget on the rest only.
                    vector<RootLossProof> provenDefenses;
                    run_scan_rounds(
                        rootLossScanStop_, lossRoot.get(),
                        SearchParams::ROOT_LOSS_SCAN_FIRST_ROUND_NODES,
                        [&](uint64_t roundNodes, bool& complete) {
                            // The defenses are independent, so the round's
                            // threads take them in turn from one queue and
                            // one node pool, each on its own board.
                            constexpr size_t threads = SearchParams::ROOT_LOSS_SCAN_THREADS;
                            LossScanShare share;
                            share.remainingNodes.store(roundNodes);
                            vector<vector<RootLossProof>> threadProofs(threads);
                            // Not vector<bool>: the threads write their own
                            // entries concurrently.
                            vector<uint8_t> threadComplete(threads, 0);
                            const auto run_thread = [&](size_t index, Board& threadBoard) {
                                bool done = false;
                                find_root_loss_proofs(
                                    threadBoard, teamSide, searchTime_,
                                    threadProofs[index], roundNodes, {},
                                    &rootLossScanStop_, &preferredRootActions,
                                    lossRoot.get(), true, &done, publish,
                                    &provenDefenses,
                                    threads > 1 ? &share : nullptr);
                                threadComplete[index] = done;
                            };
                            vector<thread> helpers;
                            vector<std::unique_ptr<Board>> helperBoards;
                            for (size_t index = 1; index < threads; ++index) {
                                helperBoards.push_back(std::make_unique<Board>(*lossBoard));
                                helpers.emplace_back(
                                    [&, index, threadBoard = helperBoards.back().get()] {
                                        lower_mate_search_thread_priority();
                                        // A failed thread leaves the round
                                        // unsettled.
                                        try {
                                            run_thread(index, *threadBoard);
                                        } catch (...) {
                                            threadComplete[index] = false;
                                        }
                                    });
                            }
                            run_thread(0, *lossBoard);
                            for (thread& helper : helpers) {
                                helper.join();
                            }
                            // Every call returns the known proofs; each new
                            // one was found by exactly one thread.
                            vector<RootLossProof> roundProofs = provenDefenses;
                            complete = true;
                            for (size_t index = 0; index < threads; ++index) {
                                complete = complete && threadComplete[index];
                                // A call stopped before it read them returns
                                // none.
                                vector<RootLossProof>& found = threadProofs[index];
                                if (found.size() > provenDefenses.size()) {
                                    roundProofs.insert(
                                        roundProofs.end(),
                                        std::make_move_iterator(
                                            found.begin() + provenDefenses.size()),
                                        std::make_move_iterator(found.end()));
                                }
                            }
                            lost = roundProofs.size() == legal_joint_actions(
                                *lossBoard, teamSide, searchTime_).size();
                            provenDefenses = roundProofs;
                            const auto delaying = std::max_element(
                                roundProofs.begin(), roundProofs.end(),
                                [](const RootLossProof& lhs,
                                   const RootLossProof& rhs) {
                                    return lhs.plyToMate < rhs.plyToMate;
                                });
                            lost = lost && delaying != roundProofs.end();
                            if (lost) {
                                lock_guard<std::mutex> guard(rootLossMutex);
                                rootLossAction = delaying->action;
                                rootLossPly = delaying->plyToMate;
                                rootLossPv = delaying->principalVariation;
                            }
                            return lost;
                        });
                    rootLossScanNanos = static_cast<uint64_t>(
                        chrono::duration_cast<chrono::nanoseconds>(
                            chrono::steady_clock::now() - searchStart).count());
                    if (lost) {
                        mateTelemetry.lossScanMs.store(search_ms());
                        // Every action loses: nothing the tree or the other
                        // scans can find changes the move.
                        rootLossProven.store(true, memory_order_release);
                        rootWinScanStop_.store(true, memory_order_release);
                        stopRootProbe.store(true, memory_order_release);
                        running.store(false, memory_order_release);
                    }
                } catch (const std::exception& error) {
                    cout << "info string root loss scan failed: "
                         << error.what() << endl;
                } catch (...) {
                    cout << "info string root loss scan failed" << endl;
                }
            });
    };
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
        // Background results and timing are read only after joining the scan.
        rootScanStats_.scanNanos += rootMateScanNanos > 0
            ? std::max(rootMateScanNanos, rootLossScanNanos)
            : static_cast<uint64_t>(
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
        stop_root_mate_scan();
        stop_root_loss_scan();
        stop_root_probe();
        halt_workers_after_scan();
        record_root_scan(true);
        result = provenAction;
        const uint64_t provenPositionHash = board.search_hash_key(
            teamSide, searchTime_);
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
            store_next_root_candidates(board, searchTime_);
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
    try {
        // A mate the immediate scan finds is on the board now: the opponents
        // never move again, so there is no race for them to win.
        if (runRootScan) {
            // This one check runs before the tree starts, so it keeps a
            // small fixed allowance; the scans proper start with the tree.
            MateSearchBudget immediateBudget;
            immediateBudget.remainingNodes =
                SearchParams::IMMEDIATE_MATE_PREFLIGHT_NODE_BUDGET;
            immediateBudget.cancelled = &stopRequested_;
            const auto immediateDeadline = chrono::steady_clock::now()
                + chrono::milliseconds(
                    SearchParams::IMMEDIATE_MATE_PREFLIGHT_MAX_MS);
            immediateBudget.deadline = moveTimeMs > 0
                ? std::min<MateSearchBudget::Clock::time_point>(
                      searchStart + chrono::milliseconds(treeMoveTimeMs),
                      immediateDeadline)
                : immediateDeadline;
            immediateRootMate = find_immediate_root_mate(
                *scanBoard, teamSide, searchTime_,
                rootMateAction, &immediateBudget);
            immediateScanComplete = !immediateBudget.exhausted;
            if (immediateRootMate) {
                rootMatePly = 1;
                rootMatePv = {{rootMateAction.moveA, rootMateAction.moveB}};
            }
        }

        cachedRootMate = runRootScan && !immediateRootMate
            && try_reuse_mate_continuation(
                *scanBoard, teamSide, searchTime_,
                rootMateAction, rootMatePly,
                options.search.drawContempt > 0.0f);
        if (cachedRootMate && loses_mate_race(*scanBoard, rootMateAction)) {
            // Forget a retained line that loses the cross-board mate race.
            cachedRootMate = false;
            mateContinuations_.clear();
        }

        if (!cachedRootMate && !immediateRootMate) {
            dispatch_workers(board, engines, searchInfo,
                             targetNodes, moveTimeMs, scanWorkerCount);
            workersDispatched = true;
            start_root_mate_scan(!immediateScanComplete);
            start_root_loss_scan();
            start_root_probe();
        }
    } catch (...) {
        stop_root_mate_scan();
        stop_root_loss_scan();
        stop_root_probe();
        halt_workers_after_scan();
        throw;
    }
    if (cachedRootMate || immediateRootMate) {
        const string proofPv = format_mate_proof_pv(
            board, rootMatePv, teamSide, searchTime_);
        return return_proven_mate(rootMateAction, rootMatePly, proofPv);
    }

    const auto return_background_mate = [&] {
        stop_root_mate_scan();
        return return_proven_mate(
            backgroundMateAction, backgroundMatePly,
            format_mate_proof_pv(
                board, backgroundMatePv, teamSide, searchTime_));
    };

    // Replace the tree with a synthetic one-edge root holding the defense
    // that delays the proven loss longest, report it, and hand it back. The
    // scans and workers are stopped and joined first, as for a mate.
    const auto return_proven_loss = [&] {
        stop_root_mate_scan();
        stop_root_loss_scan();
        stop_root_probe();
        halt_workers_after_scan();
        record_root_scan(true);
        result = rootLossAction;
        const uint64_t provenPositionHash = board.search_hash_key(
            teamSide, searchTime_);
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
            store_next_root_candidates(board, searchTime_);
            lastSearchHash_ = positionHash;
        }

        rootScanStats_.thinkNanos += static_cast<uint64_t>(
            chrono::duration_cast<chrono::nanoseconds>(
                chrono::steady_clock::now() - searchStart).count());
        if (options.verbose) {
            const string bestMoveStr = extract_best_move(board);
            const string proofPv = format_mate_proof_pv(
                board, rootLossPv, teamSide, searchTime_);
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
    };

    // The reverse scan can prove individual moves losing without proving the
    // root lost. Attach each such certificate to the corresponding MCTS child
    // as soon as that edge exists. Node selection already excludes a child
    // that is a proven WIN for the opponent while any unresolved alternative
    // remains. Workers may still be initializing or progressively widening the
    // root, so retry only the newly generated edge range at each poll.
    // The polling thread's copy of the proofs the loss scan has published so
    // far, in publication order, with the edge each one was attached to.
    vector<RootLossProof> knownLossProofs;
    vector<int> rootLossProofChildIndices;
    vector<weak_ptr<Node>> rootLossProofChildren;
    size_t rootLossEdgesProcessed = 0;
    const auto apply_root_loss_proofs = [&] {
        {
            lock_guard<std::mutex> guard(rootLossMutex);
            for (size_t index = knownLossProofs.size();
                 index < rootLossProofs.size(); ++index) {
                knownLossProofs.push_back(rootLossProofs[index]);
            }
        }
        if (knownLossProofs.size() > rootLossProofChildIndices.size()) {
            // A new proof is checked against every generated edge, not only
            // those generated since the last poll.
            rootLossProofChildIndices.resize(knownLossProofs.size(), -1);
            rootLossProofChildren.resize(knownLossProofs.size());
            rootLossEdgesProcessed = 0;
        }
        const vector<RootLossProof>& rootLossProofs = knownLossProofs;
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
        const vector<RootLossProof>& rootLossProofs = knownLossProofs;
        const auto proof = std::find_if(
            rootLossProofs.begin(), rootLossProofs.end(),
            [&](const RootLossProof& candidate) {
                return candidate.action.moveA == action.moveA
                    && candidate.action.moveB == action.moveB;
            });
        return proof == rootLossProofs.end()
            ? string{}
            : format_mate_proof_pv(
                board, proof->principalVariation, teamSide, searchTime_);
    };

    // Bestmove skips root children already proven to draw when contempt is
    // on, so every report ranks them the same way.
    const bool avoidSolvedDraw = lastRuntimeConfig_.drawContempt > 0.0f;

    // Prints `info depth ... [multipv N] ... pv ...` for up to `maxLines`
    // root children: rank 1 is the move bestmove would play now, the rest
    // follow by visits. `visitedOnly` leaves out ranks below 1 that no visit
    // has reached yet, whose q is still the unvisited -1. The block goes out
    // in one write, so no other output splits it.
    const auto print_multipv_block = [&](int maxLines, bool visitedOnly) {
        std::ostringstream out;
        double elapsedMs = searchInfo.elapsed();
        int nodes = searchInfo.get_nodes_searched();
        int depth = searchInfo.get_max_depth();
        int nps = (elapsedMs > 0) ? static_cast<int>((nodes * 1000.0) / elapsedMs) : 0;
        const bool tableOn = options.search.enableMCGS
            && options.search.enableTranspositions
            && transpositionTable;
        size_t tbhits = tableOn ? transpositionTable->getHits() : 0;
        int hashfull = tableOn ? transpositionTable->getFullness() : 0;

        // Convert Q-value [-1, 1] to centipawns using Lc0 tangent formula
        constexpr float C = 180.0f;
        constexpr float k = 1.56f;

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
                options.search.qVetoDelta, options.search.qValueWeight,
                avoidSolvedDraw);
            if (solverIdx >= 0) {
                auto it = std::find(sortedIndices.begin(), sortedIndices.end(), static_cast<size_t>(solverIdx));
                if (it != sortedIndices.end() && it != sortedIndices.begin()) {
                    sortedIndices.erase(it);
                    sortedIndices.insert(sortedIndices.begin(), static_cast<size_t>(solverIdx));
                }
            }

            if (visitedOnly && !sortedIndices.empty()) {
                sortedIndices.erase(
                    std::remove_if(sortedIndices.begin() + 1, sortedIndices.end(),
                                   [&](size_t i) { return childVisits[i] <= 0; }),
                    sortedIndices.end());
            }

            // Output up to maxLines lines
            int numPVs = min(maxLines, static_cast<int>(sortedIndices.size()));
            for (int pvIdx = 0; pvIdx < numPVs; ++pvIdx) {
                size_t childIdx = sortedIndices[pvIdx];
                string pv = exact_root_loss_pv(
                    rootNode->get_joint_action(static_cast<int>(childIdx)));
                if (pv.empty()) {
                    pv = extract_pv_from_child(
                        board, static_cast<int>(childIdx), 20,
                        teamSide, searchTime_);
                }
                float childQ = rootNode->get_child_q(static_cast<int>(childIdx));
                // A solved root's score belongs to the move that proves it;
                // the other lines keep their own.
                string scoreStr = pvIdx == 0
                    ? format_root_aware_uci_score(
                        rootNode, children[childIdx], childQ,
                        childVisits[childIdx], C, k)
                    : format_uci_score(children[childIdx].get(), childQ, true, C, k);

                out << "info depth " << depth;
                if (maxLines > 1) {
                    out << " multipv " << (pvIdx + 1);
                }
                out << " " << scoreStr
                    << " nodes " << nodes
                    << " nps " << nps
                    << " hashfull " << hashfull
                    << " tbhits " << tbhits
                    << " time " << static_cast<int>(elapsedMs);

                if (!pv.empty()) {
                    out << " pv " << pv;
                }
                out << '\n';
            }
        } else {
            // Fallback: single PV line with root Q
            string pv = extract_pv(board, 20);
            // Use root's own Q value (which is from root's perspective)
            float rootQ = rootNode ? rootNode->Q() : 0.0f;
            string scoreStr = rootNode ? format_uci_score(rootNode.get(), rootQ, false, C, k)
                                       : "score cp 0";

            out << "info depth " << depth
                << " " << scoreStr
                << " nodes " << nodes
                << " nps " << nps
                << " hashfull " << hashfull
                << " tbhits " << tbhits
                << " time " << static_cast<int>(elapsedMs);

            if (!pv.empty()) {
                out << " pv " << pv;
            }
            out << '\n';
        }
        const string block = out.str();
        if (!block.empty()) {
            cout << block << std::flush;
        }
    };

    // While a verbose search runs it prints the rank-1 line whenever the
    // maximum depth grows. With ReportIntervalMs set it prints the whole
    // MultiPV block every interval instead, so a GUI can follow every ranked
    // line as it changes, not only the first until depth stops growing.
    int lastReportedDepth = 0;
    double lastReportMs = 0.0;
    const auto report_while_searching = [&]() {
        if (!options.verbose || options.background
            || !rootNode || !rootNode->is_expanded()) {
            return;
        }
        if (options.reportIntervalMs > 0) {
            const double elapsedMs = searchInfo.elapsed();
            if (elapsedMs - lastReportMs < options.reportIntervalMs) {
                return;
            }
            lastReportMs = elapsedMs;
            print_multipv_block(options.multiPV, true);
            return;
        }
        const int depth = searchInfo.get_max_depth();
        if (depth <= lastReportedDepth) {
            return;
        }
        lastReportedDepth = depth;
        print_multipv_block(1, false);
    };

    // Periodic info output during search (UCI verbose mode only)
    // Also handles early stopping and time extension
    // When the tree, with the proofs attached to it, first holds the root
    // solved; the scans' own proofs end the search before it would.
    const auto note_tree_proof = [&] {
        if (!rootNode) {
            return;
        }
        const NodeType rootType = rootNode->get_node_type();
        int64_t& seen = rootType == NodeType::WIN ? mateTelemetry.treeWinMs
            : rootType == NodeType::LOSS ? mateTelemetry.treeLossMs
            : mateTelemetry.treeWinMs;
        if (rootType != NodeType::UNSOLVED && rootType != NodeType::DRAW
            && seen < 0) {
            seen = search_ms();
        }
    };
    constexpr int POLL_INTERVAL_MS = 5;
    bool nodeSearchStalled = false;
    int stalledCompletedNodes = 0;
    if (options.verbose && moveTimeMs > 0) {
        searchInfo.set_in_game(true);
        float lastCheckEval = 0.0f;
        bool evalInitialized = false;
        int lastBestChildIdx = -1;
        
        while (running && (isPondering_.load(std::memory_order_relaxed)
                           || searchInfo.elapsed() < searchInfo.get_effective_move_time())) {
            apply_root_loss_proofs();
            note_tree_proof();
            // A mate the probe landed mid-move ends the search the way MCTS
            // proving one does; the collapse below the loop reports it.
            if (probe_mate_ends_search()) {
                running = false;
                break;
            }
            if (isPondering_.load(std::memory_order_relaxed)
                && ponder_budget_exhausted(searchInfo)) {
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
            
            double elapsedMs = searchInfo.elapsed();
            
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

                    report_while_searching();
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
            apply_root_loss_proofs();
            note_tree_proof();
            if (probe_mate_ends_search()) {
                running = false;
                break;
            }
            if (isPondering_.load(std::memory_order_relaxed)
                && ponder_budget_exhausted(searchInfo)) {
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
            note_tree_proof();
            if (probe_mate_ends_search()) {
                running = false;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(POLL_INTERVAL_MS));
            // A verbose node search (go nodes, go infinite) reports as it
            // deepens, like a timed one: an analysis front end shows nothing
            // until bestmove otherwise.
            report_while_searching();
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
            // A node-limited ponder ignores its target the same way a timed one
            // ignores the clock, so it needs the ceilings too.
            if (isPondering_.load(std::memory_order_relaxed)
                && ponder_budget_exhausted(searchInfo)) {
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
    
    // Signal workers and the root mate scan to stop.
    note_tree_proof();
    running = false;
    rootWinScanStop_.store(true, memory_order_release);
    isPondering_.store(false, std::memory_order_release);
    currentSearchInfo_.store(nullptr, std::memory_order_release);
    
    wait_for_workers();
    stop_root_mate_scan();
    stop_root_loss_scan();
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

    if (rootMateFound.load(memory_order_acquire)) {
        return return_background_mate();
    }
    if (rootLossProven.load(memory_order_acquire)) {
        return return_proven_loss();
    }
    record_root_scan(false);

    // Nothing MCTS searched came back proven, so a mate the probe found while
    // it ran - either mid-move, which ended the loop above, or on the last
    // poll before the deadline - decides the move instead.
    if (rootNode && rootNode->get_node_type() == NodeType::UNSOLVED) {
        if (options.completeMateProbe && rootProbeThread.joinable()) {
            rootProbeThread.join();
        } else {
            stop_root_probe();
        }
        JointActionCandidate probedAction;
        int probedPly = 0;
        string probedPv;
        if (take_probe_mate(probedAction, probedPly, probedPv)
            && !loses_mate_race(board, probedAction)) {
            return return_proven_mate(probedAction, probedPly, probedPv);
        }
    }

    if (avoidSolvedDraw) {
        mark_immediate_root_repetitions(rootNode, board);
    }

    // Extract best joint action by selecting the most visited child
    if (rootNode && rootNode->is_expanded()) {
        auto visits = rootNode->get_child_visits();
        auto children = rootNode->get_children();
        if (!visits.empty() && !children.empty()) {
            size_t numChildren = min(visits.size(), children.size());
            
            // Use Q-value weighted selection (with veto and weighting)
            int bestIdx = rootNode->get_best_move_idx_with_q_weight(
                options.search.qVetoDelta, options.search.qValueWeight,
                avoidSolvedDraw);
            
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

            // A win the tree proved during this search answers to the same
            // test the scans above passed. The proof is what put this action
            // ahead of the rest, so when the race is lost the move is the best
            // of the alternatives the search actually looked at.
            if (const std::optional<JointActionCandidate> safer =
                    race_safe_alternative(
                        board, *rootNode, teamSide, searchTime_,
                        options.search.qVetoDelta, options.search.qValueWeight,
                        avoidSolvedDraw, &stopRequested_)) {
                if (options.verbose) {
                    cout << "info string Proven win loses the mate race; "
                            "playing an alternative" << endl;
                }
                result = *safer;
                certifiedChoice_ = true;
            }

            // A root the search left unsolved chose this action on visits
            // and Q. Hold it to a proof before it is played: the reserve
            // taken off the move time above is spent here, up to the move's
            // original deadline. A solved root has already been held to one.
            const auto leaderVerdict = verifierVerdicts.find(action_key(result));
            const bool leaderSettled = leaderVerdict != verifierVerdicts.end()
                && leaderVerdict->second.state
                    != RootActionVerdict::State::UNKNOWN;
            if (runConcurrentVerifier) {
                verifierStats.leaderVerdict =
                    leaderVerdict == verifierVerdicts.end() ? "unseen"
                    : leaderVerdict->second.state
                            == RootActionVerdict::State::PROVEN_LOSS
                        ? "proven"
                    : leaderVerdict->second.state
                            == RootActionVerdict::State::REFUTED_AT_BOUND
                        ? "refuted"
                        : "unknown";
                if (options.verbose) {
                    cout << "info string concurrent verifier: "
                         << verifierStats.actions << " actions, "
                         << verifierStats.slices << " slices, "
                         << verifierStats.probes << " probes ("
                         << verifierStats.probeHits << " hits), "
                         << verifierStats.proven << " proven ("
                         << verifierStats.provenNodes << " nodes), "
                         << verifierStats.refuted << " refuted, "
                         << verifierStats.nodes << " nodes, leader "
                         << verifierStats.leaderVerdict << endl;
                }
            }
            if (selectedMoveCertReserveMs > 0 && !leaderSettled
                && rootNode->get_node_type() == NodeType::UNSOLVED) {
                const auto certStart = MateSearchBudget::Clock::now();
                // The reserve follows the tree's allocation, extensions included.
                const auto certDeadline = certStart + chrono::milliseconds(
                    std::max(1, searchInfo.get_effective_move_time()
                                    + selectedMoveCertReserveMs
                                    - static_cast<int>(searchInfo.elapsed())));
                SelectedMoveCertStats certStats;
                const std::optional<JointActionCandidate> mateFree =
                    certified_mate_free_alternative(
                        board, *rootNode, result, teamSide,
                        searchTime_, certDeadline, &stopRequested_,
                        &certStats);
                if (mateFree) {
                    result = *mateFree;
                    certifiedChoice_ = true;
                }
                if (options.verbose) {
                    const auto certMs = chrono::duration_cast<
                        chrono::milliseconds>(
                        MateSearchBudget::Clock::now() - certStart).count();
                    cout << "info string selected move certification: "
                         << certStats.candidates << " candidates, "
                         << certStats.probeHits << " probe hits, "
                         << certStats.certificates << " certified"
                         << (certStats.vetoed
                                 ? certStats.replaced
                                       ? "; chosen move vetoed, alternative played"
                                       : "; chosen move vetoed, no alternative"
                                 : "")
                         << ", " << certStats.nodes << " probe nodes, "
                         << certStats.certificateNodes << " certificate nodes, "
                         << certMs << "ms" << endl;
                }
            }
        }
    }
    
    if (!options.background) {
        rootScanStats_.thinkNanos += static_cast<uint64_t>(
            chrono::duration_cast<chrono::nanoseconds>(
                chrono::steady_clock::now() - searchStart).count());
    }

    // Output UCI info if verbose
    if (options.verbose) {
        const int nodes = searchInfo.get_nodes_searched();
        // Always report final up-to-date info line(s) before bestmove.
        print_multipv_block(options.multiPV, false);

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
        // Report the action this search returns rather than re-deriving it:
        // the two agree everywhere except where the mate-race veto replaced a
        // proven win, and there the caller's move is the one to announce.
        string bestMoveStr = rootNode && rootNode->is_expanded()
            ? "(" + (result.moveA == Stockfish::MOVE_NONE
                         ? string("pass")
                         : board.uci_move(BOARD_A, result.moveA))
                + "," + (result.moveB == Stockfish::MOVE_NONE
                             ? string("pass")
                             : board.uci_move(BOARD_B, result.moveB))
                + ")"
            : extract_best_move(board);
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
    //
    // A ponder that was stopped rather than hit guessed the reply wrong, so
    // every position below its chosen move is unreachable. Indexing them
    // anyway costs the real search, which is already waiting to join this
    // thread, around ten milliseconds before it can even check for a mate.
    // The permanent brain also runs as a ponder, to ignore the clock, but it
    // guessed nothing: it is always stopped, and its index is the whole point.
    const bool missedPonder = options.isPonder && !options.background
        && stopRequested_.load(std::memory_order_acquire);
    if (SearchParams::ENABLE_TREE_REUSE && missedPonder) {
        nextRootCandidates_.clear();
        lastSearchHash_ = 0;
    } else if (SearchParams::ENABLE_TREE_REUSE) {
        if (options.background) {
            // No move is played from a background root - the position we are
            // next asked about lies below it - so retain the root itself.
            nextRootCandidates_.clear();
            retain_reuse_candidates(rootNode, board, searchTime_);
        } else {
            store_next_root_candidates(board, searchTime_);
        }
        lastSearchHash_ = board.search_hash_key(teamSide, searchTime_);
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

    // The position after our move carries the clock after it: our action is
    // charged to us if we played it on turn on both boards.
    SearchOptions nextOptions = options;
    if (nextOptions.clock) {
        nextOptions.clock->advance(teamSide, board.on_turn_on_both_boards(teamSide));
    }
    Board nextBoard(board);
    nextBoard.make_moves(playedAction.moveA, playedAction.moveB);
    run_background_search(
        nextBoard, engines, teamSide, teamHasTimeAdvantage, nextOptions);
}

void Agent::clear_stop_request() {
    stopRequested_.store(false, std::memory_order_release);
    rootWinScanStop_.store(false, std::memory_order_release);
    rootLossScanStop_.store(false, std::memory_order_release);
}

void Agent::run_background_search(Board& board,
                                  const vector<Engine*>& engines,
                                  Stockfish::Color teamSide,
                                  bool teamHasTimeAdvantage,
                                  const SearchOptions& options) {
    if (!SearchParams::ENABLE_PERMANENT_BRAIN
        || !SearchParams::ENABLE_TREE_REUSE
        || engines.empty()
        || stopRequested_.load(std::memory_order_acquire)) {
        return;
    }

    SearchOptions backgroundOptions;
    backgroundOptions.search = options.search;
    // A next-move time budgets the move a front end is about to search. This
    // tree starts with the opponents' reply, so our next move in it has no
    // budget yet and costs our usual time.
    if (options.clock) {
        backgroundOptions.clock = options.clock->without_next_move_time();
    }
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
        run_search(board, engines, ~teamSide, !teamHasTimeAdvantage,
                   backgroundOptions);
    } catch (const std::exception& error) {
        cout << "info string permanent brain stopped: " << error.what() << endl;
    } catch (...) {
        cout << "info string permanent brain stopped" << endl;
    }
}

vector<RootEdgeStats> Agent::root_edge_stats() const {
    vector<RootEdgeStats> stats;
    if (searchStatus_ != GameStatus::ONGOING || !rootNode || !rootNode->is_expanded()) {
        return stats;
    }

    const auto visits = rootNode->get_child_visits();
    const size_t edgeCount = min(visits.size(), rootNode->get_num_generated());
    stats.reserve(edgeCount);
    for (size_t index = 0; index < edgeCount; ++index) {
        stats.push_back({rootNode->get_joint_action(static_cast<int>(index)), visits[index],
                         rootNode->get_child_q(static_cast<int>(index)),
                         rootNode->get_child(static_cast<int>(index))
                             ? rootNode->get_child(static_cast<int>(index))->get_node_type()
                             : NodeType::UNSOLVED});
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
        lastRuntimeConfig_.qVetoDelta, lastRuntimeConfig_.qValueWeight,
        lastRuntimeConfig_.drawContempt > 0.0f);
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
        lastRuntimeConfig_.qVetoDelta, lastRuntimeConfig_.qValueWeight,
        lastRuntimeConfig_.drawContempt > 0.0f);
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
    Stockfish::Color rootTeam, const TimeState& rootTime) {
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
    TimeState lineTime = rootTime.after(
        rootTeam, tempBoard.on_turn_on_both_boards(rootTeam));
    
    // Apply moves to temp board
    tempBoard.make_moves(action.moveA, action.moveB);
    
    // Continue extracting PV from this child. The workers keep growing the
    // tree while this runs, and a transposition can replace a child, so the
    // walk holds a reference to the node it stands on rather than a raw
    // pointer a replaced child could leave dangling.
    std::shared_ptr<Node> currentNode = children[childIdx];
    
    for (int depth = 1; depth < maxDepth; depth++) {
        if (!currentNode || !currentNode->is_expanded()) {
            break;
        }
        
        const auto nodeChildren = currentNode->get_children();
        const auto childVisits = currentNode->get_child_visits();
        if (nodeChildren.empty() || childVisits.empty()) {
            break;
        }
        
        // Find best child: prefer solver-proven path, fallback to most visits
        int bestIdx = currentNode->get_best_move_idx_with_q_weight(
            lastRuntimeConfig_.qVetoDelta, lastRuntimeConfig_.qValueWeight);
        
        // Fallback to most-visited (handles unsolved nodes and edge cases).
        // The node may have generated children since the snapshots above were
        // taken, so an index past either of them falls back too.
        if (bestIdx < 0 || static_cast<size_t>(bestIdx) >= nodeChildren.size()
            || static_cast<size_t>(bestIdx) >= childVisits.size()) {
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
        const Stockfish::Color mover = currentNode->get_team_to_play();
        lineTime.advance(mover, tempBoard.on_turn_on_both_boards(mover));
        
        // Apply moves to temp board for next iteration
        tempBoard.make_moves(nextAction.moveA, nextAction.moveB);
        
        // Move to best child
        currentNode = nodeChildren[bestIdx];
    }

    if (currentNode
        && currentNode->get_node_type() != NodeType::UNSOLVED) {
        append_waiting_mate_suffix(
            tempBoard, currentNode->get_team_to_play(),
            rootTeam, lineTime, boardSearchPlies, pv);
    }
    
    return pv;
}

void Agent::set_is_running(bool value) {
    if (!value) {
        // Latched, not just mirrored into `running`: a background search that
        // has not dispatched yet would otherwise set `running` back to true
        // after this and never see the stop.
        stopRequested_.store(true, std::memory_order_release);
        rootWinScanStop_.store(true, std::memory_order_release);
        rootLossScanStop_.store(true, std::memory_order_release);
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

InternalMateProbeStats Agent::internal_mate_probe_stats() const {
    std::lock_guard statsLock(internalMateProbeStatsMutex_);
    return lastInternalMateProbeStats_;
}

std::string Agent::board_signature(Board& board) {
    return board.fen(BOARD_A) + "|" + board.fen(BOARD_B);
}

void Agent::discard(std::shared_ptr<void> item, bool background) {
    gcThread_.enqueue(std::move(item), !background,
                      background ? &stopRequested_ : nullptr);
}

void Agent::reindex_reused_subtree(const std::shared_ptr<Node>& reusedRoot,
                                   bool background) {
    if (!transpositionTable || !reusedRoot) {
        return;
    }

    // The retained root owns the graph while this runs before workers start.
    // Traverse raw pointers to avoid copying every node's shared_ptr vector;
    // acquire one owner only when inserting the node into the table.
    std::vector<Node*> pending = {reusedRoot.get()};
    std::unordered_set<const Node*> visited;
    // A background search is stopped for the position the next search needs,
    // and that search is waiting on this thread. Nodes left unindexed stay
    // reusable through ordinary edges, as they do past the node cap.
    while (!pending.empty()
           && visited.size() < SearchParams::TREE_REUSE_REINDEX_MAX_NODES
           && !(background
                && stopRequested_.load(std::memory_order_acquire))) {
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
                                            const std::string& signature,
                                            bool background) {
    // The hash locates the candidate; the signature is what admits it. Retained
    // edges were generated against that exact board, and a stale pocket would
    // make reused drops illegal. Under a clock the hash is the exact clock's,
    // so a drifted lead finds nothing: deeper down it can change who may sit,
    // and with it the actions a retained node was expanded with.
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
        discard(rootNode, background);
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
    if (!rootNode) {
        nextRootCandidates_.clear();
        return;
    }
    store_next_root_candidates(
        board, TimeState::legacy(rootNode->get_team_to_play(), teamHasTimeAdvantage));
}

void Agent::store_next_root_candidates(Board& board,
                                       const TimeState& rootTime) {
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
    const Stockfish::Color rootTeam = rootNode->get_team_to_play();
    TimeState ownNextTime = rootTime;
    ownNextTime.advance(rootTeam, board.on_turn_on_both_boards(rootTeam));
    Board ownNextBoard(board);
    ownNextBoard.make_moves(ownAction.moveA, ownAction.moveB);
    retain_reuse_candidates(children[bestIdx], ownNextBoard, ownNextTime);
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
 * @param subtreeTime Clock state at subtreeRoot. The walk charges every action
 *        it descends through, as the search did, so each retained hash is the
 *        one the next search computes for that position.
 */
void Agent::retain_reuse_candidates(const std::shared_ptr<Node>& subtreeRoot,
                                    Board& board,
                                    const TimeState& subtreeTime) {
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
            subtreeRoot, board, subtreeTime, 0, level);
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
                               const TimeState& nodeTime,
                               int depth,
                               int level) {
    if (!node
        || nextRootCandidates_.size()
               >= SearchParams::TREE_REUSE_MAX_CANDIDATES) {
        return;
    }

    if (depth == level) {
        const uint64_t positionHash = board.search_hash_key(
            node->get_team_to_play(), nodeTime);
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
    const Stockfish::Color actingTeam = node->get_team_to_play();
    TimeState childTime = nodeTime;
    childTime.advance(actingTeam, board.on_turn_on_both_boards(actingTeam));
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
            children[index], board, childTime, depth + 1, level);
        board.unmake_moves(reply.moveA, reply.moveB);
        if (nextRootCandidates_.size()
                >= SearchParams::TREE_REUSE_MAX_CANDIDATES) {
            return;
        }
    }
}

Agent::ExhaustiveMateResult Agent::prove_joint_mate_exhaustive(
    Board& board, Stockfish::Color teamSide, const TimeState& time,
    int maxAttackerMoves, uint64_t maxNodes) {
    using Status = ExhaustiveMateResult::Status;
    static constexpr uint32_t INFINITE =
        std::numeric_limits<uint32_t>::max() / 4;
    static constexpr uint32_t NO_NODE = std::numeric_limits<uint32_t>::max();
    const auto add = [](uint32_t lhs, uint32_t rhs) {
        return std::min(INFINITE, lhs + rhs);
    };
    // The search is a graph: positions reached by different move orders - the
    // king shuffles a sat board forces are the usual case - share one node,
    // found by position, clock and remaining attacker actions. A node's moves
    // live on its edges, and the board follows the search down from the root
    // and back.
    struct PnEdge {
        uint32_t child = 0;
        Stockfish::Move moveA = Stockfish::MOVE_NONE;
        Stockfish::Move moveB = Stockfish::MOVE_NONE;
    };
    struct PnNode {
        std::vector<PnEdge> edges;
        // An attacker node first gets its checks and captures, the rest of
        // its actions only once those look no easier than this many proofs.
        uint32_t pendingQuiet = 0;
        uint32_t proof = 1;
        uint32_t disproof = 1;
        TimeState time;
        int movesLeft = 0;  // Attacker actions still allowed.
        bool attackerToPlay = true;
        bool expanded = false;
    };
    static constexpr uint32_t QUIET_PROOF = 16;
    std::vector<PnNode> nodes;
    std::unordered_map<uint64_t, uint32_t> table;
    const size_t reserve = static_cast<size_t>(
        std::min<uint64_t>(maxNodes, uint64_t{1} << 22));
    nodes.reserve(reserve + 1);
    table.reserve(reserve);

    // Moves made on each board inside the search, for the per-board
    // repetition rule.
    std::array<int, 2> plies{0, 0};
    const auto make = [&](const PnEdge& edge) {
        board.make_moves(edge.moveA, edge.moveB);
        plies[BOARD_A] += edge.moveA != Stockfish::MOVE_NONE;
        plies[BOARD_B] += edge.moveB != Stockfish::MOVE_NONE;
    };
    const auto unmake = [&](const PnEdge& edge) {
        board.unmake_moves(edge.moveA, edge.moveB);
        plies[BOARD_A] -= edge.moveA != Stockfish::MOVE_NONE;
        plies[BOARD_B] -= edge.moveB != Stockfish::MOVE_NONE;
    };
    const auto terminal = [&](const TimeState& clock) {
        Board::LegalMoveCache legalMoveCache;
        if (board.is_checkmate(
                ~teamSide, clock.can_sit(~teamSide), &legalMoveCache)) {
            return JointMateStatus::PROVEN;
        }
        if (board.is_checkmate(
                teamSide, clock.can_sit(teamSide), &legalMoveCache)
            || board.is_draw(plies)) {
            return JointMateStatus::REFUTED;
        }
        return JointMateStatus::UNKNOWN;
    };
    const auto settle = [&](PnNode& node, JointMateStatus status) {
        if (status == JointMateStatus::PROVEN) {
            node.proof = 0;
            node.disproof = INFINITE;
        } else if (status == JointMateStatus::REFUTED) {
            node.proof = INFINITE;
            node.disproof = 0;
        }
    };
    const auto team_of = [&](const PnNode& node) {
        return node.attackerToPlay ? teamSide : ~teamSide;
    };
    // An open node starts from roughly its team's number of actions: one is
    // enough for the attackers, every one is needed against the defenders.
    // The product of the boards' move counts is only a guide, so it is not
    // worth the joint action list each node builds when it is expanded.
    const auto initialise = [&](PnNode& node) {
        const Stockfish::Color team = team_of(node);
        const auto choices = [&](int boardNum, Stockfish::Color side) {
            return board.side_to_move(boardNum) == side
                ? static_cast<uint32_t>(board.legal_moves(boardNum).size()) + 1
                : 1u;
        };
        const uint32_t actions = std::max(
            1u, choices(BOARD_A, team) * choices(BOARD_B, ~team) - 1);
        node.proof = node.attackerToPlay ? 1 : actions;
        node.disproof = node.attackerToPlay ? actions : 1;
    };
    const auto key_of = [&](const PnNode& node) {
        return board.search_hash_key(team_of(node), node.time)
            ^ (static_cast<uint64_t>(node.movesLeft + 1)
               * 0x9e3779b97f4a7c15ULL);
    };

    ExhaustiveMateResult result;
    PnNode root;
    root.time = time;
    root.movesLeft = maxAttackerMoves;
    nodes.push_back(root);
    const JointMateStatus rootStatus = terminal(time);
    if (rootStatus != JointMateStatus::UNKNOWN || maxAttackerMoves <= 0) {
        settle(nodes[0], rootStatus == JointMateStatus::PROVEN
            ? rootStatus : JointMateStatus::REFUTED);
    } else {
        initialise(nodes[0]);
    }

    std::vector<uint32_t> path;  // Nodes from the root.
    std::vector<PnEdge> pathEdges;  // The edge into each node but the root.
    // Checks on either board and captures, which are what attackers most
    // often mate with; the root takes every action at once.
    const auto forcing = [&](const JointActionCandidate& action) {
        const auto forces = [&](int boardNum, Stockfish::Move move) {
            return move != Stockfish::MOVE_NONE
                && (board.gives_check(boardNum, move)
                    || board.is_capture(boardNum, move));
        };
        return forces(BOARD_A, action.moveA) || forces(BOARD_B, action.moveB);
    };
    // The board stands at the node. Adds the node's actions - an attacker
    // node's forcing ones on expansion and its quiet ones when widened -
    // as edges to new or shared children.
    const auto add_children = [&](uint32_t index, bool widen) {
        const PnNode parent = PnNode{
            {}, 0, 0, 0, nodes[index].time, nodes[index].movesLeft,
            nodes[index].attackerToPlay, false};
        const Stockfish::Color toPlay = team_of(parent);
        const vector<JointActionCandidate> actions =
            legal_joint_actions(board, toPlay, parent.time);
        const TimeState childTime = parent.time.after(
            toPlay, board.on_turn_on_both_boards(toPlay));
        const bool split = parent.attackerToPlay && index != 0;
        std::vector<PnEdge> added;
        uint32_t quiet = 0;
        for (const JointActionCandidate& action : actions) {
            if (split) {
                const bool isForcing = forcing(action);
                if (isForcing == widen) {
                    quiet += isForcing ? 0 : 1;
                    continue;
                }
            }
            PnEdge edge{NO_NODE, action.moveA, action.moveB};
            PnNode child;
            child.time = childTime;
            child.attackerToPlay = !parent.attackerToPlay;
            child.movesLeft = parent.movesLeft
                - (parent.attackerToPlay ? 1 : 0);
            JointMateStatus status = JointMateStatus::UNKNOWN;
            // At the root, a first action the opponents answer with a mate
            // they already hold never gets its follow-up.
            if (index == 0
                && action_loses_mate_race(
                    board, action, teamSide, parent.time)) {
                status = JointMateStatus::REFUTED;
            }
            const bool defenderFeeds = !parent.attackerToPlay
                && ((action.moveA != Stockfish::MOVE_NONE
                     && board.is_capture(BOARD_A, action.moveA))
                    || (action.moveB != Stockfish::MOVE_NONE
                        && board.is_capture(BOARD_B, action.moveB)));
            make(edge);
            if (status == JointMateStatus::UNKNOWN) {
                // A defender's capture is in their partner's hand at once: a
                // drop it makes a mate ends the line before we move again.
                JointActionCandidate raceMate;
                status = defenderFeeds && find_immediate_root_mate(
                        board, ~teamSide, childTime, raceMate)
                    ? JointMateStatus::REFUTED : terminal(childTime);
            }
            if (status == JointMateStatus::UNKNOWN && child.attackerToPlay
                && child.movesLeft <= 0) {
                status = JointMateStatus::REFUTED;
            }
            if (status != JointMateStatus::UNKNOWN) {
                settle(child, status);
                edge.child = static_cast<uint32_t>(nodes.size());
                nodes.push_back(child);
            } else {
                const uint64_t key = key_of(child);
                const auto found = table.find(key);
                // A node on the current line would close a cycle; the
                // repetition rule ends most of those, and the rest get a
                // node of their own.
                if (found != table.end()
                    && std::find(path.begin(), path.end(), found->second)
                        == path.end()) {
                    edge.child = found->second;
                } else {
                    initialise(child);
                    edge.child = static_cast<uint32_t>(nodes.size());
                    nodes.push_back(child);
                    if (found == table.end()) {
                        table.emplace(key, edge.child);
                    }
                }
            }
            unmake(edge);
            added.push_back(edge);
        }
        PnNode& expanded = nodes[index];
        expanded.expanded = true;
        expanded.pendingQuiet = widen ? 0 : quiet;
        expanded.edges.insert(expanded.edges.end(), added.begin(), added.end());
    };
    // Attacker nodes need one proven child, defender nodes all of them.
    const auto update = [&](uint32_t index) {
        PnNode& node = nodes[index];
        if (!node.expanded) {
            return;
        }
        if (node.edges.empty() && node.pendingQuiet == 0) {
            settle(node, JointMateStatus::REFUTED);
            return;
        }
        uint32_t minimum = INFINITE;
        uint32_t sum = 0;
        for (const PnEdge& edge : node.edges) {
            const PnNode& next = nodes[edge.child];
            minimum = std::min(
                minimum, node.attackerToPlay ? next.proof : next.disproof);
            sum = add(sum, node.attackerToPlay ? next.disproof : next.proof);
        }
        if (node.pendingQuiet > 0) {
            // The quiet actions not yet added: each one more child to refute.
            minimum = std::min(minimum, QUIET_PROOF);
            sum = add(sum, node.pendingQuiet);
        }
        node.proof = node.attackerToPlay ? minimum : sum;
        node.disproof = node.attackerToPlay ? sum : minimum;
    };

    while (nodes[0].proof != 0 && nodes[0].disproof != 0
           && nodes.size() < maxNodes) {
        path.assign(1, 0);
        pathEdges.clear();
        uint32_t current = 0;
        bool widen = false;
        while (nodes[current].expanded) {
            const PnNode& node = nodes[current];
            uint32_t best = 0;
            uint32_t bestValue = INFINITE + 1;
            for (uint32_t edge = 0; edge < node.edges.size(); ++edge) {
                const PnNode& next = nodes[node.edges[edge].child];
                const uint32_t value = node.attackerToPlay
                    ? next.proof : next.disproof;
                if (value < bestValue) {
                    best = edge;
                    bestValue = value;
                }
            }
            if (node.attackerToPlay && node.pendingQuiet > 0
                && QUIET_PROOF <= bestValue) {
                widen = true;
                break;
            }
            const PnEdge edge = node.edges[best];
            make(edge);
            current = edge.child;
            path.push_back(current);
            pathEdges.push_back(edge);
        }
        add_children(current, widen);
        for (size_t depth = path.size(); depth-- > 0;) {
            update(path[depth]);
            if (depth > 0) {
                unmake(pathEdges[depth - 1]);
            }
        }
    }
    result.nodes = nodes.size();
    result.status = nodes[0].proof == 0 ? Status::PROVEN
        : nodes[0].disproof == 0 ? Status::NO_WIN : Status::UNKNOWN;
    if (result.status != Status::PROVEN) {
        return result;
    }

    // Attacker actions to mate from a proven node: the attackers take their
    // quickest proven action, the defenders their longest defence.
    constexpr int IN_PROGRESS = -2;
    std::vector<int> length(nodes.size(), -1);
    std::function<int(uint32_t)> measure = [&](uint32_t index) {
        if (length[index] == IN_PROGRESS) {
            return std::numeric_limits<int>::max() / 2;
        }
        if (length[index] >= 0) {
            return length[index];
        }
        length[index] = IN_PROGRESS;
        const PnNode& node = nodes[index];
        int moves = 0;
        if (node.expanded && !node.edges.empty()) {
            moves = node.attackerToPlay ? std::numeric_limits<int>::max() / 2 : 0;
            for (const PnEdge& edge : node.edges) {
                const uint32_t child = edge.child;
                if (nodes[child].proof != 0) {
                    continue;
                }
                const int childMoves = measure(child)
                    + (node.attackerToPlay ? 1 : 0);
                moves = node.attackerToPlay ? std::min(moves, childMoves)
                                            : std::max(moves, childMoves);
            }
        }
        return length[index] = moves;
    };
    result.attackerMoves = measure(0);

    // Walk the proof again down the attackers' quickest actions: every node
    // on it must be open under this line's own history, each chosen attacker
    // action legal, every legal defence an edge of its defender node and
    // proven, and every leaf mate.
    bool verified = result.attackerMoves <= maxAttackerMoves;
    const auto quickest = [&](const PnNode& node) {
        const PnEdge* chosen = nullptr;
        for (const PnEdge& edge : node.edges) {
            if (nodes[edge.child].proof != 0) {
                continue;
            }
            if (!chosen || (node.attackerToPlay
                    ? measure(edge.child) < measure(chosen->child)
                    : measure(edge.child) > measure(chosen->child))) {
                chosen = &edge;
            }
        }
        return chosen;
    };
    std::function<void(uint32_t, int)> verify = [&](uint32_t index, int depth) {
        if (!verified) {
            return;
        }
        const PnNode& node = nodes[index];
        const JointMateStatus status = terminal(node.time);
        if (!node.expanded) {
            verified = status == JointMateStatus::PROVEN;
            return;
        }
        if (status != JointMateStatus::UNKNOWN || depth > 2 * maxAttackerMoves) {
            verified = false;
            return;
        }
        const vector<JointActionCandidate> actions =
            legal_joint_actions(board, team_of(node), node.time);
        const auto listed = [&](const PnEdge& edge) {
            return std::any_of(
                actions.begin(), actions.end(),
                [&](const JointActionCandidate& action) {
                    return action.moveA == edge.moveA
                        && action.moveB == edge.moveB;
                });
        };
        if (node.attackerToPlay) {
            const PnEdge* chosen = quickest(node);
            if (!chosen || !listed(*chosen)) {
                verified = false;
                return;
            }
            const PnEdge edge = *chosen;
            make(edge);
            verify(edge.child, depth + 1);
            unmake(edge);
            return;
        }
        if (actions.size() != node.edges.size() || node.pendingQuiet != 0) {
            verified = false;
            return;
        }
        for (const PnEdge& edge : std::vector<PnEdge>(node.edges)) {
            if (!listed(edge) || nodes[edge.child].proof != 0) {
                verified = false;
                return;
            }
            make(edge);
            verify(edge.child, depth + 1);
            unmake(edge);
            if (!verified) {
                return;
            }
        }
    };
    verify(0, 0);
    result.verified = verified;

    uint32_t index = 0;
    for (int steps = 0; nodes[index].expanded
             && steps <= 2 * std::max(1, maxAttackerMoves); ++steps) {
        const PnEdge* next = quickest(nodes[index]);
        if (!next) {
            break;
        }
        result.line.push_back(MateProofPly{next->moveA, next->moveB});
        index = next->child;
    }
    result.pv = format_mate_proof_pv(board, result.line, teamSide, time);
    return result;
}

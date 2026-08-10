#include <gtest/gtest.h>
#include <mutex>
#include <thread>
#include "../src/board.h"
#include "../src/constants.h"
#include "../src/joint_action.h"
#include "../src/node.h"
#include "../src/planes.h"
#include "../src/search_params.h"
#include "../src/searchthread.h"
#include "../src/transposition_table.h"
#include "../src/utils.h"
#include "Fairy-Stockfish/src/position.h"
#include "Fairy-Stockfish/src/types.h"
#include "Fairy-Stockfish/src/bitboard.h"
#include "Fairy-Stockfish/src/piece.h"
#include "Fairy-Stockfish/src/thread.h"

// Fixture for engine initialization
class EngineTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        Stockfish::pieceMap.init();
        Stockfish::variants.init();
        Stockfish::Bitboards::init();
        Stockfish::Position::init();
        Stockfish::Threads.set(1);
        init_policy_index();
    }
};

namespace {

Stockfish::Move find_move(Board& board, int boardNum, const std::string& uci) {
    for (Stockfish::Move move : board.legal_moves(boardNum)) {
        if (board.uci_move(boardNum, move) == uci) {
            return move;
        }
    }
    return Stockfish::MOVE_NONE;
}

float plane_value(const std::vector<float>& planes, int channel, int square = 0) {
    return planes[channel * BOARD_HEIGHT * BOARD_WIDTH + square];
}

uint64_t recomputed_position_key(Board& board, int boardNum) {
    Board reconstructed;
    reconstructed.set_fen(boardNum, board.fen(boardNum));
    return reconstructed.pos[boardNum]->key();
}

}

TEST_F(EngineTest, NewInputRepresentationStartsWithEmptyHistoryPlanes) {
    EXPECT_EQ(NB_INPUT_CHANNELS, 74);
    EXPECT_EQ(NB_INPUT_CHANNELS_PER_BOARD, 37);

    Board board;
    std::vector<float> planes(NB_INPUT_VALUES());
    board_to_planes(board, planes.data(), Stockfish::WHITE, false);

    for (int boardOffset : {0, NB_INPUT_CHANNELS_PER_BOARD}) {
        for (int channel = 32; channel <= 36; ++channel) {
            for (int square = 0; square < 64; ++square) {
                EXPECT_FLOAT_EQ(plane_value(planes, boardOffset + channel, square), 0.0f);
            }
        }
    }
}

TEST_F(EngineTest, NewInputRepresentationEncodesOrientedMoveAndHalfmoveClock) {
    Board board;
    Stockfish::Move move = find_move(board, BOARD_A, "g1f3");
    ASSERT_NE(move, Stockfish::MOVE_NONE);
    board.push_move(BOARD_A, move);

    std::vector<float> planes(NB_INPUT_VALUES());
    board_to_planes(board, planes.data(), Stockfish::WHITE, false);
    EXPECT_FLOAT_EQ(plane_value(planes, 32, Stockfish::SQ_G1), 1.0f);
    EXPECT_FLOAT_EQ(plane_value(planes, 33, Stockfish::SQ_F3), 1.0f);
    EXPECT_FLOAT_EQ(plane_value(planes, 34), 1.0f / 50.0f);

    Board copiedBoard(board);
    board_to_planes(copiedBoard, planes.data(), Stockfish::WHITE, false);
    EXPECT_FLOAT_EQ(plane_value(planes, 32, Stockfish::SQ_G1), 1.0f);
    EXPECT_FLOAT_EQ(plane_value(planes, 33, Stockfish::SQ_F3), 1.0f);

    board_to_planes(board, planes.data(), Stockfish::BLACK, false);
    EXPECT_FLOAT_EQ(plane_value(planes, 32, Stockfish::SQ_G8), 1.0f);
    EXPECT_FLOAT_EQ(plane_value(planes, 33, Stockfish::SQ_F6), 1.0f);
}

TEST_F(EngineTest, SettingCurrentFenClearsSearchHistory) {
    Board board;
    Stockfish::Move move = find_move(board, BOARD_A, "g1f3");
    ASSERT_NE(move, Stockfish::MOVE_NONE);
    board.push_move(BOARD_A, move);

    const std::string fenA = board.fen(BOARD_A);
    const std::string fenB = board.fen(BOARD_B);
    ASSERT_NE(board.last_move(BOARD_A), Stockfish::MOVE_NONE);
    ASSERT_GT(board.positionHistory[BOARD_A].size(), 1U);

    board.set(fenA + "|" + fenB);

    EXPECT_EQ(board.last_move(BOARD_A), Stockfish::MOVE_NONE);
    EXPECT_EQ(board.positionHistory[BOARD_A].size(), 1U);
    EXPECT_EQ(board.positionHistory[BOARD_B].size(), 1U);
}

TEST_F(EngineTest, NewInputRepresentationEncodesRepetitionContext) {
    Board board;
    for (const std::string& uci : {"g1f3", "g8f6", "f3g1", "f6g8"}) {
        Stockfish::Move move = find_move(board, BOARD_A, uci);
        ASSERT_NE(move, Stockfish::MOVE_NONE) << uci;
        board.push_move(BOARD_A, move);
    }

    std::vector<float> planes(NB_INPUT_VALUES());
    board_to_planes(board, planes.data(), Stockfish::WHITE, false);
    EXPECT_FLOAT_EQ(plane_value(planes, 35), 1.0f);
    EXPECT_FLOAT_EQ(plane_value(planes, 36), 0.0f);
}

TEST(JointActionTest, DoubleSitRequiresTimeAdvantageAndAnIdleBoard) {
    EXPECT_TRUE(is_double_sit_legal(true, true, false));
    EXPECT_TRUE(is_double_sit_legal(true, false, true));
    EXPECT_FALSE(is_double_sit_legal(false, true, false));
    EXPECT_FALSE(is_double_sit_legal(true, true, true));
    EXPECT_FALSE(is_double_sit_legal(true, false, false));

    JointActionCandidate disadvantaged(
        Stockfish::MOVE_NONE, 0.5f, 0,
        Stockfish::MOVE_NONE, 0.5f, 0,
        true, false, false);
    JointActionCandidate advantaged(
        Stockfish::MOVE_NONE, 0.5f, 0,
        Stockfish::MOVE_NONE, 0.5f, 0,
        true, false, true);
    JointActionCandidate bothBoardsOnTurn(
        Stockfish::MOVE_NONE, 0.5f, 0,
        Stockfish::MOVE_NONE, 0.5f, 0,
        true, true, true);

    EXPECT_LT(disadvantaged.jointPrior, 0.0f);
    EXPECT_FLOAT_EQ(advantaged.jointPrior, 0.25f);
    EXPECT_LT(bothBoardsOnTurn.jointPrior, 0.0f);
}

TEST(JointActionTest, JointCandidatesFollowPriorOrdering) {
    Stockfish::Move quietA = static_cast<Stockfish::Move>(1);
    Stockfish::Move checkingA = static_cast<Stockfish::Move>(2);
    Stockfish::Move bestB = static_cast<Stockfish::Move>(3);
    Stockfish::Move secondB = static_cast<Stockfish::Move>(4);
    JointCandidateGenerator generator;
    generator.initialize(
        {quietA, checkingA}, {bestB, secondB},
        {0.9f, 0.01f}, {0.8f, 0.2f},
        false, true, true);

    JointActionCandidate first = generator.getNext();
    EXPECT_EQ(first.moveA, quietA);
    EXPECT_EQ(first.moveB, bestB);
    EXPECT_FLOAT_EQ(first.jointPrior, 0.72f);

    JointActionCandidate second = generator.getNext();
    EXPECT_EQ(second.moveA, quietA);
    EXPECT_EQ(second.moveB, secondB);
    EXPECT_FLOAT_EQ(second.jointPrior, 0.18f);
}

TEST(JointActionTest, LowerPolicyChecksDoNotLeapfrogQuietMoves) {
    Stockfish::Move quietA = static_cast<Stockfish::Move>(1);
    Stockfish::Move firstCheckingA = static_cast<Stockfish::Move>(2);
    Stockfish::Move secondCheckingA = static_cast<Stockfish::Move>(3);
    JointCandidateGenerator generator;
    generator.initialize(
        {quietA, firstCheckingA, secondCheckingA}, {Stockfish::MOVE_NONE},
        {0.9f, 0.02f, 0.01f}, {1.0f},
        false, true, false);

    EXPECT_EQ(generator.getNext().moveA, quietA);
    EXPECT_EQ(generator.getNext().moveA, firstCheckingA);
    EXPECT_EQ(generator.getNext().moveA, secondCheckingA);
}

TEST(NodeTest, DoubleSitPassesTurnToOtherTeam) {
    Node node(Stockfish::BLACK);
    std::vector<Stockfish::Move> actions = {Stockfish::MOVE_NONE};
    std::vector<float> priors = {1.0f};

    ASSERT_TRUE(node.try_init_and_expand(
        actions, actions, priors, priors, true, true, false,
        SearchParams::RuntimeConfig{}));

    ASSERT_EQ(node.get_children().size(), 1U);
    EXPECT_EQ(node.get_children().front()->get_team_to_play(), Stockfish::WHITE);
}

TEST_F(EngineTest, DoubleSitLeavesBoardPositionUnchanged) {
    Board board;
    const std::string fenA = board.fen(BOARD_A);
    const std::string fenB = board.fen(BOARD_B);
    const uint64_t hash = board.hash_key(true);

    board.make_moves(Stockfish::MOVE_NONE, Stockfish::MOVE_NONE);

    EXPECT_EQ(board.fen(BOARD_A), fenA);
    EXPECT_EQ(board.fen(BOARD_B), fenB);
    EXPECT_EQ(board.hash_key(true), hash);
}

TEST_F(EngineTest, SearchMakeMovesUpdatesAndRestoresHistoryPlanes) {
    Board board;
    const Stockfish::Move move = find_move(board, BOARD_A, "g1f3");
    ASSERT_NE(move, Stockfish::MOVE_NONE);

    board.make_moves(move, Stockfish::MOVE_NONE);
    EXPECT_EQ(board.last_move(BOARD_A), move);

    std::vector<float> planes(NB_INPUT_VALUES());
    board_to_planes(board, planes.data(), Stockfish::BLACK, false);
    EXPECT_FLOAT_EQ(plane_value(planes, 32, Stockfish::SQ_G8), 1.0f);
    EXPECT_FLOAT_EQ(plane_value(planes, 33, Stockfish::SQ_F6), 1.0f);

    board.unmake_moves(move, Stockfish::MOVE_NONE);
    EXPECT_EQ(board.last_move(BOARD_A), Stockfish::MOVE_NONE);
    board_to_planes(board, planes.data(), Stockfish::WHITE, false);
    for (int square = 0; square < 64; ++square) {
        EXPECT_FLOAT_EQ(plane_value(planes, 32, square), 0.0f);
        EXPECT_FLOAT_EQ(plane_value(planes, 33, square), 0.0f);
    }
}

TEST_F(EngineTest, JointCaptureDropRoundTripPreservesPocketKeys) {
    Board board;
    board.set_fen(BOARD_A, "4k3/8/8/8/8/8/p7/R3K3 w - - 0 1");
    board.set_fen(BOARD_B, "4k3/8/8/8/8/8/8/4K3[p] b - - 0 1");

    const Stockfish::Move capture = find_move(board, BOARD_A, "a1a2");
    const Stockfish::Move drop = find_move(board, BOARD_B, "P@a7");
    ASSERT_NE(capture, Stockfish::MOVE_NONE);
    ASSERT_NE(drop, Stockfish::MOVE_NONE);

    const std::string initialFenA = board.fen(BOARD_A);
    const std::string initialFenB = board.fen(BOARD_B);
    EXPECT_EQ(board.pos[BOARD_A]->key(), recomputed_position_key(board, BOARD_A));
    EXPECT_EQ(board.pos[BOARD_B]->key(), recomputed_position_key(board, BOARD_B));

    board.make_moves(capture, drop);
    EXPECT_EQ(board.pos[BOARD_A]->key(), recomputed_position_key(board, BOARD_A));
    EXPECT_EQ(board.pos[BOARD_B]->key(), recomputed_position_key(board, BOARD_B));

    board.unmake_moves(capture, drop);
    EXPECT_EQ(board.fen(BOARD_A), initialFenA);
    EXPECT_EQ(board.fen(BOARD_B), initialFenB);
    EXPECT_EQ(board.pos[BOARD_A]->key(), recomputed_position_key(board, BOARD_A));
    EXPECT_EQ(board.pos[BOARD_B]->key(), recomputed_position_key(board, BOARD_B));
}

TEST_F(EngineTest, RejectsDropFromEmptyPocketBeforeMutation) {
    Board board;
    board.set_fen(BOARD_A, "4k3/8/8/8/8/8/8/4K3[P] w - - 0 1");
    const Stockfish::Move drop = find_move(board, BOARD_A, "P@a3");
    ASSERT_NE(drop, Stockfish::MOVE_NONE);

    board.remove_from_hand(
        BOARD_A, Stockfish::make_piece(Stockfish::WHITE, Stockfish::PAWN));
    const std::string fenA = board.fen(BOARD_A);
    const std::string fenB = board.fen(BOARD_B);

    EXPECT_THROW(
        board.make_moves(drop, Stockfish::MOVE_NONE),
        std::logic_error);
    EXPECT_EQ(board.fen(BOARD_A), fenA);
    EXPECT_EQ(board.fen(BOARD_B), fenB);
    EXPECT_EQ(board.count_in_hand(BOARD_A, Stockfish::WHITE, Stockfish::PAWN), 0);
}

TEST_F(EngineTest, InterleavedGameReplayPreservesPocketKeys) {
    Board board;
    const std::vector<std::string> moves = {
        "1g1h3", "1f7f6", "2a2a4", "1e2e4", "2b8c6", "1e7e6",
        "2a4a5", "1d2d4", "2e7e5", "1b8c6", "2e2e3", "1b1c3",
        "2g8f6", "1f8b4", "2a5a6", "1h3f4", "2d7d5", "1g8e7",
        "2a6b7", "1f4h5", "2c8b7", "1e8g8", "2g1f3", "1P@h6",
        "2f8d6", "1g7h6", "2b1c3", "1c1h6", "2e5e4", "1e7g6",
        "2f3g5", "2e8g8", "2P@f4", "2h7h6", "2g5h3", "2P@g4",
        "2h3g1", "2d5d4", "2c3b5", "2d4e3", "2d2e3", "1P@g7",
        "2d6b4", "1f8f7", "1f1d3", "1P@a3", "1e1g1", "1a3b2",
        "2c1d2", "1e4e5", "2b4d2", "1b2a1q",
    };

    for (const std::string& token : moves) {
        const int boardNum = token[0] - '1';
        const std::string uci = token.substr(1);
        const Stockfish::Move move = find_move(board, boardNum, uci);
        ASSERT_NE(move, Stockfish::MOVE_NONE) << token;
        board.push_move(boardNum, move);
        EXPECT_EQ(board.pos[BOARD_A]->key(), recomputed_position_key(board, BOARD_A))
            << token << " changed board A key incorrectly";
        EXPECT_EQ(board.pos[BOARD_B]->key(), recomputed_position_key(board, BOARD_B))
            << token << " changed board B key incorrectly";
    }
}

TEST_F(EngineTest, DoubleSitBackupChangesValuePerspective) {
    Node parent(Stockfish::BLACK);
    std::vector<Stockfish::Move> actions = {Stockfish::MOVE_NONE};
    std::vector<float> priors = {1.0f};
    ASSERT_TRUE(parent.try_init_and_expand(
        actions, actions, priors, priors, true, true, false,
        SearchParams::RuntimeConfig{}));

    auto child = parent.get_children().front();
    JointActionCandidate sit = parent.get_joint_action(0);
    std::vector<TrajectoryEntry> trajectory = {
        TrajectoryEntry(&parent, JointActionCandidate(), 0),
        TrajectoryEntry(child.get(), sit, -1),
    };

    Board board;
    SearchThread searchThread;
    searchThread.backup(trajectory, board, 0.5f);

    EXPECT_FLOAT_EQ(parent.get_child_q(0), -0.5f);
}

TEST_F(EngineTest, CommonBackupPropagatesProvenLeafState) {
    Node parent(Stockfish::WHITE);
    std::vector<Stockfish::Move> actions = {Stockfish::MOVE_NONE};
    std::vector<float> priors = {1.0f};
    ASSERT_TRUE(parent.try_init_and_expand(
        actions, actions, priors, priors, true, true, false,
        SearchParams::RuntimeConfig{}));

    auto child = parent.get_children().front();
    child->mark_as_loss(3);
    parent.apply_virtual_loss(0);

    std::vector<TrajectoryEntry> trajectory = {
        TrajectoryEntry(&parent, JointActionCandidate(), 0),
        TrajectoryEntry(child.get(), JointActionCandidate(), -1),
    };
    Board board;
    SearchThread searchThread;
    searchThread.set_root_node(&parent);
    searchThread.backup(trajectory, board, 0.25f);

    EXPECT_EQ(parent.get_node_type(), NodeType::WIN);
    EXPECT_EQ(parent.get_end_in_ply(), 4);
    EXPECT_FLOAT_EQ(parent.get_child_q(0), 1.0f);
}

TEST_F(EngineTest, SelectionStopsAtSolvedExpandedNode) {
    Node root(Stockfish::WHITE);
    std::vector<Stockfish::Move> actions = {Stockfish::MOVE_NONE};
    std::vector<float> priors = {1.0f};
    ASSERT_TRUE(root.try_init_and_expand(
        actions, actions, priors, priors, true, true, false,
        SearchParams::RuntimeConfig{}));
    root.mark_as_win(4);

    Board board;
    SearchThread searchThread;
    searchThread.set_root_node(&root);

    EXPECT_EQ(searchThread.select_and_expand(board, true), &root);
    EXPECT_EQ(root.get_child_visits().front(), 0);
}

TEST_F(EngineTest, CheckmatePrecedesFiftyMoveDraw) {
    Board board;
    board.set_fen(
        BOARD_A,
        "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 100 4");
    board.set_fen(BOARD_B, "4k3/8/8/8/8/8/8/4K3 w - - 0 1");

    ASSERT_TRUE(board.is_draw());
    ASSERT_TRUE(board.is_checkmate(Stockfish::BLACK, false));
    EXPECT_EQ(classify_terminal_position(
                  board, Stockfish::BLACK, Stockfish::BLACK, false, 0),
              TerminalOutcome::LOSS);
}

TEST_F(EngineTest, ClassifiesUnavoidableWaitingBoardMateAsLoss) {
    Board board;
    board.set(
        "3q1r1k/1p4b1/p1r2p1p/3p1b1n/1npP1pB1/N1N1Q2P/PPP2PP1/1R3KR1[BPp] w - - 0 1|"
        "5r1k/1p2q1b1/p1r2p1p/3p1b1n/1npP1pB1/N1N4P/PPPQ1PP1/1R3KR1[BPp] w - - 0 1");

    for (const auto& [boardNum, moveUci] :
         std::vector<std::pair<int, std::string>>{{BOARD_B, "g4h5"}, {BOARD_A, "g4h5"}}) {
        std::string uci = moveUci;
        Stockfish::Move move = Stockfish::UCI::to_move(*board.pos[boardNum], uci);
        ASSERT_NE(move, Stockfish::MOVE_NONE) << moveUci;
        board.push_move(boardNum, move);
    }

    const std::string fenA = board.fen(BOARD_A);
    const std::string fenB = board.fen(BOARD_B);
    const uint64_t hash = board.hash_key(false);
    int endInPly = 0;
    EXPECT_EQ(classify_terminal_position(
                  board, Stockfish::BLACK, Stockfish::BLACK, false, 2, &endInPly),
              TerminalOutcome::LOSS);
    EXPECT_EQ(endInPly, 3);
    EXPECT_EQ(board.fen(BOARD_A), fenA);
    EXPECT_EQ(board.fen(BOARD_B), fenB);
    EXPECT_EQ(board.hash_key(false), hash);

    endInPly = 0;
    EXPECT_EQ(classify_terminal_position(
                  board, Stockfish::BLACK, Stockfish::BLACK, true, 2, &endInPly),
              TerminalOutcome::LOSS);
    EXPECT_EQ(endInPly, 3);

    EXPECT_EQ(classify_terminal_position(
                  board, Stockfish::BLACK, Stockfish::BLACK, false, 0),
              TerminalOutcome::NONE);
}

TEST_F(EngineTest, BlockedWaitingBoardMateIsNotClassifiedAsLoss) {
    Board board;
    board.set(
        "3q1r1k/1p4b1/p1r2p1p/3p1b1n/1npP1pB1/N1N1Q2P/PPP2PP1/1R3KR1[BPp] w - - 0 1|"
        "5r1k/1p2q1b1/p1r2p1p/3p1b1n/1npP1pB1/N1N4P/PPPQ1PP1/1R3KR1[BPp] w - - 0 1");

    for (const auto& [boardNum, moveUci] :
         std::vector<std::pair<int, std::string>>{{BOARD_B, "B@h2"}, {BOARD_A, "g4h5"}}) {
        std::string uci = moveUci;
        Stockfish::Move move = Stockfish::UCI::to_move(*board.pos[boardNum], uci);
        ASSERT_NE(move, Stockfish::MOVE_NONE) << moveUci;
        board.push_move(boardNum, move);
    }

    EXPECT_EQ(classify_terminal_position(
                  board, Stockfish::BLACK, Stockfish::BLACK, false, 2),
              TerminalOutcome::NONE);
}

TEST_F(EngineTest, WaitingBoardMateRequiresSplitTurns) {
    Board board;
    board.set(
        "3q1r1k/1p4b1/p1r2p1p/3p1b1n/1npP1pB1/N1N1Q2P/PPP2PP1/1R3KR1[BPp] w - - 0 1|"
        "5r1k/1p2q1b1/p1r2p1p/3p1b1n/1npP1pB1/N1N4P/PPPQ1PP1/1R3KR1[BPp] w - - 0 1");

    std::string uci = "g4h5";
    Stockfish::Move move = Stockfish::UCI::to_move(*board.pos[BOARD_A], uci);
    ASSERT_NE(move, Stockfish::MOVE_NONE);
    board.push_move(BOARD_A, move);

    ASSERT_EQ(board.side_to_move(BOARD_A), Stockfish::BLACK);
    ASSERT_EQ(board.side_to_move(BOARD_B), Stockfish::WHITE);
    EXPECT_EQ(classify_terminal_position(
                  board, Stockfish::BLACK, Stockfish::BLACK, false, 1),
              TerminalOutcome::NONE);
}

TEST_F(EngineTest, CurrentDrawPrecedesFutureWaitingBoardMate) {
    Board board;
    board.set(
        "3q1r1k/1p4b1/p1r2p1p/3p1b1n/1npP1pB1/N1N1Q2P/PPP2PP1/1R3KR1[BPp] w - - 0 1|"
        "5r1k/1p2q1b1/p1r2p1p/3p1b1n/1npP1pB1/N1N4P/PPPQ1PP1/1R3KR1[BPp] w - - 0 1");

    for (const auto& [boardNum, moveUci] :
         std::vector<std::pair<int, std::string>>{{BOARD_B, "g4h5"}, {BOARD_A, "g4h5"}}) {
        std::string uci = moveUci;
        Stockfish::Move move = Stockfish::UCI::to_move(*board.pos[boardNum], uci);
        ASSERT_NE(move, Stockfish::MOVE_NONE) << moveUci;
        board.push_move(boardNum, move);
    }

    board.record_position(BOARD_A);
    ASSERT_TRUE(board.is_draw(2));
    EXPECT_EQ(classify_terminal_position(
                  board, Stockfish::BLACK, Stockfish::BLACK, false, 2),
              TerminalOutcome::DRAW);
}

TEST_F(EngineTest, CancellingCollisionDoesNotCreateAVisit) {
    Node parent(Stockfish::WHITE);
    std::vector<Stockfish::Move> actions = {Stockfish::MOVE_NONE};
    std::vector<float> priors = {1.0f};
    ASSERT_TRUE(parent.try_init_and_expand(
        actions, actions, priors, priors, true, true, false,
        SearchParams::RuntimeConfig{}));

    auto child = parent.get_children().front();
    std::vector<TrajectoryEntry> trajectory = {
        TrajectoryEntry(&parent, JointActionCandidate(), 0),
        TrajectoryEntry(child.get(), parent.get_joint_action(0), -1),
    };

    parent.apply_virtual_loss(0);
    Board board;
    SearchThread searchThread;
    searchThread.cancel_virtual_losses(trajectory);

    EXPECT_EQ(parent.get_child_visits()[0], 0);
    EXPECT_EQ(parent.get_visits(), 0);
    EXPECT_FLOAT_EQ(parent.get_child_q(0), SearchParams::Q_INIT);
}

TEST(PolicyTest, NormalizesExtremeAndNonFiniteLogits) {
    auto probabilities = normalize_logits({1000.0f, 999.0f, -1000.0f});
    ASSERT_EQ(probabilities.size(), 3);
    EXPECT_TRUE(std::all_of(probabilities.begin(), probabilities.end(), [](float value) {
        return std::isfinite(value);
    }));
    EXPECT_NEAR(std::accumulate(probabilities.begin(), probabilities.end(), 0.0f), 1.0f, 1e-6f);
    EXPECT_GT(probabilities[0], probabilities[1]);

    auto fallback = normalize_logits({
        std::numeric_limits<float>::quiet_NaN(),
        -std::numeric_limits<float>::infinity()});
    EXPECT_FLOAT_EQ(fallback[0], 0.5f);
    EXPECT_FLOAT_EQ(fallback[1], 0.5f);
}

TEST(PolicyTest, PassFloorPreservesNonPassRatios) {
    std::vector<float> probabilities = {0.01f, 0.09f, 0.90f};

    apply_probability_floor(probabilities, 0, 0.10f);

    EXPECT_FLOAT_EQ(probabilities[0], 0.10f);
    EXPECT_NEAR(probabilities[1] / probabilities[2], 0.1f, 1e-6f);
    EXPECT_NEAR(std::accumulate(probabilities.begin(), probabilities.end(), 0.0f),
                1.0f, 1e-6f);
}

TEST(PolicyTest, SelectsPassFloorByBughouseContext) {
    SearchParams::RuntimeConfig config;
    EXPECT_FLOAT_EQ(get_pass_prior_floor(true, true, false, config), 0.0f);
    EXPECT_FLOAT_EQ(get_pass_prior_floor(false, true, true, config), 0.0f);

    config.waitPassPriorFloor = 0.12f;
    config.coordinationPassPriorFloor = 0.04f;

    EXPECT_FLOAT_EQ(get_pass_prior_floor(true, true, false, config), 0.12f);
    EXPECT_FLOAT_EQ(get_pass_prior_floor(false, true, true, config), 0.04f);
    EXPECT_FLOAT_EQ(get_pass_prior_floor(false, true, false, config), 0.0f);
}

TEST_F(EngineTest, AppliesPassFloorToNetworkPolicy) {
    Board board;
    auto actions = board.legal_moves(BOARD_A);
    actions.push_back(Stockfish::MOVE_NONE);
    std::vector<float> policyOutput(NB_POLICY_VALUES(), 0.0f);
    policyOutput[POLICY_INDEX.at("pass")] = -20.0f;

    auto probabilities = get_normalized_probability(
        policyOutput.data(), actions, BOARD_A, board, 0.10f);

    ASSERT_EQ(probabilities.size(), actions.size());
    EXPECT_FLOAT_EQ(probabilities.back(), 0.10f);
    EXPECT_NEAR(std::accumulate(probabilities.begin(), probabilities.end(), 0.0f),
                1.0f, 1e-6f);
}

TEST_F(EngineTest, LowPriorCheckingMoveDoesNotBypassPolicyOrdering) {
    Board board;
    board.set_fen(
        BOARD_A,
        "7k/ppp2rpP/8/4p1qP/3nP1N1/2NP1P2/PPP2K2/3R3R[Qpp] w - - 0 24");

    auto actions = board.legal_moves(BOARD_A);
    actions.push_back(Stockfish::MOVE_NONE);
    std::vector<float> policyOutput(NB_POLICY_VALUES(), -10.0f);
    policyOutput[POLICY_INDEX.at("h1h3")] = 5.0f;
    policyOutput[POLICY_INDEX.at("Q@g8")] = -5.0f;

    auto probabilities = get_normalized_probability(
        policyOutput.data(), actions, BOARD_A, board);
    size_t quietIdx = actions.size();
    size_t checkingIdx = actions.size();
    for (size_t i = 0; i < actions.size(); ++i) {
        std::string move = board.uci_move(BOARD_A, actions[i]);
        if (move == "h1h3") quietIdx = i;
        if (move == "Q@g8") checkingIdx = i;
    }

    ASSERT_LT(quietIdx, actions.size());
    ASSERT_LT(checkingIdx, actions.size());
    ASSERT_TRUE(board.gives_check(BOARD_A, actions[checkingIdx]));
    EXPECT_LT(probabilities[checkingIdx], probabilities[quietIdx]);

    JointCandidateGenerator generator;
    generator.initialize(
        actions, {Stockfish::MOVE_NONE}, probabilities, {1.0f},
        false, true, false);
    JointActionCandidate first = generator.getNext();
    EXPECT_EQ(board.uci_move(BOARD_A, first.moveA), "h1h3");
    EXPECT_FLOAT_EQ(first.priorA, probabilities[quietIdx]);
}

TEST(SearchConfigTest, RuntimeValuesChangeSearchCalculations) {
    EXPECT_NE(SearchParams::get_cpuct(100.0f, 1.0f, 100.0f),
              SearchParams::get_cpuct(100.0f, 3.0f, 100.0f));
}

TEST(SearchConfigTest, DefaultsPreferObjectiveAndSolverProvenResults) {
    SearchParams::RuntimeConfig config;

    EXPECT_FLOAT_EQ(config.drawContempt, 0.0f);
    EXPECT_TRUE(SearchParams::ENABLE_MATE_EARLY_EXIT);
    EXPECT_FALSE(SearchParams::ENABLE_Q_EARLY_EXIT);
    EXPECT_FALSE(SearchParams::ENABLE_TIME_EXTENSION);
    EXPECT_FALSE(SearchParams::ENABLE_TREE_REUSE);
    EXPECT_FALSE(config.enableTranspositions);
    EXPECT_EQ(SearchParams::TT_MAX_SIZE, TranspositionTable::kDefaultMaxCapacity);
}

TEST(SearchConfigTest, ProgressiveWideningScheduleIsExplicit) {
    SearchParams::RuntimeConfig config;
    config.pwCoefficient = 1.0f;
    config.rootPwCoefficient = 4.0f;

    EXPECT_EQ(SearchParams::get_allowed_children(
                  1000, config.pwCoefficient, config.pwExponent), 8);
    EXPECT_EQ(SearchParams::get_allowed_children(
                  1000, config.rootPwCoefficient, config.pwExponent), 32);
    EXPECT_EQ(SearchParams::get_allowed_children(
                  10000, config.pwCoefficient, config.pwExponent), 16);
    EXPECT_EQ(SearchParams::get_allowed_children(
                  10000, config.rootPwCoefficient, config.pwExponent), 64);
}

TEST(NodeTest, ProgressiveWideningGatesJointActionExpansion) {
    SearchParams::RuntimeConfig config;
    config.pwCoefficient = 1.0f;
    Node node(Stockfish::WHITE);
    node.set_depth(1);
    std::vector<Stockfish::Move> actionsA = {
        Stockfish::Move(1), Stockfish::Move(2), Stockfish::Move(3)};
    std::vector<Stockfish::Move> actionsB = {Stockfish::MOVE_NONE};
    std::vector<float> priorsA = {0.9f, 0.09f, 0.01f};
    std::vector<float> priorsB = {1.0f};

    ASSERT_TRUE(node.try_init_and_expand(
        actionsA, actionsB, priorsA, priorsB, false, true, false,
        config));
    ASSERT_EQ(node.get_children().size(), 1U);
    EXPECT_FLOAT_EQ(node.get_child_q(0), SearchParams::Q_INIT);
    EXPECT_FLOAT_EQ(node.get_children().front()->Q(), 0.0f);
    ASSERT_TRUE(node.has_unexpanded_joint_actions());
    EXPECT_FALSE(node.should_expand_new_child(config));

    node.update(0, 1.0f);
    EXPECT_FALSE(node.should_expand_new_child(config));
    node.update(0, 1.0f);
    EXPECT_TRUE(node.should_expand_new_child(config));

    JointActionCandidate action;
    ASSERT_NE(node.expand_next_joint_child(
        nullptr, 0, action, config), nullptr);

    EXPECT_TRUE(node.has_unexpanded_joint_actions());
    EXPECT_FALSE(node.should_expand_new_child(config));
}

TEST(NodeTest, RootProgressiveWideningExploresMoreCandidates) {
    SearchParams::RuntimeConfig config;
    config.pwCoefficient = 1.0f;
    config.rootPwCoefficient = 4.0f;
    EXPECT_GT(SearchParams::get_allowed_children(
                  10000, config.rootPwCoefficient, config.pwExponent),
              SearchParams::get_allowed_children(
                  10000, config.pwCoefficient, config.pwExponent));
}

TEST(NodeTest, RootVisitsGeneratedChildBeforeWidening) {
    Node node(Stockfish::WHITE);
    std::vector<Stockfish::Move> actionsA = {
        Stockfish::Move(1), Stockfish::Move(2), Stockfish::Move(3)};
    std::vector<Stockfish::Move> actionsB = {Stockfish::MOVE_NONE};

    ASSERT_TRUE(node.try_init_and_expand(
        actionsA, actionsB, {0.8f, 0.15f, 0.05f}, {1.0f}, false, true, false,
        SearchParams::RuntimeConfig{}));
    ASSERT_EQ(node.get_children().size(), 1U);
    EXPECT_FALSE(node.should_expand_new_child(SearchParams::RuntimeConfig{}));

    node.update(0, 0.25f);

    EXPECT_TRUE(node.should_expand_new_child(SearchParams::RuntimeConfig{}));
}

TEST(NodeTest, InFlightVisitAllowsBatchToWiden) {
    Node node(Stockfish::WHITE);
    node.set_depth(1);
    std::vector<Stockfish::Move> actionsA = {
        Stockfish::Move(1), Stockfish::Move(2), Stockfish::Move(3)};

    ASSERT_TRUE(node.try_init_and_expand(
        actionsA, {Stockfish::MOVE_NONE}, {0.8f, 0.15f, 0.05f}, {1.0f},
        false, true, false, SearchParams::RuntimeConfig{}));
    node.update_terminal(0.0f);
    EXPECT_FALSE(node.should_expand_new_child(SearchParams::RuntimeConfig{}));

    node.apply_virtual_loss(0);
    EXPECT_TRUE(node.should_expand_new_child(SearchParams::RuntimeConfig{}));
    node.remove_virtual_loss(0);
}

TEST(NodeTest, AtomicVirtualLossDivertsNextSelection) {
    Node node(Stockfish::WHITE);
    std::vector<Stockfish::Move> actionsA = {
        Stockfish::Move(1), Stockfish::Move(2)};
    SearchParams::RuntimeConfig config;

    ASSERT_TRUE(node.try_init_and_expand(
        actionsA, {Stockfish::MOVE_NONE}, {0.5f, 0.5f}, {1.0f},
        false, true, false, config));
    node.update(0, 0.0f);

    JointActionCandidate action;
    ASSERT_NE(node.expand_next_joint_child(nullptr, 0, action, config), nullptr);

    auto [firstChild, firstIdx] = node.select_child_and_apply_virtual_loss(config);
    auto [secondChild, secondIdx] = node.select_child_and_apply_virtual_loss(config);
    ASSERT_NE(firstChild, nullptr);
    ASSERT_NE(secondChild, nullptr);
    EXPECT_EQ(firstIdx, 0);
    EXPECT_EQ(secondIdx, 1);

    node.remove_virtual_loss(firstIdx);
    node.remove_virtual_loss(secondIdx);
}

TEST(NodeTest, ConcurrentExpansionReturnsMatchingActionIndex) {
    Node node(Stockfish::WHITE);
    std::vector<Stockfish::Move> actionsA;
    std::vector<float> priorsA;
    for (int i = 1; i <= 64; ++i) {
        actionsA.push_back(Stockfish::Move(i));
        priorsA.push_back(static_cast<float>(65 - i));
    }
    std::vector<Stockfish::Move> actionsB = {Stockfish::MOVE_NONE};
    std::vector<float> priorsB = {1.0f};

    ASSERT_TRUE(node.try_init_and_expand(
        actionsA, actionsB, priorsA, priorsB, false, true, false,
        SearchParams::RuntimeConfig{}));

    std::mutex resultsMutex;
    std::vector<std::pair<int, Stockfish::Move>> results;
    std::vector<std::thread> workers;
    for (int i = 0; i < 8; ++i) {
        workers.emplace_back([&]() {
            while (node.has_unexpanded_joint_actions()) {
                JointActionCandidate action;
                int childIdx = -1;
                auto child = node.expand_next_joint_child(
                    nullptr, 0, action, SearchParams::RuntimeConfig{}, &childIdx);
                if (child) {
                    std::lock_guard<std::mutex> guard(resultsMutex);
                    results.emplace_back(childIdx, action.moveA);
                }
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }

    ASSERT_EQ(results.size(), actionsA.size() - 1);
    for (const auto& [childIdx, move] : results) {
        EXPECT_EQ(node.get_joint_action(childIdx).moveA, move);
    }
}

TEST(NodeTest, TranspositionEdgeUsesParentPerspectiveWithoutInheritedVisits) {
    Node parent(Stockfish::WHITE);
    std::vector<Stockfish::Move> actionsA = {
        static_cast<Stockfish::Move>(1), static_cast<Stockfish::Move>(2)};
    std::vector<Stockfish::Move> actionsB = {Stockfish::MOVE_NONE};
    ASSERT_TRUE(parent.try_init_and_expand(
        actionsA, actionsB, {0.75f, 0.25f}, {1.0f}, false, true, false,
        SearchParams::RuntimeConfig{}));

    parent.update_terminal(1.0f);
    parent.update_terminal(1.0f);

    auto existing = std::make_shared<Node>(Stockfish::BLACK);
    existing->set_value(0.75f);
    JointActionCandidate action;
    int childIdx = -1;
    ASSERT_NE(parent.expand_next_joint_child(
        existing, 123, action, SearchParams::RuntimeConfig{}, &childIdx), nullptr);

    EXPECT_FLOAT_EQ(parent.get_child_q(childIdx), -0.75f);
    EXPECT_EQ(parent.get_child_visits()[childIdx], 1);
}

TEST(NodeTest, EvaluationReservationIsExclusiveUntilReleased) {
    Node node(Stockfish::WHITE);

    EXPECT_TRUE(node.try_reserve_evaluation());
    EXPECT_FALSE(node.try_reserve_evaluation());

    node.release_evaluation_reservation();
    EXPECT_TRUE(node.try_reserve_evaluation());
    node.release_evaluation_reservation();
}

TEST(NodeTest, QAveragesOnlyRealVisits) {
    Node node(Stockfish::WHITE);

    node.update_terminal(0.8f);
    EXPECT_FLOAT_EQ(node.Q(), 0.8f);

    node.update_terminal(0.2f);
    EXPECT_FLOAT_EQ(node.Q(), 0.5f);
}

TEST(NodeTest, SolvedQIsExactBeforeBackup) {
    Node winning(Stockfish::WHITE);
    Node losing(Stockfish::WHITE);
    Node drawn(Stockfish::WHITE);

    winning.mark_as_win(1);
    losing.mark_as_loss(1);
    drawn.mark_as_draw(1);

    EXPECT_FLOAT_EQ(winning.Q(), 1.0f);
    EXPECT_FLOAT_EQ(losing.Q(), -1.0f);
    EXPECT_FLOAT_EQ(drawn.Q(), 0.0f);
}

TEST(MctsSolverTest, PropagatesMateAtArbitraryDepth) {
    Node parent(Stockfish::WHITE);
    std::vector<Stockfish::Move> actions = {Stockfish::MOVE_NONE};
    std::vector<float> priors = {1.0f};

    ASSERT_TRUE(parent.try_init_and_expand(
        actions, actions, priors, priors, true, true, false,
        SearchParams::RuntimeConfig{}));

    auto child = parent.get_children().front();
    child->mark_as_loss(7);
    parent.init_child_node_types();

    EXPECT_TRUE(parent.update_child_node_type(0, child->get_node_type()));
    EXPECT_EQ(parent.get_node_type(), NodeType::WIN);
    EXPECT_EQ(parent.get_end_in_ply(), 8);
}

TEST(MctsSolverTest, PropagatesDrawAfterAllMovesAreSolved) {
    Node parent(Stockfish::WHITE);
    std::vector<Stockfish::Move> actions = {Stockfish::MOVE_NONE};
    std::vector<float> priors = {1.0f};

    ASSERT_TRUE(parent.try_init_and_expand(
        actions, actions, priors, priors, true, true, false,
        SearchParams::RuntimeConfig{}));

    auto child = parent.get_children().front();
    child->mark_as_draw(1);
    parent.init_child_node_types();

    EXPECT_TRUE(parent.update_child_node_type(0, child->get_node_type()));
    EXPECT_EQ(parent.get_node_type(), NodeType::DRAW);
}

TEST(MctsSolverTest, AvoidsProvenLosingChildWhileDefenseRemains) {
    Node parent(Stockfish::WHITE);
    std::vector<Stockfish::Move> actionsA = {
        Stockfish::Move(1), Stockfish::Move(2)};
    std::vector<float> priorsA = {0.9f, 0.1f};
    SearchParams::RuntimeConfig config;

    ASSERT_TRUE(parent.try_init_and_expand(
        actionsA, {Stockfish::MOVE_NONE}, priorsA, {1.0f},
        false, true, false, config));
    for (int visit = 0; visit < 10; ++visit) {
        parent.update(0, 0.5f);
    }

    JointActionCandidate action;
    int defenseIdx = -1;
    ASSERT_NE(parent.expand_next_joint_child(
        nullptr, 0, action, config, &defenseIdx), nullptr);
    ASSERT_EQ(defenseIdx, 1);
    parent.update(defenseIdx, -0.5f);

    auto children = parent.get_children();
    children[0]->mark_as_win(3);
    parent.init_child_node_types();
    EXPECT_FALSE(parent.update_child_node_type(0, children[0]->get_node_type()));
    ASSERT_EQ(parent.get_node_type(), NodeType::UNSOLVED);

    EXPECT_EQ(parent.get_best_move_idx_with_q_weight(), defenseIdx);
    auto [selectedChild, selectedIdx] = parent.select_child_and_apply_virtual_loss(config);
    ASSERT_NE(selectedChild, nullptr);
    EXPECT_EQ(selectedIdx, defenseIdx);
    parent.remove_virtual_loss(selectedIdx);
}

TEST(MctsSolverTest, WidensImmediatelyWhenAllExpandedChildrenLose) {
    Node parent(Stockfish::WHITE);
    std::vector<Stockfish::Move> actionsA = {
        Stockfish::Move(1), Stockfish::Move(2)};
    SearchParams::RuntimeConfig config;

    ASSERT_TRUE(parent.try_init_and_expand(
        actionsA, {Stockfish::MOVE_NONE}, {0.9f, 0.1f}, {1.0f},
        false, true, false, config));
    ASSERT_TRUE(parent.has_unexpanded_joint_actions());

    auto firstChild = parent.get_children().front();
    firstChild->mark_as_win(3);
    parent.init_child_node_types();
    EXPECT_FALSE(parent.update_child_node_type(0, firstChild->get_node_type()));
    ASSERT_EQ(parent.get_node_type(), NodeType::UNSOLVED);

    EXPECT_TRUE(parent.should_expand_new_child(config));
}

TEST(TranspositionTableTest, InsertOrGetReturnsCanonicalNode) {
    TranspositionTable table;
    auto first = std::make_shared<Node>(Stockfish::WHITE, 42);
    auto duplicate = std::make_shared<Node>(Stockfish::WHITE, 42);

    EXPECT_EQ(table.insertOrGet(42, first), first);
    EXPECT_EQ(table.insertOrGet(42, duplicate), first);
    EXPECT_EQ(table.getHits(), 1);
}

TEST(SearchParamsTest, EarlyStoppingRequiresFactoredVisitLead) {
    EXPECT_TRUE(SearchParams::has_insurmountable_visit_lead(100.0f, 40.0f, 2.0f));
    EXPECT_FALSE(SearchParams::has_insurmountable_visit_lead(100.0f, 60.0f, 2.0f));
    EXPECT_FALSE(SearchParams::has_insurmountable_visit_lead(100.0f, 50.0f, 2.0f));
}

TEST_F(EngineTest, InitialMoves) {
    Board board;
    // Assuming Board() default ctor sets up start position
    // If not, we can explicitly set it:
    board.set_fen(BOARD_A, board.startingFen);
    
    std::vector<Stockfish::Move> movesA = board.legal_moves(BOARD_A);
    // Standard chess start position has 20 moves.
    EXPECT_EQ(movesA.size(), 20);

    // Test a specific move exists (e.g., e2e4)
    bool found_e2e4 = false;
    for (const auto& move : movesA) {
        if (board.uci_move(BOARD_A, move) == "e2e4") {
            found_e2e4 = true;
            break;
        }
    }
    EXPECT_TRUE(found_e2e4);
}

TEST_F(EngineTest, MakeMove) {
    Board board;
    board.set_fen(BOARD_A, board.startingFen);
    
    // Find e2e4
    Stockfish::Move e2e4 = Stockfish::Move::MOVE_NONE;
    auto moves = board.legal_moves(BOARD_A);
    for (const auto& m : moves) {
        if (board.uci_move(BOARD_A, m) == "e2e4") {
            e2e4 = m;
            break;
        }
    }
    ASSERT_NE(e2e4, Stockfish::Move::MOVE_NONE);

    // Apply move on board A
    board.push_move(BOARD_A, e2e4);
    
    std::string fen = board.fen(BOARD_A);
    // e4 should be occupied by a white pawn. Checking FEN substring roughly.
    // FEN after e2e4: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1
    // Note: FEN might vary slightly with en passant target.
    EXPECT_NE(fen.find("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"), std::string::npos);
}

TEST_F(EngineTest, BughouseDrop) {
    Board board;
    board.set_fen(BOARD_A, board.startingFen);

    // Manually add pawn to hand
    board.add_to_hand(BOARD_A, Stockfish::make_piece(Stockfish::WHITE, Stockfish::PAWN));
    
    // Verify pawn is in hand
    int pawn_count = board.count_in_hand(BOARD_A, Stockfish::WHITE, Stockfish::PAWN);
    EXPECT_EQ(pawn_count, 1);
    
    auto moves = board.legal_moves(BOARD_A);
    
    bool found_drop = false;
    Stockfish::Move drop_e4 = Stockfish::MOVE_NONE;

    for (const auto& m : moves) {
        std::string uci = board.uci_move(BOARD_A, m);
        // UCI format uses lowercase for pieces in drops
        if (uci == "P@e4") {
            found_drop = true;
            drop_e4 = m;
            break;
        }
    }
    
    EXPECT_TRUE(found_drop);
    ASSERT_NE(drop_e4, Stockfish::MOVE_NONE);
    
    board.push_move(BOARD_A, drop_e4);
    std::string fen = board.fen(BOARD_A);
    // After drop p@e4, the board should have a pawn on e4
    EXPECT_NE(fen.find("4P3"), std::string::npos) << "FEN: " << fen;
}

TEST_F(EngineTest, BughouseCaptureTransfer) {
    Board board;
    board.set_fen(BOARD_A, "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2");
    board.set_fen(BOARD_B, board.startingFen);
    
    // Find and make the capture move exd5 on board A
    auto moves = board.legal_moves(BOARD_A);
    Stockfish::Move capture_move = Stockfish::MOVE_NONE;
    
    for (const auto& m : moves) {
        std::string uci = board.uci_move(BOARD_A, m);
        if (uci == "e4d5") {
            capture_move = m;
            break;
        }
    }
    
    ASSERT_NE(capture_move, Stockfish::MOVE_NONE) << "Capture move e4d5 not found";
    
    // Before capture, board B should have no pawns in hand
    EXPECT_EQ(board.count_in_hand(BOARD_B, Stockfish::BLACK, Stockfish::PAWN), 0);
    EXPECT_EQ(board.count_in_hand(BOARD_B, Stockfish::WHITE, Stockfish::PAWN), 0);
    
    // Make the capture on board A (White captures Black's pawn)
    board.push_move(BOARD_A, capture_move);
    
    // After capture, the captured pawn should appear on the partner board for the opponent's team
    // White captured Black's pawn on board A, so Black gets a pawn on board B
    int black_pawn_in_hand = board.count_in_hand(BOARD_B, Stockfish::BLACK, Stockfish::PAWN);
    EXPECT_EQ(black_pawn_in_hand, 1) << "Captured pawn should transfer to partner board";
    
    // Verify that if it's Black's turn, they can drop the pawn
    // Since board B starts with White to move, we need to make a move first
    auto board_b_moves = board.legal_moves(BOARD_B);
    EXPECT_GT(board_b_moves.size(), 0);
    
    // Make a move on board B so Black can move
    board.push_move(BOARD_B, board_b_moves[0]);
    
    // Now check if Black can drop the pawn
    auto black_moves = board.legal_moves(BOARD_B);
    bool has_black_pawn_drop = false;
    for (const auto& m : black_moves) {
        std::string uci = board.uci_move(BOARD_B, m);
        if (uci.find("P@") == 0) {  // Black pawn drop
            has_black_pawn_drop = true;
            break;
        }
    }
    EXPECT_TRUE(has_black_pawn_drop) << "Black should be able to drop the captured pawn";
}

TEST_F(EngineTest, PerftDepth1) {
    Board board;
    board.set_fen(BOARD_A, board.startingFen);
    auto moves = board.legal_moves(BOARD_A);
    EXPECT_EQ(moves.size(), 20) << "Starting position should have 20 legal moves";
}

TEST_F(EngineTest, PerftDepth2) {
    Board board;
    board.set_fen(BOARD_A, board.startingFen);
    
    auto moves = board.legal_moves(BOARD_A);
    long long total_nodes = 0;
    
    for (const auto& move : moves) {
        board.push_move(BOARD_A, move);
        auto responses = board.legal_moves(BOARD_A);
        total_nodes += responses.size();
        board.pop_move(BOARD_A);
    }
    
    EXPECT_EQ(total_nodes, 400) << "Perft(2) from starting position should be 400";
}

TEST_F(EngineTest, PerftDepth3) {
    Board board;
    board.set_fen(BOARD_A, board.startingFen);
    
    std::function<long long(int)> perft = [&](int depth) -> long long {
        if (depth == 0) return 1;
        
        auto moves = board.legal_moves(BOARD_A);
        if (depth == 1) return moves.size();
        
        long long nodes = 0;
        for (const auto& move : moves) {
            board.push_move(BOARD_A, move);
            nodes += perft(depth - 1);
            board.pop_move(BOARD_A);
        }
        return nodes;
    };
    
    long long nodes = perft(3);
    EXPECT_EQ(nodes, 8902) << "Perft(3) from starting position should be 8902";
}

TEST_F(EngineTest, ZobristHashConsistency) {
    Board board;
    board.set_fen(BOARD_A, board.startingFen);
    
    uint64_t initial_hash = board.hash_key(BOARD_A);
    
    auto moves = board.legal_moves(BOARD_A);
    ASSERT_GT(moves.size(), 0);
    
    for (const auto& move : moves) {
        board.push_move(BOARD_A, move);
        uint64_t after_move_hash = board.hash_key(BOARD_A);
        
        EXPECT_NE(after_move_hash, initial_hash) << "Hash should change after move " << board.uci_move(BOARD_A, move);
        
        board.pop_move(BOARD_A);
        uint64_t after_unmake_hash = board.hash_key(BOARD_A);
        
        EXPECT_EQ(after_unmake_hash, initial_hash) << "Hash should restore after unmake of move " << board.uci_move(BOARD_A, move);
    }
}

TEST_F(EngineTest, CombinedHashUsesRule50AndTimeAdvantageNotGamePly) {
    Board early;
    Board late;
    early.set_fen(BOARD_A, "4k3/8/8/8/8/8/8/4K3 w - - 7 1");
    late.set_fen(BOARD_A, "4k3/8/8/8/8/8/8/4K3 w - - 7 900");

    EXPECT_EQ(early.hash_key(false), late.hash_key(false));
    EXPECT_NE(early.hash_key(false), early.hash_key(true));

    late.set_fen(BOARD_A, "4k3/8/8/8/8/8/8/4K3 w - - 8 900");
    EXPECT_NE(early.hash_key(false), late.hash_key(false));
}

TEST_F(EngineTest, CombinedHashIncludesRepetitionContext) {
    Board historical;
    auto findMove = [&historical](const std::string& uci) {
        for (Stockfish::Move move : historical.legal_moves(BOARD_A)) {
            if (historical.uci_move(BOARD_A, move) == uci) {
                return move;
            }
        }
        return Stockfish::MOVE_NONE;
    };

    for (const std::string& uci : {"g1f3", "b8c6", "f3g1", "c6b8"}) {
        Stockfish::Move move = findMove(uci);
        ASSERT_NE(move, Stockfish::MOVE_NONE);
        historical.push_move(BOARD_A, move);
    }

    Board fresh;
    fresh.set_fen(
        BOARD_A,
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 4 3");

    EXPECT_EQ(historical.board_only_key(BOARD_A), fresh.board_only_key(BOARD_A));
    EXPECT_EQ(historical.rule50_count(BOARD_A), fresh.rule50_count(BOARD_A));
    EXPECT_NE(historical.hash_key(false), fresh.hash_key(false));
}

TEST_F(EngineTest, RepetitionPrefixHashRoundTripsWithSearchMoves) {
    Board board;
    const uint64_t initialHash = board.hash_key(false);
    const size_t initialPrefixCount = board.positionHistoryPrefixes[BOARD_A].size();
    Stockfish::Move move = find_move(board, BOARD_A, "g1f3");
    ASSERT_NE(move, Stockfish::MOVE_NONE);

    board.make_moves(move, Stockfish::MOVE_NONE);
    EXPECT_EQ(board.positionHistoryPrefixes[BOARD_A].size(), initialPrefixCount + 1);
    EXPECT_NE(board.hash_key(false), initialHash);

    board.unmake_moves(move, Stockfish::MOVE_NONE);
    EXPECT_EQ(board.positionHistoryPrefixes[BOARD_A].size(), initialPrefixCount);
    EXPECT_EQ(board.hash_key(false), initialHash);
}

TEST_F(EngineTest, CombinedHashIncludesTransferredPocketPieces) {
    Board board;
    const uint64_t emptyPocketHash = board.hash_key(false);

    board.add_to_hand(
        BOARD_A, Stockfish::make_piece(Stockfish::BLACK, Stockfish::QUEEN));

    EXPECT_NE(board.hash_key(false), emptyPocketHash);
}

TEST_F(EngineTest, CapturedDroppedQueenTransfersToPartnerWithUpdatedHash) {
    Board board;
    board.set_fen(BOARD_A, "4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    board.set_fen(BOARD_B, "6k1/8/8/8/8/8/8/7K[Q] w - - 0 1");

    auto find_move = [&](int boardNum, const std::string& uci) {
        for (Stockfish::Move move : board.legal_moves(boardNum)) {
            if (board.uci_move(boardNum, move) == uci) {
                return move;
            }
        }
        return Stockfish::MOVE_NONE;
    };

    Stockfish::Move queenDrop = find_move(BOARD_B, "Q@h8");
    ASSERT_NE(queenDrop, Stockfish::MOVE_NONE);
    board.push_move(BOARD_B, queenDrop);

    const uint64_t beforeCaptureHash = board.hash_key(false);
    Stockfish::Move kingCapture = find_move(BOARD_B, "g8h8");
    ASSERT_NE(kingCapture, Stockfish::MOVE_NONE);
    board.push_move(BOARD_B, kingCapture);

    EXPECT_EQ(board.count_in_hand(BOARD_A, Stockfish::WHITE, Stockfish::QUEEN), 1);
    EXPECT_NE(board.hash_key(false), beforeCaptureHash);

    board.pop_move(BOARD_B);
    EXPECT_EQ(board.count_in_hand(BOARD_A, Stockfish::WHITE, Stockfish::QUEEN), 0);
    EXPECT_EQ(board.hash_key(false), beforeCaptureHash);
}

TEST_F(EngineTest, ZobristHashUniqueness) {
    Board board;
    std::set<uint64_t> seen_hashes;
    
    board.set_fen(BOARD_A, board.startingFen);
    seen_hashes.insert(board.hash_key(BOARD_A));
    
    auto moves = board.legal_moves(BOARD_A);
    for (const auto& move : moves) {
        board.push_move(BOARD_A, move);
        uint64_t hash = board.hash_key(BOARD_A);
        
        EXPECT_EQ(seen_hashes.count(hash), 0) << "Hash collision detected for move " << board.uci_move(BOARD_A, move);
        seen_hashes.insert(hash);
        
        board.pop_move(BOARD_A);
    }
}

TEST_F(EngineTest, EnPassantCapture) {
    Board board;
    board.set_fen(BOARD_A, "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2");
    
    auto moves = board.legal_moves(BOARD_A);
    Stockfish::Move ep_capture = Stockfish::MOVE_NONE;
    
    for (const auto& m : moves) {
        std::string uci = board.uci_move(BOARD_A, m);
        if (uci == "e5d6") {
            ep_capture = m;
            break;
        }
    }
    
    ASSERT_NE(ep_capture, Stockfish::MOVE_NONE) << "En passant capture should be legal";
    
    board.push_move(BOARD_A, ep_capture);
    std::string fen = board.fen(BOARD_A);
    
    EXPECT_NE(fen.find("3P4"), std::string::npos) << "White pawn should be on d6 after en passant";
    EXPECT_EQ(fen.find("3pP3"), std::string::npos) << "Black pawn on d5 should be captured";
}

TEST_F(EngineTest, Castling) {
    Board board;
    board.set_fen(BOARD_A, "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1");
    
    auto moves = board.legal_moves(BOARD_A);
    
    bool found_kingside = false;
    bool found_queenside = false;
    
    for (const auto& m : moves) {
        std::string uci = board.uci_move(BOARD_A, m);
        if (uci == "e1g1") found_kingside = true;
        if (uci == "e1c1") found_queenside = true;
    }
    
    EXPECT_TRUE(found_kingside) << "Kingside castling (e1g1) should be legal";
    EXPECT_TRUE(found_queenside) << "Queenside castling (e1c1) should be legal";
}

TEST_F(EngineTest, Promotion) {
    Board board;
    board.set_fen(BOARD_A, "8/P7/8/8/8/8/8/4K2k w - - 0 1");
    
    auto moves = board.legal_moves(BOARD_A);
    
    bool found_queen_promo = false;
    bool found_rook_promo = false;
    bool found_bishop_promo = false;
    bool found_knight_promo = false;
    
    for (const auto& m : moves) {
        std::string uci = board.uci_move(BOARD_A, m);
        if (uci == "a7a8q") found_queen_promo = true;
        if (uci == "a7a8r") found_rook_promo = true;
        if (uci == "a7a8b") found_bishop_promo = true;
        if (uci == "a7a8n") found_knight_promo = true;
    }
    
    EXPECT_TRUE(found_queen_promo) << "Queen promotion should be legal";
    EXPECT_TRUE(found_rook_promo) << "Rook promotion should be legal";
    EXPECT_TRUE(found_bishop_promo) << "Bishop promotion should be legal";
    EXPECT_TRUE(found_knight_promo) << "Knight promotion should be legal";
}

TEST_F(EngineTest, PolicySupportsQueenAndKnightPromotionsOnly) {
    Board board;
    board.set_fen(BOARD_A, "8/P7/8/8/8/8/8/4K2k w - - 0 1");

    for (const Stockfish::Move move : board.legal_moves(BOARD_A)) {
        const std::string uci = board.uci_move(BOARD_A, move);
        if (uci == "a7a8q" || uci == "a7a8n") {
            EXPECT_TRUE(is_policy_move_representable(board, BOARD_A, move)) << uci;
        } else if (uci == "a7a8r" || uci == "a7a8b") {
            EXPECT_FALSE(is_policy_move_representable(board, BOARD_A, move)) << uci;
        }
    }
}

TEST_F(EngineTest, MoveUnmakeSymmetry) {
    Board board;
    board.set_fen(BOARD_A, board.startingFen);
    
    std::string initial_fen = board.fen(BOARD_A);
    uint64_t initial_hash = board.hash_key(BOARD_A);
    
    auto moves = board.legal_moves(BOARD_A);
    
    for (const auto& move : moves) {
        board.push_move(BOARD_A, move);
        board.pop_move(BOARD_A);
        
        EXPECT_EQ(board.fen(BOARD_A), initial_fen) << "FEN should match after unmake of " << board.uci_move(BOARD_A, move);
        EXPECT_EQ(board.hash_key(BOARD_A), initial_hash) << "Hash should match after unmake of " << board.uci_move(BOARD_A, move);
    }
}

TEST_F(EngineTest, BughouseMoveIndependence) {
    Board board;
    board.set_fen(BOARD_A, board.startingFen);
    board.set_fen(BOARD_B, board.startingFen);
    
    auto moves_b_initial = board.legal_moves(BOARD_B).size();
    
    auto moves_a = board.legal_moves(BOARD_A);
    ASSERT_GT(moves_a.size(), 0);
    
    Stockfish::Move non_capture = Stockfish::MOVE_NONE;
    for (const auto& m : moves_a) {
        std::string uci = board.uci_move(BOARD_A, m);
        if (uci == "e2e4") {
            non_capture = m;
            break;
        }
    }
    ASSERT_NE(non_capture, Stockfish::MOVE_NONE);
    
    board.push_move(BOARD_A, non_capture);
    
    EXPECT_EQ(board.legal_moves(BOARD_B).size(), moves_b_initial) 
        << "Board B move count should not change for non-capture moves on Board A";
    
    EXPECT_EQ(board.count_in_hand(BOARD_B, Stockfish::WHITE, Stockfish::PAWN), 0)
        << "No pieces should be added to hand for non-capture moves";
    EXPECT_EQ(board.count_in_hand(BOARD_B, Stockfish::BLACK, Stockfish::PAWN), 0)
        << "No pieces should be added to hand for non-capture moves";
}

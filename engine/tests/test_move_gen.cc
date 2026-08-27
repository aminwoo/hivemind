#include <gtest/gtest.h>
#include <chrono>
#include <mutex>
#include <set>
#include <thread>
#include "environment/board.h"
#include "environment/constants.h"
#include "environment/joint_action.h"
#include "search/node.h"
#include "environment/planes.h"
#include "search/search_params.h"
#include "search/searchthread.h"
#include "search/agent.h"
#include "search/transposition_table.h"
#include "common/utils.h"
#include "common/globals.h"
#include "Fairy-Stockfish/src/position.h"
#include "Fairy-Stockfish/src/types.h"
#include "Fairy-Stockfish/src/bitboard.h"
#include "Fairy-Stockfish/src/piece.h"
#include "Fairy-Stockfish/src/thread.h"

class AgentTreeReuseTestPeer {
public:
    static void set_root(Agent& agent, const std::shared_ptr<Node>& root) {
        agent.rootNode = root;
    }

    static size_t retained_candidate_count(const Agent& agent) {
        return agent.nextRootCandidates_.size();
    }

    static void reindex_reused_subtree(
        Agent& agent, const std::shared_ptr<Node>& root) {
        agent.transpositionTable->clear();
        agent.reindex_reused_subtree(root);
    }

    static std::shared_ptr<Node> lookup(Agent& agent, uint64_t hash) {
        return agent.transpositionTable->lookup(hash);
    }

    static bool find_root_mate_and_retain(
        Agent& agent, Board& board, Stockfish::Color teamSide,
        bool teamHasTimeAdvantage, JointActionCandidate& action,
        int& plyToMate) {
        return Agent::find_root_mate_impl(
            board, teamSide, teamHasTimeAdvantage, action, plyToMate,
            SearchParams::MATE_SEARCH_NODE_BUDGET,
            &agent.mateContinuations_);
    }

    static bool reuse_mate_continuation(
        const Agent& agent, Board& board, Stockfish::Color teamSide,
        bool teamHasTimeAdvantage, JointActionCandidate& action,
        int& plyToMate) {
        return agent.try_reuse_mate_continuation(
            board, teamSide, teamHasTimeAdvantage, action, plyToMate);
    }

    static std::string format_root_aware_uci_score(
        const std::shared_ptr<Node>& root,
        const std::shared_ptr<Node>& pvChild,
        float childQ) {
        return Agent::format_root_aware_uci_score(root, pvChild, childQ);
    }
};

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

TEST(NodeTest, ProvenRootDrawOverridesUnvisitedPvChildScore) {
    auto root = std::make_shared<Node>(Stockfish::BLACK);
    auto child = std::make_shared<Node>(Stockfish::WHITE);
    root->mark_as_draw(1);

    EXPECT_EQ(
        AgentTreeReuseTestPeer::format_root_aware_uci_score(
            root, child, SearchParams::Q_INIT),
        "score cp 0");
}

TEST(NodeTest, ProvenPvChildDrawOverridesItsProvisionalQ) {
    auto root = std::make_shared<Node>(Stockfish::BLACK);
    auto child = std::make_shared<Node>(Stockfish::WHITE);
    child->mark_as_draw(1);

    EXPECT_EQ(
        AgentTreeReuseTestPeer::format_root_aware_uci_score(
            root, child, SearchParams::Q_INIT),
        "score cp 0");
}

TEST_F(EngineTest, HalfInputRepresentationMatchesFloatConversion) {
    Board board;
    for (const char* uci : {"g1f3", "g8f6", "f3g1"}) {
        Stockfish::Move move = find_move(board, BOARD_A, uci);
        ASSERT_NE(move, Stockfish::MOVE_NONE) << uci;
        board.push_move(BOARD_A, move);
    }

    std::vector<float> floatPlanes(NB_INPUT_VALUES());
    std::vector<__half> halfPlanes(NB_INPUT_VALUES());
    board_to_planes(board, floatPlanes.data(), Stockfish::BLACK, true);
    board_to_planes(board, halfPlanes.data(), Stockfish::BLACK, true);

    for (size_t index = 0; index < floatPlanes.size(); ++index) {
        EXPECT_FLOAT_EQ(
            __half2float(halfPlanes[index]),
            __half2float(__float2half_rn(floatPlanes[index]))) << index;
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
    for (const char* uci : {"g1f3", "g8f6", "f3g1", "f6g8"}) {
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
        JointActionRules{true, false, false});
    JointActionCandidate advantaged(
        Stockfish::MOVE_NONE, 0.5f, 0,
        Stockfish::MOVE_NONE, 0.5f, 0,
        JointActionRules{true, false, true});
    JointActionCandidate bothBoardsOnTurn(
        Stockfish::MOVE_NONE, 0.5f, 0,
        Stockfish::MOVE_NONE, 0.5f, 0,
        JointActionRules{true, true, true});

    EXPECT_LT(disadvantaged.jointPrior, 0.0f);
    EXPECT_FLOAT_EQ(advantaged.jointPrior, 0.25f);
    EXPECT_LT(bothBoardsOnTurn.jointPrior, 0.0f);
}

TEST(JointActionTest, SinglePassWithoutTimeAdvantageRequiresAPartnerCapture) {
    EXPECT_FALSE(is_single_pass_legal(false, true, true, false));
    EXPECT_TRUE(is_single_pass_legal(false, true, true, true));
    EXPECT_TRUE(is_single_pass_legal(true, true, true, false));
    EXPECT_TRUE(is_single_pass_legal(false, true, false, false));

    const Stockfish::Move moveB = static_cast<Stockfish::Move>(7);
    const JointActionRules bothOnTurn{true, true, false, true, true};

    JointActionCandidate passWithQuietPartner(
        Stockfish::MOVE_NONE, 0.5f, 0,
        moveB, 0.5f, 0,
        bothOnTurn, false, false);
    JointActionCandidate passWithCapturingPartner(
        Stockfish::MOVE_NONE, 0.5f, 0,
        moveB, 0.5f, 0,
        bothOnTurn, false, true);
    // A board that is on turn without legal moves must wait; that is not a choice.
    JointActionCandidate forcedPass(
        Stockfish::MOVE_NONE, 0.5f, 0,
        moveB, 0.5f, 0,
        JointActionRules{true, true, false, false, true}, false, false);

    EXPECT_LT(passWithQuietPartner.jointPrior, 0.0f);
    EXPECT_FLOAT_EQ(passWithCapturingPartner.jointPrior, 0.25f);
    EXPECT_FLOAT_EQ(forcedPass.jointPrior, 0.25f);
}

TEST(JointActionTest, GeneratorSkipsQuietPassPairsWhenBothBoardsAreOnTurn) {
    const Stockfish::Move quietA = static_cast<Stockfish::Move>(1);
    const Stockfish::Move captureB = static_cast<Stockfish::Move>(2);
    const Stockfish::Move quietB = static_cast<Stockfish::Move>(3);
    JointCandidateGenerator generator;
    generator.initialize(
        {quietA, Stockfish::MOVE_NONE}, {quietB, captureB, Stockfish::MOVE_NONE},
        {0.6f, 0.4f}, {0.5f, 0.3f, 0.2f},
        false, true, true,
        {0, 0}, {0, 1, 0});

    std::vector<std::pair<Stockfish::Move, Stockfish::Move>> generated;
    while (generator.hasNext()) {
        JointActionCandidate candidate = generator.getNext();
        generated.emplace_back(candidate.moveA, candidate.moveB);
    }

    EXPECT_NE(std::find(generated.begin(), generated.end(),
                        std::make_pair(Stockfish::MOVE_NONE, captureB)),
              generated.end());
    EXPECT_EQ(std::find(generated.begin(), generated.end(),
                        std::make_pair(Stockfish::MOVE_NONE, quietB)),
              generated.end());
    EXPECT_EQ(std::find(generated.begin(), generated.end(),
                        std::make_pair(quietA, Stockfish::MOVE_NONE)),
              generated.end());
    EXPECT_EQ(std::find(generated.begin(), generated.end(),
                        std::make_pair(Stockfish::MOVE_NONE, Stockfish::MOVE_NONE)),
              generated.end());
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

TEST(JointActionTest, RepreparingGumbelPoolPreservesEveryCandidateOnce) {
    JointCandidateGenerator generator;
    generator.initialize(
        {Stockfish::Move(1), Stockfish::Move(2),
         Stockfish::Move(3), Stockfish::Move(4)},
        {Stockfish::Move(5), Stockfish::Move(6), Stockfish::Move(7)},
        {0.40f, 0.30f, 0.20f, 0.10f},
        {0.50f, 0.30f, 0.20f},
        false, true, true);

    generator.prepareGumbelPool(6, 11);
    std::unordered_set<std::pair<size_t, size_t>, PairHash> generated;
    for (int count = 0; count < 2; ++count) {
        const JointActionCandidate candidate = generator.getNext();
        EXPECT_TRUE(generated.insert({candidate.idxA, candidate.idxB}).second);
    }

    generator.prepareGumbelPool(2, 29);
    while (generator.hasNext()) {
        const JointActionCandidate candidate = generator.getNext();
        EXPECT_TRUE(generated.insert({candidate.idxA, candidate.idxB}).second);
    }
    EXPECT_EQ(generated.size(), 12U);
}

TEST(JointActionTest, DisablingGumbelRestoresFactorizedOrder) {
    JointCandidateGenerator generator;
    generator.initialize(
        {Stockfish::Move(1), Stockfish::Move(2), Stockfish::Move(3)},
        {Stockfish::MOVE_NONE}, {0.60f, 0.30f, 0.10f}, {1.0f},
        false, true, false);

    generator.prepareGumbelPool(3, 41);
    generator.restoreFactorizedOrder();

    EXPECT_FLOAT_EQ(generator.getNext().jointPrior, 0.60f);
    EXPECT_FLOAT_EQ(generator.getNext().jointPrior, 0.30f);
    EXPECT_FLOAT_EQ(generator.getNext().jointPrior, 0.10f);
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

#ifndef NDEBUG
    EXPECT_THROW(board.make_moves(drop, Stockfish::MOVE_NONE), std::logic_error);
#else
    EXPECT_FALSE(board.is_legal_move(BOARD_A, drop));
#endif
    EXPECT_EQ(board.fen(BOARD_A), fenA);
    EXPECT_EQ(board.fen(BOARD_B), fenB);
    EXPECT_EQ(board.count_in_hand(BOARD_A, Stockfish::WHITE, Stockfish::PAWN), 0);
}

TEST_F(EngineTest, HasAnyLegalMoveMatchesFullGeneration) {
    Board board;
    auto expectMatchesFullGeneration = [&](int boardNum) {
        EXPECT_EQ(
            board.has_any_legal_move(boardNum),
            !board.legal_moves(boardNum).empty());
    };

    // Ordinary positions return on the first legal board move.
    expectMatchesFullGeneration(BOARD_A);
    expectMatchesFullGeneration(BOARD_B);

    // A real pocket piece can establish mobility without board-move generation.
    board.set_fen(BOARD_A, "4k3/8/8/8/8/8/8/4K3[P] w - - 0 1");
    expectMatchesFullGeneration(BOARD_A);

    // Check evasions and positions with no legal move take the fallback path.
    board.set_fen(
        BOARD_A,
        "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4");
    expectMatchesFullGeneration(BOARD_A);

    board.set_fen(BOARD_A, "7k/5K1p/7P/8/8/8/8/8[] b - - 0 1");
    ASSERT_FALSE(board.is_in_check(BOARD_A));
    expectMatchesFullGeneration(BOARD_A);
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

TEST_F(EngineTest, PromotedPieceCaptureGivesPawnAndBoardCopyPreservesPromotedState) {
    Board board;
    board.set(
        "r1bBk2r/pppn1ppp/4p3/3p4/3PB3/P1P1P3/2P2PPP/q~2QK1NR[NPp] w Kkq - 0 10|"
        "r1bq1bnr/ppp1k1pp/8/3pn1N1/5B2/8/PPP1PPPP/RN1QKB1R[Rqbn] b KQ - 1 7");

    // Board copy constructor must preserve ~ on promoted piece
    Board copy(board);
    EXPECT_NE(copy.fen(BOARD_A).find("q~"), std::string::npos);
    EXPECT_EQ(copy.fen(BOARD_A), board.fen(BOARD_A));
    EXPECT_EQ(copy.fen(BOARD_B), board.fen(BOARD_B));

    // Capturing promoted queen on a1 should add a PAWN to partner on Board B, NOT a Queen
    const Stockfish::Move capturePromotedQueen = find_move(board, BOARD_A, "d1a1");
    ASSERT_NE(capturePromotedQueen, Stockfish::MOVE_NONE);

    const int initialPawnsInHandB = board.count_in_hand(BOARD_B, Stockfish::BLACK, Stockfish::PAWN);
    const int initialQueensInHandB = board.count_in_hand(BOARD_B, Stockfish::BLACK, Stockfish::QUEEN);

    board.push_move(BOARD_A, capturePromotedQueen);

    EXPECT_EQ(board.count_in_hand(BOARD_B, Stockfish::BLACK, Stockfish::PAWN), initialPawnsInHandB + 1);
    EXPECT_EQ(board.count_in_hand(BOARD_B, Stockfish::BLACK, Stockfish::QUEEN), initialQueensInHandB);

    // Black on Board B had 1 Queen in hand ([Rqbn]). After dropping that 1 Queen,
    // dropping a 2nd Queen must be illegal because capturing q~ gave a Pawn, not a 2nd Queen.
    const Stockfish::Move firstQueenDrop = Stockfish::make_drop(
        Stockfish::SQ_D6, Stockfish::QUEEN, Stockfish::QUEEN);
    EXPECT_TRUE(board.is_legal_move(BOARD_B, firstQueenDrop));
    board.push_move(BOARD_B, firstQueenDrop);
    EXPECT_EQ(board.count_in_hand(BOARD_B, Stockfish::BLACK, Stockfish::QUEEN), 0);
    const Stockfish::Move secondQueenDrop = Stockfish::make_drop(
        Stockfish::SQ_D5, Stockfish::QUEEN, Stockfish::QUEEN);
    EXPECT_FALSE(board.is_legal_move(BOARD_B, secondQueenDrop));

    // Test that the copy behaves identically
    const Stockfish::Move copyCapture = find_move(copy, BOARD_A, "d1a1");
    ASSERT_NE(copyCapture, Stockfish::MOVE_NONE);
    copy.push_move(BOARD_A, copyCapture);
    EXPECT_EQ(copy.count_in_hand(BOARD_B, Stockfish::BLACK, Stockfish::PAWN), initialPawnsInHandB + 1);
    EXPECT_EQ(copy.count_in_hand(BOARD_B, Stockfish::BLACK, Stockfish::QUEEN), initialQueensInHandB);
    copy.push_move(BOARD_B, firstQueenDrop);
    EXPECT_EQ(copy.count_in_hand(BOARD_B, Stockfish::BLACK, Stockfish::QUEEN), 0);
    EXPECT_FALSE(copy.is_legal_move(BOARD_B, secondQueenDrop));
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

    SearchThread searchThread;
    searchThread.backup(trajectory, 0.5f);

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
    SearchThread searchThread;
    searchThread.backup(trajectory, 0.25f);

    EXPECT_EQ(parent.get_node_type(), NodeType::WIN);
    EXPECT_EQ(parent.get_end_in_ply(), 4);
    EXPECT_FLOAT_EQ(parent.get_child_q(0), 1.0f);
}

TEST_F(EngineTest, SelectionStopsAtSolvedExpandedNode) {
    auto root = std::make_shared<Node>(Stockfish::WHITE);
    std::vector<Stockfish::Move> actions = {Stockfish::MOVE_NONE};
    std::vector<float> priors = {1.0f};
    ASSERT_TRUE(root->try_init_and_expand(
        actions, actions, priors, priors, true, true, false,
        SearchParams::RuntimeConfig{}));
    root->mark_as_win(4);

    Board board;
    SearchThread searchThread;
    searchThread.set_root_node(root);

    EXPECT_EQ(searchThread.select_and_expand(board, true).leaf, root);
    EXPECT_EQ(root->get_child_visits().front(), 0);
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

TEST_F(EngineTest, PassProbabilityUsesNetworkLogitWithoutFloor) {
    Board board;
    auto actions = board.legal_moves(BOARD_A);
    actions.push_back(Stockfish::MOVE_NONE);
    std::vector<float> policyOutput(NB_POLICY_VALUES(), 0.0f);
    policyOutput[POLICY_INDEX.at("pass")] = -20.0f;

    const auto probabilities = get_normalized_probability(
        policyOutput.data(), actions, BOARD_A, board);

    ASSERT_EQ(probabilities.size(), actions.size());
    EXPECT_LT(probabilities.back(), 1e-6f);
}

TEST_F(EngineTest, HalfPolicyNormalizationMatchesFloatPolicy) {
    Board board;
    auto actions = board.legal_moves(BOARD_A);
    actions.push_back(Stockfish::MOVE_NONE);
    std::vector<float> floatPolicy(NB_POLICY_VALUES());
    std::vector<__half> halfPolicy(NB_POLICY_VALUES());
    for (size_t index = 0; index < floatPolicy.size(); ++index) {
        floatPolicy[index] = static_cast<float>(static_cast<int>(index % 17) - 8) * 0.25f;
        halfPolicy[index] = __float2half_rn(floatPolicy[index]);
    }

    const auto floatProbabilities = get_normalized_probability(
        floatPolicy.data(), actions, BOARD_A, board);
    const auto halfProbabilities = get_normalized_probability(
        halfPolicy.data(), actions, BOARD_A, board);

    ASSERT_EQ(halfProbabilities.size(), floatProbabilities.size());
    for (size_t index = 0; index < floatProbabilities.size(); ++index) {
        EXPECT_FLOAT_EQ(halfProbabilities[index], floatProbabilities[index]);
    }
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
    EXPECT_TRUE(SearchParams::ENABLE_TIME_EXTENSION);
    EXPECT_TRUE(SearchParams::ENABLE_TREE_REUSE);
    EXPECT_TRUE(config.enableTranspositions);
    EXPECT_FALSE(config.enableGumbelRootSearch);
    EXPECT_EQ(config.rootGumbelPoolSize, 2);
    EXPECT_EQ(config.rootGumbelInitialCandidates, 1);
    EXPECT_EQ(config.rootGumbelReplenishment, 1);
    EXPECT_EQ(config.rootGumbelMaxRoundVisits, 2);
    EXPECT_EQ(SearchParams::TT_MAX_SIZE, TranspositionTable::kDefaultMaxCapacity);
}

TEST_F(EngineTest, FastPolicyIndexMatchesMapLookupForAllMoves) {
    Board board;
    board.set("2rq1rk1/pppnb1p1/4p1p1/3pP1pp/4P3/2N1P1B1/PPP2NPP/R2Q1RK1/NN b - - 0 3|r4rk1/ppp2p1p/4bB1p/8/6b1/2P5/P1PB1PPP/R3R1K1/qbbnnppPB w");

    for (int boardNum : {BOARD_A, BOARD_B}) {
        const Stockfish::Color stm = board.side_to_move(boardNum);
        for (Stockfish::Move move : board.legal_moves(boardNum)) {
            std::string uci = board.uci_move(boardNum, move);
            if (uci.size() == 5 && uci.back() == 'q') {
                uci.pop_back();
            }
            std::string policyMove = (stm == Stockfish::BLACK && move != Stockfish::MOVE_NONE)
                ? mirror_move(uci)
                : uci;

            int expectedIndex = POLICY_INDEX.count(policyMove) ? POLICY_INDEX[policyMove] : -1;
            int fastIndex = get_fast_policy_index(move, stm);
            EXPECT_EQ(fastIndex, expectedIndex) << "Mismatch for move " << uci << " (policy " << policyMove << ")";
        }
    }
}

TEST(SearchConfigTest, ProgressiveWideningScheduleIsExplicit) {
    SearchParams::RuntimeConfig config;
    config.pwCoefficient = 1.0f;
    config.rootPwCoefficient = 4.0f;

    EXPECT_EQ(SearchParams::get_allowed_children(
                  1000, config.pwCoefficient, config.pwExponent), 16);
    EXPECT_EQ(SearchParams::get_allowed_children(
                  1000, config.rootPwCoefficient, config.pwExponent), 64);
    EXPECT_EQ(SearchParams::get_allowed_children(
                  10000, config.pwCoefficient, config.pwExponent), 40);
    EXPECT_EQ(SearchParams::get_allowed_children(
                  10000, config.rootPwCoefficient, config.pwExponent), 160);
}

TEST(SearchConfigTest, MovesLeftDiscountingPrefersFastWinAndDistantLoss) {
    const float discount = 0.20f;
    const float fastPlies = 0.05f;   // 5 plies
    const float distantPlies = 0.95f; // 95 plies

    // Positive evaluation (winning): fast win is preferred over distant win
    float winFast = 1.0f * (1.0f - discount * fastPlies);
    float winDistant = 1.0f * (1.0f - discount * distantPlies);
    EXPECT_GT(winFast, winDistant);
    EXPECT_GT(winFast, 0.98f);
    EXPECT_LT(winDistant, 0.82f);

    // Negative evaluation (losing): distant loss (survival) is preferred over fast loss
    float lossFast = -1.0f * (1.0f - discount * fastPlies);
    float lossDistant = -1.0f * (1.0f - discount * distantPlies);
    EXPECT_GT(lossDistant, lossFast); // lossDistant is less negative (closer to 0)
    EXPECT_LT(lossFast, -0.98f);
    EXPECT_GT(lossDistant, -0.82f);
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

TEST(JointCandidateGeneratorTest, JointFactorsRescorePrefixAndPreserveFallback) {
    JointCandidateGenerator generator;
    const std::vector<Stockfish::Move> actionsA = {
        Stockfish::Move(1), Stockfish::Move(2)};
    const std::vector<Stockfish::Move> actionsB = {
        Stockfish::Move(3), Stockfish::Move(4)};
    generator.initialize(
        actionsA, actionsB, {0.9f, 0.1f}, {0.9f, 0.1f},
        false, true, true, {}, {},
        {0.0f, 10.0f}, {0.0f, 1.0f}, 1, 2, 1.0f);

    ASSERT_TRUE(generator.hasNext());
    const JointActionCandidate first = generator.getNext();
    EXPECT_EQ(first.moveA, actionsA[1]);
    EXPECT_EQ(first.moveB, actionsB[1]);

    std::set<std::pair<uint32_t, uint32_t>> generated;
    generated.emplace(
        static_cast<uint32_t>(first.moveA),
        static_cast<uint32_t>(first.moveB));
    while (generator.hasNext()) {
        const JointActionCandidate candidate = generator.getNext();
        generated.emplace(
            static_cast<uint32_t>(candidate.moveA),
            static_cast<uint32_t>(candidate.moveB));
    }
    EXPECT_EQ(generated.size(), 4U);
}

TEST(JointCandidateGeneratorTest, JointPoolMergesWithHigherPriorityFallback) {
    JointCandidateGenerator generator;
    const std::vector<Stockfish::Move> actionsA = {
        Stockfish::Move(1), Stockfish::Move(2),
        Stockfish::Move(3), Stockfish::Move(4)};
    const std::vector<Stockfish::Move> actionsB = {
        Stockfish::Move(5), Stockfish::Move(6), Stockfish::Move(7)};
    const std::vector<float> priorsA = {0.4f, 0.3f, 0.2f, 0.1f};
    const std::vector<float> priorsB = {0.7f, 0.2f, 0.1f};
    const std::vector<float> factorsA(actionsA.size(), 0.0f);
    const std::vector<float> factorsB(actionsB.size(), 0.0f);
    generator.initialize(
        actionsA, actionsB, priorsA, priorsB,
        false, true, true, {}, {}, factorsA, factorsB, 1, 2, 1.0f);

    const JointActionCandidate first = generator.getNext();
    EXPECT_EQ(first.moveA, actionsA[0]);
    EXPECT_EQ(first.moveB, actionsB[0]);
    EXPECT_FLOAT_EQ(first.expansionPriority, 0.28f);

    const JointActionCandidate second = generator.getNext();
    EXPECT_EQ(second.moveA, actionsA[1]);
    EXPECT_EQ(second.moveB, actionsB[0]);
    EXPECT_FLOAT_EQ(second.expansionPriority, 0.21f);

    // This pair is outside the learned top-2 x top-2 pool, but its
    // factorized prior exceeds every remaining learned-pool candidate.
    const JointActionCandidate fallback = generator.peekNext();
    EXPECT_EQ(fallback.moveA, actionsA[2]);
    EXPECT_EQ(fallback.moveB, actionsB[0]);
    EXPECT_FLOAT_EQ(fallback.expansionPriority, 0.14f);
    const JointActionCandidate third = generator.getNext();
    EXPECT_EQ(third.moveA, fallback.moveA);
    EXPECT_EQ(third.moveB, fallback.moveB);

    std::set<std::pair<uint32_t, uint32_t>> generated = {{
        static_cast<uint32_t>(first.moveA),
        static_cast<uint32_t>(first.moveB)}, {
        static_cast<uint32_t>(second.moveA),
        static_cast<uint32_t>(second.moveB)}, {
        static_cast<uint32_t>(third.moveA),
        static_cast<uint32_t>(third.moveB)}};
    while (generator.hasNext()) {
        const JointActionCandidate candidate = generator.getNext();
        generated.emplace(
            static_cast<uint32_t>(candidate.moveA),
            static_cast<uint32_t>(candidate.moveB));
    }
    EXPECT_EQ(generated.size(), actionsA.size() * actionsB.size());
}

TEST(JointCandidateGeneratorTest, RepreparingLargerPoolPreservesExpandedIndices) {
    JointCandidateGenerator generator;
    const std::vector<Stockfish::Move> actionsA = {
        Stockfish::Move(1), Stockfish::Move(2), Stockfish::Move(3)};
    const std::vector<Stockfish::Move> actionsB = {
        Stockfish::Move(4), Stockfish::Move(5), Stockfish::Move(6)};
    const std::vector<float> priors = {0.6f, 0.3f, 0.1f};
    const std::vector<float> factorsA = {0.0f, 0.0f, 10.0f};
    const std::vector<float> factorsB = {0.0f, 0.0f, 1.0f};
    generator.initialize(
        actionsA, actionsB, priors, priors,
        false, true, true, {}, {}, factorsA, factorsB, 1, 1, 1.0f);

    const JointActionCandidate first = generator.getNext();
    ASSERT_EQ(first.moveA, actionsA[0]);
    ASSERT_EQ(first.moveB, actionsB[0]);
    const std::vector<float> promotedPriors =
        generator.reprepareJointPolicyPool(3, 1.0f);

    ASSERT_EQ(promotedPriors.size(), 1U);
    EXPECT_FLOAT_EQ(
        promotedPriors[0], generator.getGenerated(0).jointPrior);
    EXPECT_NE(promotedPriors[0], first.jointPrior);
    const JointActionCandidate promotedNext = generator.peekNext();
    EXPECT_EQ(promotedNext.moveA, actionsA[2]);
    EXPECT_EQ(promotedNext.moveB, actionsB[2]);

    std::set<std::pair<uint32_t, uint32_t>> generated = {{
        static_cast<uint32_t>(first.moveA),
        static_cast<uint32_t>(first.moveB)}};
    const JointActionCandidate second = generator.getNext();
    generated.emplace(
        static_cast<uint32_t>(second.moveA),
        static_cast<uint32_t>(second.moveB));
    generator.reprepareJointPolicyPool(3, 1.0f);
    while (generator.hasNext()) {
        const JointActionCandidate candidate = generator.getNext();
        generated.emplace(
            static_cast<uint32_t>(candidate.moveA),
            static_cast<uint32_t>(candidate.moveB));
    }
    EXPECT_EQ(generated.size(), 9U);
}

TEST(NodeTest, ReusedRootRebuildsJointPolicyPool) {
    Node node(Stockfish::WHITE);
    node.set_depth(1);
    SearchParams::RuntimeConfig config;
    config.enableGumbelRootSearch = false;
    config.jointPolicyTopK = 1;
    config.rootJointPolicyTopK = 3;
    const std::vector<Stockfish::Move> actionsA = {
        Stockfish::Move(1), Stockfish::Move(2), Stockfish::Move(3)};
    const std::vector<Stockfish::Move> actionsB = {
        Stockfish::Move(4), Stockfish::Move(5), Stockfish::Move(6)};
    const std::vector<float> priors = {0.6f, 0.3f, 0.1f};

    ASSERT_TRUE(node.try_init_and_expand(
        actionsA, actionsB, priors, priors, false, true, true, config,
        {}, {}, {0.0f, 0.0f, 10.0f}, {0.0f, 0.0f, 1.0f}, 1));
    const JointActionCandidate originalChild = node.get_joint_action(0);
    ASSERT_EQ(originalChild.moveA, actionsA[0]);
    ASSERT_EQ(originalChild.moveB, actionsB[0]);

    node.set_depth(0);
    node.configure_root_search(config);

    const JointActionCandidate rescoredChild = node.get_joint_action(0);
    EXPECT_EQ(rescoredChild.moveA, originalChild.moveA);
    EXPECT_EQ(rescoredChild.moveB, originalChild.moveB);
    EXPECT_NE(rescoredChild.jointPrior, originalChild.jointPrior);
    const JointActionCandidate promotedNext = node.peek_next_joint_action();
    EXPECT_EQ(promotedNext.moveA, actionsA[2]);
    EXPECT_EQ(promotedNext.moveB, actionsB[2]);
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
    SearchParams::RuntimeConfig config;
    config.enableGumbelRootSearch = false;
    std::vector<Stockfish::Move> actionsA = {
        Stockfish::Move(1), Stockfish::Move(2), Stockfish::Move(3)};
    std::vector<Stockfish::Move> actionsB = {Stockfish::MOVE_NONE};

    ASSERT_TRUE(node.try_init_and_expand(
        actionsA, actionsB, {0.8f, 0.15f, 0.05f}, {1.0f}, false, true, false,
        config));
    ASSERT_EQ(node.get_children().size(), 1U);
    EXPECT_FALSE(node.should_expand_new_child(config));

    node.update(0, 0.25f);

    EXPECT_TRUE(node.should_expand_new_child(config));
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
    config.enableGumbelRootSearch = false;

    ASSERT_TRUE(node.try_init_and_expand(
        actionsA, {Stockfish::MOVE_NONE}, {0.5f, 0.5f}, {1.0f},
        false, true, false, config));
    node.update(0, 0.0f);

    JointActionCandidate action;
    ASSERT_NE(node.expand_next_joint_child(nullptr, 0, action, config), nullptr);

    Node::ChildSelection firstSelection =
        node.select_child_and_apply_virtual_loss(config);
    Node::ChildSelection secondSelection =
        node.select_child_and_apply_virtual_loss(config);
    auto& [firstChild, firstIdx, firstReserved, firstPending] = firstSelection;
    auto& [secondChild, secondIdx, secondReserved, secondPending] = secondSelection;
    ASSERT_NE(firstChild, nullptr);
    ASSERT_NE(secondChild, nullptr);
    EXPECT_EQ(firstIdx, 0);
    EXPECT_EQ(secondIdx, 1);
    EXPECT_TRUE(firstReserved);
    EXPECT_TRUE(secondReserved);
    EXPECT_EQ(firstPending, nullptr);
    EXPECT_EQ(secondPending, nullptr);

    node.remove_virtual_loss(firstIdx);
    node.remove_virtual_loss(secondIdx);
    firstChild->release_evaluation_reservation();
    secondChild->release_evaluation_reservation();
}

TEST(NodeTest, PendingEvaluationDivertsSelectionToAvailableSibling) {
    Node node(Stockfish::WHITE);
    std::vector<Stockfish::Move> actionsA = {
        Stockfish::Move(1), Stockfish::Move(2)};
    SearchParams::RuntimeConfig config;
    config.enableGumbelRootSearch = false;

    ASSERT_TRUE(node.try_init_and_expand(
        actionsA, {Stockfish::MOVE_NONE}, {0.9f, 0.1f}, {1.0f},
        false, true, false, config));
    JointActionCandidate action;
    ASSERT_NE(node.expand_next_joint_child(nullptr, 0, action, config), nullptr);

    auto children = node.get_children();
    ASSERT_TRUE(children[0]->try_reserve_evaluation());

    auto [selectedChild, selectedIdx, evaluationReserved, pendingEvaluation] =
        node.select_child_and_apply_virtual_loss(config);
    EXPECT_EQ(selectedChild, children[1]);
    EXPECT_EQ(selectedIdx, 1);
    EXPECT_TRUE(evaluationReserved);
    EXPECT_EQ(pendingEvaluation, nullptr);

    node.remove_virtual_loss(selectedIdx);
    selectedChild->release_evaluation_reservation();
    children[0]->release_evaluation_reservation();
}

TEST(NodeTest, BlockedBatchLeafDivertsSelectionToAvailableSibling) {
    Node node(Stockfish::WHITE);
    std::vector<Stockfish::Move> actionsA = {
        Stockfish::Move(1), Stockfish::Move(2)};
    SearchParams::RuntimeConfig config;
    config.enableGumbelRootSearch = false;

    ASSERT_TRUE(node.try_init_and_expand(
        actionsA, {Stockfish::MOVE_NONE}, {0.9f, 0.1f}, {1.0f},
        false, true, false, config));
    JointActionCandidate action;
    ASSERT_NE(node.expand_next_joint_child(nullptr, 0, action, config), nullptr);

    const auto children = node.get_children();
    const std::unordered_set<const Node*> blocked = {children[0].get()};
    auto [selectedChild, selectedIdx, evaluationReserved, pendingEvaluation] =
        node.select_child_and_apply_virtual_loss(config, &blocked);

    EXPECT_EQ(selectedChild, children[1]);
    EXPECT_EQ(selectedIdx, 1);
    EXPECT_TRUE(evaluationReserved);
    EXPECT_EQ(pendingEvaluation, nullptr);

    node.remove_virtual_loss(selectedIdx);
    selectedChild->release_evaluation_reservation();
}

TEST(NodeTest, SelectionWaitsWhenEveryChildEvaluationIsPending) {
    Node node(Stockfish::WHITE);
    SearchParams::RuntimeConfig config;
    config.enableGumbelRootSearch = false;

    ASSERT_TRUE(node.try_init_and_expand(
        {Stockfish::Move(1)}, {Stockfish::MOVE_NONE}, {1.0f}, {1.0f},
        false, true, false, config));
    auto children = node.get_children();
    ASSERT_TRUE(children[0]->try_reserve_evaluation());

    auto [selectedChild, selectedIdx, evaluationReserved, pendingEvaluation] =
        node.select_child_and_apply_virtual_loss(config);
    EXPECT_EQ(selectedChild, nullptr);
    EXPECT_EQ(selectedIdx, -1);
    EXPECT_FALSE(evaluationReserved);
    EXPECT_EQ(pendingEvaluation, children[0]);

    children[0]->release_evaluation_reservation();
}

TEST(NodeTest, DynamicFpuBoostsUnvisitedChildInWinningParent) {
    Node node(Stockfish::WHITE);
    std::vector<Stockfish::Move> actionsA = {
        Stockfish::Move(1), Stockfish::Move(2)};
    SearchParams::RuntimeConfig config;
    config.enableDynamicFpu = true;
    config.enableGumbelRootSearch = false;
    config.fpuReduction = 0.5f;

    ASSERT_TRUE(node.try_init_and_expand(
        actionsA, {Stockfish::MOVE_NONE}, {0.6f, 0.4f}, {1.0f},
        false, true, false, config));
    // Parent is strongly winning: +0.9
    node.update(0, 0.9f);

    JointActionCandidate action;
    ASSERT_NE(node.expand_next_joint_child(nullptr, 0, action, config), nullptr);

    // With dynamic FPU (parentQ=0.9, fpuReduction=0.5, visitedPolicy=0.6),
    // unvisited child 1 has FPU Q = 0.9 - 0.5 * sqrt(0.6) = ~0.51 > -1.0.
    auto selection = node.select_child_and_apply_virtual_loss(config);
    EXPECT_NE(selection.child, nullptr);
    EXPECT_TRUE(selection.hasEvaluationReservation);
    node.remove_virtual_loss(selection.childIdx);
    selection.child->release_evaluation_reservation();
}

TEST(NodeTest, DynamicFpuIsInvariantToParentValueOffset) {
    Node node(Stockfish::WHITE);
    SearchParams::RuntimeConfig config;
    config.enableDynamicFpu = true;
    config.enableGumbelRootSearch = false;
    config.fpuReduction = 1.0f;
    config.cpuctInit = 0.0f;
    config.cpuctBase = 1.0e20f;

    ASSERT_TRUE(node.try_init_and_expand(
        {Stockfish::Move(1), Stockfish::Move(2)},
        {Stockfish::MOVE_NONE}, {0.1f, 0.9f}, {1.0f},
        false, true, false, config));
    JointActionCandidate action;
    ASSERT_NE(node.expand_next_joint_child(nullptr, 0, action, config), nullptr);

    // Offset the parent and its visited edge to -1. An absolute [-1, 1]
    // clamp would flatten the unvisited edge to -1 too and select it by index.
    node.update(1, -1.0f);
    Node::ChildSelection selection =
        node.select_child_and_apply_virtual_loss(config);
    EXPECT_EQ(selection.childIdx, 1);
    if (selection.hasEvaluationReservation) {
        selection.child->release_evaluation_reservation();
    }
    node.remove_virtual_loss(selection.childIdx);
}

TEST(NodeTest, OrdinaryPuctFindsBestUnvisitedPriorAfterGumbelExpansion) {
    Node node(Stockfish::WHITE);
    SearchParams::RuntimeConfig config;
    config.enableGumbelRootSearch = true;
    config.rootGumbelPoolSize = 8;
    config.rootGumbelInitialCandidates = 8;
    config.rootNoiseSeed = 17;
    const std::vector<Stockfish::Move> actionsA = {
        Stockfish::Move(1), Stockfish::Move(2), Stockfish::Move(3),
        Stockfish::Move(4), Stockfish::Move(5), Stockfish::Move(6),
        Stockfish::Move(7), Stockfish::Move(8)};

    ASSERT_TRUE(node.try_init_and_expand(
        actionsA, {Stockfish::MOVE_NONE},
        {0.130f, 0.129f, 0.128f, 0.127f,
         0.124f, 0.123f, 0.120f, 0.119f},
        {1.0f}, false, true, false, config));
    JointActionCandidate action;
    while (node.get_num_generated() < actionsA.size()) {
        ASSERT_NE(node.expand_next_joint_child(nullptr, 0, action, config),
                  nullptr);
    }

    int expectedIdx = -1;
    float highestPrior = -1.0f;
    for (size_t index = 0; index < node.get_num_generated(); ++index) {
        const float prior = node.get_joint_action(static_cast<int>(index)).jointPrior;
        if (prior > highestPrior) {
            highestPrior = prior;
            expectedIdx = static_cast<int>(index);
        }
    }
    ASSERT_NE(expectedIdx, 0);  // The Gumbel pool actually reordered this seed.

    config.enableGumbelRootSearch = false;
    node.configure_root_search(config);
    node.update_terminal(0.0f);  // Give the PUCT exploration term a nonzero N.
    Node::ChildSelection selection =
        node.select_child_and_apply_virtual_loss(config);
    EXPECT_EQ(selection.childIdx, expectedIdx);
    if (selection.hasEvaluationReservation) {
        selection.child->release_evaluation_reservation();
    }
    node.remove_virtual_loss(selection.childIdx);
}

TEST(NodeTest, GumbelRootBalancesCandidatesBeforeSequentialHalving) {
    Node node(Stockfish::WHITE);
    SearchParams::RuntimeConfig config;
    config.enableGumbelRootSearch = true;
    config.rootGumbelPoolSize = 8;
    config.rootGumbelInitialCandidates = 4;
    config.rootGumbelReplenishment = 2;
    config.rootGumbelMaxRoundVisits = 4;
    config.rootNoiseSeed = 17;
    const std::vector<Stockfish::Move> actionsA = {
        Stockfish::Move(1), Stockfish::Move(2), Stockfish::Move(3),
        Stockfish::Move(4), Stockfish::Move(5), Stockfish::Move(6),
        Stockfish::Move(7), Stockfish::Move(8)};

    ASSERT_TRUE(node.try_init_and_expand(
        actionsA, {Stockfish::MOVE_NONE},
        {0.30f, 0.20f, 0.15f, 0.12f, 0.09f, 0.06f, 0.05f, 0.03f},
        {1.0f}, false, true, false, config));
    while (node.get_children().size() < 4U) {
        JointActionCandidate action;
        ASSERT_NE(node.expand_next_joint_child(nullptr, 0, action, config), nullptr);
    }
    EXPECT_FALSE(node.should_expand_new_child(config));

    auto completeSimulation = [&](float value) {
        Node::ChildSelection selection =
            node.select_child_and_apply_virtual_loss(config);
        EXPECT_NE(selection.child, nullptr);
        if (!selection.child) {
            return -1;
        }
        if (selection.hasEvaluationReservation) {
            selection.child->release_evaluation_reservation();
        }
        node.update_and_remove_virtual_loss(selection.childIdx, value);
        return selection.childIdx;
    };

    std::unordered_set<int> firstRound;
    for (int simulation = 0; simulation < 4; ++simulation) {
        firstRound.insert(completeSimulation(0.0f));
    }
    EXPECT_EQ(firstRound.size(), 4U);

    std::unordered_set<int> secondRound;
    for (int simulation = 0; simulation < 4; ++simulation) {
        secondRound.insert(completeSimulation(0.25f));
    }
    EXPECT_EQ(secondRound.size(), 2U);
}

TEST(NodeTest, GumbelRootProgressivelyReplenishesAfterFinalist) {
    Node node(Stockfish::WHITE);
    SearchParams::RuntimeConfig config;
    config.enableGumbelRootSearch = true;
    config.rootGumbelPoolSize = 6;
    config.rootGumbelInitialCandidates = 2;
    config.rootGumbelReplenishment = 1;
    config.rootGumbelMaxRoundVisits = 1;
    config.rootNoiseSeed = 23;

    ASSERT_TRUE(node.try_init_and_expand(
        {Stockfish::Move(1), Stockfish::Move(2), Stockfish::Move(3),
         Stockfish::Move(4), Stockfish::Move(5), Stockfish::Move(6)},
        {Stockfish::MOVE_NONE},
        {0.30f, 0.24f, 0.18f, 0.12f, 0.09f, 0.07f}, {1.0f},
        false, true, false, config));
    JointActionCandidate action;
    ASSERT_NE(node.expand_next_joint_child(nullptr, 0, action, config), nullptr);

    auto completeSimulation = [&]() {
        Node::ChildSelection selection =
            node.select_child_and_apply_virtual_loss(config);
        ASSERT_NE(selection.child, nullptr);
        if (selection.hasEvaluationReservation) {
            selection.child->release_evaluation_reservation();
        }
        node.update_and_remove_virtual_loss(selection.childIdx, 0.0f);
    };
    completeSimulation();
    completeSimulation();
    completeSimulation();

    const Node::ChildSelection replenishSignal =
        node.select_child_and_apply_virtual_loss(config);
    EXPECT_EQ(replenishSignal.child, nullptr);
    EXPECT_TRUE(node.should_expand_new_child(config));

    ASSERT_NE(node.expand_next_joint_child(nullptr, 0, action, config), nullptr);
    EXPECT_EQ(node.get_children().size(), 3U);
}

TEST(NodeTest, GumbelRootEliminatesSolverProvenLosingAction) {
    Node node(Stockfish::WHITE);
    SearchParams::RuntimeConfig config;
    config.enableGumbelRootSearch = true;
    config.rootGumbelPoolSize = 4;
    config.rootGumbelInitialCandidates = 2;
    config.rootNoiseSeed = 31;

    ASSERT_TRUE(node.try_init_and_expand(
        {Stockfish::Move(1), Stockfish::Move(2),
         Stockfish::Move(3), Stockfish::Move(4)},
        {Stockfish::MOVE_NONE}, {0.4f, 0.3f, 0.2f, 0.1f}, {1.0f},
        false, true, false, config));
    JointActionCandidate action;
    ASSERT_NE(node.expand_next_joint_child(nullptr, 0, action, config), nullptr);
    const auto children = node.get_children();
    children[0]->mark_as_win(3);

    Node::ChildSelection selection =
        node.select_child_and_apply_virtual_loss(config);
    ASSERT_NE(selection.child, nullptr);
    EXPECT_EQ(selection.child, children[1]);
    if (selection.hasEvaluationReservation) {
        selection.child->release_evaluation_reservation();
    }
    node.remove_virtual_loss(selection.childIdx);
}

TEST(NodeTest, GumbelRootReconfiguresReusedSubtree) {
    Node node(Stockfish::WHITE);
    node.set_depth(2);
    SearchParams::RuntimeConfig config;
    config.enableGumbelRootSearch = true;
    config.rootGumbelPoolSize = 6;
    config.rootGumbelInitialCandidates = 2;
    config.rootNoiseSeed = 47;

    ASSERT_TRUE(node.try_init_and_expand(
        {Stockfish::Move(1), Stockfish::Move(2), Stockfish::Move(3),
         Stockfish::Move(4), Stockfish::Move(5), Stockfish::Move(6)},
        {Stockfish::MOVE_NONE},
        {0.30f, 0.24f, 0.18f, 0.12f, 0.09f, 0.07f}, {1.0f},
        false, true, false, config));
    JointActionCandidate action;
    ASSERT_NE(node.expand_next_joint_child(nullptr, 0, action, config), nullptr);

    node.set_depth(0);
    node.configure_root_search(config);
    EXPECT_FALSE(node.should_expand_new_child(config));

    std::unordered_set<int> selected;
    for (int simulation = 0; simulation < 2; ++simulation) {
        Node::ChildSelection choice =
            node.select_child_and_apply_virtual_loss(config);
        ASSERT_NE(choice.child, nullptr);
        selected.insert(choice.childIdx);
        if (choice.hasEvaluationReservation) {
            choice.child->release_evaluation_reservation();
        }
        node.update_and_remove_virtual_loss(choice.childIdx, 0.0f);
    }
    EXPECT_EQ(selected.size(), 2U);
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

TEST_F(EngineTest, ReservedCanonicalExpansionRestoresBoardAndEdgeState) {
    Board board;
    const uint64_t initialHash = board.hash_key(false);
    const std::string initialFenA = board.fen(BOARD_A);
    const std::string initialFenB = board.fen(BOARD_B);
    const auto legalMoves = board.legal_moves(BOARD_A);
    ASSERT_GE(legalMoves.size(), 2U);

    auto root = std::make_shared<Node>(Stockfish::WHITE);
    SearchParams::RuntimeConfig config;
    config.enableTranspositions = true;
    ASSERT_TRUE(root->try_init_and_expand(
        {legalMoves[0], legalMoves[1]}, {Stockfish::MOVE_NONE},
        {0.9f, 0.1f}, {1.0f}, false, true, false, config));
    root->update(0, 0.0f);

    const JointActionCandidate action = root->peek_next_joint_action();
    board.make_moves(action.moveA, action.moveB);
    const uint64_t childHash = board.search_hash_key(Stockfish::BLACK, true);
    board.unmake_moves(action.moveA, action.moveB);

    auto canonical = std::make_shared<Node>(Stockfish::BLACK, childHash);
    ASSERT_TRUE(canonical->try_reserve_evaluation());
    TranspositionTable table;
    ASSERT_EQ(table.insertOrGet(childHash, canonical), canonical);

    SearchThread searchThread;
    searchThread.set_root_node(root);
    searchThread.set_runtime_config(config);
    searchThread.set_transposition_table(&table);
    const LeafSelection selection = searchThread.select_and_expand(board, false);

    EXPECT_EQ(selection.leaf, nullptr);
    EXPECT_EQ(selection.pendingEvaluation, canonical);
    EXPECT_EQ(board.hash_key(false), initialHash);
    EXPECT_EQ(board.fen(BOARD_A), initialFenA);
    EXPECT_EQ(board.fen(BOARD_B), initialFenB);
    ASSERT_EQ(root->get_children().size(), 2U);
    EXPECT_EQ(root->get_children()[1], canonical);
    EXPECT_EQ(root->get_child_visits()[1], 0);

    canonical->release_evaluation_reservation();
}

TEST_F(EngineTest, InitialGeneratedChildUsesCanonicalTransposition) {
    Board board;
    const auto legalMoves = board.legal_moves(BOARD_A);
    ASSERT_FALSE(legalMoves.empty());

    auto root = std::make_shared<Node>(Stockfish::WHITE);
    SearchParams::RuntimeConfig config;
    config.enableTranspositions = true;
    ASSERT_TRUE(root->try_init_and_expand(
        {legalMoves[0]}, {Stockfish::MOVE_NONE}, {1.0f}, {1.0f},
        false, true, false, config));
    const JointActionCandidate action = root->get_joint_action(0);

    board.make_moves(action.moveA, action.moveB);
    const uint64_t childHash = board.search_hash_key(Stockfish::BLACK, true);
    board.unmake_moves(action.moveA, action.moveB);

    auto canonical = std::make_shared<Node>(Stockfish::BLACK, childHash);
    TranspositionTable table;
    ASSERT_EQ(table.insertOrGet(childHash, canonical), canonical);

    SearchThread searchThread;
    searchThread.set_root_node(root);
    searchThread.set_runtime_config(config);
    searchThread.set_transposition_table(&table);
    const LeafSelection selection = searchThread.select_and_expand(board, false);

    EXPECT_EQ(selection.leaf, canonical);
    EXPECT_TRUE(selection.hasEvaluationReservation);
    EXPECT_EQ(selection.pendingEvaluation, nullptr);
    EXPECT_EQ(root->get_children()[0], canonical);
    EXPECT_EQ(board.search_hash_key(Stockfish::BLACK, true), childHash);
    EXPECT_EQ(table.getHits(), 1U);

    root->remove_virtual_loss(0);
    canonical->release_evaluation_reservation();
    board.unmake_moves(action.moveA, action.moveB);
}

TEST_F(EngineTest, BlockedCanonicalLeafIsExcludedAndReservationIsReleased) {
    Board board;
    const uint64_t initialHash = board.hash_key(false);
    const auto legalMoves = board.legal_moves(BOARD_A);
    ASSERT_FALSE(legalMoves.empty());

    auto root = std::make_shared<Node>(Stockfish::WHITE);
    SearchParams::RuntimeConfig config;
    config.enableTranspositions = true;
    ASSERT_TRUE(root->try_init_and_expand(
        {legalMoves[0]}, {Stockfish::MOVE_NONE}, {1.0f}, {1.0f},
        false, true, false, config));
    const JointActionCandidate action = root->get_joint_action(0);

    board.make_moves(action.moveA, action.moveB);
    const uint64_t childHash = board.search_hash_key(Stockfish::BLACK, true);
    board.unmake_moves(action.moveA, action.moveB);

    auto canonical = std::make_shared<Node>(Stockfish::BLACK, childHash);
    TranspositionTable table;
    ASSERT_EQ(table.insertOrGet(childHash, canonical), canonical);
    const std::unordered_set<const Node*> blocked = {canonical.get()};

    SearchThread searchThread;
    searchThread.set_root_node(root);
    searchThread.set_runtime_config(config);
    searchThread.set_transposition_table(&table);
    const LeafSelection collision = searchThread.select_and_expand(
        board, false, &blocked);

    EXPECT_EQ(collision.leaf, nullptr);
    EXPECT_EQ(collision.pendingEvaluation, nullptr);
    EXPECT_EQ(collision.exhaustedSubtree, canonical);
    EXPECT_FALSE(canonical->is_evaluation_pending());
    EXPECT_EQ(board.hash_key(false), initialHash);
    EXPECT_EQ(root->get_children()[0], canonical);

    const LeafSelection exhausted = searchThread.select_and_expand(
        board, false, &blocked);
    EXPECT_EQ(exhausted.leaf, nullptr);
    EXPECT_EQ(exhausted.exhaustedSubtree, root);
    EXPECT_EQ(board.hash_key(false), initialHash);
}

TEST(NodeTest, EvaluationReservationIsExclusiveUntilReleased) {
    Node node(Stockfish::WHITE);

    EXPECT_TRUE(node.try_reserve_evaluation());
    EXPECT_FALSE(node.try_reserve_evaluation());

    node.release_evaluation_reservation();
    EXPECT_TRUE(node.try_reserve_evaluation());
    node.release_evaluation_reservation();
}

TEST(NodeTest, PendingEvaluationRetainsReplacedChild) {
    Node parent(Stockfish::WHITE);
    ASSERT_TRUE(parent.try_init_and_expand(
        {Stockfish::Move(1)}, {Stockfish::MOVE_NONE}, {1.0f}, {1.0f},
        false, true, false, SearchParams::RuntimeConfig{}));

    auto child = parent.get_children().front();
    ASSERT_TRUE(child->try_reserve_evaluation());

    auto selection = parent.select_child_and_apply_virtual_loss();
    ASSERT_EQ(selection.child, nullptr);
    ASSERT_EQ(selection.pendingEvaluation, child);

    std::weak_ptr<Node> childLifetime = child;
    parent.replace_child(0, std::make_shared<Node>(Stockfish::BLACK));
    child.reset();
    EXPECT_FALSE(childLifetime.expired());

    selection.pendingEvaluation->release_evaluation_reservation();
    selection.pendingEvaluation.reset();
    EXPECT_TRUE(childLifetime.expired());
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

TEST(MctsSolverTest, ReverseProofReachesEveryTranspositionParent) {
    const std::vector<Stockfish::Move> actions = {Stockfish::MOVE_NONE};
    const std::vector<float> priors = {1.0f};
    auto firstParent = std::make_shared<Node>(Stockfish::WHITE);
    auto secondParent = std::make_shared<Node>(Stockfish::WHITE);

    ASSERT_TRUE(firstParent->try_init_and_expand(
        actions, actions, priors, priors, true, true, false,
        SearchParams::RuntimeConfig{}));
    ASSERT_TRUE(secondParent->try_init_and_expand(
        actions, actions, priors, priors, true, true, false,
        SearchParams::RuntimeConfig{}));

    auto canonicalChild = std::make_shared<Node>(Stockfish::BLACK, 42);
    firstParent->replace_child(0, canonicalChild);
    secondParent->replace_child(0, canonicalChild);

    canonicalChild->mark_as_loss(3);

    EXPECT_EQ(firstParent->get_node_type(), NodeType::WIN);
    EXPECT_EQ(firstParent->get_end_in_ply(), 4);
    EXPECT_EQ(secondParent->get_node_type(), NodeType::WIN);
    EXPECT_EQ(secondParent->get_end_in_ply(), 4);
}

TEST(MctsSolverTest, ReverseProofPropagatesTransitively) {
    const std::vector<Stockfish::Move> actions = {Stockfish::MOVE_NONE};
    const std::vector<float> priors = {1.0f};
    auto grandparent = std::make_shared<Node>(Stockfish::WHITE);
    auto parent = std::make_shared<Node>(Stockfish::BLACK);

    ASSERT_TRUE(grandparent->try_init_and_expand(
        actions, actions, priors, priors, true, true, false,
        SearchParams::RuntimeConfig{}));
    ASSERT_TRUE(parent->try_init_and_expand(
        actions, actions, priors, priors, true, true, false,
        SearchParams::RuntimeConfig{}));
    grandparent->replace_child(0, parent);

    const auto leaf = parent->get_children().front();
    leaf->mark_as_loss(5);

    ASSERT_EQ(parent->get_node_type(), NodeType::WIN);
    EXPECT_EQ(parent->get_end_in_ply(), 6);
    EXPECT_EQ(grandparent->get_node_type(), NodeType::LOSS);
    EXPECT_EQ(grandparent->get_end_in_ply(), 7);
}

TEST(MctsSolverTest, ReverseProofIgnoresReplacedChild) {
    const std::vector<Stockfish::Move> actions = {Stockfish::MOVE_NONE};
    const std::vector<float> priors = {1.0f};
    auto parent = std::make_shared<Node>(Stockfish::WHITE);

    ASSERT_TRUE(parent->try_init_and_expand(
        actions, actions, priors, priors, true, true, false,
        SearchParams::RuntimeConfig{}));
    const auto replacedChild = parent->get_children().front();
    auto canonicalChild = std::make_shared<Node>(Stockfish::BLACK, 84);

    parent->replace_child(0, canonicalChild);
    replacedChild->mark_as_loss(9);
    EXPECT_EQ(parent->get_node_type(), NodeType::UNSOLVED);

    canonicalChild->mark_as_loss(2);
    ASSERT_EQ(parent->get_node_type(), NodeType::WIN);
    EXPECT_EQ(parent->get_end_in_ply(), 3);
}

TEST(MctsSolverTest, ReverseProofCatchesAlreadySolvedCanonicalChild) {
    const std::vector<Stockfish::Move> actions = {Stockfish::MOVE_NONE};
    const std::vector<float> priors = {1.0f};
    auto parent = std::make_shared<Node>(Stockfish::WHITE);

    ASSERT_TRUE(parent->try_init_and_expand(
        actions, actions, priors, priors, true, true, false,
        SearchParams::RuntimeConfig{}));
    auto solvedCanonical = std::make_shared<Node>(Stockfish::BLACK, 126);
    solvedCanonical->mark_as_loss(2);

    parent->replace_child(0, solvedCanonical);

    EXPECT_EQ(parent->get_node_type(), NodeType::WIN);
    EXPECT_EQ(parent->get_end_in_ply(), 3);
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
    auto [selectedChild, selectedIdx, evaluationReserved, pendingEvaluation] =
        parent.select_child_and_apply_virtual_loss(config);
    ASSERT_NE(selectedChild, nullptr);
    EXPECT_EQ(selectedIdx, defenseIdx);
    EXPECT_TRUE(evaluationReserved);
    EXPECT_EQ(pendingEvaluation, nullptr);
    parent.remove_virtual_loss(selectedIdx);
    selectedChild->release_evaluation_reservation();
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

TEST(PonderModeTest, SearchInfoResetStartTime) {
    auto start = std::chrono::steady_clock::now() - std::chrono::milliseconds(500);
    SearchInfo info(start, 1000);
    EXPECT_GE(info.elapsed(), 400.0);

    info.reset_start_time();
    EXPECT_LT(info.elapsed(), 100.0);
}

TEST(PonderModeTest, EffectiveMoveTimePublishesExtensionsLockFree) {
    SearchInfo info(std::chrono::steady_clock::now(), 1000);
    EXPECT_EQ(info.get_effective_move_time(), 1000);

    ASSERT_TRUE(info.try_extend_time(2.0f, 1));
    EXPECT_GT(info.get_effective_move_time(), 1000);
    EXPECT_EQ(info.get_extension_count(), 1);
    EXPECT_FALSE(info.try_extend_time(2.0f, 1));
}

TEST(PonderModeTest, SearchOptionsPonderFlags) {
    SearchOptions normalOpts = SearchOptions::uci(1000, 1, false);
    EXPECT_FALSE(normalOpts.isPonder);
    EXPECT_TRUE(normalOpts.enablePonder);

    SearchOptions ponderOpts = SearchOptions::uci(1000, 1, true);
    EXPECT_TRUE(ponderOpts.isPonder);
    EXPECT_TRUE(ponderOpts.enablePonder);
}

TEST(PonderModeTest, AgentPonderHitTransitions) {
    Agent agent;
    EXPECT_FALSE(agent.is_pondering());

    SearchInfo info(std::chrono::steady_clock::now() - std::chrono::milliseconds(300), 1000);
    agent.ponderhit(); // Safe to call when not running
    EXPECT_FALSE(agent.is_pondering());
}

TEST_F(EngineTest, TreeReuseRetainsNonPrincipalOpponentReplies) {
    Board board;
    constexpr bool teamHasTimeAdvantage = false;
    auto root = std::make_shared<Node>(
        Stockfish::WHITE,
        board.search_hash_key(Stockfish::WHITE, teamHasTimeAdvantage));

    const Stockfish::Move ownMove = find_move(board, BOARD_A, "e2e4");
    ASSERT_NE(ownMove, Stockfish::MOVE_NONE);
    ASSERT_TRUE(root->try_init_and_expand(
        {ownMove}, {Stockfish::MOVE_NONE}, {1.0f}, {1.0f},
        teamHasTimeAdvantage, true, false, SearchParams::RuntimeConfig{}));

    const auto rootChildren = root->get_children();
    ASSERT_EQ(rootChildren.size(), 1U);
    const std::shared_ptr<Node>& opponentNode = rootChildren.front();

    Board afterOwnMove(board);
    afterOwnMove.make_moves(ownMove, Stockfish::MOVE_NONE);
    const Stockfish::Move firstReply =
        find_move(afterOwnMove, BOARD_A, "e7e5");
    const Stockfish::Move secondReply =
        find_move(afterOwnMove, BOARD_A, "c7c5");
    ASSERT_NE(firstReply, Stockfish::MOVE_NONE);
    ASSERT_NE(secondReply, Stockfish::MOVE_NONE);
    ASSERT_TRUE(opponentNode->try_init_and_expand(
        {firstReply, secondReply}, {Stockfish::MOVE_NONE},
        {0.75f, 0.25f}, {1.0f}, !teamHasTimeAdvantage, true, true,
        SearchParams::RuntimeConfig{}));

    JointActionCandidate expandedReply;
    int expandedReplyIndex = -1;
    ASSERT_TRUE(opponentNode->expand_next_joint_child(
        nullptr, 0, expandedReply, SearchParams::RuntimeConfig{},
        &expandedReplyIndex));
    ASSERT_EQ(expandedReplyIndex, 1);

    const auto replyNodes = opponentNode->get_children();
    ASSERT_EQ(replyNodes.size(), 2U);
    replyNodes[0]->mark_as_win(5);
    replyNodes[1]->mark_as_win(3);

    Agent agent;
    AgentTreeReuseTestPeer::set_root(agent, root);
    agent.store_next_root_candidates(board, teamHasTimeAdvantage);
    ASSERT_EQ(AgentTreeReuseTestPeer::retained_candidate_count(agent), 3U);

    const JointActionCandidate nonPrincipalReply =
        opponentNode->get_joint_action(1);
    Board actualPosition(afterOwnMove);
    actualPosition.make_moves(
        nonPrincipalReply.moveA, nonPrincipalReply.moveB);

    const std::shared_ptr<Node> reused = agent.try_reuse_tree(
        actualPosition.search_hash_key(
            Stockfish::WHITE, teamHasTimeAdvantage), Stockfish::WHITE,
        Agent::board_signature(actualPosition));
    EXPECT_EQ(reused, replyNodes[1]);
    ASSERT_NE(reused, nullptr);
    EXPECT_EQ(reused->get_node_type(), NodeType::WIN);
    EXPECT_EQ(reused->get_end_in_ply(), 3);
}

TEST_F(EngineTest, TreeReuseIndexesPositionsThreeJointPliesDeep) {
    // Ponder retains one predicted reply. A partner-board exchange while we
    // think - opponent moves, partner answers, opponent moves again - lands
    // three joint plies down, which is where reuse has to still find a node.
    Board board;
    constexpr bool teamHasTimeAdvantage = false;
    const SearchParams::RuntimeConfig config{};
    auto root = std::make_shared<Node>(
        Stockfish::WHITE,
        board.search_hash_key(Stockfish::WHITE, teamHasTimeAdvantage));

    // Our move, the root edge that store_next_root_candidates walks below.
    const Stockfish::Move ownMove = find_move(board, BOARD_A, "e2e4");
    ASSERT_NE(ownMove, Stockfish::MOVE_NONE);
    ASSERT_TRUE(root->try_init_and_expand(
        {ownMove}, {Stockfish::MOVE_NONE}, {1.0f}, {1.0f},
        teamHasTimeAdvantage, true, false, config));
    Board afterOwnMove(board);
    afterOwnMove.make_moves(ownMove, Stockfish::MOVE_NONE);

    // Level 0: the opponents answer on our board and open the partner board.
    const std::shared_ptr<Node> opponentNode = root->get_children().front();
    const Stockfish::Move replyA = find_move(afterOwnMove, BOARD_A, "e7e5");
    const Stockfish::Move replyB = find_move(afterOwnMove, BOARD_B, "d2d4");
    ASSERT_NE(replyA, Stockfish::MOVE_NONE);
    ASSERT_NE(replyB, Stockfish::MOVE_NONE);
    ASSERT_TRUE(opponentNode->try_init_and_expand(
        {replyA}, {replyB}, {1.0f}, {1.0f},
        !teamHasTimeAdvantage, true, true, config));
    const std::shared_ptr<Node> replyNode = opponentNode->get_children().front();
    Board afterReply(afterOwnMove);
    afterReply.make_moves(opponentNode->get_joint_action(0).moveA,
                          opponentNode->get_joint_action(0).moveB);

    // Level 1: our team moves again, our partner included.
    const Stockfish::Move ourA = find_move(afterReply, BOARD_A, "g1f3");
    const Stockfish::Move partnerB = find_move(afterReply, BOARD_B, "d7d5");
    ASSERT_NE(ourA, Stockfish::MOVE_NONE);
    ASSERT_NE(partnerB, Stockfish::MOVE_NONE);
    ASSERT_TRUE(replyNode->try_init_and_expand(
        {ourA}, {partnerB}, {1.0f}, {1.0f},
        teamHasTimeAdvantage, true, true, config));
    replyNode->update(0, 0.0f);
    const std::shared_ptr<Node> ourNode = replyNode->get_children().front();
    Board afterOurs(afterReply);
    afterOurs.make_moves(replyNode->get_joint_action(0).moveA,
                         replyNode->get_joint_action(0).moveB);

    // Level 2: the opponents move once more on both boards.
    const Stockfish::Move deepA = find_move(afterOurs, BOARD_A, "b8c6");
    const Stockfish::Move deepB = find_move(afterOurs, BOARD_B, "c1f4");
    ASSERT_NE(deepA, Stockfish::MOVE_NONE);
    ASSERT_NE(deepB, Stockfish::MOVE_NONE);
    ASSERT_TRUE(ourNode->try_init_and_expand(
        {deepA}, {deepB}, {1.0f}, {1.0f},
        !teamHasTimeAdvantage, true, true, config));
    ourNode->update(0, 0.0f);
    const std::shared_ptr<Node> deepNode = ourNode->get_children().front();
    Board deepPosition(afterOurs);
    deepPosition.make_moves(ourNode->get_joint_action(0).moveA,
                            ourNode->get_joint_action(0).moveB);

    Agent agent;
    AgentTreeReuseTestPeer::set_root(agent, root);
    agent.store_next_root_candidates(board, teamHasTimeAdvantage);
    // Levels 0, 1 and 3. Level 2 has the opposing team on move, so no search
    // can ever be rooted there and it is not worth an index entry.
    EXPECT_EQ(AgentTreeReuseTestPeer::retained_candidate_count(agent), 3U);
    ASSERT_EQ(deepNode->get_team_to_play(), Stockfish::WHITE);

    const std::shared_ptr<Node> reused = agent.try_reuse_tree(
        deepPosition.search_hash_key(
            Stockfish::WHITE, teamHasTimeAdvantage), Stockfish::WHITE,
        Agent::board_signature(deepPosition));
    EXPECT_EQ(reused, deepNode);
}

TEST_F(EngineTest, TreeReuseRejectsUnrelatedPositionAtSameDepth) {
    Board board;
    constexpr bool teamHasTimeAdvantage = false;
    const SearchParams::RuntimeConfig config{};
    auto root = std::make_shared<Node>(
        Stockfish::WHITE,
        board.search_hash_key(Stockfish::WHITE, teamHasTimeAdvantage));

    const Stockfish::Move ownMove = find_move(board, BOARD_A, "e2e4");
    ASSERT_NE(ownMove, Stockfish::MOVE_NONE);
    ASSERT_TRUE(root->try_init_and_expand(
        {ownMove}, {Stockfish::MOVE_NONE}, {1.0f}, {1.0f},
        teamHasTimeAdvantage, true, false, config));

    Agent agent;
    AgentTreeReuseTestPeer::set_root(agent, root);
    agent.store_next_root_candidates(board, teamHasTimeAdvantage);

    // A position the retained walk never reached must not be adopted.
    Board elsewhere(board);
    const Stockfish::Move otherMove = find_move(elsewhere, BOARD_A, "d2d4");
    ASSERT_NE(otherMove, Stockfish::MOVE_NONE);
    elsewhere.make_moves(otherMove, Stockfish::MOVE_NONE);

    EXPECT_EQ(agent.try_reuse_tree(
                  elsewhere.search_hash_key(
                      Stockfish::BLACK, !teamHasTimeAdvantage), Stockfish::BLACK,
                  Agent::board_signature(elsewhere)),
              nullptr);
}

TEST_F(EngineTest, RootMateProofRetainsEveryDefenderContinuation) {
    Board board;
    board.set(
        "r2q3r/ppp5/2n1Npkp/3p4/3P4/2P1P3/P1P2PPP/R2QK1NR[PNpppnbbrq] w KQ - 0 2"
        "|"
        "r1bk4/ppp1npNp/2nb3B/3B2B1/3P4/2P5/P1P2PPP/R2QK2R[bpp] b KQ");

    Agent agent;
    JointActionCandidate rootAction;
    int rootPlyToMate = 0;
    ASSERT_TRUE(AgentTreeReuseTestPeer::find_root_mate_and_retain(
        agent, board, Stockfish::WHITE, true,
        rootAction, rootPlyToMate));
    ASSERT_EQ(board.uci_move(BOARD_A, rootAction.moveA), "d1h5");
    ASSERT_EQ(rootAction.moveB, Stockfish::MOVE_NONE);
    ASSERT_EQ(rootPlyToMate, 9);

    Board afterRoot(board);
    afterRoot.make_moves(rootAction.moveA, rootAction.moveB);
    const std::vector<Stockfish::Move> defenderReplies =
        afterRoot.legal_moves(BOARD_A);
    ASSERT_FALSE(defenderReplies.empty());

    for (Stockfish::Move reply : defenderReplies) {
        Board continuation(afterRoot);
        continuation.push_move(BOARD_A, reply);

        JointActionCandidate cachedAction;
        int cachedPlyToMate = 0;
        ASSERT_TRUE(AgentTreeReuseTestPeer::reuse_mate_continuation(
            agent, continuation, Stockfish::WHITE, true,
            cachedAction, cachedPlyToMate))
            << "missing continuation after "
            << afterRoot.uci_move(BOARD_A, reply);
        EXPECT_GT(cachedPlyToMate, 0);
        EXPECT_LE(cachedPlyToMate, rootPlyToMate - 2);
        EXPECT_NE(cachedAction.moveA, Stockfish::MOVE_NONE);
        EXPECT_EQ(cachedAction.moveB, Stockfish::MOVE_NONE);
        EXPECT_TRUE(continuation.is_legal_move(
            BOARD_A, cachedAction.moveA));
    }
}

TEST_F(EngineTest, RootMateContinuationRejectsPartnerBoardChange) {
    Board board;
    board.set(
        "r2q3r/ppp5/2n1Npkp/3p4/3P4/2P1P3/P1P2PPP/R2QK1NR[PNpppnbbrq] w KQ - 0 2"
        "|"
        "r1bk4/ppp1npNp/2nb3B/3B2B1/3P4/2P5/P1P2PPP/R2QK2R[bpp] b KQ");

    Agent agent;
    JointActionCandidate rootAction;
    int rootPlyToMate = 0;
    ASSERT_TRUE(AgentTreeReuseTestPeer::find_root_mate_and_retain(
        agent, board, Stockfish::WHITE, true,
        rootAction, rootPlyToMate));

    board.make_moves(rootAction.moveA, rootAction.moveB);
    const std::vector<Stockfish::Move> defenderReplies =
        board.legal_moves(BOARD_A);
    ASSERT_FALSE(defenderReplies.empty());
    board.push_move(BOARD_A, defenderReplies.front());

    JointActionCandidate cachedAction;
    int cachedPlyToMate = 0;
    ASSERT_TRUE(AgentTreeReuseTestPeer::reuse_mate_continuation(
        agent, board, Stockfish::WHITE, true,
        cachedAction, cachedPlyToMate));

    const std::vector<Stockfish::Move> partnerMoves =
        board.legal_moves(BOARD_B);
    ASSERT_FALSE(partnerMoves.empty());
    board.push_move(BOARD_B, partnerMoves.front());

    EXPECT_FALSE(AgentTreeReuseTestPeer::reuse_mate_continuation(
        agent, board, Stockfish::WHITE, true,
        cachedAction, cachedPlyToMate));
}

TEST_F(EngineTest, ReusedTreeIsReindexedIntoTranspositionTable) {
    SearchParams::RuntimeConfig config;
    auto root = std::make_shared<Node>(Stockfish::WHITE, 101);
    ASSERT_TRUE(root->try_init_and_expand(
        {static_cast<Stockfish::Move>(1)}, {Stockfish::MOVE_NONE},
        {1.0f}, {1.0f}, false, true, false, config));
    const std::shared_ptr<Node> child = root->get_children().front();
    child->set_hash(202);

    ASSERT_TRUE(child->try_init_and_expand(
        {static_cast<Stockfish::Move>(2)}, {Stockfish::MOVE_NONE},
        {1.0f}, {1.0f}, false, true, false, config));
    const std::shared_ptr<Node> grandchild = child->get_children().front();
    grandchild->set_hash(303);

    Agent agent;
    AgentTreeReuseTestPeer::reindex_reused_subtree(agent, root);

    EXPECT_EQ(AgentTreeReuseTestPeer::lookup(agent, 101), root);
    EXPECT_EQ(AgentTreeReuseTestPeer::lookup(agent, 202), child);
    EXPECT_EQ(AgentTreeReuseTestPeer::lookup(agent, 303), grandchild);
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

    for (const char* uci : {"g1f3", "b8c6", "f3g1", "c6b8"}) {
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

TEST_F(EngineTest, RepetitionHistoryRoundTripsWithSearchMoves) {
    Board board;
    const uint64_t initialHash = board.hash_key(false);
    const size_t initialHistoryCount = board.positionHistory[BOARD_A].size();
    Stockfish::Move move = find_move(board, BOARD_A, "g1f3");
    ASSERT_NE(move, Stockfish::MOVE_NONE);

    board.make_moves(move, Stockfish::MOVE_NONE);
    EXPECT_EQ(board.positionHistory[BOARD_A].size(), initialHistoryCount + 1);
    EXPECT_NE(board.hash_key(false), initialHash);

    board.unmake_moves(move, Stockfish::MOVE_NONE);
    EXPECT_EQ(board.positionHistory[BOARD_A].size(), initialHistoryCount);
    EXPECT_EQ(board.hash_key(false), initialHash);
}

TEST_F(EngineTest, CombinedHashMergesInterleavedCrossBoardOrders) {
    // The sound merge class: each board saw the same moves in the same order,
    // only the interleaving between the boards differs. Same last move, same
    // positions stood on, same repetition future - one node.
    auto play = [](Board& board, int boardNum,
                   std::initializer_list<const char*> ucis) {
        for (const char* uci : ucis) {
            Stockfish::Move move = find_move(board, boardNum, uci);
            ASSERT_NE(move, Stockfish::MOVE_NONE) << uci;
            board.push_move(boardNum, move);
        }
    };

    Board boardFirst;
    play(boardFirst, BOARD_A, {"g1f3", "b8c6"});
    play(boardFirst, BOARD_B, {"e2e4", "e7e5"});

    Board interleaved;
    play(interleaved, BOARD_B, {"e2e4"});
    play(interleaved, BOARD_A, {"g1f3"});
    play(interleaved, BOARD_B, {"e7e5"});
    play(interleaved, BOARD_A, {"b8c6"});

    EXPECT_EQ(boardFirst.hash_key(false), interleaved.hash_key(false));
    EXPECT_EQ(boardFirst.hash_key(true), interleaved.hash_key(true));
}

TEST_F(EngineTest, CombinedHashSeparatesPositionsReachedByDifferentLastMove) {
    // Same placement, different move just played. The network reads the last
    // move as two input planes, so a shared node would serve one path an
    // evaluation computed for the other.
    auto play = [](Board& board, std::initializer_list<const char*> ucis) {
        for (const char* uci : ucis) {
            Stockfish::Move move = find_move(board, BOARD_A, uci);
            ASSERT_NE(move, Stockfish::MOVE_NONE) << uci;
            board.push_move(BOARD_A, move);
        }
    };

    Board endsWithKnightMove;
    play(endsWithKnightMove, {"g1f3", "b8c6", "b1c3", "g8f6"});
    Board endsWithOtherKnightMove;
    play(endsWithOtherKnightMove, {"b1c3", "g8f6", "g1f3", "b8c6"});

    ASSERT_EQ(endsWithKnightMove.board_only_key(BOARD_A),
              endsWithOtherKnightMove.board_only_key(BOARD_A));
    ASSERT_NE(endsWithKnightMove.last_move(BOARD_A),
              endsWithOtherKnightMove.last_move(BOARD_A));
    EXPECT_NE(endsWithKnightMove.hash_key(false),
              endsWithOtherKnightMove.hash_key(false));
}

TEST_F(EngineTest, CombinedHashSeparatesDifferentRepetitionMaps) {
    // The current placement and last move agree, but the intermediate
    // positions visited by the two move orders do not. Merging would make a
    // later repetition transition depend on which parent reached the node.
    auto play = [](Board& board, std::initializer_list<const char*> ucis) {
        for (const char* uci : ucis) {
            Stockfish::Move move = find_move(board, BOARD_A, uci);
            ASSERT_NE(move, Stockfish::MOVE_NONE) << uci;
            board.push_move(BOARD_A, move);
        }
    };

    Board kingsideKnightFirst;
    play(kingsideKnightFirst, {"g1f3", "b8c6", "b1c3", "g8f6"});
    Board queensideKnightFirst;
    play(queensideKnightFirst, {"b1c3", "b8c6", "g1f3", "g8f6"});

    ASSERT_EQ(kingsideKnightFirst.last_move(BOARD_A),
              queensideKnightFirst.last_move(BOARD_A));
    ASSERT_EQ(kingsideKnightFirst.rule50_count(BOARD_A),
              queensideKnightFirst.rule50_count(BOARD_A));
    EXPECT_NE(kingsideKnightFirst.hash_key(false),
              queensideKnightFirst.hash_key(false));
    EXPECT_NE(kingsideKnightFirst.hash_key(true),
              queensideKnightFirst.hash_key(true));
}

TEST_F(EngineTest, CombinedHashSeparatesDifferentRepetitionStatus) {
    // Same placement, same last move, same fifty-move counter - but one has
    // stood here before. The network reads that as an input plane, so these
    // are two nodes.
    auto play = [](Board& board, std::initializer_list<const char*> ucis) {
        for (const char* uci : ucis) {
            Stockfish::Move move = find_move(board, BOARD_A, uci);
            ASSERT_NE(move, Stockfish::MOVE_NONE) << uci;
            board.push_move(BOARD_A, move);
        }
    };

    Board secondVisit;
    play(secondVisit, {"g1f3", "b8c6", "f3g1", "c6b8", "g1f3"});
    Board firstVisit;
    play(firstVisit, {"b1c3", "b8c6", "c3b1", "c6b8", "g1f3"});

    ASSERT_EQ(secondVisit.board_only_key(BOARD_A),
              firstVisit.board_only_key(BOARD_A));
    ASSERT_EQ(secondVisit.last_move(BOARD_A), firstVisit.last_move(BOARD_A));
    ASSERT_EQ(secondVisit.rule50_count(BOARD_A), firstVisit.rule50_count(BOARD_A));
    ASSERT_EQ(secondVisit.repetition_status(BOARD_A), 1);
    ASSERT_EQ(firstVisit.repetition_status(BOARD_A), 0);
    EXPECT_NE(secondVisit.hash_key(false), firstVisit.hash_key(false));
}

TEST_F(EngineTest, RepetitionHashSeparatesDifferentFutureDrawContexts) {
    // Both paths stand on the start position for the second time, with the
    // same current repetition status. Only one has visited Nf3 before. They
    // must not share a graph node because playing Nf3 has different repetition
    // semantics below the two states.
    auto play = [](Board& board, std::initializer_list<const char*> ucis) {
        for (const char* uci : ucis) {
            Stockfish::Move move = find_move(board, BOARD_A, uci);
            ASSERT_NE(move, Stockfish::MOVE_NONE) << uci;
            board.push_move(BOARD_A, move);
        }
    };

    Board loopedThroughNf3;
    play(loopedThroughNf3, {"g1f3", "b8c6", "f3g1", "c6b8"});
    Board loopedThroughNc3;
    play(loopedThroughNc3, {"b1c3", "b8c6", "c3b1", "c6b8"});

    ASSERT_EQ(loopedThroughNf3.repetition_count(BOARD_A), 2);
    ASSERT_EQ(loopedThroughNc3.repetition_count(BOARD_A), 2);
    EXPECT_NE(loopedThroughNf3.hash_key(false), loopedThroughNc3.hash_key(false));

    const std::array<int, 2> inSearch{1, 1};
    play(loopedThroughNf3, {"g1f3"});
    play(loopedThroughNc3, {"g1f3"});
    EXPECT_TRUE(loopedThroughNf3.is_repetition_draw(inSearch));
    EXPECT_FALSE(loopedThroughNc3.is_repetition_draw(inSearch));
    EXPECT_TRUE(loopedThroughNf3.is_draw(inSearch));
    EXPECT_FALSE(loopedThroughNc3.is_draw(inSearch));
}

TEST_F(EngineTest, FiftyMoveDrawStaysProvableBecauseTheCounterIsInTheKey) {
    Board board;
    board.set_fen(BOARD_A, "4k3/8/8/8/8/8/8/4K3 w - - 100 200");
    const std::array<int, 2> inSearch{1, 1};

    EXPECT_TRUE(board.is_draw_on_board(BOARD_A, 1));
    EXPECT_FALSE(board.is_repetition_draw(inSearch));
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

TEST_F(EngineTest, SingleBoardForcedMateOnPosition4) {
    Board board;
    board.set(
        "r2q3r/ppp5/2n1Npkp/3p4/3P4/2P1P3/P1P2PPP/R2QK1NR[PNpppnbbrq] w KQ - 0 2"
        "|"
        "r1bk4/ppp1npNp/2nb3B/3B2B1/3P4/2P5/P1P2PPP/R2QK2R[bpp] b KQ");

    EXPECT_EQ(board.side_to_move(BOARD_A), Stockfish::WHITE);
    EXPECT_EQ(board.side_to_move(BOARD_B), Stockfish::BLACK);

    JointActionCandidate rootMateAction;
    int rootMatePly = 0;
    bool rootFound = Agent::find_root_mate(board, Stockfish::WHITE, true, rootMateAction, rootMatePly);
    EXPECT_TRUE(rootFound);
    EXPECT_NE(rootMateAction.moveA, Stockfish::MOVE_NONE);
    EXPECT_EQ(board.uci_move(BOARD_A, rootMateAction.moveA), "d1h5");
    EXPECT_EQ(rootMateAction.moveB, Stockfish::MOVE_NONE);

    // Test on starting position (should quickly return false)
    Board startBoard;
    JointActionCandidate startMateAction;
    int startMatePly = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    bool startFound = Agent::find_root_mate(startBoard, Stockfish::WHITE, true, startMateAction, startMatePly);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Starting position mate search time: " << ms << " ms, found=" << startFound << std::endl;
    EXPECT_FALSE(startFound);
}

TEST_F(EngineTest, JointForcedMateWithOnlyOneBoardOnTurn) {
    Board board;
    board.set(
        "r6k/6pp/8/8/8/8/5Q2/4KR2[] w - - 0 1"
        "|"
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1");

    ASSERT_EQ(board.side_to_move(BOARD_A), Stockfish::WHITE);
    ASSERT_EQ(board.side_to_move(BOARD_B), Stockfish::WHITE);

    Stockfish::Move queenCheck = Stockfish::MOVE_NONE;
    for (Stockfish::Move move : board.legal_moves(BOARD_A)) {
        if (board.uci_move(BOARD_A, move) == "f2f8") {
            queenCheck = move;
            break;
        }
    }
    ASSERT_NE(queenCheck, Stockfish::MOVE_NONE);
    ASSERT_TRUE(board.gives_check(BOARD_A, queenCheck));

    JointActionCandidate mateAction;
    int matePly = 0;
    EXPECT_TRUE(Agent::find_root_mate(
        board, Stockfish::WHITE, true, mateAction, matePly));
    EXPECT_TRUE(mateAction.moveA != Stockfish::MOVE_NONE
                || mateAction.moveB != Stockfish::MOVE_NONE);
    EXPECT_EQ(board.uci_move(BOARD_A, mateAction.moveA), "f2f8");
    EXPECT_EQ(mateAction.moveB, Stockfish::MOVE_NONE);
    EXPECT_EQ(matePly, 3);
}

TEST_F(EngineTest, JointForcedMateAccountsForCrossBoardDefense) {
    Board board;
    board.set(
        "r6k/6pp/8/8/8/8/5Q2/4KR2[] w - - 0 1"
        "|"
        "7k/8/8/8/8/8/n7/R6K[] w - - 0 1");

    auto find_move = [&](int boardNum, const std::string& uci) {
        for (Stockfish::Move move : board.legal_moves(boardNum)) {
            if (board.uci_move(boardNum, move) == uci) {
                return move;
            }
        }
        return Stockfish::MOVE_NONE;
    };

    const Stockfish::Move queenCheck = find_move(BOARD_A, "f2f8");
    ASSERT_NE(queenCheck, Stockfish::MOVE_NONE);
    board.make_moves(queenCheck, Stockfish::MOVE_NONE);

    const Stockfish::Move forcedRookCapture = find_move(BOARD_A, "a8f8");
    const Stockfish::Move partnerKnightCapture = find_move(BOARD_B, "a1a2");
    ASSERT_NE(forcedRookCapture, Stockfish::MOVE_NONE);
    ASSERT_NE(partnerKnightCapture, Stockfish::MOVE_NONE);
    board.make_moves(forcedRookCapture, partnerKnightCapture);
    EXPECT_EQ(board.count_in_hand(
        BOARD_A, Stockfish::BLACK, Stockfish::KNIGHT), 1);

    const Stockfish::Move rookCheck = find_move(BOARD_A, "f1f8");
    ASSERT_NE(rookCheck, Stockfish::MOVE_NONE);
    board.make_moves(rookCheck, Stockfish::MOVE_NONE);
    EXPECT_FALSE(board.is_checkmate(Stockfish::BLACK, false));
    board.unmake_moves(rookCheck, Stockfish::MOVE_NONE);
    board.unmake_moves(forcedRookCapture, partnerKnightCapture);
    board.unmake_moves(queenCheck, Stockfish::MOVE_NONE);

    JointActionCandidate mateAction;
    int matePly = 0;
    EXPECT_FALSE(Agent::find_root_mate(
        board, Stockfish::WHITE, true, mateAction, matePly));
}

TEST_F(EngineTest, JointForcedMateBudgetExhaustionIsConservative) {
    Board board;
    board.set(
        "r6k/6pp/8/8/8/8/5Q2/4KR2[] w - - 0 1"
        "|"
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] b KQkq - 0 1");

    const uint64_t hashBefore = board.hash_key(false);
    JointActionCandidate mateAction;
    int matePly = 0;
    EXPECT_FALSE(Agent::find_root_mate(
        board, Stockfish::WHITE, false, mateAction, matePly, 1));
    EXPECT_EQ(board.hash_key(false), hashBefore);
}

TEST_F(EngineTest, ImmediateMateIn1DetectedInAllModesAndTurnConfigurations) {
    // 1. Capture Mate in 1 on Board A, Board B not on turn (Team White)
    // Board A has Scholar's mate: 1. Qxf7#
    // Board B is Black to move (not our turn)
    {
        Board board;
        board.set(
            "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
            "|"
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");

        // Test in 'go' mode (teamHasTimeAdvantage = false)
        JointActionCandidate mateGo;
        int plyGo = 0;
        bool foundGo = Agent::find_root_mate(board, Stockfish::WHITE, false, mateGo, plyGo);
        EXPECT_TRUE(foundGo);
        EXPECT_EQ(board.uci_move(BOARD_A, mateGo.moveA), "h5f7");
        EXPECT_EQ(mateGo.moveB, Stockfish::MOVE_NONE);
        EXPECT_EQ(plyGo, 1);

        // Test in 'sit' mode (teamHasTimeAdvantage = true)
        JointActionCandidate mateSit;
        int plySit = 0;
        bool foundSit = Agent::find_root_mate(board, Stockfish::WHITE, true, mateSit, plySit);
        EXPECT_TRUE(foundSit);
        EXPECT_EQ(board.uci_move(BOARD_A, mateSit.moveA), "h5f7");
        EXPECT_EQ(mateSit.moveB, Stockfish::MOVE_NONE);
        EXPECT_EQ(plySit, 1);
    }

    // 2. Mate in 1 on Board B for Team White (White on A, Black on B)
    // Board A is Black to move (not our turn)
    // Board B is Black to move (our turn for partner) with Fool's mate: 1... Qh4#
    {
        Board board;
        board.set(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
            "|"
            "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2");

        JointActionCandidate mateGo;
        int plyGo = 0;
        bool foundGo = Agent::find_root_mate(board, Stockfish::WHITE, false, mateGo, plyGo);
        EXPECT_TRUE(foundGo);
        EXPECT_EQ(mateGo.moveA, Stockfish::MOVE_NONE);
        EXPECT_EQ(board.uci_move(BOARD_B, mateGo.moveB), "d8h4");

        JointActionCandidate mateSit;
        int plySit = 0;
        bool foundSit = Agent::find_root_mate(board, Stockfish::WHITE, true, mateSit, plySit);
        EXPECT_TRUE(foundSit);
        EXPECT_EQ(mateSit.moveA, Stockfish::MOVE_NONE);
        EXPECT_EQ(board.uci_move(BOARD_B, mateSit.moveB), "d8h4");
    }

    // 3. Both boards on turn with non-capture drop mate in 1 on Board A: Q@f7#
    // Board A is White to move (Q@f7#)
    // Board B is Black to move (our turn)
    {
        Board board;
        board.set(
            "r1bqkb1r/ppppp1pp/2n5/8/2B1P3/8/PPPP1PPP/RNBQK1NR[Q] w KQkq - 0 1"
            "|"
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");

        // In 'sit' mode (time advantage): (Q@f7#, pass) is legal and selected
        JointActionCandidate mateSit;
        int plySit = 0;
        bool foundSit = Agent::find_root_mate(board, Stockfish::WHITE, true, mateSit, plySit);
        EXPECT_TRUE(foundSit);
        EXPECT_EQ(board.uci_move(BOARD_A, mateSit.moveA), "Q@f7");
        EXPECT_EQ(mateSit.moveB, Stockfish::MOVE_NONE);

        // In 'go' mode (no time advantage): non-capture on A with pass on B is not legal,
        // so joint action (Q@f7#, legalMoveOnB) is selected and immediately delivers mate
        JointActionCandidate mateGo;
        int plyGo = 0;
        bool foundGo = Agent::find_root_mate(board, Stockfish::WHITE, false, mateGo, plyGo);
        EXPECT_TRUE(foundGo);
        EXPECT_EQ(board.uci_move(BOARD_A, mateGo.moveA), "Q@f7");
        EXPECT_NE(mateGo.moveB, Stockfish::MOVE_NONE);
    }
}

// Regression for Chess.com match 182144901205/182144901207. White has just
// mated on Board A, but its down-time partner still has to move on Board B.
// Captures such as Qxd2 would feed Black a bishop for B@d8, so the engine must
// return a quiet mate-preserving move instead of treating the root as already
// terminal and emitting bestmove (none).
TEST_F(EngineTest, DownTimeTeamMustPreserveMateBeforeRootIsTerminal) {
    Board board;
    board.set(
        "r1k1Q3/ppp3pp/3Pp1np/b2p4/8/2P1P3/PP1Q1PPP/R3K1R1[P] b Q - 0 27|"
        "r4knr/p1p3pp/2p1Pp2/6B1/1b1qp1b1/2N5/PPPBPPBP/R1BK3R[NNNqrbnnp] b - - 3 17");

    ASSERT_TRUE(board.is_checkmate(Stockfish::BLACK, true));
    ASSERT_FALSE(board.legal_moves(Stockfish::WHITE, false).empty());

    JointActionCandidate mateAction;
    int matePly = 0;
    ASSERT_TRUE(Agent::find_root_mate(
        board, Stockfish::WHITE, false, mateAction, matePly));
    EXPECT_EQ(mateAction.moveA, Stockfish::MOVE_NONE);
    ASSERT_NE(mateAction.moveB, Stockfish::MOVE_NONE);
    EXPECT_FALSE(board.is_capture(BOARD_B, mateAction.moveB));
    EXPECT_EQ(matePly, 1);
}

// Regression: is_checkmate() identifies a team by the color it plays on Board A,
// so a mate delivered on Board B must be tested with the attacker's own team id.
// Using ~attackerColor there searched for a mate against our own team, which both
// hid every Board B mate and reported our own losses as forced wins.
TEST_F(EngineTest, SingleBoardForcedMateFoundOnBoardB) {
    // Same forced mate as SingleBoardForcedMateOnPosition4, mirrored onto Board B
    // so that our partner (Black) is the attacker.
    Board board;
    board.set(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        "|"
        "r2qk1nr/p1p2ppp/2p1p3/3p4/3P4/2N1nPKP/PPP5/R2Q3R[pnPPPNBBRQ] b kq - 0 2");

    EXPECT_EQ(board.side_to_move(BOARD_A), Stockfish::WHITE);
    EXPECT_EQ(board.side_to_move(BOARD_B), Stockfish::BLACK);

    JointActionCandidate mateAction;
    int matePly = 0;
    bool found = Agent::find_root_mate(board, Stockfish::WHITE, true, mateAction, matePly);
    EXPECT_TRUE(found);
    EXPECT_EQ(mateAction.moveA, Stockfish::MOVE_NONE);
    EXPECT_EQ(board.uci_move(BOARD_B, mateAction.moveB), "d8h4");
    EXPECT_EQ(matePly, 9);
}

// A quiet capture can be the first move of a mate on the partner board. Both
// queen captures send a black knight to Board B, enabling
// N@f2+ Bxf2 Nxf2# after Black's forced intervening move on Board A.
TEST_F(EngineTest, RootMateScanFindsCrossBoardCaptureFeedMate) {
    Board board;
    board.set(
        "r2q1r2/p1p2pkp/3n1bp1/7n/3P2Qn/7P/PPPp1PP1/4R1K1[QRBNr] w - - 0 1"
        "|"
        "r4q1k/ppp2p1p/2P2Nnp/4p3/8/2N1Bq~1b/PPPPB2P/R1BKR2n~[NPPPpbb] b - - 0 1");
    const std::string originalFenA = board.fen(BOARD_A);
    const std::string originalFenB = board.fen(BOARD_B);

    JointActionCandidate mateAction;
    int matePly = 0;
    ASSERT_TRUE(Agent::find_root_mate(
        board, Stockfish::WHITE, true, mateAction, matePly, 2000));
    ASSERT_NE(mateAction.moveA, Stockfish::MOVE_NONE);
    EXPECT_EQ(mateAction.moveB, Stockfish::MOVE_NONE);
    const std::string feedMove = board.uci_move(BOARD_A, mateAction.moveA);
    EXPECT_EQ(feedMove, "g4h4");
    EXPECT_EQ(matePly, 5);

    // Probing every capture and reply must leave the caller's board untouched.
    EXPECT_EQ(board.fen(BOARD_A), originalFenA);
    EXPECT_EQ(board.fen(BOARD_B), originalFenB);

    // With h4 empty, the equivalent Qxh5 knight feed must also be recognized.
    Board alternate;
    alternate.set(
        "r2q1r2/p1p2pkp/3n1bp1/7n/3P2Q1/7P/PPPp1PP1/4R1K1[QRBNr] w - - 0 1"
        "|"
        "r4q1k/ppp2p1p/2P2Nnp/4p3/8/2N1Bq~1b/PPPPB2P/R1BKR2n~[NPPPpbb] b - - 0 1");
    JointActionCandidate alternateMate;
    int alternateMatePly = 0;
    ASSERT_TRUE(Agent::find_root_mate(
        alternate, Stockfish::WHITE, true,
        alternateMate, alternateMatePly, 2000));
    ASSERT_NE(alternateMate.moveA, Stockfish::MOVE_NONE);
    EXPECT_EQ(alternate.uci_move(BOARD_A, alternateMate.moveA), "g4h5");
    EXPECT_EQ(alternateMate.moveB, Stockfish::MOVE_NONE);
    EXPECT_EQ(alternateMatePly, 5);

    // Swapping the physical boards changes the team id from White to Black,
    // but must not change the proof or let A-to-B candidates starve B-to-A.
    Board swapped;
    swapped.set(
        "r4q1k/ppp2p1p/2P2Nnp/4p3/8/2N1Bq~1b/PPPPB2P/R1BKR2n~[NPPPpbb] b - - 0 1"
        "|"
        "r2q1r2/p1p2pkp/3n1bp1/7n/3P2Qn/7P/PPPp1PP1/4R1K1[QRBNr] w - - 0 1");
    JointActionCandidate swappedMate;
    int swappedMatePly = 0;
    ASSERT_TRUE(Agent::find_root_mate(
        swapped, Stockfish::BLACK, true, swappedMate, swappedMatePly, 2000));
    EXPECT_EQ(swappedMate.moveA, Stockfish::MOVE_NONE);
    ASSERT_NE(swappedMate.moveB, Stockfish::MOVE_NONE);
    EXPECT_EQ(swapped.uci_move(BOARD_B, swappedMate.moveB), "g4h4");
    EXPECT_EQ(swappedMatePly, 5);
}

// A proven capture-feed mate must not make the prepass return before checking
// for a shorter mate that is already available on either board.
TEST_F(EngineTest, RootMateScanPrefersShortestDirectMateOverCaptureFeed) {
    Board queenDropOnA;
    queenDropOnA.set(
        "r1bq1b1r/ppp1k1pp/3npp2/4N1B1/3Q4/2N2N2/PPP2KPP/R6R[NQqbnPPP] w - - 0 1"
        "|"
        "r1b1k2r/ppp2ppp/2p1p3/6B1/B2nn3/2P1P3/P4PPP/R1B1K2R[pPP] b kq - 0 1");
    JointActionCandidate mateOnA;
    int matePlyOnA = 0;
    ASSERT_TRUE(Agent::find_root_mate(
        queenDropOnA, Stockfish::WHITE, true,
        mateOnA, matePlyOnA, 15000));
    ASSERT_NE(mateOnA.moveA, Stockfish::MOVE_NONE);
    EXPECT_EQ(queenDropOnA.uci_move(BOARD_A, mateOnA.moveA), "Q@f7")
        << "reported ply " << matePlyOnA;
    EXPECT_EQ(mateOnA.moveB, Stockfish::MOVE_NONE);
    EXPECT_EQ(matePlyOnA, 5);

    Board queenDropOnB;
    queenDropOnB.set(
        "r2q1rk1/ppp1bpBp/3p4/3B4/3pP3/1B1P4/PPP2PPP/R2b1RK1[NNNnnp] b - - 0 1"
        "|"
        "r2q1r1k/p1pb1pnp/2pbp2N/3n4/3Pp2P/2B1P3/PPP2PP1/R2QK2R[Qp] w KQ - 0 1");
    JointActionCandidate mateOnB;
    int matePlyOnB = 0;
    ASSERT_TRUE(Agent::find_root_mate(
        queenDropOnB, Stockfish::BLACK, true,
        mateOnB, matePlyOnB, 15000));
    EXPECT_EQ(mateOnB.moveA, Stockfish::MOVE_NONE);
    ASSERT_NE(mateOnB.moveB, Stockfish::MOVE_NONE);
    EXPECT_EQ(queenDropOnB.uci_move(BOARD_B, mateOnB.moveB), "Q@g8")
        << "reported ply " << matePlyOnB;
    EXPECT_EQ(matePlyOnB, 3);
}

TEST_F(EngineTest, RootMateScanFindsDeepBishopCaptureFeedMate) {
    Board board;
    board.set(
        "r4r2/1ppbn1pk/p3p2q/3pRn1B/5pP1/1BPP4/P1PB1PPP/1R4K1[QNp] b - - 0 1"
        "|"
        "rnbN1b1r/pppk1Ppp/4pp2/4p3/4B3/8/PPPn1PPP/R3K1NR[QNPqb] w - - 0 1");

    JointActionCandidate mateAction;
    int matePly = 0;
    // Seven plies deep behind a queen capture, so this one needs the budget of
    // an ordinary timed search rather than the minimum allocation a few-hundred
    // node self-play search receives.
    ASSERT_TRUE(Agent::find_root_mate(
        board, Stockfish::BLACK, true,
        mateAction, matePly, SearchParams::MATE_SEARCH_NODE_BUDGET));
    ASSERT_NE(mateAction.moveA, Stockfish::MOVE_NONE);
    EXPECT_EQ(board.uci_move(BOARD_A, mateAction.moveA), "h6h5");
    EXPECT_EQ(mateAction.moveB, Stockfish::MOVE_NONE);
    EXPECT_EQ(matePly, 7);
}

TEST_F(EngineTest, RootForcedLossScanProvesEveryDefenseAndSelectsDelay) {
    Board board;
    board.set(
        "rQ2k2r/p1B1qppp/b1p1pn2/3p4/3P4/2PbPNB1/PR2NPPP/3Q1K1R[Npn] b - - 0 1"
        "|"
        "r1b2k1r/pp3ppp/3p4/1N1Pq3/6n1/B5P1/P1P2PPP/3R1RK1[BPPbnpp] b - - 0 1");
    const std::string originalFenA = board.fen(BOARD_A);
    const std::string originalFenB = board.fen(BOARD_B);

    JointActionCandidate delayingAction;
    int lossPly = 0;
    ASSERT_TRUE(Agent::find_root_forced_loss(
        board, Stockfish::BLACK, false,
        delayingAction, lossPly, 40000));
    EXPECT_NE(delayingAction.moveA, Stockfish::MOVE_NONE);
    EXPECT_EQ(delayingAction.moveB, Stockfish::MOVE_NONE);
    EXPECT_TRUE(board.is_legal_move(BOARD_A, delayingAction.moveA));
    EXPECT_GT(lossPly, 0);
    EXPECT_EQ(board.fen(BOARD_A), originalFenA);
    EXPECT_EQ(board.fen(BOARD_B), originalFenB);

    JointActionCandidate unprovenAction;
    int unprovenPly = 0;
    EXPECT_FALSE(Agent::find_root_forced_loss(
        board, Stockfish::BLACK, false,
        unprovenAction, unprovenPly, 1));
    EXPECT_EQ(board.fen(BOARD_A), originalFenA);
    EXPECT_EQ(board.fen(BOARD_B), originalFenB);

    Board safePosition;
    JointActionCandidate falseLossAction;
    int falseLossPly = 0;
    EXPECT_FALSE(Agent::find_root_forced_loss(
        safePosition, Stockfish::WHITE, true,
        falseLossAction, falseLossPly, 2000));
}

TEST_F(EngineTest, RootMateScanFindsDownTimeCaptureFeedMateInOne) {
    const std::string targetFen =
        "7k/8/8/8/8/8/2PPB3/2BKR3[] b - - 0 1";

    Board board;
    board.set(
        "r2q1r2/p1p2pkp/3n1bp1/7n/3P2Qn/7P/PPPp1PP1/4R1K1[QRBNr] w - - 0 1"
        "|" + targetFen);
    JointActionCandidate mateAction;
    int matePly = 0;
    ASSERT_TRUE(Agent::find_root_mate(
        board, Stockfish::WHITE, false, mateAction, matePly, 2000));
    ASSERT_NE(mateAction.moveA, Stockfish::MOVE_NONE);
    EXPECT_EQ(mateAction.moveB, Stockfish::MOVE_NONE);
    const std::string feedMove = board.uci_move(BOARD_A, mateAction.moveA);
    EXPECT_TRUE(feedMove == "g4h4" || feedMove == "g4h5") << feedMove;
    EXPECT_EQ(matePly, 3);

    Board swapped;
    swapped.set(
        targetFen + "|"
        "r2q1r2/p1p2pkp/3n1bp1/7n/3P2Qn/7P/PPPp1PP1/4R1K1[QRBNr] w - - 0 1");
    JointActionCandidate swappedMate;
    int swappedMatePly = 0;
    ASSERT_TRUE(Agent::find_root_mate(
        swapped, Stockfish::BLACK, false,
        swappedMate, swappedMatePly, 2000));
    EXPECT_EQ(swappedMate.moveA, Stockfish::MOVE_NONE);
    ASSERT_NE(swappedMate.moveB, Stockfish::MOVE_NONE);
    const std::string swappedFeed =
        swapped.uci_move(BOARD_B, swappedMate.moveB);
    EXPECT_TRUE(swappedFeed == "g4h4" || swappedFeed == "g4h5")
        << swappedFeed;
    EXPECT_EQ(swappedMatePly, 3);

    // The same knight feed is not a forced win when the time-ahead opponent
    // can answer by mating the board on which the capture was made.
    Board losingCapture;
    losingCapture.set(
        "3qk3/8/8/8/8/5P2/n2PP2P/R2BKB2[] w - - 0 1"
        "|" + targetFen);
    JointActionCandidate rejectedAction;
    int rejectedPly = 0;
    EXPECT_FALSE(Agent::find_root_mate(
        losingCapture, Stockfish::WHITE, false,
        rejectedAction, rejectedPly, 2000));
}

// Regression: our own team being mated on Board A must never be reported as a
// forced mate for us just because our partner has a checking move on Board B.
TEST_F(EngineTest, RootMateScanDoesNotReportOurOwnLossAsAMate) {
    Board board;
    // Board A: White is in check with no legal move, saved only by the partner's
    // capture on Board B under the time-advantage rule. Board B: Black (our
    // partner) has checking moves but no mate.
    board.set(
        "4k3/8/8/8/8/8/5PPP/4r1K1[] w - - 0 1"
        "|"
        "1r2k3/8/8/3q4/8/8/6N1/R6K[] b - - 0 1");

    ASSERT_TRUE(board.is_in_check(BOARD_A));
    ASSERT_EQ(board.legal_moves(BOARD_A).size(), 0u);
    ASSERT_FALSE(board.is_checkmate(Stockfish::WHITE, true));

    JointActionCandidate mateAction;
    int matePly = 0;
    EXPECT_FALSE(Agent::find_root_mate(board, Stockfish::WHITE, true, mateAction, matePly));
}

// Bughouse scores a team with no legal action as a loss, and a quiet move can
// create that state, so the root scan must not be limited to checking moves.
TEST_F(EngineTest, RootMateScanFindsQuietStalemateWin) {
    Board board;
    // Board A: 1.Kf7 leaves Black without a legal move (h7 is blocked by h6).
    board.set(
        "7k/7p/5K1P/8/8/8/8/8[] w - - 0 1"
        "|"
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");

    JointActionCandidate mateAction;
    int matePly = 0;
    bool found = Agent::find_root_mate(board, Stockfish::WHITE, true, mateAction, matePly);
    EXPECT_TRUE(found);
    EXPECT_EQ(board.uci_move(BOARD_A, mateAction.moveA), "f6f7");
    EXPECT_EQ(matePly, 1);

    // The scan reported a terminal position, so verify it really is one.
    board.push_move(BOARD_A, mateAction.moveA);
    EXPECT_EQ(board.legal_moves(BOARD_A).size(), 0u);
    EXPECT_TRUE(board.is_checkmate(Stockfish::BLACK, false));
    board.pop_move(BOARD_A);
}

// The root scan runs on the calling CPU thread alongside MCTS, so it must stay
// bounded even with an exposed king and a full hand on both boards.
TEST_F(EngineTest, RootMateScanStaysBounded) {
    Board board;
    board.set(
        "6k1/5ppp/8/8/8/8/5PPP/6K1[QRRBBNNPqrrbbnnp] w - - 0 1"
        "|"
        "6k1/5ppp/8/8/8/8/5PPP/6K1[QRRBBNNPqrrbbnnp] b - - 0 1");

    JointActionCandidate mateAction;
    int matePly = 0;
    auto start = std::chrono::steady_clock::now();
    bool found = Agent::find_root_mate(board, Stockfish::WHITE, true, mateAction, matePly);
    double elapsedMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << "Bounded mate scan: found=" << found << " ply=" << matePly
              << " time=" << elapsedMs << " ms" << std::endl;
    EXPECT_LT(elapsedMs, 1000.0);
    if (found) {
        // Deepening reports the shortest proof it can find, not the first one.
        EXPECT_LE(matePly, 9);
        EXPECT_TRUE(mateAction.moveA != Stockfish::MOVE_NONE
                    || mateAction.moveB != Stockfish::MOVE_NONE);
    }
}

// The pre-pass runs inside the move time, so a node budget is not enough on its
// own: joint proof nodes cost two orders of magnitude more than single-board
// ones. This position keeps the joint scan busy for most of a second on its
// full node budget; the deadline has to cut it short regardless.
TEST_F(EngineTest, RootMateScanStopsAtItsDeadline) {
    Board board;
    board.set(
        "4rk2/1pp2ppp/1pp2b2/8/1PP3B1/7P/P1P2PPP/4R1K1[] b - - 0 1"
        "|"
        "r2qnrk1/pp3ppp/2np2Bb/3NpPb1/3n1p2/3BpN1P/PPPB1PPP/R2Q1RK1"
        "[QRNNqrbn] b - - 0 1");

    JointActionCandidate mateAction;
    int matePly = 0;
    const auto start = std::chrono::steady_clock::now();
    const bool found = Agent::find_root_mate(
        board, Stockfish::BLACK, false, mateAction, matePly,
        SearchParams::MATE_SEARCH_NODE_BUDGET,
        start + std::chrono::milliseconds(20));
    const double elapsedMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();

    EXPECT_FALSE(found);
    EXPECT_LT(elapsedMs, 200.0);
}

// The capture-feed allowance is derived from the caller's budget instead of a
// fixed floor, so a caller that scaled its budget down to a short search is not
// billed for the worst case of a long one.
TEST_F(EngineTest, RootMateScanFeedProbesRespectCallerBudget) {
    Board board;
    board.set(
        "r4r2/1ppbn1pk/p3p2q/3pRn1B/5pP1/1BPP4/P1PB1PPP/1R4K1[QNp] b - - 0 1"
        "|"
        "rnbN1b1r/pppk1Ppp/4pp2/4p3/4B3/8/PPPn1PPP/R3K1NR[QNPqb] w - - 0 1");

    JointActionCandidate mateAction;
    int matePly = 0;
    const auto start = std::chrono::steady_clock::now();
    const bool found = Agent::find_root_mate(
        board, Stockfish::BLACK, true, mateAction, matePly,
        SearchParams::MATE_SEARCH_MIN_NODE_BUDGET);
    const double elapsedMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();

    // The same position is proven under a full budget by
    // RootMateScanFindsDeepBishopCaptureFeedMate; at the minimum allocation it
    // must give up quickly rather than run the feed scan to a fixed floor.
    EXPECT_FALSE(found);
    EXPECT_LT(elapsedMs, 50.0);
}

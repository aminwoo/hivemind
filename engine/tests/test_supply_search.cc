#include <gtest/gtest.h>
#include <numeric>
#include "common/utils.h"
#include "search/supply_search.h"

namespace {
using namespace Stockfish;
class SupplySearchTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { init_fairy_stockfish(); }
    const std::string target =
        "r1bq1b1r/ppp1pkpp/2n2n2/3p4/3P4/8/PPP1PPPP/RNBQKB1R[Pp] w KQ - 0 5";
    const std::string feeder =
        "r1bqk2r/ppp1bppp/2n1pn2/1B2N3/3P4/2N5/PPP2PPP/R1BQK2R[Np] b KQkq - 2 7";

    float score(Board& board, int number, Color team, const std::string& uci) {
        auto moves = board.legal_moves(number);
        moves.push_back(MOVE_NONE);
        const auto weights = SupplySearch::weights(board, number, team, moves);
        for (size_t i = 0; i < moves.size(); ++i) {
            if (moves[i] != MOVE_NONE && board.uci_move(number, moves[i]) == uci) {
                return weights.empty() ? 0.0f : weights[i];
            }
        }
        ADD_FAILURE() << "Missing legal move " << uci;
        return -1;
    }
};

TEST_F(SupplySearchTest, FindsQuietPreparationFromArchivedOpeningWithoutMutation) {
    Board board;
    board.set(target + "|" + feeder);
    const auto key = board.search_hash_key(WHITE, true);
    const auto fenA = board.fen(BOARD_A);
    const auto fenB = board.fen(BOARD_B);
    const auto historyA = board.moveHistory[0];
    const auto historyB = board.moveHistory[1];
    EXPECT_GT(score(board, BOARD_B, WHITE, "e7b4"), 0.0f);
    EXPECT_EQ(score(board, BOARD_B, WHITE, "h7h6"), 0.0f);
    EXPECT_EQ(board.search_hash_key(WHITE, true), key);
    EXPECT_EQ(board.moveHistory[0], historyA);
    EXPECT_EQ(board.moveHistory[1], historyB);
    EXPECT_EQ(board.fen(BOARD_A), fenA);
    EXPECT_EQ(board.fen(BOARD_B), fenB);
}

TEST_F(SupplySearchTest, BoardSwapPreservesTeamAndSupplyDirection) {
    Board original, swapped;
    original.set(target + "|" + feeder);
    swapped.set(feeder + "|" + target);
    EXPECT_FLOAT_EQ(score(original, BOARD_B, WHITE, "e7b4"),
                    score(swapped, BOARD_A, BLACK, "e7b4"));
    EXPECT_TRUE(SupplySearch::weights(original, BOARD_B, BLACK,
                                    original.legal_moves(BOARD_B)).empty());
}

TEST_F(SupplySearchTest, QuietOpeningWithShelteredKingsKeepsOriginalPolicy) {
    Board board;
    // The archived feeder has Ne5 attacking f7, so it is not a quiet shield.
    board.set(board.startingFen + "|" + board.startingFen);
    auto actions = board.legal_moves(BOARD_B);
    const auto weights = SupplySearch::weights(board, BOARD_B, BLACK, actions);
    EXPECT_TRUE(weights.empty());
    std::vector<float> policy(actions.size(), 1.0f / actions.size());
    const auto before = policy;
    SupplySearch::mix_policy(policy, weights, 0.2f);
    EXPECT_EQ(policy, before);
}

TEST_F(SupplySearchTest, CapturedPromotionSuppliesPawnRatherThanKnight) {
    Board ordinary, promoted;
    const std::string captureBoard =
        "r1bqk2r/p1p1bppp/2N1pn2/8/1b1P4/2N5/PPP2PPP/R1BQK2R[] b KQkq - 0 9";
    std::string promotedBoard = captureBoard;
    promotedBoard.replace(promotedBoard.find("/2N5/"), 5, "/2N~5/");
    ordinary.set(target + "|" + captureBoard);
    promoted.set(target + "|" + promotedBoard);
    EXPECT_GT(score(ordinary, BOARD_B, WHITE, "b4c3"),
              score(promoted, BOARD_B, WHITE, "b4c3"));
}

TEST_F(SupplySearchTest, RealCheckingDropReceivesFollowupExploration) {
    Board board;
    board.set(
        "r1bq1b1r/ppp1p1pp/2n2nk1/3p2N1/3P4/8/PPP1PPPP/RNBQKB1R[BPbnpp] w KQ - 2 6|"
        + feeder);
    EXPECT_GT(score(board, BOARD_A, WHITE, "B@f7"), 0.0f);
    EXPECT_EQ(score(board, BOARD_A, WHITE, "a2a3"), 0.0f);
}

TEST_F(SupplySearchTest, MixingPreservesMassAndDisabledPolicyExactly) {
    std::vector<float> policy{0.90f, 0.01f, 0.09f};
    const auto before = policy;
    SupplySearch::mix_policy(policy, {0, 3, 0}, 0.0f);
    EXPECT_EQ(policy, before);
    SupplySearch::mix_policy(policy, {0, 3, 0}, 0.2f);
    EXPECT_NEAR(std::accumulate(policy.begin(), policy.end(), 0.0f), 1.0f, 1e-6f);
    EXPECT_GT(policy[1], 0.20f);
    EXPECT_GT(policy[0], 0.0f);
    EXPECT_GT(policy[2], 0.0f);
}

TEST_F(SupplySearchTest, PressureIsBoundedAndChangesSignWithTeam) {
    Board board, swapped;
    board.set(target + "|" + feeder);
    swapped.set(feeder + "|" + target);
    const float attack = SupplySearch::pressure(board, WHITE);
    EXPECT_GT(attack, 0.0f);
    EXPECT_LE(attack, 1.0f);
    EXPECT_FLOAT_EQ(attack, -SupplySearch::pressure(board, BLACK));
    EXPECT_FLOAT_EQ(attack, SupplySearch::pressure(swapped, BLACK));
}

TEST_F(SupplySearchTest, ActualSuppliedKnightRaisesPressureWithoutFabrication) {
    Board emptyHand, knightHand;
    emptyHand.set(target + "|" + feeder);
    std::string withKnight = target;
    withKnight.replace(withKnight.find("[Pp]"), 4, "[PNp]");
    knightHand.set(withKnight + "|" + feeder);
    const auto key = knightHand.hash_key();
    const auto fen = knightHand.fen(BOARD_A);
    EXPECT_GT(SupplySearch::pressure(knightHand, WHITE),
              SupplySearch::pressure(emptyHand, WHITE));
    EXPECT_EQ(knightHand.hash_key(), key);
    EXPECT_EQ(knightHand.fen(BOARD_A), fen);
}

TEST_F(SupplySearchTest, BareKingsHaveNoSupplyPressure) {
    Board board;
    board.set("8/5k2/8/8/8/8/8/K7[] w - - 0 1|"
              "8/8/8/8/8/8/7k/K7[] b - - 0 1");
    EXPECT_FLOAT_EQ(SupplySearch::pressure(board, WHITE), 0.0f);
}

TEST_F(SupplySearchTest, CheckingDropPressureIsBoundedByPieceType) {
    Board restricted, open;
    const std::string knightSupply =
        "4k3/8/8/8/8/8/N7/4K3[] b - - 0 1";
    // The pawns occupy five of the six knight checking-drop squares.
    restricted.set("3p3p/5k2/3p4/4p1p1/8/8/8/K7[] w - - 0 1|"
                   + knightSupply);
    open.set("8/5k2/8/8/8/8/8/K7[] w - - 0 1|" + knightSupply);
    EXPECT_FLOAT_EQ(SupplySearch::pressure(open, WHITE),
                    SupplySearch::pressure(restricted, WHITE));
}

TEST_F(SupplySearchTest, PrioritizesFlightSquareAgainstShieldSacrifice) {
    Board board;
    board.set(
        "rnbqkb1r/ppp1pppp/5n2/3p2N1/3P4/8/PPP1PPPP/RNBQKB1R[] b KQkq - 3 3|"
        "rnbqkb1r/ppp2ppp/3ppn2/8/3PP3/2N5/PPP2PPP/R1BQKBNR[] w KQkq - 2 4");
    const auto key = board.hash_key();
    EXPECT_GT(score(board, BOARD_A, BLACK, "e7e6"),
              score(board, BOARD_A, BLACK, "c8e6"));
    EXPECT_EQ(board.hash_key(), key);
}

TEST_F(SupplySearchTest, ShieldFlightIsBoardAndColorSymmetric) {
    const std::string threatened =
        "rnbqkb1r/ppp1pppp/5n2/3p2N1/3P4/8/PPP1PPPP/RNBQKB1R[] b KQkq - 3 3";
    const std::string mirrored =
        "rnbqkb1r/ppp1pppp/8/3p4/3P2n1/5N2/PPP1PPPP/RNBQKB1R[] w KQkq - 3 3";
    Board original, swapped, colorMirror;
    original.set(threatened + "|" + original.startingFen);
    swapped.set(swapped.startingFen + "|" + threatened);
    colorMirror.set(mirrored + "|" + colorMirror.startingFen);
    const float flight = score(original, BOARD_A, BLACK, "e7e6");
    EXPECT_GT(flight, 0.0f);
    EXPECT_FLOAT_EQ(flight, score(swapped, BOARD_B, WHITE, "e7e6"));
    EXPECT_FLOAT_EQ(flight, score(colorMirror, BOARD_A, WHITE, "e2e3"));
    EXPECT_FLOAT_EQ(score(original, BOARD_A, BLACK, "b8c6"), 0.0f);
}

TEST_F(SupplySearchTest, AlreadyDefendedShieldKeepsOriginalPolicy) {
    Board board;
    // An extra knight on h8 already defends f7; there is no king-only capture.
    board.set(
        "rnbqkb1n/ppp1pppp/5n2/3p2N1/3P4/8/PPP1PPPP/RNBQKB1R[] b KQkq - 3 3|"
        + board.startingFen);
    EXPECT_TRUE(SupplySearch::weights(board, BOARD_A, BLACK,
                                     board.legal_moves(BOARD_A)).empty());
}

TEST_F(SupplySearchTest, ShieldDefenseLeavesSafeKingCaptureAvailable) {
    Board board;
    board.set("4k3/5n2/8/8/8/8/5r2/4K3[] w - - 0 1|" + board.startingFen);
    // There is no potential feed on the local board, nor an attacked shield.
    EXPECT_FLOAT_EQ(score(board, BOARD_A, WHITE, "e1f2"), 0.0f);
}

TEST_F(SupplySearchTest, ColorMirrorPreservesPressureAndPreparation) {
    Board original, mirrored;
    original.set(target + "|" + feeder);
    mirrored.set(
        "rnbqkb1r/ppp1pppp/8/3p4/3P4/2N2N2/PPP1PKPP/R1BQ1B1R[Pp] b kq - 0 5|"
        "r1bqk2r/ppp2ppp/2n5/3p4/1b2n3/2N1PN2/PPP1BPPP/R1BQK2R[Pn] w KQkq - 2 7");
    EXPECT_FLOAT_EQ(SupplySearch::pressure(original, WHITE),
                    SupplySearch::pressure(mirrored, BLACK));
    EXPECT_FLOAT_EQ(score(original, BOARD_B, WHITE, "e7b4"),
                    score(mirrored, BOARD_B, BLACK, "e2b5"));
}

TEST_F(SupplySearchTest, EnPassantAndPromotionCapturesUseActualSuppliedType) {
    Board enPassant, promotion;
    const std::string defendedPawnDrop =
        "8/5k2/8/6N1/8/8/8/K7[] w - - 0 1";
    enPassant.set(defendedPawnDrop + "|"
        "4k3/8/8/8/3Pp3/8/8/4K3[] b - d3 0 1");
    const auto key = enPassant.hash_key();
    EXPECT_GT(score(enPassant, BOARD_B, WHITE, "e4d3"), 0.0f);
    EXPECT_EQ(enPassant.hash_key(), key);
    promotion.set(defendedPawnDrop + "|"
        "4k3/8/8/8/8/8/1p6/R3K3[] b - - 0 1");
    EXPECT_GT(score(promotion, BOARD_B, WHITE, "b2a1q"), 0.0f);
}
} // namespace

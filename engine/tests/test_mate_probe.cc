#include <gtest/gtest.h>

#include <string>

#include "common/globals.h"
#include "environment/board.h"
#include "environment/constants.h"
#include "search/mate_probe.h"

#include "Fairy-Stockfish/src/types.h"

class MateProbeTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        init_fairy_stockfish();
        init_policy_index();
    }
};

// The scanner in agent.cc searches checking moves only, so this mate - which
// needs the quiet Qe3 in the middle of the sequence - is invisible to it.
TEST_F(MateProbeTest, FindsAMateThatNeedsAQuietMove) {
    const std::string fen =
        "r1bq1b1r/ppp1p1pp/2n2nk1/3p2N1/3P4/8/PPP1PPPP/RNBQKB1R[Bb] w KQ - 2 2";

    const MateProbe::Result result = MateProbe::probe(fen, 6, 3000);
    ASSERT_TRUE(result.found);
    EXPECT_GE(result.mateInMoves, 1);
    EXPECT_LE(result.mateInMoves, 6);
    ASSERT_FALSE(result.principalVariation.empty());

    // The move handed back is the one hivemind would play, so it has to be
    // legal on a real board - unlike later plies, which may block with a piece
    // the defender's partner is assumed to supply.
    Board board;
    board.set(fen + "|" + fen);
    ASSERT_NE(result.bestMove, Stockfish::MOVE_NONE);
    EXPECT_TRUE(board.is_legal_move(BOARD_A, result.bestMove));
    EXPECT_EQ(board.uci_move(BOARD_A, result.bestMove), "B@f7");

    EXPECT_EQ(result.principalVariation.front(), "B@f7");
}

// A quiet position with no mate must come back empty rather than guessing.
TEST_F(MateProbeTest, ReportsNothingWithoutAMate) {
    const MateProbe::Result result = MateProbe::probe(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1", 4, 500);
    EXPECT_FALSE(result.found);
    EXPECT_TRUE(result.principalVariation.empty());
}

// The probe holds process-wide Fairy-Stockfish state, so repeated calls have to
// leave it in a state the next probe can use.
TEST_F(MateProbeTest, RepeatedProbesStaySound) {
    const std::string fen =
        "r1bq1b1r/ppp1p1pp/2n2nk1/3p2N1/3P4/8/PPP1PPPP/RNBQKB1R[Bb] w KQ - 2 2";
    for (int attempt = 0; attempt < 3; ++attempt) {
        const MateProbe::Result result = MateProbe::probe(fen, 6, 1000);
        EXPECT_TRUE(result.found) << "attempt " << attempt;
    }
}

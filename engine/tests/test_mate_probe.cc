#include <gtest/gtest.h>

#include <chrono>
#include <string>

#include "common/globals.h"
#include "environment/board.h"
#include "environment/constants.h"
#include "search/mate_probe.h"
#include "search/search_params.h"

#include "Fairy-Stockfish/src/types.h"

// The root probe accepts mates beyond the exact check-only scan's horizon.
// Those answers must not be held until movetime expires by a second, smaller
// early-exit limit (notably mate in 7 and mate in 13 in the Nachos suite).
TEST(MateProbeEarlyExitTest, AcceptsEverySupportedMateDistance) {
    ASSERT_TRUE(SearchParams::ENABLE_MATE_EARLY_EXIT);
    for (int moves = 1; moves <= SearchParams::MATE_PROBE_MAX_MATE_MOVES;
         ++moves) {
        EXPECT_TRUE(SearchParams::mate_probe_can_end_search(2 * moves - 1))
            << "mate in " << moves;
    }
}

TEST(MateProbeEarlyExitTest, RejectsInvalidAndOutOfRangeDistances) {
    EXPECT_FALSE(SearchParams::mate_probe_can_end_search(-1));
    EXPECT_FALSE(SearchParams::mate_probe_can_end_search(0));
    EXPECT_FALSE(SearchParams::mate_probe_can_end_search(
        2 * SearchParams::MATE_PROBE_MAX_MATE_MOVES));
    EXPECT_FALSE(SearchParams::mate_probe_can_end_search(
        2 * (SearchParams::MATE_PROBE_MAX_MATE_MOVES + 1) - 1));
}

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

    // A generous budget the probe should not need: it returns on the first
    // mate it proves rather than spending what is left shortening it.
    const auto started = std::chrono::steady_clock::now();
    const MateProbe::Result result = MateProbe::probe(fen, 16, 8000000, 10000);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started).count();
    ASSERT_TRUE(result.found);
    EXPECT_LT(elapsed, 2000) << "probe kept searching after proving a mate";
    EXPECT_GE(result.mateInMoves, 1);
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
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1", 16, 8000000, 500);
    EXPECT_FALSE(result.found);
    EXPECT_TRUE(result.principalVariation.empty());
}

// The caller's own search runs while the probe does, and the loop that notices
// it has solved the root only resumes once the probe returns. A probe that
// ignored the abort would hold its whole budget past the answer.
TEST_F(MateProbeTest, StopsWhenTheCallerSaysTheAnswerIsIn) {
    const std::string fen =
        "r1bq1b1r/ppp1p1pp/2n2nk1/3p2N1/3P4/8/PPP1PPPP/RNBQKB1R[Bb] w KQ - 2 2";
    const auto started = std::chrono::steady_clock::now();
    const MateProbe::Result result =
        MateProbe::probe(fen, 16, 8000000, 10000, [] { return true; });
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started).count();
    EXPECT_LT(elapsed, 2000) << "probe ran on past its abort";
    (void)result;
}

// The probe holds process-wide Fairy-Stockfish state, so repeated calls have to
// leave it in a state the next probe can use.
TEST_F(MateProbeTest, RepeatedProbesStaySound) {
    const std::string fen =
        "r1bq1b1r/ppp1p1pp/2n2nk1/3p2N1/3P4/8/PPP1PPPP/RNBQKB1R[Bb] w KQ - 2 2";
    for (int attempt = 0; attempt < 3; ++attempt) {
        const MateProbe::Result result = MateProbe::probe(fen, 16, 8000000, 1000);
        EXPECT_TRUE(result.found) << "attempt " << attempt;
    }
}

#include <gtest/gtest.h>

#include <chrono>
#include <string>

#include "common/globals.h"
#include "environment/board.h"
#include "environment/constants.h"
#include "search/mate_probe.h"
#include "search/search_params.h"

#include "Fairy-Stockfish/src/tt.h"
#include "Fairy-Stockfish/src/types.h"
#include "Fairy-Stockfish/src/uci.h"

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

// The worker count is process-wide, so a test that changes it puts it back
// for the tests after it.
class MateThreadsGuard {
public:
    explicit MateThreadsGuard(int count) : previous_(MateProbe::threads()) {
        MateProbe::set_threads(count);
    }
    ~MateThreadsGuard() { MateProbe::set_threads(previous_); }

private:
    int previous_;
};

// The scanner in agent.cc searches checking moves only, so this mate - which
// needs the quiet Qe3 in the middle of the sequence - is invisible to it.
TEST_F(MateProbeTest, FindsAMateThatNeedsAQuietMove) {
    const std::string fen =
        "r1bq1b1r/ppp1p1pp/2n2nk1/3p2N1/3P4/8/PPP1PPPP/RNBQKB1R[Bb] w KQ - 2 2";

    // A generous budget the probe should not need: it returns on the first
    // mate it proves rather than spending what is left shortening it.
    const auto started = std::chrono::steady_clock::now();
    const MateProbe::Result result = MateProbe::probe(
        fen, SearchParams::MATE_PROBE_MAX_MATE_MOVES, 8000000, 10000);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started).count();
    ASSERT_TRUE(result.found);
    EXPECT_LT(elapsed, 2000) << "probe kept searching after proving a mate";
    EXPECT_GE(result.mateInMoves, 1);
    ASSERT_FALSE(result.principalVariation.empty());

    // The move handed back is the one hivemind would play, so it has to be
    // legal on a real board.
    Board board;
    board.set(fen + "|" + fen);
    ASSERT_NE(result.bestMove, Stockfish::MOVE_NONE);
    EXPECT_TRUE(board.is_legal_move(BOARD_A, result.bestMove));
    EXPECT_EQ(board.uci_move(BOARD_A, result.bestMove), "B@f7");

    EXPECT_EQ(result.principalVariation.front(), "B@f7");
}

// chess.com 186251789181 ply 105, board B: a mate in eight or so for Black.
// The search proves it at a depth well short of the mate, its own line
// stopping where the mate score came from the table, so the probe reads the
// rest of the line from the table: whole, and ending in checkmate.
TEST_F(MateProbeTest, HandsBackTheWholeLineOfALongMate) {
    const std::string fen =
        "1Q~Nq3r/p6p/1bB1p3/kN1p4/1b6/2N4P/P1P2PKP/R7[Bppq] b - - 0 31";
    // Threads sharing the table overwrite one another's entries, and earlier
    // probes leave theirs, so which survive varies; one thread on a cleared
    // table makes the test deterministic.
    const int threads = MateProbe::threads();
    MateProbe::set_threads(1);
    Stockfish::TT.clear();
    const MateProbe::Result result = MateProbe::probe(
        fen, SearchParams::MATE_PROBE_MAX_MATE_MOVES, 8000000, 10000);
    MateProbe::set_threads(threads);
    ASSERT_TRUE(result.found);
    ASSERT_GE(result.mateInMoves, 6);

    Board board;
    board.set(fen + "|" + fen);
    for (const std::string& text : result.principalVariation) {
        std::string moveText = text;
        const Stockfish::Move move =
            Stockfish::UCI::to_move(*board.pos[BOARD_A], moveText);
        ASSERT_TRUE(move != Stockfish::MOVE_NONE
                    && board.is_legal_move(BOARD_A, move)) << text;
        board.push_move(BOARD_A, move);
    }
    EXPECT_EQ(result.principalVariation.size(),
              static_cast<size_t>(2 * result.mateInMoves - 1))
        << result.tablePlies << " plies from the table";
    EXPECT_TRUE(board.is_in_check(BOARD_A)
                && board.legal_moves(BOARD_A).empty())
        << "the line does not end in checkmate";
}

// The probe searches one board with the other sitting, and a partner who
// sits feeds nobody. Fairy-Stockfish's bughouse variant would otherwise let
// the defender block Ra8+ with a piece their partner is assumed to capture -
// and score the position as a "virtual" mate the probe does not report.
TEST_F(MateProbeTest, DoesNotLetTheDefenderDropAPieceTheyDoNotHold) {
    const std::string fen = "6k1/5ppp/8/8/8/8/8/R5K1[] w - - 0 1";
    const MateProbe::Result result = MateProbe::probe(
        fen, SearchParams::MATE_PROBE_MAX_MATE_MOVES, 8000000, 2000);
    ASSERT_TRUE(result.found);
    EXPECT_EQ(result.mateInMoves, 1);

    Board board;
    board.set(fen + "|" + fen);
    EXPECT_EQ(board.uci_move(BOARD_A, result.bestMove), "a1a8");
    ASSERT_EQ(result.principalVariation.size(), 1u);
    EXPECT_EQ(result.principalVariation.front(), "a1a8");
}

// A blocker the defender really holds still counts.
TEST_F(MateProbeTest, SeesABlockTheDefenderCanActuallyPlay) {
    const MateProbe::Result result = MateProbe::probe(
        "6k1/5ppp/8/8/8/8/8/R5K1[n] w - - 0 1",
        SearchParams::MATE_PROBE_MAX_MATE_MOVES, 200000, 1000);
    EXPECT_FALSE(result.found && result.mateInMoves == 1)
        << "Ra8+ is met by a knight drop";
}

// Fairy-Stockfish must preserve bughouse's stalemate-as-loss score. Whether
// this local stalemate ends the two-board game is decided by the caller.
TEST_F(MateProbeTest, ReportsSingleBoardStalemateForTwoBoardValidation) {
    const MateProbe::Result result = MateProbe::probe(
        "7k/7p/5K1P/8/8/8/8/8[] w - - 0 1", 16, 200000, 1000);
    ASSERT_TRUE(result.found);
    EXPECT_EQ(result.mateInMoves, 1);
    ASSERT_FALSE(result.principalVariation.empty());
    EXPECT_EQ(result.principalVariation.front(), "f6f7");
}

// Both Ra8# and Kf7 stalemate immediately. Fairy-Stockfish assigns those the
// same bughouse score, so the probe's root ordering supplies the tie-break.
TEST_F(MateProbeTest, PrefersCheckmateOverStalemate) {
    const std::string fen = "7k/7p/5K1P/8/8/8/8/R7[] w - - 0 1";
    const MateProbe::Result result = MateProbe::probe(
        fen, 16, 200000, 1000);
    ASSERT_TRUE(result.found);

    Board board;
    board.set(fen + "|" + fen);
    EXPECT_EQ(board.uci_move(BOARD_A, result.bestMove), "a1a8");
    board.push_move(BOARD_A, result.bestMove);
    EXPECT_TRUE(board.is_in_check(BOARD_A));
    EXPECT_TRUE(board.legal_moves(BOARD_A).empty());
    board.pop_move(BOARD_A);
}

// Every ply the probe hands back is a move of the probed board alone, so the
// caller can replay the whole line against the two-board model.
TEST_F(MateProbeTest, EveryPlyOfTheLineIsPlayable) {
    const std::string fen =
        "r1bq1b1r/ppp1p1pp/2n2nk1/3p2N1/3P4/8/PPP1PPPP/RNBQKB1R[Bb] w KQ - 2 2";
    const MateProbe::Result result = MateProbe::probe(
        fen, SearchParams::MATE_PROBE_MAX_MATE_MOVES, 8000000, 10000);
    ASSERT_TRUE(result.found);
    Board board;
    board.set(fen + "|" + fen);
    for (const std::string& moveText : result.principalVariation) {
        std::string parsed = moveText;
        const Stockfish::Move move = Stockfish::UCI::to_move(
            *board.pos[BOARD_A], parsed);
        ASSERT_NE(move, Stockfish::MOVE_NONE) << moveText;
        ASSERT_TRUE(board.is_legal_move(BOARD_A, move)) << moveText;
        board.push_move(BOARD_A, move);
    }
}

TEST_F(MateProbeTest, SupportsNodeOnlyBudget) {
    const std::string fen =
        "r1bq1b1r/ppp1p1pp/2n2nk1/3p2N1/3P4/8/PPP1PPPP/RNBQKB1R[Bb] w KQ - 2 2";

    const MateProbe::Result result = MateProbe::probe(
        fen, SearchParams::MATE_PROBE_MAX_MATE_MOVES,
        SearchParams::MATE_PROBE_ROOT_NODE_BUDGET, 0);

    ASSERT_TRUE(result.found);
    EXPECT_GT(result.nodes, 0);
    EXPECT_NE(result.bestMove, Stockfish::MOVE_NONE);
}

// A quiet position with no mate must come back empty rather than guessing.
TEST_F(MateProbeTest, ReportsNothingWithoutAMate) {
    const MateProbe::Result result = MateProbe::probe(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1",
        SearchParams::MATE_PROBE_MAX_MATE_MOVES, 8000000, 500);
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
    const MateProbe::Result result = MateProbe::probe(
        fen, SearchParams::MATE_PROBE_MAX_MATE_MOVES, 8000000, 10000,
        [] { return true; });
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
        const MateProbe::Result result = MateProbe::probe(
        fen, SearchParams::MATE_PROBE_MAX_MATE_MOVES, 8000000, 1000);
        EXPECT_TRUE(result.found) << "attempt " << attempt;
    }
}

// The Fairy-Stockfish worker is a helper thread, so it never raises
// Threads.stop on its own: once rootDepth reaches MAX_PLY it simply returns
// to its idle loop. A side that is being mated gets there in a few thousand
// nodes, since mate-distance pruning collapses every iteration. With no
// deadline and no abort - self-play's complete-probe mode - the poll loop used
// to spin until the process was killed.
TEST_F(MateProbeTest, ReturnsWhenTheSearchExhaustsItsDepth) {
    const auto started = std::chrono::steady_clock::now();
    const MateProbe::Result result = MateProbe::probe(
        "8/8/8/8/8/1qk5/7P/K7[] w - - 0 1",
        SearchParams::MATE_PROBE_MAX_MATE_MOVES, 8000000, 0);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started).count();
    EXPECT_FALSE(result.found);
    EXPECT_LT(result.nodes, 100000u);
    EXPECT_LT(elapsed, 5000) << "probe kept polling after the search returned";
}

TEST_F(MateProbeTest, SetThreadsResizesThePoolBothWays) {
    MateThreadsGuard guard(4);
    EXPECT_EQ(MateProbe::threads(), 4);
    MateProbe::set_threads(2);
    EXPECT_EQ(MateProbe::threads(), 2);
    MateProbe::set_threads(0);
    EXPECT_EQ(MateProbe::threads(), 1);
}

TEST_F(MateProbeTest, SeveralWorkersFindTheQuietMate) {
    MateThreadsGuard guard(4);
    const std::string fen =
        "r1bq1b1r/ppp1p1pp/2n2nk1/3p2N1/3P4/8/PPP1PPPP/RNBQKB1R[Bb] w KQ - 2 2";
    for (int attempt = 0; attempt < 3; ++attempt) {
        const auto started = std::chrono::steady_clock::now();
        const MateProbe::Result result = MateProbe::probe(
            fen, SearchParams::MATE_PROBE_MAX_MATE_MOVES, 8000000, 10000);
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - started).count();
        ASSERT_TRUE(result.found) << "attempt " << attempt;
        EXPECT_LT(elapsed, 2000) << "probe kept searching after proving a mate";

        Board board;
        board.set(fen + "|" + fen);
        EXPECT_EQ(board.uci_move(BOARD_A, result.bestMove), "B@f7");
        ASSERT_FALSE(result.principalVariation.empty());
        EXPECT_EQ(result.principalVariation.front(), "B@f7");
    }
}

// The node budget is what the workers spend between them, not each.
TEST_F(MateProbeTest, NodeBudgetCoversEveryWorker) {
    MateThreadsGuard guard(4);
    const uint64_t nodeBudget = 400000;
    const MateProbe::Result result = MateProbe::probe(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1",
        SearchParams::MATE_PROBE_MAX_MATE_MOVES, nodeBudget, 0);
    EXPECT_FALSE(result.found);
    EXPECT_GE(result.nodes, nodeBudget);
    EXPECT_LT(result.nodes, 2 * nodeBudget);
}

// One worker running out of depth stops the rest rather than leaving the poll
// waiting on them.
TEST_F(MateProbeTest, SeveralWorkersReturnWhenTheSearchExhaustsItsDepth) {
    MateThreadsGuard guard(4);
    const auto started = std::chrono::steady_clock::now();
    const MateProbe::Result result = MateProbe::probe(
        "8/8/8/8/8/1qk5/7P/K7[] w - - 0 1",
        SearchParams::MATE_PROBE_MAX_MATE_MOVES, 8000000, 0);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started).count();
    EXPECT_FALSE(result.found);
    EXPECT_LT(elapsed, 5000) << "probe kept polling after the search returned";
}

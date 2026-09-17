#include <gtest/gtest.h>

#include <atomic>

#include <string>
#include <vector>

#include "common/globals.h"
#include "environment/board.h"
#include "environment/constants.h"
#include "environment/joint_action.h"
#include "search/agent.h"
#include "search/mate_probe.h"
#include "search/searchthread.h"
#include "Fairy-Stockfish/src/uci.h"

namespace {

// Board A of a lost game: White is in check from a rook drop the knight on e2
// guards, so Rxg1 is the only legal move there. It is also the capture that
// feeds the rook mating the other board with R@c1 - which is why the search
// called the whole thing a mate in two. Playing it hands board A back to Black,
// who mates at once with Nxf2, and no second move of ours is ever made.
constexpr const char* kRaceBoardA =
    "4Q~bk1/pPp1pppp/3p4/3B4/3P1rn1/B1P5/P1P1nPPP/R4RrK"
    "[QQRBNNPPPqbbnnpp] w - - 5 25";
constexpr const char* kRaceBoardB =
    "2kr3r/ppp2ppp/4p3/6q1/1N2P3/2BP4/4RbPP/3K1N2[] b - - 1 44";

// The same two boards one joint action on: the capture is played, the rook it
// fed is in hand on board B, and Black is to move on board A.
constexpr const char* kFedBoardA =
    "4Q~bk1/pPp1pppp/3p4/3B4/3P1rn1/B1P5/P1P1nPPP/R5RK"
    "[QQRBNNPPPqbbnnpp] b - - 0 25";
constexpr const char* kFedBoardB =
    "2kr3r/ppp2ppp/4p3/6q1/1N2P3/2BP4/4RbPP/3K1N2[r] b - - 1 44";

// kFedBoardA without the knight on g4, which is the piece that mates on f2.
// Everything else about the race is identical, so the waiting mate on board B
// now survives every reply and the same classifier has to say so.
constexpr const char* kFedBoardANoMate =
    "4Q~bk1/pPp1pppp/3p4/3B4/3P1r2/B1P5/P1P1nPPP/R5RK"
    "[QQRBNNPPPqbbnnpp] b - - 0 25";

Stockfish::Move move_of(Board& board, int boardNum, std::string uci) {
    return Stockfish::UCI::to_move(*board.pos[boardNum], uci);
}

JointActionCandidate joint(Stockfish::Move moveA, Stockfish::Move moveB) {
    JointActionCandidate action;
    action.moveA = moveA;
    action.moveB = moveB;
    return action;
}

}  // namespace

class MateRaceTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        init_fairy_stockfish();
        init_policy_index();
    }
};

TEST_F(MateRaceTest, CaptureFeedThatUnfreezesAMatingBoardLosesTheRace) {
    Board board;
    board.set_fen(BOARD_A, kRaceBoardA);
    board.set_fen(BOARD_B, kRaceBoardB);

    const Stockfish::Move capture = move_of(board, BOARD_A, "f1g1");
    ASSERT_NE(capture, Stockfish::MOVE_NONE);
    // Answering the check at all means making this move: there is no other.
    ASSERT_EQ(board.legal_moves(BOARD_A).size(), 1u);

    const std::string signatureBefore = Agent::board_signature(board);
    EXPECT_TRUE(Agent::action_loses_mate_race(
        board, joint(capture, Stockfish::MOVE_NONE),
        Stockfish::WHITE, true));
    EXPECT_EQ(Agent::board_signature(board), signatureBefore);
}

TEST_F(MateRaceTest, SittingOnTheCheckedBoardKeepsTheOpponentFrozen) {
    Board board;
    board.set_fen(BOARD_A, kRaceBoardA);
    board.set_fen(BOARD_B, kRaceBoardB);

    // Board A stays ours to move, so Black never gets to play Nxf2 there.
    const Stockfish::Move quiet = move_of(board, BOARD_B, "f2c5");
    ASSERT_NE(quiet, Stockfish::MOVE_NONE);
    EXPECT_FALSE(Agent::action_loses_mate_race(
        board, joint(Stockfish::MOVE_NONE, quiet), Stockfish::WHITE, true));
}

TEST_F(MateRaceTest, OpponentsWaitingUnblockableMateVetoesOurMate) {
    Board board;
    board.set_fen(BOARD_A, kRaceBoardA);
    board.set_fen(BOARD_B, kRaceBoardB);

    // One move on from the position above, both teams hold mate in one. The
    // opponent's unblockable Nxf2 is treated as instantaneous, so our R@c1
    // does not win the race.
    board.push_move(BOARD_A, move_of(board, BOARD_A, "f1g1"));
    const Stockfish::Move drop = move_of(board, BOARD_B, "R@c1");
    ASSERT_NE(drop, Stockfish::MOVE_NONE);
    EXPECT_TRUE(Agent::action_loses_mate_race(
        board, joint(Stockfish::MOVE_NONE, drop), Stockfish::WHITE, true));
}

TEST_F(MateRaceTest, UnblockableMateBeatsOpponentsBlockableMateInOne) {
    Board board;
    board.set_fen(BOARD_A, "6rk/6pp/8/4N3/8/8/8/K7[] w - - 0 1");
    board.set_fen(
        BOARD_B, "6k1/5ppp/8/8/8/8/8/R5K1[] w - - 0 1");

    JointActionCandidate opponentMate;
    int opponentPlyToMate = 0;
    ASSERT_TRUE(Agent::find_root_mate(
        board, Stockfish::BLACK, true, opponentMate, opponentPlyToMate));
    EXPECT_EQ(board.uci_move(BOARD_B, opponentMate.moveB), "a1a8");
    board.push_move(BOARD_B, opponentMate.moveB);
    EXPECT_TRUE(board.can_partner_provide_blocking_piece(
        BOARD_B, Stockfish::BLACK, false, true));
    EXPECT_FALSE(board.can_partner_provide_blocking_piece(
        BOARD_B, Stockfish::BLACK, false));
    EXPECT_TRUE(board.is_checkmate(Stockfish::WHITE, false));
    board.pop_move(BOARD_B);

    const Stockfish::Move ourMate = move_of(board, BOARD_A, "e5f7");
    ASSERT_NE(ourMate, Stockfish::MOVE_NONE);
    EXPECT_FALSE(Agent::action_loses_mate_race(
        board, joint(ourMate, Stockfish::MOVE_NONE),
        Stockfish::WHITE, false));
}

TEST_F(MateRaceTest, OpponentsUnblockableMateInOneAlwaysWinsTheRace) {
    Board board;
    constexpr const char* smotheredMateInOne =
        "6rk/6pp/8/4N3/8/8/8/K7[] w - - 0 1";
    board.set_fen(BOARD_A, smotheredMateInOne);
    board.set_fen(BOARD_B, smotheredMateInOne);

    const Stockfish::Move ourMate = move_of(board, BOARD_A, "e5f7");
    ASSERT_NE(ourMate, Stockfish::MOVE_NONE);
    EXPECT_TRUE(Agent::action_loses_mate_race(
        board, joint(ourMate, Stockfish::MOVE_NONE),
        Stockfish::WHITE, false));
    EXPECT_TRUE(Agent::action_loses_mate_race(
        board, joint(ourMate, Stockfish::MOVE_NONE),
        Stockfish::WHITE, true));
}

TEST_F(MateRaceTest, OrdinaryPositionsAreNotVetoed) {
    Board board;
    const Stockfish::Move opening = move_of(board, BOARD_A, "e2e4");
    ASSERT_NE(opening, Stockfish::MOVE_NONE);
    EXPECT_FALSE(Agent::action_loses_mate_race(
        board, joint(opening, Stockfish::MOVE_NONE), Stockfish::WHITE, false));
}

// The root veto is the last word, not the only one: a search that reaches this
// line from further up has to value it the same way, or the root would follow
// an inflated score into it without ever claiming a win.
TEST_F(MateRaceTest, InteriorNodeDoesNotAwardTheWaitingMateItLosesTheRaceFor) {
    Board board;
    board.set_fen(BOARD_A, kFedBoardA);
    board.set_fen(BOARD_B, kFedBoardB);

    // R@c1 mates on the waiting board, and Black must move on board A, so the
    // shape is exactly the one the waiting-board classifier answers - but the
    // move Black makes there is Nxf2, which mates first.
    int endInPly = 0;
    EXPECT_EQ(classify_terminal_position(
                  board, Stockfish::BLACK, Stockfish::WHITE, true,
                  std::array<int, 2>{1, 1}, &endInPly),
              TerminalOutcome::NONE);
}

TEST_F(MateRaceTest, InteriorNodeStillAwardsAWaitingMateThatSurvives) {
    Board board;
    board.set_fen(BOARD_A, kFedBoardANoMate);
    board.set_fen(BOARD_B, kFedBoardB);

    int endInPly = 0;
    WaitingMateContinuation continuation;
    EXPECT_EQ(classify_terminal_position(
                  board, Stockfish::BLACK, Stockfish::WHITE, true,
                  std::array<int, 2>{1, 1}, &endInPly, false, false,
                  &continuation),
              TerminalOutcome::LOSS);
    // Forced reply on board A, then the mate on board B.
    EXPECT_EQ(endInPly, 3);
    EXPECT_EQ(continuation.waitingBoard, BOARD_B);
    EXPECT_EQ(board.uci_move(BOARD_B, continuation.matingMove), "R@c1");
}

TEST_F(MateRaceTest, SpentBudgetRaisesNoVeto) {
    Board board;
    board.set_fen(BOARD_A, kRaceBoardA);
    board.set_fen(BOARD_B, kRaceBoardB);

    Agent::MateSearchBudget budget;
    budget.remainingNodes = 0;
    budget.exhausted = true;
    EXPECT_FALSE(Agent::action_loses_mate_race(
        board, joint(move_of(board, BOARD_A, "f1g1"), Stockfish::MOVE_NONE),
        Stockfish::WHITE, true, &budget));
}

// chess.com 183478110885/183478110887, our team white: board A is the
// opponent's to move, they are sitting there with Q@e1# ready the moment a
// queen reaches their hand. On board B we hold a queen in hand and the probe
// sees "Qxh2+ Kxh2 Q@h4#". Kxh2 puts our queen in the A-side's hand, so
// Q@e1# is played before the second move of that line - and Q@h4 was never a
// mate anyway: the A-side captures at once and their partner blocks on h3.
constexpr const char* kQueenFeedBoardA =
    "r4rk1/pPB2ppp/1nP1pn2/3p3b/3P4/2N1PP2/PPP2PnP/R1BK2NR"
    "[QRBPbbnppp] b - - 3 16";
constexpr const char* kQueenFeedBoardB =
    "r3k2r/1pp2pp1/4p2P/pP1p4/3b3q/P1N5/2P3PN/R1BQ3K[q] b kq - 1 27";

class QueenFeedRaceTest : public MateRaceTest {
protected:
    void SetUp() override {
        board.set_fen(BOARD_A, kQueenFeedBoardA);
        board.set_fen(BOARD_B, kQueenFeedBoardB);
    }

    void play_capture_and_recapture() {
        board.push_move(BOARD_B, move_of(board, BOARD_B, "h4h2"));
        board.push_move(BOARD_B, move_of(board, BOARD_B, "h1h2"));
    }

    Board board;
};

TEST_F(QueenFeedRaceTest, ACheckWhoseRecaptureFeedsAMateLosesTheRace) {
    const Stockfish::Move check = move_of(board, BOARD_B, "h4h2");
    ASSERT_NE(check, Stockfish::MOVE_NONE);
    for (bool timeAdvantage : {false, true}) {
        EXPECT_TRUE(Agent::action_loses_mate_race(
            board, joint(Stockfish::MOVE_NONE, check),
            Stockfish::WHITE, timeAdvantage))
            << "time advantage " << timeAdvantage;
    }
}

TEST_F(QueenFeedRaceTest, ABlockableCheckIsNotMateWhileThePartnerCanCapture) {
    play_capture_and_recapture();
    board.push_move(BOARD_B, move_of(board, BOARD_B, "Q@h4"));
    for (bool timeAdvantage : {false, true}) {
        EXPECT_FALSE(board.is_checkmate(Stockfish::BLACK, !timeAdvantage));
    }
}

TEST_F(QueenFeedRaceTest, TheProbeDoesNotClaimEitherMate) {
    for (bool timeAdvantage : {false, true}) {
        JointActionCandidate action;
        int plyToMate = 0;
        std::string pv;
        EXPECT_FALSE(Agent::probe_root_mate(
            board, Stockfish::WHITE, timeAdvantage, 200000, 2000, {},
            action, plyToMate, pv))
            << "claimed " << pv << " with time advantage " << timeAdvantage;
    }

    play_capture_and_recapture();
    for (bool timeAdvantage : {false, true}) {
        JointActionCandidate action;
        int plyToMate = 0;
        std::string pv;
        EXPECT_FALSE(Agent::probe_root_mate(
            board, Stockfish::WHITE, timeAdvantage, 200000, 2000, {},
            action, plyToMate, pv))
            << "claimed " << pv << " with time advantage " << timeAdvantage;
    }
}

// A single-board mate line is only playable when the partner board may sit
// through it, and only a team ahead on time may do that: behind, the
// defender is the one who gets to sit after the first check and play the
// other board instead. So a down-time root gets nothing from the probe, even
// where the board on its own is a plain mate in one.
TEST_F(MateRaceTest, TheProbeIsForTheTimeAheadTeamOnly) {
    Board board;
    board.set_fen(BOARD_A, "6k1/5ppp/8/8/8/8/8/R5K1[] w - - 0 1");
    board.set_fen(BOARD_B, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1");

    JointActionCandidate action;
    int plyToMate = 0;
    std::string pv;
    EXPECT_FALSE(Agent::probe_root_mate(
        board, Stockfish::WHITE, false, 200000, 2000, {},
        action, plyToMate, pv))
        << "claimed " << pv << " without the time advantage";

    ASSERT_TRUE(Agent::probe_root_mate(
        board, Stockfish::WHITE, true, 200000, 2000, {},
        action, plyToMate, pv));
    EXPECT_EQ(board.uci_move(BOARD_A, action.moveA), "a1a8");
    EXPECT_EQ(action.moveB, Stockfish::MOVE_NONE);
    EXPECT_EQ(plyToMate, 1);
}

// A time-ahead attacker can sit through the defending partner's quiet moves.
// Only a capture that supplies a legal drop can unfreeze the stalemated board.
TEST_F(MateRaceTest, ProbeValidatesStalemateAgainstPartnerCaptures) {
    constexpr const char* stalemateBoard =
        "7k/7p/5K1P/8/8/8/8/8[] w - - 0 1";
    constexpr const char* otherBoard =
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[]";

    auto probe = [&](const std::string& otherSideToMove,
                     JointActionCandidate& action) {
        Board board;
        board.set_fen(BOARD_A, stalemateBoard);
        board.set_fen(
            BOARD_B, std::string(otherBoard) + " " + otherSideToMove
                + " KQkq - 0 1");
        int plyToMate = 0;
        std::string pv;
        const bool found = Agent::probe_root_mate(
            board, Stockfish::WHITE, true, 200000, 2000, {},
            action, plyToMate, pv);
        if (found) {
            EXPECT_EQ(board.uci_move(BOARD_A, action.moveA), "f6f7");
            EXPECT_EQ(plyToMate, 1);
        }
        return found;
    };

    JointActionCandidate action;
    EXPECT_TRUE(probe("b", action));
    EXPECT_TRUE(probe("w", action));

    Board board;
    board.set_fen(BOARD_A, stalemateBoard);
    board.set_fen(BOARD_B, "7k/8/8/8/8/8/r7/R6K[] w - - 0 1");
    int plyToMate = 0;
    std::string pv;
    EXPECT_FALSE(Agent::probe_root_mate(
        board, Stockfish::WHITE, true, 200000, 2000, {},
        action, plyToMate, pv));
}

// Board A has Kf7 stalemate and board B has Ra1 checkmate, both in one. The
// probe visits A first, so choosing board B verifies that the two-board result
// selection prefers checkmate rather than keeping the first equal-distance win.
TEST_F(MateRaceTest, ProbePrefersCheckmateOverStalemateAcrossBoards) {
    Board board;
    board.set_fen(
        BOARD_A, "7k/7p/5K1P/8/8/8/8/8[] w - - 0 1");
    board.set_fen(
        BOARD_B, "r5k1/8/8/8/8/8/5PPP/6K1[] b - - 0 1");

    JointActionCandidate action;
    int plyToMate = 0;
    std::string pv;
    std::vector<std::string> published;
    ASSERT_TRUE(Agent::probe_root_mate(
        board, Stockfish::WHITE, true, 400000, 2000, {},
        action, plyToMate, pv,
        [&] {
            published.push_back(action.moveB == Stockfish::MOVE_NONE
                ? board.uci_move(BOARD_A, action.moveA)
                : board.uci_move(BOARD_B, action.moveB));
        }));
    EXPECT_EQ(action.moveA, Stockfish::MOVE_NONE);
    ASSERT_NE(action.moveB, Stockfish::MOVE_NONE);
    EXPECT_EQ(board.uci_move(BOARD_B, action.moveB), "a8a1");
    EXPECT_EQ(plyToMate, 1);
    EXPECT_EQ(published, std::vector<std::string>{"a8a1"});
}

// The exact mate-in-one preflight is what normally decides this root before
// the background probe finishes. It must apply the same preference across
// boards rather than returning the first board-A stalemate it encounters.
TEST_F(MateRaceTest, ExactRootScanPrefersCheckmateOverStalemateAcrossBoards) {
    Board board;
    board.set_fen(
        BOARD_A, "7k/7p/5K1P/8/8/8/8/8[] w - - 0 1");
    board.set_fen(
        BOARD_B, "r5k1/8/8/8/8/8/5PPP/6K1[] b - - 0 1");

    JointActionCandidate action;
    int plyToMate = 0;
    ASSERT_TRUE(Agent::find_root_mate(
        board, Stockfish::WHITE, true, action, plyToMate));
    EXPECT_EQ(action.moveA, Stockfish::MOVE_NONE);
    ASSERT_NE(action.moveB, Stockfish::MOVE_NONE);
    EXPECT_EQ(board.uci_move(BOARD_B, action.moveB), "a8a1");
    EXPECT_EQ(plyToMate, 1);
}

// The probe's local line is not a win if the defending partner already has a
// mate in one. That mate is played before Hivemind reaches its next move.
TEST_F(MateRaceTest, ProbeRejectsLineWhenOpponentHasMateInOne) {
    Board board;
    board.set_fen(
        BOARD_A,
        "r1bq1b1r/ppp1p1pp/2n2nk1/3p2N1/3P4/8/PPP1PPPP/RNBQKB1R[Bb] w KQ - 2 2");
    board.set_fen(
        BOARD_B, "6k1/5ppp/8/8/8/8/8/R5K1[] w - - 0 1");

    const MateProbe::Result singleBoard = MateProbe::probe(
        board.fen(BOARD_A), 16, 8000000, 2000);
    ASSERT_TRUE(singleBoard.found);
    ASSERT_GT(singleBoard.mateInMoves, 1);

    JointActionCandidate action;
    int plyToMate = 0;
    std::string pv;
    EXPECT_FALSE(Agent::probe_root_mate(
        board, Stockfish::WHITE, true, 8000000, 2000, {},
        action, plyToMate, pv))
        << "claimed " << pv << " while the opponent had Ra8#";
}

// Benchmark position 4 has a legal capture on the defending partner board,
// but that capture does not refute B@f7's mate. An unrelated capture must not
// suppress the Fairy proof merely because it could be paired with a pass.
TEST_F(MateRaceTest, BenchmarkPositionFourStillReportsMate) {
    Board board;
    board.set_fen(
        BOARD_A,
        "r1bq1b1r/ppp1p1pp/2n2nk1/3p2N1/3P4/8/PPP1PPPP/RNBQKB1R[Bb] w KQ - 2 2");
    board.set_fen(
        BOARD_B,
        "rn2k2r/ppp1bppp/3p1n2/4pq2/3PP3/2N2N2/PPP2PPP/R1BQ1RK1/p w kq - 0 1");

    JointActionCandidate action;
    int plyToMate = 0;
    std::string pv;
    ASSERT_TRUE(Agent::probe_root_mate(
        board, Stockfish::WHITE, true, 8000000, 2000, {},
        action, plyToMate, pv));
    EXPECT_EQ(board.uci_move(BOARD_A, action.moveA), "B@f7");
    EXPECT_EQ(action.moveB, Stockfish::MOVE_NONE);
    EXPECT_GT(plyToMate, 1);
}

TEST_F(QueenFeedRaceTest, TheExactScansDoNotClaimEitherMate) {
    for (bool timeAdvantage : {false, true}) {
        JointActionCandidate action;
        int plyToMate = 0;
        EXPECT_FALSE(Agent::find_root_mate(
            board, Stockfish::WHITE, timeAdvantage, action, plyToMate));
    }
    play_capture_and_recapture();
    JointActionCandidate theirs;
    int theirPly = 0;
    ASSERT_TRUE(Agent::find_root_mate(
        board, Stockfish::BLACK, false, theirs, theirPly));
    EXPECT_EQ(board.uci_move(BOARD_A, theirs.moveA), "Q@e1");
    EXPECT_EQ(theirPly, 1);
}

// Board A of the lost game after 27.fxg5, Black (this team) to move, behind
// on time. 27...Rhg8?? loses to 28.Qf6+, a mate in eight with best defence
// whose queen is fed to board B; the exact solver cannot reach that depth,
// but the positions further down the line are within its reach.
constexpr const char* kRhg8BoardA =
    "r6r/pppnqpkp/4p3/3pP1P1/7B/2P1P1P1/PpP1nQP1/3qN2K[QNNb] b - - 0 27";
constexpr const char* kRhg8BoardB =
    "r1bkn2R/pp2np1p/3pp3/7B/3P1r2/3BbR2/PPP3PP/R4KNR[BPPbpp] b - - 3 24";

// Board A with Black (this team) to move, behind on time. The rook on a8 is
// all that guards the back rank: ...Rxa2 leaves it and walks into Rb8#, which
// the opponents ahead on time can play while sitting board B. ...h6 does not.
// Board B is bare so no partner capture can ever supply a blocker.
constexpr const char* kBackRankBoardA =
    "r5k1/5ppp/8/8/8/8/P7/1R4K1[] b - - 0 1";
constexpr const char* kBackRankBoardB =
    "7k/8/8/8/8/8/8/K7[] b - - 0 1";

class SelectedMoveCertTest : public MateRaceTest {
protected:
    static Board back_rank_board() {
        Board board;
        board.set_fen(BOARD_A, kBackRankBoardA);
        board.set_fen(BOARD_B, kBackRankBoardB);
        return board;
    }
    static Agent::MateSearchBudget::Clock::time_point deadline_in(int ms) {
        return Agent::MateSearchBudget::Clock::now()
            + std::chrono::milliseconds(ms);
    }
};

TEST_F(SelectedMoveCertTest, LeavingTheBackRankIsCertifiedLost) {
    Board board = back_rank_board();
    const JointActionCandidate rxa2 = joint(
        move_of(board, BOARD_A, "a8a2"), Stockfish::MOVE_NONE);
    const std::string before = board.fen(BOARD_A) + "|" + board.fen(BOARD_B);

    int plyToMate = 0;
    SelectedMoveCertStats stats;
    EXPECT_TRUE(Agent::action_walks_into_certified_mate(
        board, rxa2, Stockfish::BLACK, false, deadline_in(2000), nullptr,
        plyToMate, &stats));
    EXPECT_EQ(plyToMate, 1);
    EXPECT_EQ(stats.candidates, 1U);
    EXPECT_EQ(stats.probeHits, 1U);
    EXPECT_EQ(stats.certificates, 1U);
    EXPECT_EQ(board.fen(BOARD_A) + "|" + board.fen(BOARD_B), before);
}

TEST_F(SelectedMoveCertTest, MakingLuftIsNotVetoed) {
    Board board = back_rank_board();
    const JointActionCandidate h6 = joint(
        move_of(board, BOARD_A, "h7h6"), Stockfish::MOVE_NONE);

    int plyToMate = 0;
    SelectedMoveCertStats stats;
    EXPECT_FALSE(Agent::action_walks_into_certified_mate(
        board, h6, Stockfish::BLACK, false, deadline_in(2000), nullptr,
        plyToMate, &stats));
    EXPECT_EQ(plyToMate, 0);
    EXPECT_EQ(stats.candidates, 1U);
    EXPECT_EQ(stats.certificates, 0U);
}

// Ahead on time this team may sit board A after Rb8+ and play board B, so
// no single-board line is forced against it and no veto is raised.
TEST_F(SelectedMoveCertTest, TimeAdvantageDisarmsTheVeto) {
    Board board = back_rank_board();
    const JointActionCandidate rxa2 = joint(
        move_of(board, BOARD_A, "a8a2"), Stockfish::MOVE_NONE);

    int plyToMate = 0;
    SelectedMoveCertStats stats;
    EXPECT_FALSE(Agent::action_walks_into_certified_mate(
        board, rxa2, Stockfish::BLACK, true, deadline_in(2000), nullptr,
        plyToMate, &stats));
    EXPECT_EQ(stats.candidates, 0U);
}

TEST_F(SelectedMoveCertTest, SpentDeadlineRaisesNoVeto) {
    Board board = back_rank_board();
    const JointActionCandidate rxa2 = joint(
        move_of(board, BOARD_A, "a8a2"), Stockfish::MOVE_NONE);

    int plyToMate = 0;
    SelectedMoveCertStats stats;
    EXPECT_FALSE(Agent::action_walks_into_certified_mate(
        board, rxa2, Stockfish::BLACK, false, deadline_in(-1), nullptr,
        plyToMate, &stats));
    EXPECT_EQ(stats.candidates, 0U);
    EXPECT_EQ(stats.certificates, 0U);
}

// The tree's most visited alternative is held to the same test, and the
// proof lands on the vetoed child so its root edge is a solved loss after.
TEST_F(SelectedMoveCertTest, VetoedMoveIsReplacedByTheNextVisitedAlternative) {
    Board board = back_rank_board();
    const Stockfish::Move rxa2 = move_of(board, BOARD_A, "a8a2");
    const Stockfish::Move h6 = move_of(board, BOARD_A, "h7h6");

    SearchParams::RuntimeConfig config;
    auto root = std::make_shared<Node>(
        Stockfish::BLACK, board.search_hash_key(Stockfish::BLACK, false));
    ASSERT_TRUE(root->try_init_and_expand(
        {rxa2, h6}, {Stockfish::MOVE_NONE}, {0.6f, 0.4f}, {1.0f},
        false, true, false, config));
    JointActionCandidate expanded;
    ASSERT_NE(root->expand_next_joint_child(nullptr, 0, expanded, config),
              nullptr);
    ASSERT_EQ(root->get_num_generated(), 2U);
    const JointActionCandidate chosen = root->get_joint_action(0);
    ASSERT_EQ(chosen.moveA, rxa2);

    SelectedMoveCertStats stats;
    const std::optional<JointActionCandidate> replacement =
        Agent::certified_mate_free_alternative(
            board, *root, chosen, Stockfish::BLACK, false, deadline_in(4000),
            nullptr, &stats);
    ASSERT_TRUE(replacement.has_value());
    EXPECT_EQ(replacement->moveA, h6);
    EXPECT_EQ(replacement->moveB, Stockfish::MOVE_NONE);
    EXPECT_TRUE(stats.vetoed);
    EXPECT_TRUE(stats.replaced);
    EXPECT_EQ(stats.candidates, 2U);
    EXPECT_EQ(root->get_child(0)->get_node_type(), NodeType::WIN);
    EXPECT_EQ(root->get_child(1)->get_node_type(), NodeType::UNSOLVED);
}

TEST_F(SelectedMoveCertTest, SafeChosenMoveNeedsNoAlternative) {
    Board board = back_rank_board();
    const Stockfish::Move rxa2 = move_of(board, BOARD_A, "a8a2");
    const Stockfish::Move h6 = move_of(board, BOARD_A, "h7h6");

    SearchParams::RuntimeConfig config;
    auto root = std::make_shared<Node>(
        Stockfish::BLACK, board.search_hash_key(Stockfish::BLACK, false));
    ASSERT_TRUE(root->try_init_and_expand(
        {h6, rxa2}, {Stockfish::MOVE_NONE}, {0.6f, 0.4f}, {1.0f},
        false, true, false, config));
    const JointActionCandidate chosen = root->get_joint_action(0);
    ASSERT_EQ(chosen.moveA, h6);

    SelectedMoveCertStats stats;
    EXPECT_FALSE(Agent::certified_mate_free_alternative(
        board, *root, chosen, Stockfish::BLACK, false, deadline_in(4000),
        nullptr, &stats).has_value());
    EXPECT_FALSE(stats.vetoed);
    EXPECT_EQ(stats.candidates, 1U);
}

// Board B after the feed of the lost game: Black holds the queen captured on
// board A and mates in three checks while White on B has [QBPPP] to drop.
// The joint solver proves it in under a thousand nodes once its checks are
// ordered by evasions and the depth is searched in one pass; the same proof
// used to cost several hundred thousand.
TEST_F(SelectedMoveCertTest, JointSolverProvesTheFedBoardMateCheaply) {
    Board board;
    board.set_fen(BOARD_A,
        "r5rk/ppp2p1p/4pP2/3p4/7B/2P1P1P1/PpP1n1P1/3qN2K[QNNb] w - - 1 31");
    board.set_fen(BOARD_B,
        "r1bkn2R/pp2np1p/3pp3/7B/3P1r2/3BbR2/PPP3PP/R4KNR[QBPPPqbnpp] b - - 3 24");

    Agent::MateSearchBudget budget;
    budget.remainingNodes = 5000;
    JointActionCandidate action;
    int plyToMate = 0;
    std::vector<MateProofPly> line;
    ASSERT_TRUE(Agent::prove_joint_forced_mate(
        board, Stockfish::WHITE, true, 3, budget, action, plyToMate, &line,
        false));
    EXPECT_EQ(action.moveA, Stockfish::MOVE_NONE);
    EXPECT_EQ(board.uci_move(BOARD_B, action.moveB), "f4f3");
    EXPECT_EQ(plyToMate, 5);
    EXPECT_LT(5000 - budget.remainingNodes, 2000U);
}

// One move earlier on board A: the queen is still on f6 and gxf6+ has to be
// found before the board-B mate exists. Two boards, a capture that feeds the
// hand, and a switch of boards, proven in one pass.
TEST_F(SelectedMoveCertTest, JointSolverFollowsTheFeedAcrossBoards) {
    Board board;
    board.set_fen(BOARD_A,
        "r5r1/ppp2pkp/4pq2/3p2P1/7B/2P1P1P1/PpP1n1P1/3qN2K[QNNb] w - - 0 30");
    board.set_fen(BOARD_B,
        "r1bkn2R/pp2np1p/3pp3/7B/3P1r2/3BbR2/PPP3PP/R4KNR[QBPPbnpp] b - - 3 24");

    Agent::MateSearchBudget budget;
    budget.remainingNodes = 5000000;
    JointActionCandidate action;
    int plyToMate = 0;
    std::vector<MateProofPly> line;
    ASSERT_TRUE(Agent::prove_joint_forced_mate(
        board, Stockfish::WHITE, true, 4, budget, action, plyToMate, &line,
        false));
    EXPECT_EQ(board.uci_move(BOARD_A, action.moveA), "g5f6");
    EXPECT_EQ(plyToMate, 7);
}

// The same position seen from the team about to be mated: their chosen
// move on board A is irrelevant, the joint fallback vetoes it regardless of
// what Fairy sees on either board alone.
TEST_F(SelectedMoveCertTest, JointFallbackVetoesACrossBoardMate) {
    Board board;
    board.set_fen(BOARD_A,
        "r5rk/ppp2p1p/4pP2/3p4/7B/2P1P1P1/PpP1n1P1/3qN2K[QNNb] w - - 1 31");
    board.set_fen(BOARD_B,
        "r1bkn2R/pp2np1p/3pp3/7B/3P1r2/3BbR2/PPP3PP/R4KNR[QBPPPqbnpp] b - - 3 24");
    // Hand the move to Black on A so this team has an action to certify:
    // a quiet king move on board A, White to move afterwards on both boards.
    board.set_fen(BOARD_A,
        "r5r1/ppp2pkp/4pP2/3p4/7B/2P1P1P1/PpP1n1P1/3qN2K[QNNb] b - - 1 31");
    const JointActionCandidate kh8 = joint(
        move_of(board, BOARD_A, "g7h8"), Stockfish::MOVE_NONE);

    int plyToMate = 0;
    SelectedMoveCertStats stats;
    EXPECT_TRUE(Agent::action_walks_into_certified_mate(
        board, kh8, Stockfish::BLACK, false, deadline_in(2000), nullptr,
        plyToMate, &stats));
    // Fairy sees the board-B mate once the queen is in hand and reports
    // its own line; the joint solver's is shorter. Either is a certificate.
    EXPECT_GE(plyToMate, 5);
    EXPECT_EQ(stats.certificates, 1U);
}

// The verifier's slices one move before the feed: after 29.exf6+ Black's
// ...Qxf6 walks into 30.gxf6+, which hands the queen to board B for the mate
// there. Fairy sees nothing on either board alone - the queen is not in
// hand yet - and the joint slice proves it. A slice too small to finish
// leaves UNKNOWN, retryable, and the next one resumes from its cache.
class RootActionSliceTest : public MateRaceTest {
protected:
    static Board before_feed_board() {
        Board board;
        board.set_fen(BOARD_A, kRhg8BoardA);
        board.set_fen(BOARD_B, kRhg8BoardB);
        for (const char* uci : {"h8g8", "f2f6", "d7f6", "e5f6"}) {
            board.make_moves(move_of(board, BOARD_A, uci), Stockfish::MOVE_NONE);
        }
        return board;
    }
    static Board fed_board() {
        Board board;
        board.set_fen(BOARD_A,
            "r5r1/ppp2pkp/4pP2/3p4/7B/2P1P1P1/PpP1n1P1/3qN2K[QNNb] b - - 1 31");
        board.set_fen(BOARD_B,
            "r1bkn2R/pp2np1p/3pp3/7B/3P1r2/3BbR2/PPP3PP/R4KNR[QBPPPqbnpp] b - - 3 24");
        return board;
    }
    static Agent::MateSearchBudget::Clock::time_point deadline_in(int ms) {
        return Agent::MateSearchBudget::Clock::now()
            + std::chrono::milliseconds(ms);
    }
};

TEST_F(RootActionSliceTest, FairySliceThenJointSliceProvesTheLoss) {
    Board board = before_feed_board();
    const JointActionCandidate qxf6 = joint(
        move_of(board, BOARD_A, "e7f6"), Stockfish::MOVE_NONE);
    const std::string before = board.fen(BOARD_A) + "|" + board.fen(BOARD_B);
    RootActionVerdict verdict;
    ConcurrentVerifierStats stats;
    uint64_t nodes = 0;

    // No tree here, so the proof runs in generation order; bound four is
    // what that affords. The live verifier orders by the tree's priors.
    Agent::verify_root_action_slice(
        board, qxf6, Stockfish::BLACK, verdict, 800000, 30, deadline_in(5000),
        nullptr, nullptr, nodes, &stats, nullptr, 4);
    EXPECT_TRUE(verdict.fairyProbed);
    EXPECT_EQ(verdict.state, RootActionVerdict::State::UNKNOWN);
    EXPECT_EQ(verdict.slices, 1);
    EXPECT_EQ(stats.probes, 1U);
    EXPECT_EQ(stats.probeHits, 0U);

    Agent::verify_root_action_slice(
        board, qxf6, Stockfish::BLACK, verdict, 800000, 30, deadline_in(5000),
        nullptr, nullptr, nodes, &stats, nullptr, 4);
    EXPECT_EQ(verdict.state, RootActionVerdict::State::PROVEN_LOSS);
    EXPECT_GE(verdict.plyToMate, 5);
    EXPECT_EQ(verdict.slices, 2);
    EXPECT_EQ(stats.proven, 1U);
    EXPECT_GT(nodes, 0U);
    EXPECT_LT(nodes, 800000U);
    EXPECT_EQ(board.fen(BOARD_A) + "|" + board.fen(BOARD_B), before);

    // Settled verdicts are not searched again.
    Agent::verify_root_action_slice(
        board, qxf6, Stockfish::BLACK, verdict, 800000, 30, deadline_in(5000),
        nullptr, nullptr, nodes, &stats, nullptr, 4);
    EXPECT_EQ(verdict.slices, 2);
}

TEST_F(RootActionSliceTest, ASliceCutShortStaysUnknownAndResumes) {
    Board board = fed_board();
    const JointActionCandidate kh8 = joint(
        move_of(board, BOARD_A, "g7h8"), Stockfish::MOVE_NONE);
    RootActionVerdict verdict;
    verdict.fairyProbed = true;
    uint64_t nodes = 0;

    Agent::verify_root_action_slice(
        board, kh8, Stockfish::BLACK, verdict, 50, 30, deadline_in(2000),
        nullptr, nullptr, nodes, nullptr, nullptr, 3);
    EXPECT_EQ(verdict.state, RootActionVerdict::State::UNKNOWN);
    EXPECT_EQ(verdict.slices, 1);
    EXPECT_LE(nodes, 50U);
    ASSERT_TRUE(verdict.cache);

    // The next slice picks up the same cache and finishes.
    Agent::verify_root_action_slice(
        board, kh8, Stockfish::BLACK, verdict, 50000, 30, deadline_in(2000),
        nullptr, nullptr, nodes, nullptr, nullptr, 3);
    EXPECT_EQ(verdict.state, RootActionVerdict::State::PROVEN_LOSS);
}

TEST_F(RootActionSliceTest, SpentDeadlineOrStopLeavesUnknown) {
    Board board = fed_board();
    const JointActionCandidate kh8 = joint(
        move_of(board, BOARD_A, "g7h8"), Stockfish::MOVE_NONE);
    RootActionVerdict verdict;
    verdict.fairyProbed = true;
    uint64_t nodes = 0;

    Agent::verify_root_action_slice(
        board, kh8, Stockfish::BLACK, verdict, 50000, 30, deadline_in(-1),
        nullptr, nullptr, nodes);
    EXPECT_EQ(verdict.state, RootActionVerdict::State::UNKNOWN);
    EXPECT_EQ(verdict.slices, 0);

    std::atomic<bool> stop{true};
    Agent::verify_root_action_slice(
        board, kh8, Stockfish::BLACK, verdict, 50000, 30, deadline_in(2000),
        &stop, nullptr, nodes);
    EXPECT_EQ(verdict.state, RootActionVerdict::State::UNKNOWN);
    EXPECT_EQ(verdict.slices, 0);
}

// No mate for the opponents at the bound: with budget and time to spare
// the joint slice refutes, and that is final for the search.
TEST_F(RootActionSliceTest, ExhaustiveRefutationIsFinal) {
    Board board;
    board.set_fen(BOARD_A, "r5k1/5ppp/8/8/8/8/P7/1R4K1[] b - - 0 1");
    board.set_fen(BOARD_B, "7k/8/8/8/8/8/8/K7[] b - - 0 1");
    const JointActionCandidate h6 = joint(
        move_of(board, BOARD_A, "h7h6"), Stockfish::MOVE_NONE);
    RootActionVerdict verdict;
    verdict.fairyProbed = true;
    uint64_t nodes = 0;

    Agent::verify_root_action_slice(
        board, h6, Stockfish::BLACK, verdict, 2000000, 30, deadline_in(10000),
        nullptr, nullptr, nodes);
    EXPECT_EQ(verdict.state, RootActionVerdict::State::REFUTED_AT_BOUND);
    EXPECT_LT(nodes, 2000000U);
}

TEST_F(RootActionSliceTest, RefutationNeedsBudgetToSpare) {
    Board board;
    board.set_fen(BOARD_A, "r5k1/5ppp/8/8/8/8/P7/1R4K1[] b - - 0 1");
    board.set_fen(BOARD_B, "7k/8/8/8/8/8/8/K7[] b - - 0 1");
    const JointActionCandidate h6 = joint(
        move_of(board, BOARD_A, "h7h6"), Stockfish::MOVE_NONE);
    RootActionVerdict verdict;
    verdict.fairyProbed = true;
    uint64_t nodes = 0;

    Agent::verify_root_action_slice(
        board, h6, Stockfish::BLACK, verdict, 5, 30, deadline_in(10000),
        nullptr, nullptr, nodes);
    EXPECT_EQ(verdict.state, RootActionVerdict::State::UNKNOWN);
}

#include <gtest/gtest.h>

#include <string>

#include "common/globals.h"
#include "environment/board.h"
#include "environment/constants.h"
#include "environment/joint_action.h"
#include "search/agent.h"
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

TEST_F(MateRaceTest, OurOwnMateIsNeverVetoed) {
    Board board;
    board.set_fen(BOARD_A, kRaceBoardA);
    board.set_fen(BOARD_B, kRaceBoardB);

    // One move on from the position above: the rook has arrived and the drop
    // ends the game, so the mate waiting on board A is never played.
    board.push_move(BOARD_A, move_of(board, BOARD_A, "f1g1"));
    const Stockfish::Move drop = move_of(board, BOARD_B, "R@c1");
    ASSERT_NE(drop, Stockfish::MOVE_NONE);
    EXPECT_FALSE(Agent::action_loses_mate_race(
        board, joint(Stockfish::MOVE_NONE, drop), Stockfish::WHITE, true));
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

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "environment/adjudication.h"
#include "environment/board.h"
#include "environment/time_state.h"
#include "search/node.h"
#include "search/searchthread.h"
#include "common/globals.h"
#include "common/utils.h"

namespace {

class TimeStateBoardTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        init_fairy_stockfish();
        init_policy_index();
    }
};

Stockfish::Move find_move(Board& board, int boardNum, const std::string& uci) {
    for (Stockfish::Move move : board.legal_moves(boardNum)) {
        if (board.uci_move(boardNum, move) == uci) {
            return move;
        }
    }
    return Stockfish::MOVE_NONE;
}

// White to move on board A and black on board B: the white team is on turn on
// both boards, so both of its clocks run.
const char* WHITE_TEAM_ON_BOTH =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1|"
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR[] b KQkq - 0 1";

/**
 * A white-team root holding one joint action, with its child unexpanded so
 * selection stops there and reports the clock that child is reached with.
 */
std::shared_ptr<Node> one_action_root(Board& board, const std::string& moveA,
                                      const std::string& moveB) {
    auto root = std::make_shared<Node>(Stockfish::WHITE);
    const bool onA = board.side_to_move(BOARD_A) == Stockfish::WHITE;
    const bool onB = board.side_to_move(BOARD_B) == Stockfish::BLACK;
    const Stockfish::Move a = moveA.empty()
        ? Stockfish::MOVE_NONE : find_move(board, BOARD_A, moveA);
    const Stockfish::Move b = moveB.empty()
        ? Stockfish::MOVE_NONE : find_move(board, BOARD_B, moveB);
    EXPECT_TRUE(moveA.empty() || a != Stockfish::MOVE_NONE);
    EXPECT_TRUE(moveB.empty() || b != Stockfish::MOVE_NONE);
    EXPECT_TRUE(root->try_init_and_expand(
        {a}, {b}, {1.0f}, {1.0f}, true, onA, onB,
        SearchParams::RuntimeConfig{}));
    return root;
}

}  // namespace

TEST(TimeStateTest, LegacyStateReproducesThePermanentFlag) {
    for (Stockfish::Color team : {Stockfish::WHITE, Stockfish::BLACK}) {
        for (bool flag : {false, true}) {
            TimeState state = TimeState::legacy(team, flag);
            EXPECT_FALSE(state.is_clocked());
            EXPECT_EQ(state.can_sit(team), flag);
            EXPECT_EQ(state.can_sit(~team), !flag);

            // A legacy lead never moves, however long either team thinks.
            const TimeState before = state;
            state.advance(team, true);
            state.advance(~team, true);
            EXPECT_EQ(state, before);
        }
    }
}

TEST(TimeStateTest, OneLeadSharedByBothTeamsWithOppositeSigns) {
    const TimeState white = TimeState::clocked(Stockfish::WHITE, 1500);
    EXPECT_TRUE(white.is_clocked());
    EXPECT_EQ(white.lead_ms(Stockfish::WHITE), 1500);
    EXPECT_EQ(white.lead_ms(Stockfish::BLACK), -1500);
    EXPECT_TRUE(white.can_sit(Stockfish::WHITE));
    EXPECT_FALSE(white.can_sit(Stockfish::BLACK));

    const TimeState black = TimeState::clocked(Stockfish::BLACK, 1500);
    EXPECT_EQ(black.lead_ms(Stockfish::BLACK), 1500);
    EXPECT_EQ(black.lead_ms(Stockfish::WHITE), -1500);
}

TEST(TimeStateTest, EqualClocksLetNeitherTeamSit) {
    const TimeState level = TimeState::clocked(Stockfish::WHITE, 0);
    EXPECT_FALSE(level.can_sit(Stockfish::WHITE));
    EXPECT_FALSE(level.can_sit(Stockfish::BLACK));
}

TEST(TimeStateTest, MoveTimesArePerTeamDefaultToTheFloorAndStayPositive) {
    const TimeState defaults = TimeState::clocked(Stockfish::WHITE, 0);
    EXPECT_EQ(defaults.move_time_ms(Stockfish::WHITE), TimeState::DEFAULT_MOVE_TIME_MS);
    EXPECT_EQ(defaults.move_time_ms(Stockfish::BLACK), TimeState::DEFAULT_MOVE_TIME_MS);
    EXPECT_EQ(TimeState::DEFAULT_MOVE_TIME_MS, 100);

    const TimeState mixed = TimeState::clocked(Stockfish::BLACK, 0, 15000, 0);
    EXPECT_EQ(mixed.move_time_ms(Stockfish::BLACK), 15000);
    EXPECT_EQ(mixed.move_time_ms(Stockfish::WHITE), 1);
}

TEST(TimeStateTest, OnlyATeamOnTurnOnBothBoardsPaysAndAtItsOwnSpeed) {
    // White thinks for 15 s an action, black answers in 0.1 s.
    TimeState state = TimeState::clocked(Stockfish::WHITE, 20000, 15000, 100);

    // One clock of each team running: the lead holds whoever acts.
    state.advance(Stockfish::WHITE, false);
    state.advance(Stockfish::BLACK, false);
    EXPECT_EQ(state.lead_ms(Stockfish::WHITE), 20000);

    state.advance(Stockfish::WHITE, true);
    EXPECT_EQ(state.lead_ms(Stockfish::WHITE), 5000);
    EXPECT_TRUE(state.can_sit(Stockfish::WHITE));

    state.advance(Stockfish::BLACK, true);
    EXPECT_EQ(state.lead_ms(Stockfish::WHITE), 5100);

    state.advance(Stockfish::WHITE, true);
    EXPECT_EQ(state.lead_ms(Stockfish::WHITE), -9900);
    EXPECT_FALSE(state.can_sit(Stockfish::WHITE));
    EXPECT_TRUE(state.can_sit(Stockfish::BLACK));
}

TEST(TimeStateTest, LeadClassKeepsWhoMaySitExactAndSizeCoarse) {
    const auto lead_class = [](int lead) {
        return TimeState::clocked(Stockfish::WHITE, lead).lead_class(Stockfish::WHITE);
    };
    EXPECT_NE(lead_class(1), lead_class(0));
    EXPECT_NE(lead_class(-1), lead_class(0));
    EXPECT_NE(lead_class(1), lead_class(-1));
    // 10 and 11 move times of 100 ms land in the same log2 class.
    EXPECT_EQ(lead_class(1000), lead_class(1100));
    EXPECT_NE(lead_class(100), lead_class(1000));

    // Each side's class is measured in the move time of the team ahead.
    const TimeState slowWhite = TimeState::clocked(Stockfish::WHITE, 30000, 15000, 100);
    const TimeState fastWhite = TimeState::clocked(Stockfish::WHITE, 30000, 100, 100);
    EXPECT_NE(slowWhite.lead_class(Stockfish::WHITE), fastWhite.lead_class(Stockfish::WHITE));
}

TEST_F(TimeStateBoardTest, LegacyStateHashesExactlyAsItsFlag) {
    Board board;
    for (Stockfish::Color root : {Stockfish::WHITE, Stockfish::BLACK}) {
        for (bool flag : {false, true}) {
            const TimeState state = TimeState::legacy(root, flag);
            for (Stockfish::Color team : {Stockfish::WHITE, Stockfish::BLACK}) {
                EXPECT_EQ(board.search_hash_key(team, state),
                          board.search_hash_key(team, state.can_sit(team)));
            }
        }
    }
}

TEST_F(TimeStateBoardTest, ClockedHashSeparatesWhoMaySit) {
    Board board;
    const uint64_t ahead = board.search_hash_key(
        Stockfish::WHITE, TimeState::clocked(Stockfish::WHITE, 500));
    const uint64_t level = board.search_hash_key(
        Stockfish::WHITE, TimeState::clocked(Stockfish::WHITE, 0));
    const uint64_t behind = board.search_hash_key(
        Stockfish::WHITE, TimeState::clocked(Stockfish::WHITE, -500));
    EXPECT_NE(ahead, level);
    EXPECT_NE(ahead, behind);
    EXPECT_NE(level, behind);
    // A clocked node never collides with a legacy one.
    EXPECT_NE(ahead, board.search_hash_key(Stockfish::WHITE, true));
    EXPECT_NE(behind, board.search_hash_key(Stockfish::WHITE, false));
}

TEST_F(TimeStateBoardTest, AdjudicationReadsBothTeamsFromTheClock) {
    // Fool's mate on board A. The legacy overload must judge it exactly as
    // the legacy clock state it now delegates to.
    Board board;
    board.set(
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR[] w KQkq - 1 3|"
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1");
    for (bool flag : {false, true}) {
        EXPECT_EQ(adjudicate_game(board, Stockfish::BLACK, flag),
                  adjudicate_game(
                      board, Stockfish::BLACK,
                      TimeState::legacy(Stockfish::BLACK, flag)));
    }
}

TEST_F(TimeStateBoardTest, SelectionChargesTheTeamOnTurnOnBothBoards) {
    Board board;
    board.set(WHITE_TEAM_ON_BOTH);
    ASSERT_TRUE(board.on_turn_on_both_boards(Stockfish::WHITE));
    auto root = one_action_root(board, "e2e4", "e7e5");

    SearchThread searchThread;
    searchThread.set_root_node(root);
    const TimeState rootTime =
        TimeState::clocked(Stockfish::WHITE, 150, 100, 100);
    const LeafSelection selection = searchThread.select_and_expand(board, rootTime);

    ASSERT_NE(selection.leaf, nullptr);
    EXPECT_NE(selection.leaf, root);
    EXPECT_EQ(selection.time.lead_ms(Stockfish::WHITE), 50);
    EXPECT_TRUE(selection.time.can_sit(Stockfish::WHITE));
}

TEST_F(TimeStateBoardTest, SelectionChargesNothingWhenEachTeamHasOneBoard) {
    // Start position: the white team is on turn on board A only, the black
    // team on board B, so one clock of each team runs.
    Board board;
    ASSERT_FALSE(board.on_turn_on_both_boards(Stockfish::WHITE));
    auto root = one_action_root(board, "e2e4", "");

    SearchThread searchThread;
    searchThread.set_root_node(root);
    const TimeState rootTime =
        TimeState::clocked(Stockfish::WHITE, 150, 100, 100);
    const LeafSelection selection = searchThread.select_and_expand(board, rootTime);

    ASSERT_NE(selection.leaf, nullptr);
    EXPECT_EQ(selection.time.lead_ms(Stockfish::WHITE), 150);
}

TEST_F(TimeStateBoardTest, SlowTeamLosesTheRightToSitAfterOneAction) {
    Board board;
    board.set(WHITE_TEAM_ON_BOTH);
    auto root = one_action_root(board, "e2e4", "e7e5");

    SearchThread searchThread;
    searchThread.set_root_node(root);
    // Two seconds ahead, but fifteen seconds an action.
    const TimeState rootTime =
        TimeState::clocked(Stockfish::WHITE, 2000, 15000, 100);
    const LeafSelection selection = searchThread.select_and_expand(board, rootTime);

    ASSERT_NE(selection.leaf, nullptr);
    EXPECT_EQ(selection.time.lead_ms(Stockfish::WHITE), -13000);
    EXPECT_FALSE(selection.time.can_sit(Stockfish::WHITE));
    EXPECT_TRUE(selection.time.can_sit(Stockfish::BLACK));
}

TEST_F(TimeStateBoardTest, LegacySelectionKeepsTheRootFlag) {
    Board board;
    board.set(WHITE_TEAM_ON_BOTH);
    auto root = one_action_root(board, "e2e4", "e7e5");

    SearchThread searchThread;
    searchThread.set_root_node(root);
    const LeafSelection selection = searchThread.select_and_expand(board, true);

    ASSERT_NE(selection.leaf, nullptr);
    EXPECT_EQ(selection.time, TimeState::legacy(Stockfish::WHITE, true));
}

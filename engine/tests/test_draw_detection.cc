#include <gtest/gtest.h>
#include "../src/board.h"
#include "../src/constants.h"
#include "Fairy-Stockfish/src/position.h"
#include "Fairy-Stockfish/src/types.h"
#include "Fairy-Stockfish/src/bitboard.h"
#include "Fairy-Stockfish/src/piece.h"
#include "Fairy-Stockfish/src/thread.h"

class DrawDetectionTest : public ::testing::Test {
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

TEST_F(DrawDetectionTest, ThreefoldRepetition) {
    Board board;
    board.set_fen(BOARD_A, board.startingFen);
    
    auto moves = board.legal_moves(BOARD_A);
    Stockfish::Move nf3 = Stockfish::MOVE_NONE;
    for (const auto& m : moves) {
        if (board.uci_move(BOARD_A, m) == "g1f3") {
            nf3 = m;
            break;
        }
    }
    ASSERT_NE(nf3, Stockfish::MOVE_NONE);
    
    board.push_move(BOARD_A, nf3);
    moves = board.legal_moves(BOARD_A);
    Stockfish::Move nc6 = Stockfish::MOVE_NONE;
    for (const auto& m : moves) {
        if (board.uci_move(BOARD_A, m) == "b8c6") {
            nc6 = m;
            break;
        }
    }
    ASSERT_NE(nc6, Stockfish::MOVE_NONE);
    
    board.push_move(BOARD_A, nc6);
    
    EXPECT_FALSE(board.is_draw(BOARD_A)) << "Should not be draw after 2 moves";
    
    moves = board.legal_moves(BOARD_A);
    Stockfish::Move ng1 = Stockfish::MOVE_NONE;
    for (const auto& m : moves) {
        if (board.uci_move(BOARD_A, m) == "f3g1") {
            ng1 = m;
            break;
        }
    }
    ASSERT_NE(ng1, Stockfish::MOVE_NONE);
    
    board.push_move(BOARD_A, ng1);
    moves = board.legal_moves(BOARD_A);
    Stockfish::Move nb8 = Stockfish::MOVE_NONE;
    for (const auto& m : moves) {
        if (board.uci_move(BOARD_A, m) == "c6b8") {
            nb8 = m;
            break;
        }
    }
    ASSERT_NE(nb8, Stockfish::MOVE_NONE);
    
    board.push_move(BOARD_A, nb8);
    // Back to starting position (2nd occurrence)
    EXPECT_FALSE(board.is_draw(BOARD_A)) << "Two occurrences should not be draw";
    
    board.push_move(BOARD_A, nf3);
    board.push_move(BOARD_A, nc6);
    board.push_move(BOARD_A, ng1);
    board.push_move(BOARD_A, nb8);
    // Back to starting position (3rd occurrence)
    
    EXPECT_TRUE(board.is_draw(BOARD_A)) << "Should be draw by threefold repetition";
}

TEST_F(DrawDetectionTest, FiftyMoveRule) {
    Board board;
    board.set_fen(BOARD_A, "4k3/8/8/8/8/8/8/4K3 w - - 99 100");
    
    EXPECT_FALSE(board.is_draw(BOARD_A)) << "Should not be draw at 99 half-moves";
    
    auto moves = board.legal_moves(BOARD_A);
    ASSERT_GT(moves.size(), 0);
    
    board.push_move(BOARD_A, moves[0]);
    
    EXPECT_TRUE(board.is_draw(BOARD_A)) << "Should be draw by fifty-move rule at 100 half-moves";
}

TEST_F(DrawDetectionTest, DrawNotInsufficientMaterialBughouse) {
    Board board;
    
    board.set_fen(BOARD_A, "4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    EXPECT_FALSE(board.is_draw(BOARD_A)) << "Bughouse: King vs King is not draw (can receive pieces)";
    
    board.set_fen(BOARD_A, "4k3/8/8/8/8/8/8/4KB2 w - - 0 1");
    EXPECT_FALSE(board.is_draw(BOARD_A)) << "Bughouse: King+Bishop vs King is not draw (can receive pieces)";
}

TEST_F(DrawDetectionTest, ThreefoldDetectionVerified) {
    Board board;
    board.set_fen(BOARD_A, board.startingFen);
    
    auto find_move = [&](const std::string& uci) {
        auto moves = board.legal_moves(BOARD_A);
        for (const auto& m : moves) {
            if (board.uci_move(BOARD_A, m) == uci) return m;
        }
        return Stockfish::MOVE_NONE;
    };
    
    board.push_move(BOARD_A, find_move("g1f3"));
    board.push_move(BOARD_A, find_move("b8c6"));
    board.push_move(BOARD_A, find_move("f3g1"));
    board.push_move(BOARD_A, find_move("c6b8"));
    EXPECT_FALSE(board.is_draw(BOARD_A)) << "Two occurrences should not be draw";
    
    board.push_move(BOARD_A, find_move("g1f3"));
    board.push_move(BOARD_A, find_move("b8c6"));
    board.push_move(BOARD_A, find_move("f3g1"));
    board.push_move(BOARD_A, find_move("c6b8"));
    
    EXPECT_TRUE(board.is_draw(BOARD_A)) << "Should be draw after third occurrence of starting position";
}

TEST_F(DrawDetectionTest, BughouseGlobalDrawDetection) {
    Board board;
    
    board.set_fen(BOARD_A, "4k3/8/8/8/8/8/8/4K3 w - - 100 100");
    board.set_fen(BOARD_B, board.startingFen);
    
    EXPECT_TRUE(board.is_draw()) << "Global draw should be detected when board A reaches 50-move rule";
    EXPECT_TRUE(board.is_draw_on_board(BOARD_A)) << "Board A should be draw by 50-move rule";
    EXPECT_FALSE(board.is_draw_on_board(BOARD_B)) << "Board B should not be draw by itself";
}

TEST_F(DrawDetectionTest, NoDrawAfterOneRepetition) {
    // Test that position appearing twice (1 repetition) is NOT a draw
    Board board;
    board.set_fen(BOARD_A, board.startingFen);
    
    auto find_move = [&](const std::string& uci) {
        auto moves = board.legal_moves(BOARD_A);
        for (const auto& m : moves) {
            if (board.uci_move(BOARD_A, m) == uci) return m;
        }
        return Stockfish::MOVE_NONE;
    };
    
    // Starting position (occurrence #1)
    EXPECT_FALSE(board.is_draw(BOARD_A)) << "Starting position should not be draw";
    
    // Move knight out and back to return to starting position
    board.push_move(BOARD_A, find_move("g1f3"));
    EXPECT_FALSE(board.is_draw(BOARD_A)) << "After 1 move should not be draw";
    
    board.push_move(BOARD_A, find_move("b8c6"));
    EXPECT_FALSE(board.is_draw(BOARD_A)) << "After 2 moves should not be draw";
    
    board.push_move(BOARD_A, find_move("f3g1"));
    EXPECT_FALSE(board.is_draw(BOARD_A)) << "After 3 moves should not be draw";
    
    board.push_move(BOARD_A, find_move("c6b8"));
    // Starting position (occurrence #2) - only 1 repetition, should NOT be draw
    EXPECT_FALSE(board.is_draw(BOARD_A)) << "Position appearing twice (1 repetition) should NOT be draw";
}

TEST_F(DrawDetectionTest, NoDrawAfterTwoOccurrences) {
    // Test that position appearing twice total (1 repetition) is NOT a draw
    Board board;
    board.set_fen(BOARD_A, board.startingFen);
    
    auto find_move = [&](const std::string& uci) {
        auto moves = board.legal_moves(BOARD_A);
        for (const auto& m : moves) {
            if (board.uci_move(BOARD_A, m) == uci) return m;
        }
        return Stockfish::MOVE_NONE;
    };
    
    // Reach a position, leave it, and return to it once
    board.push_move(BOARD_A, find_move("e2e4"));  // Position A (occurrence #1)
    board.push_move(BOARD_A, find_move("e7e5"));
    board.push_move(BOARD_A, find_move("g1f3"));  // Position B
    board.push_move(BOARD_A, find_move("b8c6"));
    board.push_move(BOARD_A, find_move("f3g1"));  // Back to Position A (occurrence #2)
    board.push_move(BOARD_A, find_move("c6b8"));
    
    EXPECT_FALSE(board.is_draw(BOARD_A)) << "Position appearing twice should NOT be draw (need 3 occurrences)";
}

TEST_F(DrawDetectionTest, DrawAfterThreeOccurrences) {
    // Test that position appearing three times total (2 repetitions) IS a draw
    Board board;
    board.set_fen(BOARD_A, board.startingFen);
    
    auto find_move = [&](const std::string& uci) {
        auto moves = board.legal_moves(BOARD_A);
        for (const auto& m : moves) {
            if (board.uci_move(BOARD_A, m) == uci) return m;
        }
        return Stockfish::MOVE_NONE;
    };
    
    // Create a position that repeats three times
    board.push_move(BOARD_A, find_move("e2e4"));  // Position A (occurrence #1)
    board.push_move(BOARD_A, find_move("e7e5"));
    
    board.push_move(BOARD_A, find_move("g1f3"));  // Leave Position A
    board.push_move(BOARD_A, find_move("b8c6"));
    board.push_move(BOARD_A, find_move("f3g1"));  // Back to Position A (occurrence #2)
    board.push_move(BOARD_A, find_move("c6b8"));
    
    EXPECT_FALSE(board.is_draw(BOARD_A)) << "After 2 occurrences, should not yet be draw";
    
    board.push_move(BOARD_A, find_move("g1f3"));  // Leave Position A again
    board.push_move(BOARD_A, find_move("b8c6"));
    board.push_move(BOARD_A, find_move("f3g1"));  // Back to Position A (occurrence #3)
    board.push_move(BOARD_A, find_move("c6b8"));
    
    EXPECT_TRUE(board.is_draw(BOARD_A)) << "Position appearing three times (2 repetitions) should be draw";
}

TEST_F(DrawDetectionTest, ThreefoldRepetitionIntermediatePosition) {
    // Test threefold repetition of an intermediate position (not the starting position)
    Board board;
    board.set_fen(BOARD_A, board.startingFen);
    
    auto find_move = [&](const std::string& uci) {
        auto moves = board.legal_moves(BOARD_A);
        for (const auto& m : moves) {
            if (board.uci_move(BOARD_A, m) == uci) return m;
        }
        return Stockfish::MOVE_NONE;
    };
    
    // Move to a specific position, then repeat it
    board.push_move(BOARD_A, find_move("e2e4"));
    board.push_move(BOARD_A, find_move("e7e5"));
    // This is Position X (occurrence #1)
    
    board.push_move(BOARD_A, find_move("g1f3"));
    board.push_move(BOARD_A, find_move("g8f6"));
    board.push_move(BOARD_A, find_move("f3g1"));
    board.push_move(BOARD_A, find_move("f6g8"));
    // Back to Position X (occurrence #2)
    
    EXPECT_FALSE(board.is_draw(BOARD_A)) << "Two occurrences should not be draw";
    
    board.push_move(BOARD_A, find_move("g1f3"));
    board.push_move(BOARD_A, find_move("g8f6"));
    board.push_move(BOARD_A, find_move("f3g1"));
    board.push_move(BOARD_A, find_move("f6g8"));
    // Back to Position X (occurrence #3)
    
    EXPECT_TRUE(board.is_draw(BOARD_A)) << "Three occurrences should be draw by threefold repetition";
}

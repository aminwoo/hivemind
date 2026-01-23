#include <gtest/gtest.h>
#include "../src/board.h"
#include "../src/constants.h"
#include "Fairy-Stockfish/src/position.h"
#include "Fairy-Stockfish/src/types.h"
#include "Fairy-Stockfish/src/bitboard.h"
#include "Fairy-Stockfish/src/piece.h"
#include "Fairy-Stockfish/src/stubs.h"
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
    
    board.push_move(BOARD_A, nf3);
    board.push_move(BOARD_A, nc6);
    
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
    board.push_move(BOARD_A, find_move("g1f3"));
    board.push_move(BOARD_A, find_move("b8c6"));
    
    EXPECT_TRUE(board.is_draw(BOARD_A)) << "Should be draw after third repetition of starting position";
}

TEST_F(DrawDetectionTest, BughouseGlobalDrawDetection) {
    Board board;
    
    board.set_fen(BOARD_A, "4k3/8/8/8/8/8/8/4K3 w - - 100 100");
    board.set_fen(BOARD_B, board.startingFen);
    
    EXPECT_TRUE(board.is_draw()) << "Global draw should be detected when board A reaches 50-move rule";
    EXPECT_TRUE(board.is_draw(BOARD_A)) << "Board A should be draw by 50-move rule";
    EXPECT_FALSE(board.is_draw(BOARD_B)) << "Board B should not be draw by itself";
}

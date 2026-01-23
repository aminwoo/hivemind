#include <gtest/gtest.h>
#include "../src/board.h"
#include "../src/constants.h"
#include "Fairy-Stockfish/src/position.h"
#include "Fairy-Stockfish/src/types.h"
#include "Fairy-Stockfish/src/bitboard.h"
#include "Fairy-Stockfish/src/piece.h"
#include "Fairy-Stockfish/src/uci.h"
#include "Fairy-Stockfish/src/thread.h"

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

TEST_F(EngineTest, InitialMoves) {
    Board board;
    // Assuming Board() default ctor sets up start position
    // If not, we can explicitly set it:
    board.set_fen(BOARD_A, board.startingFen);
    
    std::vector<Stockfish::Move> movesA = board.legal_moves(0);
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
    board.set_fen(0, board.startingFen);
    
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

    // Apply move on board 0
    board.push_move(0, e2e4);
    
    std::string fen = board.fen(0);
    // e4 should be occupied by a white pawn. Checking FEN substring roughly.
    // FEN after e2e4: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1
    // Note: FEN might vary slightly with en passant target.
    EXPECT_NE(fen.find("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"), std::string::npos);
}

TEST_F(EngineTest, BughouseDrop) {
    Board board;
    board.set_fen(0, board.startingFen);

    // Manually add pawn to hand
    board.add_to_hand(BOARD_A, Stockfish::make_piece(Stockfish::WHITE, Stockfish::PAWN));
    
    auto moves = board.legal_moves(BOARD_A);
    bool found_drop = false;
    Stockfish::Move drop_e4 = Stockfish::Move::MOVE_NONE;

    for (const auto& m : moves) {
        std::string uci = board.uci_move(BOARD_A, m);
        if (uci == "P@e4") {
            found_drop = true;
            drop_e4 = m;
            break;
        }
    }
    
    EXPECT_TRUE(found_drop);
    ASSERT_NE(drop_e4, Stockfish::Move::MOVE_NONE);
    
    board.push_move(0, drop_e4);
    std::string fen = board.fen(0);
    // After drop P@e4, the board should have a pawn on e4.
    // rnbqkbnr/pppppppp/8/8/4P3/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1
    EXPECT_NE(fen.find("rnbqkbnr/pppppppp/8/8/4P3/8/PPPPPPPP/RNBQKBNR"), std::string::npos);
}

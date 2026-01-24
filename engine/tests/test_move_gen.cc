#include <gtest/gtest.h>
#include "../src/board.h"
#include "../src/constants.h"
#include "Fairy-Stockfish/src/position.h"
#include "Fairy-Stockfish/src/types.h"
#include "Fairy-Stockfish/src/bitboard.h"
#include "Fairy-Stockfish/src/piece.h"
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
    
    // Verify pawn is in hand
    int pawn_count = board.count_in_hand(BOARD_A, Stockfish::WHITE, Stockfish::PAWN);
    EXPECT_EQ(pawn_count, 1);
    
    auto moves = board.legal_moves(BOARD_A);
    
    bool found_drop = false;
    Stockfish::Move drop_e4 = Stockfish::MOVE_NONE;

    for (const auto& m : moves) {
        std::string uci = board.uci_move(BOARD_A, m);
        // UCI format uses lowercase for pieces in drops
        if (uci == "P@e4") {
            found_drop = true;
            drop_e4 = m;
            break;
        }
    }
    
    EXPECT_TRUE(found_drop);
    ASSERT_NE(drop_e4, Stockfish::MOVE_NONE);
    
    board.push_move(0, drop_e4);
    std::string fen = board.fen(0);
    // After drop p@e4, the board should have a pawn on e4
    EXPECT_NE(fen.find("4P3"), std::string::npos) << "FEN: " << fen;
}

TEST_F(EngineTest, BughouseCaptureTransfer) {
    Board board;
    board.set_fen(0, "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2");
    board.set_fen(1, board.startingFen);
    
    // Find and make the capture move exd5 on board A
    auto moves = board.legal_moves(BOARD_A);
    Stockfish::Move capture_move = Stockfish::MOVE_NONE;
    
    for (const auto& m : moves) {
        std::string uci = board.uci_move(BOARD_A, m);
        if (uci == "e4d5") {
            capture_move = m;
            break;
        }
    }
    
    ASSERT_NE(capture_move, Stockfish::MOVE_NONE) << "Capture move e4d5 not found";
    
    // Before capture, board B should have no pawns in hand
    EXPECT_EQ(board.count_in_hand(BOARD_B, Stockfish::BLACK, Stockfish::PAWN), 0);
    EXPECT_EQ(board.count_in_hand(BOARD_B, Stockfish::WHITE, Stockfish::PAWN), 0);
    
    // Make the capture on board A (White captures Black's pawn)
    board.push_move(BOARD_A, capture_move);
    
    // After capture, the captured pawn should appear on the partner board for the opponent's team
    // White captured Black's pawn on board A, so Black gets a pawn on board B
    int black_pawn_in_hand = board.count_in_hand(BOARD_B, Stockfish::BLACK, Stockfish::PAWN);
    EXPECT_EQ(black_pawn_in_hand, 1) << "Captured pawn should transfer to partner board";
    
    // Verify that if it's Black's turn, they can drop the pawn
    // Since board B starts with White to move, we need to make a move first
    auto board_b_moves = board.legal_moves(BOARD_B);
    EXPECT_GT(board_b_moves.size(), 0);
    
    // Make a move on board B so Black can move
    board.push_move(BOARD_B, board_b_moves[0]);
    
    // Now check if Black can drop the pawn
    auto black_moves = board.legal_moves(BOARD_B);
    bool has_black_pawn_drop = false;
    for (const auto& m : black_moves) {
        std::string uci = board.uci_move(BOARD_B, m);
        if (uci.find("P@") == 0) {  // Black pawn drop
            has_black_pawn_drop = true;
            break;
        }
    }
    EXPECT_TRUE(has_black_pawn_drop) << "Black should be able to drop the captured pawn";
}

TEST_F(EngineTest, PerftDepth1) {
    Board board;
    board.set_fen(BOARD_A, board.startingFen);
    auto moves = board.legal_moves(BOARD_A);
    EXPECT_EQ(moves.size(), 20) << "Starting position should have 20 legal moves";
}

TEST_F(EngineTest, PerftDepth2) {
    Board board;
    board.set_fen(BOARD_A, board.startingFen);
    
    auto moves = board.legal_moves(BOARD_A);
    long long total_nodes = 0;
    
    for (const auto& move : moves) {
        board.push_move(BOARD_A, move);
        auto responses = board.legal_moves(BOARD_A);
        total_nodes += responses.size();
        board.pop_move(BOARD_A);
    }
    
    EXPECT_EQ(total_nodes, 400) << "Perft(2) from starting position should be 400";
}

TEST_F(EngineTest, PerftDepth3) {
    Board board;
    board.set_fen(BOARD_A, board.startingFen);
    
    std::function<long long(int)> perft = [&](int depth) -> long long {
        if (depth == 0) return 1;
        
        auto moves = board.legal_moves(BOARD_A);
        if (depth == 1) return moves.size();
        
        long long nodes = 0;
        for (const auto& move : moves) {
            board.push_move(BOARD_A, move);
            nodes += perft(depth - 1);
            board.pop_move(BOARD_A);
        }
        return nodes;
    };
    
    long long nodes = perft(3);
    EXPECT_EQ(nodes, 8902) << "Perft(3) from starting position should be 8902";
}

TEST_F(EngineTest, ZobristHashConsistency) {
    Board board;
    board.set_fen(BOARD_A, board.startingFen);
    
    uint64_t initial_hash = board.hash_key(BOARD_A);
    
    auto moves = board.legal_moves(BOARD_A);
    ASSERT_GT(moves.size(), 0);
    
    for (const auto& move : moves) {
        board.push_move(BOARD_A, move);
        uint64_t after_move_hash = board.hash_key(BOARD_A);
        
        EXPECT_NE(after_move_hash, initial_hash) << "Hash should change after move " << board.uci_move(BOARD_A, move);
        
        board.pop_move(BOARD_A);
        uint64_t after_unmake_hash = board.hash_key(BOARD_A);
        
        EXPECT_EQ(after_unmake_hash, initial_hash) << "Hash should restore after unmake of move " << board.uci_move(BOARD_A, move);
    }
}

TEST_F(EngineTest, ZobristHashUniqueness) {
    Board board;
    std::set<uint64_t> seen_hashes;
    
    board.set_fen(BOARD_A, board.startingFen);
    seen_hashes.insert(board.hash_key(BOARD_A));
    
    auto moves = board.legal_moves(BOARD_A);
    for (const auto& move : moves) {
        board.push_move(BOARD_A, move);
        uint64_t hash = board.hash_key(BOARD_A);
        
        EXPECT_EQ(seen_hashes.count(hash), 0) << "Hash collision detected for move " << board.uci_move(BOARD_A, move);
        seen_hashes.insert(hash);
        
        board.pop_move(BOARD_A);
    }
}

TEST_F(EngineTest, EnPassantCapture) {
    Board board;
    board.set_fen(BOARD_A, "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2");
    
    auto moves = board.legal_moves(BOARD_A);
    Stockfish::Move ep_capture = Stockfish::MOVE_NONE;
    
    for (const auto& m : moves) {
        std::string uci = board.uci_move(BOARD_A, m);
        if (uci == "e5d6") {
            ep_capture = m;
            break;
        }
    }
    
    ASSERT_NE(ep_capture, Stockfish::MOVE_NONE) << "En passant capture should be legal";
    
    board.push_move(BOARD_A, ep_capture);
    std::string fen = board.fen(BOARD_A);
    
    EXPECT_NE(fen.find("3P4"), std::string::npos) << "White pawn should be on d6 after en passant";
    EXPECT_EQ(fen.find("3pP3"), std::string::npos) << "Black pawn on d5 should be captured";
}

TEST_F(EngineTest, Castling) {
    Board board;
    board.set_fen(BOARD_A, "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1");
    
    auto moves = board.legal_moves(BOARD_A);
    
    bool found_kingside = false;
    bool found_queenside = false;
    
    for (const auto& m : moves) {
        std::string uci = board.uci_move(BOARD_A, m);
        if (uci == "e1g1") found_kingside = true;
        if (uci == "e1c1") found_queenside = true;
    }
    
    EXPECT_TRUE(found_kingside) << "Kingside castling (e1g1) should be legal";
    EXPECT_TRUE(found_queenside) << "Queenside castling (e1c1) should be legal";
}

TEST_F(EngineTest, Promotion) {
    Board board;
    board.set_fen(BOARD_A, "8/P7/8/8/8/8/8/4K2k w - - 0 1");
    
    auto moves = board.legal_moves(BOARD_A);
    
    bool found_queen_promo = false;
    bool found_rook_promo = false;
    bool found_bishop_promo = false;
    bool found_knight_promo = false;
    
    for (const auto& m : moves) {
        std::string uci = board.uci_move(BOARD_A, m);
        if (uci == "a7a8q") found_queen_promo = true;
        if (uci == "a7a8r") found_rook_promo = true;
        if (uci == "a7a8b") found_bishop_promo = true;
        if (uci == "a7a8n") found_knight_promo = true;
    }
    
    EXPECT_TRUE(found_queen_promo) << "Queen promotion should be legal";
    EXPECT_TRUE(found_rook_promo) << "Rook promotion should be legal";
    EXPECT_TRUE(found_bishop_promo) << "Bishop promotion should be legal";
    EXPECT_TRUE(found_knight_promo) << "Knight promotion should be legal";
}

TEST_F(EngineTest, MoveUnmakeSymmetry) {
    Board board;
    board.set_fen(BOARD_A, board.startingFen);
    
    std::string initial_fen = board.fen(BOARD_A);
    uint64_t initial_hash = board.hash_key(BOARD_A);
    
    auto moves = board.legal_moves(BOARD_A);
    
    for (const auto& move : moves) {
        board.push_move(BOARD_A, move);
        board.pop_move(BOARD_A);
        
        EXPECT_EQ(board.fen(BOARD_A), initial_fen) << "FEN should match after unmake of " << board.uci_move(BOARD_A, move);
        EXPECT_EQ(board.hash_key(BOARD_A), initial_hash) << "Hash should match after unmake of " << board.uci_move(BOARD_A, move);
    }
}

TEST_F(EngineTest, BughouseMoveIndependence) {
    Board board;
    board.set_fen(BOARD_A, board.startingFen);
    board.set_fen(BOARD_B, board.startingFen);
    
    auto moves_a_initial = board.legal_moves(BOARD_A).size();
    auto moves_b_initial = board.legal_moves(BOARD_B).size();
    
    auto moves_a = board.legal_moves(BOARD_A);
    ASSERT_GT(moves_a.size(), 0);
    
    Stockfish::Move non_capture = Stockfish::MOVE_NONE;
    for (const auto& m : moves_a) {
        std::string uci = board.uci_move(BOARD_A, m);
        if (uci == "e2e4") {
            non_capture = m;
            break;
        }
    }
    ASSERT_NE(non_capture, Stockfish::MOVE_NONE);
    
    board.push_move(BOARD_A, non_capture);
    
    EXPECT_EQ(board.legal_moves(BOARD_B).size(), moves_b_initial) 
        << "Board B move count should not change for non-capture moves on Board A";
    
    EXPECT_EQ(board.count_in_hand(BOARD_B, Stockfish::WHITE, Stockfish::PAWN), 0)
        << "No pieces should be added to hand for non-capture moves";
    EXPECT_EQ(board.count_in_hand(BOARD_B, Stockfish::BLACK, Stockfish::PAWN), 0)
        << "No pieces should be added to hand for non-capture moves";
}

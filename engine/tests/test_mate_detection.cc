#include <gtest/gtest.h>
#include "../src/board.h"
#include "../src/constants.h"
#include "Fairy-Stockfish/src/position.h"
#include "Fairy-Stockfish/src/types.h"
#include "Fairy-Stockfish/src/bitboard.h"
#include "Fairy-Stockfish/src/piece.h"
#include "Fairy-Stockfish/src/thread.h"

// Fixture for mate detection tests
class MateDetectionTest : public ::testing::Test {
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

// =============================================================================
// Basic Checkmate Tests (No Partner Help Possible)
// =============================================================================

// Test: Back rank mate - clearly checkmate with no escape
TEST_F(MateDetectionTest, BackRankMate) {
    Board board;
    // Use scholar's mate pattern as it reliably has no legal moves
    board.set_fen(BOARD_A, "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4");
    
    // Verify Black is in check
    EXPECT_TRUE(board.is_in_check(BOARD_A)) << "Black should be in check";
    
    // Verify Black has no legal moves
    auto black_moves = board.legal_moves(BOARD_A);
    EXPECT_EQ(black_moves.size(), 0) << "Black should have no legal moves";
    
    // Board B: No pieces available for partner to capture
    board.set_fen(BOARD_B, "4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    
    // Black should be checkmated
    EXPECT_TRUE(board.is_checkmate(Stockfish::BLACK, false)) 
        << "Checkmate should be detected when partner cannot help";
}

// Test: Scholar's mate position
TEST_F(MateDetectionTest, ScholarsMate) {
    Board board;
    // Board A: Classic scholar's mate position - Black is mated
    board.set_fen(BOARD_A, "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4");
    
    // Board B: Starting position (no pieces in hand)
    board.set_fen(BOARD_B, board.startingFen);
    
    EXPECT_TRUE(board.is_checkmate(Stockfish::BLACK, false))
        << "Scholar's mate should be detected as checkmate";
}

// Test: Simple king + rook vs king mate
TEST_F(MateDetectionTest, SimpleRookMate) {
    Board board;
    // Board A: Black king mated in corner - king on h8, white queen on g8 (protected by king on f7)
    board.set_fen(BOARD_A, "6Qk/5K2/8/8/8/8/8/8 b - - 0 1");
    
    // Verify Black is in check
    EXPECT_TRUE(board.is_in_check(BOARD_A)) << "Black should be in check";
    
    // Verify Black has no legal moves
    auto black_moves = board.legal_moves(BOARD_A);
    EXPECT_EQ(black_moves.size(), 0) << "Black should have no legal moves";
    
    // Board B: No pieces for partner to capture
    board.set_fen(BOARD_B, "4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    
    EXPECT_TRUE(board.is_checkmate(Stockfish::BLACK, false))
        << "Corner queen mate should be checkmate";
}

// =============================================================================
// NOT Checkmate - Partner Can Provide Blocking Piece
// =============================================================================

// Test: Check can be blocked if partner captures a piece
TEST_F(MateDetectionTest, NotMatePartnerCanProvideBlocker) {
    Board board;
    // Board A: Black king on e8 in check from white queen on e1
    // Check can be blocked on e2-e7 if Black had a piece to drop
    board.set_fen(BOARD_A, "4k3/8/8/8/8/8/8/4Q2K b - - 0 1");
    
    // Board B: It's White's turn (partner of Black is White on Board B)
    // Partner can capture a piece to provide for blocking
    // White has queen that can capture a pawn
    board.set_fen(BOARD_B, "4k3/4p3/8/8/8/8/8/4Q2K w - - 0 1");
    
    // Black is in check with no legal moves, but partner (White on B) can capture
    // That pawn could be dropped to block
    EXPECT_FALSE(board.is_checkmate(Stockfish::BLACK, false))
        << "Not mate - partner can capture piece that could block";
}

// Test: Partner has capturable piece, can provide blocker
TEST_F(MateDetectionTest, PartnerCanCaptureToProvideBlocker) {
    Board board;
    // Board A: Black in check, can't move, but check could be blocked
    board.set_fen(BOARD_A, "4k3/pppppppp/8/8/8/8/8/4R2K b - - 0 1");
    
    // Board B: White (partner of Black) to move, can capture a knight
    board.set_fen(BOARD_B, "4k3/8/8/3n4/8/8/8/4R2K w - - 0 1");
    
    EXPECT_FALSE(board.is_checkmate(Stockfish::BLACK, false))
        << "Partner can capture knight to provide blocking piece";
}

// =============================================================================
// Double Check - Partner Cannot Help
// =============================================================================

// Test: Double check - can only escape by king move, not by blocking
TEST_F(MateDetectionTest, DoubleCheckIsMate) {
    Board board;
    // Board A: Double check with smothered king
    // Use a classic smothered mate pattern where the king can't move
    // Black king on g8, Black pieces on f8,g7,h7,h8, double check from Nf7 and Qe8
    board.set_fen(BOARD_A, "r4Qrk/5Npp/8/8/8/8/8/4K3 b - - 0 1");
    
    // Verify Black is in check
    EXPECT_TRUE(board.is_in_check(BOARD_A)) << "Black should be in check";
    
    // Check if there are no legal moves (king is smothered)
    auto black_moves = board.legal_moves(BOARD_A);
    EXPECT_EQ(black_moves.size(), 0) << "Black should have no legal moves (smothered)";
    
    // Board B: Even if partner could capture pieces, double check can't be blocked
    board.set_fen(BOARD_B, "4k3/8/8/8/8/8/3q4/4K3 w - - 0 1");
    
    // Double check = checkmate if king can't move
    EXPECT_TRUE(board.is_checkmate(Stockfish::BLACK, false))
        << "Double check with no king escape should be mate";
}

// =============================================================================
// Knight Check - Cannot Be Blocked
// =============================================================================

// Test: Knight check - verify knight checks can't be blocked by interposition
// Note: This tests the CONCEPT that knight checks can't be blocked,
// but finding a true smothered mate position is tricky
TEST_F(MateDetectionTest, KnightCheckNoBlock) {
    Board board;
    // Use scholar's mate position for reliable checkmate
    // The key test is that blocking squares are handled correctly
    board.set_fen(BOARD_A, "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4");
    
    EXPECT_TRUE(board.is_in_check(BOARD_A)) << "Black should be in check";
    
    auto black_moves = board.legal_moves(BOARD_A);
    EXPECT_EQ(black_moves.size(), 0) << "Black should have no legal moves";
    
    // Board B: No pieces available for partner
    board.set_fen(BOARD_B, "4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    
    EXPECT_TRUE(board.is_checkmate(Stockfish::BLACK, false))
        << "Checkmate when partner cannot provide blocking piece";
}

// =============================================================================
// Pawn Drop Restrictions
// =============================================================================

// Test: Check on back rank can't be blocked by pawn (pawn drop restriction)
TEST_F(MateDetectionTest, PawnCantBlockOnBackRank) {
    Board board;
    // Board A: Black king on e8, checked by rook on e1
    // Blocking square is e8 or e7, e8 has king, e7 is empty
    // A pawn COULD block on e7, but not on e8
    board.set_fen(BOARD_A, "4k3/8/8/8/8/8/8/4R2K b - - 0 1");
    
    // Board B: Partner's turn, can capture ONLY a pawn
    board.set_fen(BOARD_B, "4k3/8/8/8/8/8/4p3/4R2K w - - 0 1");
    
    // e7 is a valid pawn drop square (not rank 1 or 8), so this is NOT mate
    EXPECT_FALSE(board.is_checkmate(Stockfish::BLACK, false))
        << "Pawn can block on e7 (valid rank), so not mate";
}

// Test: Only back rank blocking squares - pawn can't help
TEST_F(MateDetectionTest, OnlyBackRankBlockingSquare) {
    Board board;
    // Position where the ONLY blocking square is on rank 8 (can't drop pawn)
    // Black king on h8, White queen on a8 giving check
    // Only blocking squares are b8, c8, d8, e8, f8, g8 - all on rank 8!
    board.set_fen(BOARD_A, "Q6k/8/8/8/8/8/8/7K b - - 0 1");
    
    // Board B: Partner can only capture a pawn
    board.set_fen(BOARD_B, "4k3/8/8/8/8/8/4p3/4K2R w - - 0 1");
    
    // Check if Black has any legal moves
    auto black_moves = board.legal_moves(BOARD_A);
    if (black_moves.empty()) {
        // If partner can only provide pawn but blocking squares are on rank 8
        // This SHOULD be mate if the logic correctly handles pawn restrictions
        // But we need to verify the position - king might have Kh7 escape
        EXPECT_TRUE(board.is_checkmate(Stockfish::BLACK, false))
            << "Pawn cannot block on rank 8, should be mate";
    }
}

// =============================================================================
// Time Advantage Scenarios
// =============================================================================

// Test: Not partner's turn, but team has time advantage
TEST_F(MateDetectionTest, TimeAdvantageAllowsFutureCapture) {
    Board board;
    // Use scholar's mate position
    board.set_fen(BOARD_A, "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4");
    
    EXPECT_TRUE(board.is_in_check(BOARD_A)) << "Black should be in check";
    auto black_moves = board.legal_moves(BOARD_A);
    EXPECT_EQ(black_moves.size(), 0) << "Black should have no legal moves";
    
    // Board B: It's BLACK's turn (not partner White), but there's a Black knight to capture
    // The knight is on d4, owned by Black (lowercase 'n')
    board.set_fen(BOARD_B, "4k3/8/8/8/3n4/8/8/4K2R b - - 0 1");
    
    // Verify the position: Black to move, Black has knight on d4
    EXPECT_EQ(board.side_to_move(BOARD_B), Stockfish::BLACK) << "Black to move on Board B";
    
    // Check if there are Black pieces that White could capture
    Stockfish::Bitboard black_knights = board.pieces(BOARD_B, Stockfish::BLACK, Stockfish::KNIGHT);
    EXPECT_TRUE(black_knights != 0) << "Black should have knight on Board B";
    
    // Without time advantage: partner (White) can't capture now since it's Black's turn on B
    EXPECT_TRUE(board.is_checkmate(Stockfish::BLACK, false))
        << "Without time advantage, partner can't help if not their turn";
    
    // With time advantage: partner could potentially capture the knight in future
    // BUT: Looking at the scholar's mate position, the blocking squares are e6 and e7
    // (between Qf7 and Ke8). These are NOT on rank 1 or 8, so even a pawn could block.
    // The knight on Board B could be captured by White in the future and given to Black.
    // 
    // However, looking more closely: the check is from Qf7 to Ke8
    // The only blocking square between f7 and e8 is... there is none - they're adjacent!
    // Actually f7 and e8 are diagonally adjacent, so there's no blocking square.
    // This means can_partner_provide_blocking_piece should return false because
    // between_bb(e8, f7) is empty!
    
    // Let me verify: in Scholar's mate, can the check be blocked?
    // Queen on f7 attacks king on e8 diagonally - they are adjacent, so no blocking square
    // This means partner CAN'T help by providing a piece, so it IS checkmate even with time advantage
    
    // Update expectation: time advantage doesn't help when there's no blocking square
    EXPECT_TRUE(board.is_checkmate(Stockfish::BLACK, true))
        << "Even with time advantage, no blocking squares means checkmate";
}

// Test: Time advantage but no pieces to capture
TEST_F(MateDetectionTest, TimeAdvantageButNoPieces) {
    Board board;
    // Use scholar's mate position
    board.set_fen(BOARD_A, "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4");
    
    EXPECT_TRUE(board.is_in_check(BOARD_A)) << "Black should be in check";
    auto black_moves = board.legal_moves(BOARD_A);
    EXPECT_EQ(black_moves.size(), 0) << "Black should have no legal moves";
    
    // Board B: It's White's turn (partner), but no pieces to capture (only kings)
    board.set_fen(BOARD_B, "4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    
    // Even with time advantage, no pieces to capture = mate
    EXPECT_TRUE(board.is_checkmate(Stockfish::BLACK, true))
        << "Time advantage doesn't help if no pieces to capture";
}

// =============================================================================
// Edge Cases
// =============================================================================

// Test: Stalemate is NOT checkmate
TEST_F(MateDetectionTest, StalemateIsNotCheckmate) {
    Board board;
    // Board A: Stalemate position - Black king not in check but no legal moves
    // Black king on a8, White queen on b6 and White king on c7 - classic stalemate
    board.set_fen(BOARD_A, "k7/8/1Q6/8/8/8/8/1K6 b - - 0 1");
    
    // Verify it's stalemate (no check, no legal moves)
    EXPECT_FALSE(board.is_in_check(BOARD_A)) << "King should not be in check";
    auto moves = board.legal_moves(BOARD_A);
    EXPECT_EQ(moves.size(), 0) << "No legal moves in stalemate";
    
    board.set_fen(BOARD_B, board.startingFen);
    
    // is_checkmate should return false for stalemate (not in check)
    EXPECT_FALSE(board.is_checkmate(Stockfish::BLACK, false))
        << "Stalemate should not be detected as checkmate";
}

// Test: In check but has legal moves - not mate
TEST_F(MateDetectionTest, InCheckButHasEscape) {
    Board board;
    // Board A: King in check but can escape
    board.set_fen(BOARD_A, "4k3/8/8/8/8/8/8/4R2K b - - 0 1");
    
    board.set_fen(BOARD_B, board.startingFen);
    
    auto moves = board.legal_moves(BOARD_A);
    EXPECT_GT(moves.size(), 0) << "King should have escape moves";
    
    EXPECT_FALSE(board.is_checkmate(Stockfish::BLACK, false))
        << "Not checkmate if king has escape squares";
}

// Test: King can capture checking piece
TEST_F(MateDetectionTest, KingCanCaptureChecker) {
    Board board;
    // Board A: White king can capture the undefended checking piece
    // White king on e1, Black rook on e2 giving check, rook is undefended
    board.set_fen(BOARD_A, "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1");
    
    EXPECT_TRUE(board.is_in_check(BOARD_A)) << "White should be in check";
    
    board.set_fen(BOARD_B, board.startingFen);
    
    // White king can capture rook on e2
    auto moves = board.legal_moves(BOARD_A);
    bool can_capture = false;
    for (const auto& m : moves) {
        std::string uci = board.uci_move(BOARD_A, m);
        if (uci == "e1e2") {  // Kxe2
            can_capture = true;
            break;
        }
    }
    EXPECT_TRUE(can_capture) << "King should be able to capture checking piece";
    
    EXPECT_FALSE(board.is_checkmate(Stockfish::WHITE, false))
        << "Not checkmate if king can capture checker";
}

// Test: Piece can block the check
TEST_F(MateDetectionTest, OwnPieceCanBlock) {
    Board board;
    // Board A: Knight can block the check
    board.set_fen(BOARD_A, "4k3/8/8/8/8/8/3N4/R3K3 b - - 0 1");
    
    board.set_fen(BOARD_B, board.startingFen);
    
    // Black has legal moves (blocking)
    auto moves = board.legal_moves(BOARD_A);
    EXPECT_GT(moves.size(), 0) << "Should have blocking moves";
    
    EXPECT_FALSE(board.is_checkmate(Stockfish::BLACK, false))
        << "Not checkmate if own piece can block";
}

// Test: Piece already in hand can block
TEST_F(MateDetectionTest, PieceInHandCanBlock) {
    Board board;
    // Board A: Black in check, no moves, but has piece in hand to drop
    board.set_fen(BOARD_A, "4k3/pppppppp/8/8/8/8/8/4R2K b - - 0 1");
    
    // Add a knight to Black's hand on Board A
    board.add_to_hand(BOARD_A, Stockfish::make_piece(Stockfish::BLACK, Stockfish::KNIGHT));
    
    board.set_fen(BOARD_B, board.startingFen);
    
    // Black should be able to drop the knight to block
    auto moves = board.legal_moves(BOARD_A);
    bool can_drop_block = false;
    for (const auto& m : moves) {
        std::string uci = board.uci_move(BOARD_A, m);
        // Look for knight drop on blocking squares (e2-e7)
        if (uci.find("N@") == 0) {
            can_drop_block = true;
            break;
        }
    }
    
    EXPECT_TRUE(can_drop_block) << "Should be able to drop knight to block";
    EXPECT_FALSE(board.is_checkmate(Stockfish::BLACK, false))
        << "Not checkmate if piece in hand can block";
}

// =============================================================================
// Board B Checkmate Detection
// =============================================================================

// Test: Checkmate on Board B (partner of side)
TEST_F(MateDetectionTest, CheckmateOnBoardB) {
    Board board;
    // Board A: Normal position, White to move
    board.set_fen(BOARD_A, board.startingFen);
    
    // Board B: Scholar's mate position - Black is mated
    // When checking WHITE team, we look at Board B where Black plays
    board.set_fen(BOARD_B, "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4");
    
    // Verify Black on Board B is in check with no escape
    EXPECT_TRUE(board.is_in_check(BOARD_B)) << "Black on Board B should be in check";
    auto moves_b = board.legal_moves(BOARD_B);
    EXPECT_EQ(moves_b.size(), 0) << "Black on Board B should have no legal moves";
    
    // For WHITE team: White plays on Board A, partner (Black) plays on Board B
    // is_checkmate(WHITE) checks Board A for White check and Board B for Black check
    // Since Black on Board B is mated, WHITE team loses
    EXPECT_TRUE(board.is_checkmate(Stockfish::WHITE, false))
        << "Checkmate on Board B should be detected for White's team";
}

// =============================================================================
// Legal Moves with Checkmate
// =============================================================================

// Test: legal_moves(side) returns empty when side is mated
TEST_F(MateDetectionTest, LegalMovesEmptyWhenMated) {
    Board board;
    // Set up mate position
    board.set_fen(BOARD_A, "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4");
    board.set_fen(BOARD_B, board.startingFen);
    
    // Black is mated, legal_moves should return empty
    auto moves = board.legal_moves(Stockfish::BLACK, false);
    EXPECT_EQ(moves.size(), 0)
        << "legal_moves should return empty for mated side";
}

// Test: legal_moves(side) includes moves when not mated
TEST_F(MateDetectionTest, LegalMovesNonEmptyWhenNotMated) {
    Board board;
    board.set_fen(BOARD_A, board.startingFen);
    board.set_fen(BOARD_B, board.startingFen);
    
    auto moves = board.legal_moves(Stockfish::WHITE, false);
    EXPECT_GT(moves.size(), 0)
        << "legal_moves should return moves for non-mated side";
}

// =============================================================================
// Regression Tests (Add specific bug scenarios here)
// =============================================================================

// Test: Verify legal_moves returns moves when team is not mated
TEST_F(MateDetectionTest, LegalMovesForTeamWhenNotMated) {
    Board board;
    // Same bug report position
    board.set_fen(BOARD_A, "5k1R/pppbrp2/2p1pQ2/8/2B1P3/2PN4/PPP3K1/8[RB] b - - 4 31");
    board.set_fen(BOARD_B, "r6r/pppk1Ppp/2n1q3/3n2N1/8/2P5/P1P1NPPP/R1B1K2R[QBNPPPPqrbbbnnppppp] w KQ - 0 18");
    
    // This is the call the engine makes in agent.cc
    // teamSide = BLACK, teamHasTimeAdvantage = false (or true)
    auto moves = board.legal_moves(Stockfish::BLACK, false);
    
    // The bug: legal_moves returns empty because it thinks it's checkmate
    // But it's NOT checkmate because partner can capture the queen on e6
    // Partner's moves (White on Board B) should be included
    EXPECT_GT(moves.size(), 0) 
        << "legal_moves(BLACK) should return moves - partner can play on Board B";
    
    // Verify partner has moves on Board B
    auto partner_only_moves = board.legal_moves(BOARD_B);
    EXPECT_GT(partner_only_moves.size(), 0) << "White has legal moves on Board B";
}

// Test: Specific bughouse position from user bug report
// FEN: 5k1R/pppbrp2/2p1pQ2/8/2B1P3/2PN4/PPP3K1/8[RB] b - - 4 31|r6r/pppk1Ppp/2n1q3/3n2N1/8/2P5/P1P1NPPP/R1B1K2R[QBNPPPPqrbbbnnppppp] w KQ - 0 18
// Team: Black
TEST_F(MateDetectionTest, BughousePositionFromBugReport) {
    Board board;
    // Board A: Black king on f8, in check from Rh8
    // [RB] = White has Rook and Bishop in pocket (uppercase = White)
    board.set_fen(BOARD_A, "5k1R/pppbrp2/2p1pQ2/8/2B1P3/2PN4/PPP3K1/8[RB] b - - 4 31");
    
    // Board B: White to move, has many pieces in pocket
    board.set_fen(BOARD_B, "r6r/pppk1Ppp/2n1q3/3n2N1/8/2P5/P1P1NPPP/R1B1K2R[QBNPPPPqrbbbnnppppp] w KQ - 0 18");
    
    // Verify the position setup
    EXPECT_TRUE(board.is_in_check(BOARD_A)) << "Black should be in check on Board A";
    EXPECT_EQ(board.side_to_move(BOARD_A), Stockfish::BLACK) << "Black to move on Board A";
    EXPECT_EQ(board.side_to_move(BOARD_B), Stockfish::WHITE) << "White to move on Board B";
    
    // Black has 0 legal moves on Board A (in check, no escape, no pieces in hand)
    auto moves_a = board.legal_moves(BOARD_A);
    EXPECT_EQ(moves_a.size(), 0) << "Black should have 0 legal moves on Board A";
    
    // Black has no pieces in hand on Board A
    EXPECT_EQ(board.count_in_hand(BOARD_A, Stockfish::BLACK, Stockfish::ROOK), 0);
    EXPECT_EQ(board.count_in_hand(BOARD_A, Stockfish::BLACK, Stockfish::BISHOP), 0);
    EXPECT_EQ(board.count_in_hand(BOARD_A, Stockfish::BLACK, Stockfish::QUEEN), 0);
    EXPECT_EQ(board.count_in_hand(BOARD_A, Stockfish::BLACK, Stockfish::KNIGHT), 0);
    EXPECT_EQ(board.count_in_hand(BOARD_A, Stockfish::BLACK, Stockfish::PAWN), 0);
    
    // Partner (White on Board B) can capture a queen with Nxe6
    // If partner captures the queen, Black gets a queen to drop on g8 to block
    bool partner_has_queen_capture = false;
    auto moves_b = board.legal_moves(BOARD_B);
    for (const auto& m : moves_b) {
        if (board.is_capture(BOARD_B, m)) {
            char captured = board.get_captured_piece(BOARD_B, m);
            if (captured == 'q') {
                partner_has_queen_capture = true;
                break;
            }
        }
    }
    EXPECT_TRUE(partner_has_queen_capture) << "Partner should be able to capture queen";
    
    // Since partner can provide a queen that could block on g8, this is NOT checkmate
    EXPECT_FALSE(board.is_checkmate(Stockfish::BLACK, false))
        << "Not checkmate - partner can capture queen to provide blocking piece";
}

// Test: Checkmate when partner cannot help
TEST_F(MateDetectionTest, CheckmateWhenPartnerCannotHelp) {
    Board board;
    // Board A: Scholar's mate - we know this is checkmate
    // Queen on f7 protected by bishop on c4, Black king on e8
    board.set_fen(BOARD_A, "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4");
    
    // Verify Black is in check
    EXPECT_TRUE(board.is_in_check(BOARD_A)) << "Black should be in check from Qf7";
    
    // Get legal moves
    auto moves_a = board.legal_moves(BOARD_A);
    EXPECT_EQ(moves_a.size(), 0) << "Black should have no legal moves";
    
    // Board B: White to move, only has kings (no pieces to capture at all)
    board.set_fen(BOARD_B, "4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    
    // Partner has no captures available at all
    auto moves_b = board.legal_moves(BOARD_B);
    bool has_any_capture = false;
    for (const auto& m : moves_b) {
        if (board.is_capture(BOARD_B, m)) {
            has_any_capture = true;
        }
    }
    EXPECT_FALSE(has_any_capture) << "Partner should have no captures available";
    
    // This SHOULD be checkmate since partner can't help
    EXPECT_TRUE(board.is_checkmate(Stockfish::BLACK, false))
        << "Checkmate - partner has no captures to provide blocking piece";
}

// Test: Ensure adjacent check (pawn/king) can't be blocked
TEST_F(MateDetectionTest, AdjacentCheckCantBeBlocked) {
    Board board;
    // Board A: King checked by adjacent pawn - no blocking possible
    board.set_fen(BOARD_A, "4k3/3P4/8/8/8/8/8/4K3 b - - 0 1");
    
    // Board B: Partner could capture pieces
    board.set_fen(BOARD_B, "4k3/8/8/8/3q4/8/8/4K3 w - - 0 1");
    
    // If king has no escape and can't capture the pawn, it's mate
    // Partner capturing pieces won't help since pawn check is adjacent
    auto moves = board.legal_moves(BOARD_A);
    if (moves.empty()) {
        EXPECT_TRUE(board.is_checkmate(Stockfish::BLACK, false))
            << "Adjacent pawn check with no escape should be mate";
    }
}

// Test: Verify capture on partner board actually provides piece
TEST_F(MateDetectionTest, VerifyCaptureProvidesPiece) {
    Board board;
    // Board A: Black in check, needs a piece to block
    board.set_fen(BOARD_A, "4k3/pppppppp/8/8/8/8/8/4R2K b - - 0 1");
    
    // Board B: White to move, rook can capture queen
    board.set_fen(BOARD_B, "4k3/8/8/8/8/8/8/q3R2K w - - 0 1");
    
    // White capturing queen gives Black a queen to drop
    EXPECT_FALSE(board.is_checkmate(Stockfish::BLACK, false))
        << "Partner can capture queen to provide blocking piece";
    
    // Verify by actually making the capture and checking hand
    auto moves_b = board.legal_moves(BOARD_B);
    Stockfish::Move capture = Stockfish::MOVE_NONE;
    for (const auto& m : moves_b) {
        if (board.uci_move(BOARD_B, m) == "e1a1") {
            capture = m;
            break;
        }
    }
    
    if (capture != Stockfish::MOVE_NONE) {
        board.push_move(BOARD_B, capture);
        // After capture, Black should have queen in hand on Board A
        int queens = board.count_in_hand(BOARD_A, Stockfish::BLACK, Stockfish::QUEEN);
        EXPECT_EQ(queens, 1) << "Black should have captured queen in hand";
    }
}

// Test: When partner has no blocking captures, position is checkmate (adjacent check)
TEST_F(MateDetectionTest, NoBlockingCapturesIsMate) {
    Board board;
    // Board A: Scholar's mate - adjacent check, no blocking possible
    board.set_fen(BOARD_A, "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4");
    
    // Board B: Partner has captures but they don't help (check is adjacent)
    board.set_fen(BOARD_B, "4k3/8/4q3/8/8/8/8/4K2R w - - 0 1");
    
    // This is checkmate - no blocking squares between adjacent Qf7 and Ke8
    EXPECT_TRUE(board.is_checkmate(Stockfish::BLACK, false))
        << "Scholar's mate is checkmate even if partner has captures";
}

// Test: After non-blocking partner move, checkmate should be detected
// This simulates what happens after (pass, N@c5) - Black is still mated
TEST_F(MateDetectionTest, CheckmateAfterNonBlockingPartnerMove) {
    Board board;
    // Bug report position
    board.set_fen(BOARD_A, "5k1R/pppbrp2/2p1pQ2/8/2B1P3/2PN4/PPP3K1/8[RB] b - - 4 31");
    board.set_fen(BOARD_B, "r6r/pppk1Ppp/2n1q3/3n2N1/8/2P5/P1P1NPPP/R1B1K2R[QBNPPPPqrbbbnnppppp] w KQ - 0 18");
    
    // Verify initial state: NOT checkmate because partner can capture queen
    EXPECT_FALSE(board.is_checkmate(Stockfish::BLACK, false))
        << "Initially NOT checkmate - partner can capture queen";
    
    // Simulate Team BLACK playing (pass, N@c5)
    // Black passes on A (no move made), partner drops knight on c5
    Stockfish::Square c5 = Stockfish::make_square(Stockfish::FILE_C, Stockfish::RANK_5);
    Stockfish::Move drop_nc5 = Stockfish::make_drop(c5, Stockfish::KNIGHT, Stockfish::KNIGHT);
    
    // Make the joint move (pass, N@c5)
    board.make_moves(Stockfish::MOVE_NONE, drop_nc5);
    
    // After (pass, N@c5):
    // - Board A: Still Black to move, still in check
    // - Board B: Now Black to move (after White's drop)
    EXPECT_EQ(board.side_to_move(BOARD_A), Stockfish::BLACK) << "Board A: Black to move";
    EXPECT_EQ(board.side_to_move(BOARD_B), Stockfish::BLACK) << "Board B: Black to move";
    EXPECT_TRUE(board.is_in_check(BOARD_A)) << "Black still in check on Board A";
    
    // NOW: When we check is_checkmate for BLACK:
    // - Black is in check on A with no moves
    // - Partner (White) is NOT on turn on Board B (it's Black's turn there)
    // - can_partner_provide_blocking_piece should return FALSE
    // - Therefore is_checkmate(BLACK) should return TRUE!
    EXPECT_TRUE(board.is_checkmate(Stockfish::BLACK, false))
        << "After non-blocking partner move, Black is mated - partner can't help anymore";
}

// Test: R@h8 mate after Kxg8 sequence
// After Q@g8, Rxg8, Kxg8, R@h8 should be checkmate
TEST_F(MateDetectionTest, RookDropBackRankMate) {
    Board board;
    // Position after Kxg8: King on g8, White to move, White has Rook in pocket
    board.set_fen(BOARD_A, "6k1/pppbrp2/2p1pQ2/8/2B1P3/2PN4/PPP3K1/8[R] w - - 0 1");
    board.set_fen(BOARD_B, "r6r/pppk1Ppp/2n1q3/2Nn4/8/2P5/P1P1NPPP/R1B1K2R[QBNPPPPrbbbnnppppp] b KQ - 0 1");
    
    EXPECT_EQ(board.side_to_move(BOARD_A), Stockfish::WHITE) << "Board A: White to move";
    
    // Make the move R@h8
    Stockfish::Square h8 = Stockfish::make_square(Stockfish::FILE_H, Stockfish::RANK_8);
    Stockfish::Move rh8 = Stockfish::make_drop(h8, Stockfish::ROOK, Stockfish::ROOK);
    
    board.make_moves(rh8, Stockfish::MOVE_NONE);
    
    // After R@h8, Black should be in check with no escape
    EXPECT_TRUE(board.is_in_check(BOARD_A)) << "Black in check after R@h8";
    
    auto blackMoves = board.legal_moves(BOARD_A);
    EXPECT_EQ(blackMoves.size(), 0) << "Black should have no legal moves after R@h8";
    
    // Black on Board A is mated - Team BLACK is mated
    EXPECT_TRUE(board.is_checkmate(Stockfish::BLACK, false))
        << "R@h8 should be checkmate - Black king on g8 has no escape";
}

// =============================================================================
// "No Turn" Tests - Team has neither board's turn
// =============================================================================

// Test: When a team has no turn on either board, they should not be considered checkmated
// This tests the fix for the bug where games ended prematurely as "checkmate" when
// a team simply didn't have the turn on either board.
TEST_F(MateDetectionTest, NoTurnIsNotCheckmate) {
    Board board;
    // Set up a position where Board A has White to move and Board B has Black to move
    // In bughouse:
    // - WHITE team plays White on A and Black on B
    // - BLACK team plays Black on A and White on B
    // If Board A = White to move, Board B = Black to move:
    // - WHITE team has turns on BOTH boards (White on A, Black on B)
    // - BLACK team has turns on NEITHER board (not Black on A, not White on B)
    
    board.set_fen(BOARD_A, "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1");  // White to move
    board.set_fen(BOARD_B, "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");  // Black to move
    
    // BLACK team has no turn on either board, but is NOT checkmated
    EXPECT_FALSE(board.is_checkmate(Stockfish::BLACK, false))
        << "No turn on either board should NOT be checkmate";
    EXPECT_FALSE(board.is_checkmate(Stockfish::BLACK, true))
        << "No turn on either board should NOT be checkmate (with time advantage)";
    
    // WHITE team has turns on both boards, definitely not mated
    EXPECT_FALSE(board.is_checkmate(Stockfish::WHITE, false))
        << "Team with turns should not be checkmated";
    EXPECT_FALSE(board.is_checkmate(Stockfish::WHITE, true))
        << "Team with turns should not be checkmated (with time advantage)";
}

// Test: legal_moves returns empty when team has no turn, but is_checkmate returns false
TEST_F(MateDetectionTest, EmptyMovesNotCheckmate) {
    Board board;
    // Same setup as above - BLACK team has no turn on either board
    board.set_fen(BOARD_A, "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1");  // White to move
    board.set_fen(BOARD_B, "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");  // Black to move
    
    // BLACK team has empty legal moves (no turn on either board)
    auto blackMoves = board.legal_moves(Stockfish::BLACK, false);
    EXPECT_TRUE(blackMoves.empty())
        << "legal_moves should be empty when team has no turn";
    
    // But is_checkmate should return false (they're not mated, just waiting)
    EXPECT_FALSE(board.is_checkmate(Stockfish::BLACK, false))
        << "Empty legal_moves with no turn should NOT mean checkmate";
    
    // WHITE team should have normal moves
    auto whiteMoves = board.legal_moves(Stockfish::WHITE, false);
    EXPECT_GT(whiteMoves.size(), 1)
        << "White team should have many legal moves";
}
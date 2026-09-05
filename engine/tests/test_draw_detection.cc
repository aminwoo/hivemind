#include <gtest/gtest.h>
#include <algorithm>
#include <sstream>
#include "environment/board.h"
#include "environment/constants.h"
#include "common/globals.h"
#include "Fairy-Stockfish/src/position.h"
#include "Fairy-Stockfish/src/types.h"
#include "Fairy-Stockfish/src/bitboard.h"
#include "Fairy-Stockfish/src/piece.h"
#include "Fairy-Stockfish/src/thread.h"

class DrawDetectionTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        init_fairy_stockfish();
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

TEST_F(DrawDetectionTest, WaitingBoardTwofoldIsNotSearchDraw) {
    Board board;
    board.set_fen(BOARD_B, board.startingFen);

    auto push = [&](const std::string& uci) {
        std::string moveText = uci;
        const Stockfish::Move move = Stockfish::UCI::to_move(
            *board.pos[BOARD_B], moveText);
        ASSERT_NE(move, Stockfish::MOVE_NONE) << uci;
        board.push_move(BOARD_B, move);
    };
    push("g1f3");
    push("g8f6");
    push("f3g1");
    push("f6g8");

    ASSERT_EQ(board.repetition_count(BOARD_B), 2);
    EXPECT_FALSE(board.is_draw(std::array<int, 2>{1, 0}))
        << "A move on board A must not turn board B's existing twofold into a draw";
    EXPECT_TRUE(board.is_draw(std::array<int, 2>{0, 1}))
        << "A repetition reached on board B inside the search is a draw";
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

TEST_F(DrawDetectionTest, RepetitionKeyIgnoresPocketPieces) {
    Board board;

    board.set_fen(BOARD_A, "4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    uint64_t emptyPocketKey = board.board_only_key(BOARD_A);

    board.set_fen(BOARD_A, "4k3/8/8/8/8/8/8/4K3[P] w - - 0 1");
    uint64_t pawnPocketKey = board.board_only_key(BOARD_A);

    EXPECT_EQ(emptyPocketKey, pawnPocketKey);
}

TEST_F(DrawDetectionTest, ReportedKnightKingCycleIsOnlyTwofold) {
    Board board;
    const std::string moves =
        "1e2e4 1e7e5 1g1f3 1b8c6 1f1c4 1f8e7 1b1c3 2e2e4 2g8f6 2b1c3 "
        "2d7d5 2e4d5 1P@e6 1d2d3 1g8f6 2f6d5 1P@h6 1h8g8 1h6g7 1g8g7 "
        "1c1h6 1g7g2 2g1f3 2b8c6 2d2d4 2e7e6 2f1d3 2f8b4 2c1d2 2P@f4 "
        "2P@h6 2g7h6 1P@g7 1g2g7 1h6g7 2c3d5 1N@g2 1e1d2 2d8d5 2d2b4 "
        "1B@f4 2c6b4 1B@e3 1g2e3 1f2e3 1f4e3 1d2e3 1f6g4 1e3d2 1e7g5 "
        "1d2e1 2P@g7 2b4d3 2d1d3 1N@g2 1e1e2 1g2f4 1e2e1 2h8g8 2B@e4 "
        "2d5d8 2e4h7 1P@f2 1e1d2 1f4g2 1d2e2 1g2f4 1e2d2 1f4g2 1d2e2 "
        "2g8g7 2P@g6";

    std::istringstream stream(moves);
    std::string token;
    while (stream >> token) {
        const int boardNum = token[0] - '1';
        std::string moveText = token.substr(1);
        Stockfish::Move move = Stockfish::UCI::to_move(*board.pos[boardNum], moveText);
        ASSERT_NE(move, Stockfish::MOVE_NONE) << "Invalid move: " << token;
        board.push_move(boardNum, move);
    }

    const uint64_t currentKey = board.board_only_key(BOARD_A);
    const int occurrences = static_cast<int>(std::count(
        board.positionHistory[BOARD_A].begin(),
        board.positionHistory[BOARD_A].end(), currentKey));

    EXPECT_EQ(occurrences, 2);
    EXPECT_FALSE(board.is_draw_on_board(BOARD_A));
    EXPECT_FALSE(board.is_draw());
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

TEST_F(DrawDetectionTest, ReportedBoardTwoMoveCompletesThreefoldRepetition) {
    Board board;
    const std::string moves =
        "1g1f3 1d7d5 1d2d4 1b8c6 1b1c3 1c8g4 1c1f4 1e7e6 1h2h3 1g4f3 "
        "1e2f3 1f8d6 1f4d6 1c7d6 2e2e4 2g8f6 2b1c3 2b8c6 2g1f3 2d7d5 "
        "2e4d5 2f6d5 2d2d4 2e7e5 2c3d5 2d8d5 1N@h5 1N@f5 1P@g4 1P@e3 "
        "1g4f5 2N@e3 2B@a5 2B@c3 2a5c3 2b2c3 1B@h4 1g2g3 1e3f2 1e1f2 "
        "1h4g3 1f2g3 1d8g5 2d5a5 2d4e5 2a5c3 1P@g4 1g5h5 1g4h5 1g8f6 "
        "1d1e2 1f6h5 1g3f2 1e8g8 1h1g1 1P@g3 1g1g3 1h5g3 1f2g3 1c6d4 "
        "1e2e3 2c1d2 2c3c5 2N@e4 2c5e7 2P@f6 2e7d8 2f6g7 2f8g7 2P@f6 "
        "2B@f8 2f6g7 2f8g7 2P@f6 2B@f8";

    std::istringstream stream(moves);
    std::string token;
    while (stream >> token) {
        const int boardNum = token[0] - '1';
        std::string moveText = token.substr(1);
        Stockfish::Move move = Stockfish::UCI::to_move(*board.pos[boardNum], moveText);
        ASSERT_NE(move, Stockfish::MOVE_NONE) << "Invalid move: " << token;
        board.push_move(boardNum, move);
    }

    EXPECT_FALSE(board.is_draw());

    std::string repetitionText = "f6g7";
    Stockfish::Move repetition = Stockfish::UCI::to_move(*board.pos[BOARD_B], repetitionText);
    ASSERT_NE(repetition, Stockfish::MOVE_NONE);
    board.push_move(BOARD_B, repetition);

    EXPECT_TRUE(board.is_draw_on_board(BOARD_B));
    EXPECT_TRUE(board.is_draw());
}

TEST_F(DrawDetectionTest, GptSolQc7CompletesThreefoldRepetition) {
    Board board;
    const std::string moves =
        "1e2e4 1e7e5 1g1f3 1b8c6 1d2d4 1e5d4 1f3d4 1f8c5 1d4f5 1c5f2 "
        "1e1f2 1g8f6 1f5g7 1e8f8 1g7f5 1f6e4 1f2e1 1d8h4 1f5h4 2e2e4 "
        "2e7e6 2e4e5 2B@d4 2P@e3 2d4e5 2g1f3 2e5d6 2d2d4 2b8c6 2f1b5 "
        "2P@e4 2b5c6 1N@f2 1P@g7 1f8g7 2b7c6 1B@d4 1c6d4 1d1d4 1f7f6 "
        "2f3e5 2d6e5 1N@f5 1g7f7 1f1c4 2d4e5 1B@e6 1c4e6 1d7e6 1f5h6 "
        "1f7g7 1b1c3 1f2h1 1c3e4 2P@f3 2g2f3 1P@f2 1e4f2 1h1f2 1h6f5 "
        "1e6f5 1c1h6 1g7h6 1d4f6 1h6h5 1f6h8 1h5h4 1h8h7 1h4g4 "
        "1h7g7 1g4f4 1g7c7 1f4g5 1c7g7 1g5f4 1g7c7 1f4g5 1c7g7 "
        "1g5f4 2N@g2 2e1f1";

    std::istringstream stream(moves);
    std::string token;
    while (stream >> token) {
        const int boardNum = token[0] - '1';
        std::string moveText = token.substr(1);
        const Stockfish::Move move = Stockfish::UCI::to_move(
            *board.pos[boardNum], moveText);
        ASSERT_NE(move, Stockfish::MOVE_NONE) << "Invalid move: " << token;
        board.push_move(boardNum, move);
    }

    EXPECT_FALSE(board.is_draw());
    std::string repetitionText = "g7c7";
    const Stockfish::Move repetition = Stockfish::UCI::to_move(
        *board.pos[BOARD_A], repetitionText);
    ASSERT_NE(repetition, Stockfish::MOVE_NONE);
    board.push_move(BOARD_A, repetition);

    EXPECT_TRUE(board.is_repetition_draw({0, 0}));
    EXPECT_TRUE(board.is_draw());
}

// The repetition key is a Zobrist over the board fields rather than a hashed
// FEN string. Beyond ignoring pockets (covered above), it must still separate
// promoted pieces, side to move, castling rights and the en passant file, and
// must be restored exactly by unmake.
TEST_F(DrawDetectionTest, RepetitionKeyTracksEveryBoardField) {
    Board withEmptyHand;
    withEmptyHand.set_fen(BOARD_A, "4k3/8/8/8/8/8/8/4K3[] w - - 0 1");

    Board blackToMove;
    blackToMove.set_fen(BOARD_A, "4k3/8/8/8/8/8/8/4K3[] b - - 0 1");
    EXPECT_NE(withEmptyHand.board_only_key(BOARD_A),
              blackToMove.board_only_key(BOARD_A))
        << "Side to move must affect repetition identity";

    Board promotedQueen;
    promotedQueen.set_fen(BOARD_A, "4k3/8/8/8/8/8/8/3QK3[] w - - 0 1");
    Board originalQueen;
    originalQueen.set_fen(BOARD_A, "4k3/8/8/8/8/8/8/3Q~K3[] w - - 0 1");
    EXPECT_NE(promotedQueen.board_only_key(BOARD_A),
              originalQueen.board_only_key(BOARD_A))
        << "A promoted piece is not the same piece for repetition purposes";

    Board withCastling;
    withCastling.set_fen(BOARD_A, "r3k2r/8/8/8/8/8/8/R3K2R[] w KQkq - 0 1");
    Board withoutCastling;
    withoutCastling.set_fen(BOARD_A, "r3k2r/8/8/8/8/8/8/R3K2R[] w - - 0 1");
    EXPECT_NE(withCastling.board_only_key(BOARD_A),
              withoutCastling.board_only_key(BOARD_A))
        << "Castling rights must affect repetition identity";

    Board withEnPassant;
    withEnPassant.set_fen(BOARD_A, "4k3/8/8/3pP3/8/8/8/4K3[] w - d6 0 2");
    Board withoutEnPassant;
    withoutEnPassant.set_fen(BOARD_A, "4k3/8/8/3pP3/8/8/8/4K3[] w - - 0 2");
    EXPECT_NE(withEnPassant.board_only_key(BOARD_A),
              withoutEnPassant.board_only_key(BOARD_A))
        << "The en passant file must affect repetition identity";

    // The key must survive make/unmake unchanged.
    Board roundTrip;
    roundTrip.set_fen(BOARD_A, "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R[Qp] w KQkq - 0 1");
    const uint64_t before = roundTrip.board_only_key(BOARD_A);
    for (Stockfish::Move move : roundTrip.legal_moves(BOARD_A)) {
        roundTrip.push_move(BOARD_A, move);
        roundTrip.pop_move(BOARD_A);
        ASSERT_EQ(roundTrip.board_only_key(BOARD_A), before)
            << "Key changed after make/unmake of " << roundTrip.uci_move(BOARD_A, move);
    }
}

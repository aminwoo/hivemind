#include <gtest/gtest.h>

#include <atomic>
#include <stdexcept>
#include <vector>

#include "common/globals.h"
#include "environment/adjudication.h"
#include "environment/planes.h"
#include "search/single_board.h"

namespace {

class SingleBoardTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        init_fairy_stockfish();
        init_policy_index();
    }
};

Stockfish::Move push(Board& board, std::string uci) {
    const auto move = Stockfish::UCI::to_move(*board.pos[BOARD_A], uci);
    EXPECT_NE(move, Stockfish::MOVE_NONE);
    board.push_move(BOARD_A, move);
    return move;
}

single_board::Evaluation uniform(Board&, const std::vector<Stockfish::Move>& moves) {
    return {0.0f, std::vector<float>(moves.size(), 1.0f / moves.size())};
}

size_t perft(Board& board, int depth) {
    if (!depth) return 1;
    size_t nodes = 0;
    for (auto move : board.legal_moves(BOARD_A)) {
        board.push_move(BOARD_A, move);
        nodes += perft(board, depth - 1);
        board.pop_move(BOARD_A);
    }
    return nodes;
}

TEST_F(SingleBoardTest, VariantMoveCountsMatchPythonChess) {
    // Independent reference counts from python-chess, including mandatory
    // captures, king promotions, exploding kings, and shuffled castling.
    struct Case { Board::Variant variant; const char* fen; int depth; size_t count; };
    const Case cases[] = {
        {Board::Variant::ANTICHESS, nullptr, 3, 8067},
        {Board::Variant::ATOMIC, nullptr, 3, 8902},
        {Board::Variant::THREE_CHECK, nullptr, 3, 8902},
        {Board::Variant::CHESS960, "4k3/8/8/8/8/8/8/RK5R w HA - 0 1", 2, 106},
        {Board::Variant::ATOMIC, "7k/6p1/8/8/8/8/8/K5R1 w - - 0 1", 2, 50},
        {Board::Variant::ANTICHESS, "7k/P7/8/8/8/8/8/8 w - - 0 1", 2, 15},
    };
    for (const auto& test : cases) {
        Board board(test.variant);
        if (test.fen) board.set(test.fen);
        const auto before = board.search_hash_key(Stockfish::WHITE);
        EXPECT_EQ(perft(board, test.depth), test.count) << board.variant_name();
        EXPECT_EQ(board.search_hash_key(Stockfish::WHITE), before);
    }
}

TEST_F(SingleBoardTest, AntichessForcesCapturesAndWinsByExtinctionOrStalemate) {
    Board board(Board::Variant::ANTICHESS);
    board.set("7k/8/8/8/8/8/p7/R6K w - - 0 1");
    const auto moves = board.legal_moves(BOARD_A);
    ASSERT_EQ(moves.size(), 1u);
    EXPECT_EQ(board.uci_move(BOARD_A, moves.front()), "a1a2");
    push(board, "a1a2");
    EXPECT_EQ(board.count_in_hand(BOARD_B, Stockfish::BLACK, Stockfish::PAWN), 0);
    board.set("7k/8/8/8/8/8/8/8 w - - 0 1");
    EXPECT_EQ(adjudicate_game(board, Stockfish::WHITE, false), GameStatus::WIN);
    EXPECT_FALSE(board.is_draw());
    board.set("p6k/P7/8/8/8/8/8/8 w - - 0 1");
    EXPECT_TRUE(board.legal_moves(BOARD_A).empty());
    EXPECT_EQ(adjudicate_game(board, Stockfish::WHITE, false), GameStatus::WIN);
}

TEST_F(SingleBoardTest, Chess960CastlingAndCopiesPreserveTheRookSquares) {
    for (const auto& uci : {"b1h1", "b1a1"}) {
        Board board(Board::Variant::CHESS960);
        board.set("4k3/8/8/8/8/8/8/RK5R w HA - 0 1");
        Board copy(board);
        EXPECT_TRUE(copy.pos[BOARD_A]->is_chess960());
        EXPECT_EQ(copy.search_hash_key(Stockfish::WHITE), board.search_hash_key(Stockfish::WHITE));
        const auto before = copy.fen(BOARD_A);
        const auto move = push(copy, uci);
        EXPECT_EQ(Stockfish::type_of(move), Stockfish::CASTLING);
        EXPECT_EQ(copy.pos[BOARD_A]->square<Stockfish::KING>(Stockfish::WHITE),
            std::string(uci) == "b1h1" ? Stockfish::SQ_G1 : Stockfish::SQ_C1);
        copy.pop_move(BOARD_A);
        EXPECT_EQ(copy.fen(BOARD_A), before);
    }
}

TEST_F(SingleBoardTest, AtomicExplosionsAndThirdChecksWinImmediatelyWithoutInference) {
    std::atomic<bool> stop{false};
    const single_board::Evaluator noInference = [](Board&, const auto&) -> single_board::Evaluation {
        throw std::runtime_error("immediate variant win must not need inference");
    };
    for (auto variant : {Board::Variant::ATOMIC, Board::Variant::THREE_CHECK}) {
        Board board(variant);
        board.set(variant == Board::Variant::ATOMIC
            ? "7k/6p1/8/8/8/8/8/K5R1 w - - 0 1"
            : "4k3/8/8/8/8/8/8/R3K3 w - - 1+3 0 1");
        const auto before = board.fen(BOARD_A);
        const auto result = single_board::search(board, noInference, {}, stop);
        ASSERT_TRUE(result.mateInOne);
        EXPECT_EQ(board.fen(BOARD_A), before);
        board.push_move(BOARD_A, result.move);
        EXPECT_EQ(adjudicate_game(board, Stockfish::BLACK, false), GameStatus::LOSS);
        EXPECT_FALSE(board.is_draw());
        EXPECT_TRUE(board.legal_moves(BOARD_A).empty());
        board.pop_move(BOARD_A);
        EXPECT_EQ(board.fen(BOARD_A), before);
    }
}

TEST_F(SingleBoardTest, AtomicBlastLeavesAdjacentPawnsAndRestoresExplodedPieces) {
    Board board(Board::Variant::ATOMIC);
    board.set("7k/8/8/2pnp3/2BRP3/8/8/K7 w - - 0 1");
    const auto before = board.fen(BOARD_A);
    push(board, "d4d5");
    EXPECT_EQ(board.pos[BOARD_A]->piece_on(Stockfish::SQ_C4), Stockfish::NO_PIECE);
    EXPECT_EQ(board.pos[BOARD_A]->piece_on(Stockfish::SQ_D5), Stockfish::NO_PIECE);
    EXPECT_EQ(board.pos[BOARD_A]->piece_on(Stockfish::SQ_C5), Stockfish::B_PAWN);
    EXPECT_EQ(board.pos[BOARD_A]->piece_on(Stockfish::SQ_E5), Stockfish::B_PAWN);
    EXPECT_EQ(board.pos[BOARD_A]->piece_on(Stockfish::SQ_E4), Stockfish::W_PAWN);
    board.pop_move(BOARD_A);
    EXPECT_EQ(board.fen(BOARD_A), before);
}

TEST_F(SingleBoardTest, ThreeCheckCountersAffectRepetitionAndSurviveCopyAndLichessFen) {
    Board board(Board::Variant::THREE_CHECK), other(Board::Variant::THREE_CHECK);
    EXPECT_EQ(board.pos[BOARD_A]->checks_remaining(Stockfish::WHITE), 3);
    board.set("4k3/8/8/8/8/8/8/R3K3 w - - 0 1 +2+1");
    other.set("4k3/8/8/8/8/8/8/R3K3 w - - 2+2 0 1");
    EXPECT_NE(board.board_only_key(BOARD_A), other.board_only_key(BOARD_A));
    EXPECT_EQ(board.pos[BOARD_A]->checks_remaining(Stockfish::WHITE), 1);
    EXPECT_EQ(board.pos[BOARD_A]->checks_remaining(Stockfish::BLACK), 2);
    Board copy(board);
    EXPECT_EQ(copy.search_hash_key(Stockfish::WHITE), board.search_hash_key(Stockfish::WHITE));
    push(copy, "a1a8");
    EXPECT_EQ(copy.pos[BOARD_A]->checks_remaining(Stockfish::WHITE), 0);
    EXPECT_EQ(adjudicate_game(copy, Stockfish::BLACK, false), GameStatus::LOSS);
}

TEST_F(SingleBoardTest, VariantDrawRulesAvoidOrthodoxInsufficientMaterialAssumptions) {
    Board board(Board::Variant::THREE_CHECK);
    board.set("7k/8/8/8/8/8/8/KB6 w - - 3+3 0 1");
    EXPECT_FALSE(board.is_draw());
    board.set("7k/8/8/8/8/8/8/K7 w - - 3+3 0 1");
    EXPECT_TRUE(board.is_draw());
    board.set_variant(Board::Variant::ANTICHESS);
    board.set("7k/8/8/8/8/8/8/K7 w - - 0 1");
    EXPECT_FALSE(board.is_draw());
    board.set("8/8/8/8/8/8/8/B6b w - - 0 1");
    EXPECT_TRUE(board.is_draw());
    board.set_variant(Board::Variant::ATOMIC);
    board.set("7k/6n1/8/8/8/8/8/KB6 w - - 0 1");
    EXPECT_FALSE(board.is_draw());
    board.set("7k/8/8/8/8/8/8/KR6 w - - 0 1");
    EXPECT_TRUE(board.is_draw());
    board.set("7k/8/8/8/8/8/8/KQ6 w - - 0 1");
    EXPECT_FALSE(board.is_draw());
}

TEST_F(SingleBoardTest, NewVariantsKeepStandardPartnerAndEncodeNonRoyalKings) {
    Board reference;
    for (auto variant : {Board::Variant::ANTICHESS, Board::Variant::CHESS960,
                         Board::Variant::ATOMIC, Board::Variant::THREE_CHECK}) {
        Board board(variant);
        for (auto side : {Stockfish::WHITE, Stockfish::BLACK}) {
            std::vector<float> planes(NB_INPUT_VALUES()), expected(NB_INPUT_VALUES());
            board_to_planes(board, planes.data(), side, true);
            board_to_planes(reference, expected.data(), side, false);
            for (size_t i = 37 * 64; i < planes.size(); ++i) EXPECT_FLOAT_EQ(planes[i], expected[i]);
            for (int plane : {5, 11}) {
                for (int square = 0; square < 64; ++square) {
                    EXPECT_FLOAT_EQ(planes[plane * 64 + square], expected[plane * 64 + square]);
                }
            }
        }
    }
}

TEST_F(SingleBoardTest, ChessCapturesDoNotCreatePockets) {
    Board board(Board::Variant::CHESS);
    push(board, "e2e4"); push(board, "d7d5"); push(board, "e4d5");
    EXPECT_EQ(board.count_in_hand(BOARD_A, Stockfish::WHITE, Stockfish::PAWN), 0);
    EXPECT_EQ(board.count_in_hand(BOARD_B, Stockfish::BLACK, Stockfish::PAWN), 0);
    EXPECT_TRUE(board.legal_moves(BOARD_B).empty());
    EXPECT_FALSE(board.is_legal_move(BOARD_A, Stockfish::MOVE_NONE));
}

TEST_F(SingleBoardTest, CrazyhouseCapturesAndDropsUseTheSameBoardAndUndo) {
    Board board(Board::Variant::CRAZYHOUSE);
    push(board, "e2e4"); push(board, "d7d5");
    const auto before = board.search_hash_key(Stockfish::WHITE);
    push(board, "e4d5");
    EXPECT_EQ(board.count_in_hand(BOARD_A, Stockfish::WHITE, Stockfish::PAWN), 1);
    EXPECT_EQ(board.count_in_hand(BOARD_B, Stockfish::BLACK, Stockfish::PAWN), 0);
    Board copy(board);
    EXPECT_EQ(copy.variant, Board::Variant::CRAZYHOUSE);
    EXPECT_EQ(copy.search_hash_key(Stockfish::BLACK), board.search_hash_key(Stockfish::BLACK));
    push(copy, "g8f6"); push(copy, "P@e4");
    EXPECT_EQ(copy.count_in_hand(BOARD_A, Stockfish::WHITE, Stockfish::PAWN), 0);
    copy.pop_move(BOARD_A);
    EXPECT_EQ(copy.count_in_hand(BOARD_A, Stockfish::WHITE, Stockfish::PAWN), 1);
    board.pop_move(BOARD_A);
    EXPECT_EQ(board.search_hash_key(Stockfish::WHITE), before);
    EXPECT_EQ(board.count_in_hand(BOARD_A, Stockfish::WHITE, Stockfish::PAWN), 0);
}

TEST_F(SingleBoardTest, CapturedPromotionsBecomePawnsAndEnPassantFeedsOwnPocket) {
    Board board(Board::Variant::CRAZYHOUSE);
    board.set("4k3/8/8/8/8/8/4q~3/4R1K1[] w - - 0 1");
    const auto before = board.fen(BOARD_A);
    push(board, "e1e2");
    EXPECT_EQ(board.count_in_hand(BOARD_A, Stockfish::WHITE, Stockfish::PAWN), 1);
    EXPECT_EQ(board.count_in_hand(BOARD_A, Stockfish::WHITE, Stockfish::QUEEN), 0);
    board.pop_move(BOARD_A);
    EXPECT_EQ(board.fen(BOARD_A), before);
    board.set("4k3/8/8/3pP3/8/8/8/4K3[] w - d6 0 1");
    push(board, "e5d6");
    EXPECT_EQ(board.count_in_hand(BOARD_A, Stockfish::WHITE, Stockfish::PAWN), 1);
}

TEST_F(SingleBoardTest, AllFourPromotionsAreLegalInSingleBoardVariants) {
    for (auto variant : {Board::Variant::CHESS, Board::Variant::CRAZYHOUSE}) {
        Board board(variant);
        board.set("4k3/P7/8/8/8/8/8/4K3 w - - 0 1");
        int promotions = 0;
        for (auto move : board.legal_moves(BOARD_A)) {
            promotions += Stockfish::type_of(move) == Stockfish::PROMOTION;
        }
        EXPECT_EQ(promotions, 4);
    }
}

TEST_F(SingleBoardTest, ChessAndCrazyhouseUseCorrectTerminalRules) {
    for (auto variant : {Board::Variant::CHESS, Board::Variant::CRAZYHOUSE}) {
        Board board(variant);
        board.set("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1");
        EXPECT_FALSE(board.is_checkmate(Stockfish::BLACK));
        EXPECT_EQ(adjudicate_game(board, Stockfish::BLACK, false), GameStatus::DRAW);
        board.set("7k/6Q1/5K2/8/8/8/8/8 b - - 100 1");
        EXPECT_EQ(adjudicate_game(board, Stockfish::BLACK, true), GameStatus::LOSS);
        board.set("4k3/8/8/8/8/8/8/R3K3 w - - 100 1");
        EXPECT_EQ(board.is_draw(), variant == Board::Variant::CHESS);
    }
}

TEST_F(SingleBoardTest, PocketPiecesAndPromotionsArePartOfCrazyhouseRepetition) {
    Board first(Board::Variant::CRAZYHOUSE), second(Board::Variant::CRAZYHOUSE);
    first.set("4k3/8/8/8/8/8/8/4K3[P] w - - 0 1");
    second.set("4k3/8/8/8/8/8/8/4K3[N] w - - 0 1");
    EXPECT_NE(first.board_only_key(BOARD_A), second.board_only_key(BOARD_A));
    EXPECT_FALSE(first.is_draw());
    first.set("4k3/8/8/8/8/8/4Q3/4K3[] w - - 0 1");
    second.set("4k3/8/8/8/8/8/4Q~3/4K3[] w - - 0 1");
    EXPECT_NE(first.board_only_key(BOARD_A), second.board_only_key(BOARD_A));
    first.set("4k3/8/8/8/8/8/8/4K3[] w - - 0 1");
    EXPECT_FALSE(first.is_draw());
    first.set_variant(Board::Variant::CHESS);
    for (int cycle = 0; cycle < 2; ++cycle) {
        push(first, "g1f3"); push(first, "g8f6"); push(first, "f3g1"); push(first, "f6g8");
    }
    EXPECT_TRUE(first.is_draw());
    Board copy(first);
    EXPECT_TRUE(copy.is_draw());
}

TEST_F(SingleBoardTest, SharedNetworkEncodingKeepsPartnerAtStartAndZerosTimeAdvantage) {
    Board reference;
    for (auto variant : {Board::Variant::CHESS, Board::Variant::CRAZYHOUSE}) {
        Board board(variant);
        auto checkPartner = [&] {
            for (auto side : {Stockfish::WHITE, Stockfish::BLACK}) {
                std::vector<float> planes(NB_INPUT_VALUES(), 123.0f);
                std::vector<float> expected(NB_INPUT_VALUES());
                board_to_planes(board, planes.data(), side, true);
                board_to_planes(reference, expected.data(), side, false);
                for (size_t i = 37 * 64; i < planes.size(); ++i) {
                    EXPECT_FLOAT_EQ(planes[i], expected[i]) << "input index " << i;
                }
                for (size_t square = 0; square < 64; ++square) {
                    EXPECT_FLOAT_EQ(planes[(37 + 26) * 64 + square], 1.0f);
                    EXPECT_FLOAT_EQ(planes[(37 + 25) * 64 + square],
                                    side == Stockfish::BLACK ? 1.0f : 0.0f);
                    EXPECT_FLOAT_EQ(planes[31 * 64 + square], 0.0f);
                    EXPECT_FLOAT_EQ(planes[(37 + 31) * 64 + square], 0.0f);
                }
            }
        };
        checkPartner();
        push(board, "e2e4"); push(board, "d7d5"); push(board, "e4d5");
        checkPartner();
        std::vector<float> planes(NB_INPUT_VALUES());
        board_to_planes(board, planes.data(), Stockfish::WHITE, true);
        EXPECT_FLOAT_EQ(planes[12 * 64],
                        variant == Board::Variant::CRAZYHOUSE ? 1.0f / 16.0f : 0.0f);
        EXPECT_FALSE(board.can_partner_provide_blocking_piece(BOARD_A, Stockfish::WHITE, true));
        board.set("4k3/8/8/8/8/8/8/4K3 b - - 0 1");
        checkPartner();
    }
}

TEST_F(SingleBoardTest, SearchFindsChessAndDropMatesWithoutInference) {
    std::atomic<bool> stop{false};
    const single_board::Evaluator noInference = [](Board&, const auto&) -> single_board::Evaluation {
        throw std::runtime_error("mate in one must not need inference");
    };
    for (auto variant : {Board::Variant::CHESS, Board::Variant::CRAZYHOUSE}) {
        Board board(variant);
        board.set(variant == Board::Variant::CHESS
            ? "7k/8/5KQ1/8/8/8/8/8 w - - 0 1"
            : "7k/8/5K2/8/8/8/8/8[Q] w - - 0 1");
        const auto before = board.fen(BOARD_A);
        const auto result = single_board::search(board, noInference, {}, stop);
        ASSERT_TRUE(result.mateInOne);
        EXPECT_EQ(board.fen(BOARD_A), before);
        board.push_move(BOARD_A, result.move);
        EXPECT_TRUE(board.is_checkmate(Stockfish::BLACK));
    }
}

TEST_F(SingleBoardTest, NodeLimitsCancellationAndInferenceErrorsRestoreTheRoot) {
    Board board(Board::Variant::CHESS);
    const auto before = board.search_hash_key(Stockfish::WHITE);
    std::atomic<bool> stop{false};
    single_board::Limits limits;
    limits.nodes = 64;
    limits.moveTimeMs = 0;
    const auto result = single_board::search(board, uniform, limits, stop);
    EXPECT_EQ(result.nodes, 64u);
    EXPECT_TRUE(board.is_legal_move(BOARD_A, result.move));
    EXPECT_EQ(board.search_hash_key(Stockfish::WHITE), before);
    stop.store(true);
    EXPECT_TRUE(board.is_legal_move(BOARD_A, single_board::search(board, uniform, limits, stop).move));
    stop.store(false);
    int calls = 0;
    EXPECT_THROW(single_board::search(board, [&](Board& b, const auto& moves) {
        if (++calls > 1) throw std::runtime_error("inference error");
        return uniform(b, moves);
    }, limits, stop), std::runtime_error);
    EXPECT_EQ(board.search_hash_key(Stockfish::WHITE), before);
}

TEST_F(SingleBoardTest, ClaimableRootDrawStillReturnsALegalUciMove) {
    Board board(Board::Variant::CHESS);
    board.set("4k3/8/8/8/8/8/8/R3K3 w - - 100 1");
    std::atomic<bool> stop{false};
    const auto result = single_board::search(board, [](Board&, const auto&) -> single_board::Evaluation {
        throw std::runtime_error("drawn position does not need inference");
    }, {}, stop);
    EXPECT_FLOAT_EQ(result.value, 0.0f);
    EXPECT_TRUE(board.is_legal_move(BOARD_A, result.move));
}

}  // namespace

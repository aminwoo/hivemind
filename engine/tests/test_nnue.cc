#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#include "common/globals.h"
#include "environment/board.h"
#include "environment/constants.h"
#include "nnue/features.h"

class NnueFeatureTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        init_fairy_stockfish();
        init_policy_index();
    }
};

namespace {

std::vector<int> sorted_features(Board& board, Stockfish::Color team, bool advantage) {
    std::array<uint16_t, nnue::MAX_ACTIVE_FEATURES> buffer{};
    const int count = nnue::extract_features(board, team, advantage, buffer.data());
    std::vector<int> features(buffer.begin(), buffer.begin() + count);
    std::sort(features.begin(), features.end());
    return features;
}

}  // namespace

TEST_F(NnueFeatureTest, MirrorIsAnInvolutionOverTheFeatureSpace) {
    std::vector<bool> seen(nnue::NUM_FEATURES, false);
    for (int f = 0; f < nnue::NUM_FEATURES; ++f) {
        const int mirrored = nnue::mirror_feature(f);
        ASSERT_GE(mirrored, 0);
        ASSERT_LT(mirrored, nnue::NUM_FEATURES);
        EXPECT_EQ(nnue::mirror_feature(mirrored), f) << "feature " << f;
        EXPECT_FALSE(seen[mirrored]);
        seen[mirrored] = true;
    }
}

// Training augments with swapped boards through swap_boards_feature(); it must
// match extracting features from the actually swapped position.
TEST_F(NnueFeatureTest, BoardSwapIsAFeaturePermutation) {
    std::mt19937_64 rng(777);
    for (int game = 0; game < 20; ++game) {
        Board board;
        for (int ply = 0; ply < 120; ++ply) {
            Board swapped;
            swapped.set(board.fen(BOARD_B) + "|" + board.fen(BOARD_A));
            for (Stockfish::Color team : {Stockfish::WHITE, Stockfish::BLACK}) {
                for (bool advantage : {false, true}) {
                    std::vector<int> permuted;
                    for (int f : sorted_features(board, team, advantage)) {
                        permuted.push_back(nnue::swap_boards_feature(f));
                    }
                    std::sort(permuted.begin(), permuted.end());
                    ASSERT_EQ(permuted, sorted_features(swapped, ~team, advantage));
                }
            }
            const int boardNumber = static_cast<int>(rng() & 1ULL);
            std::vector<Stockfish::Move> moves = board.legal_moves(boardNumber);
            if (moves.empty()) {
                break;
            }
            board.push_move(boardNumber, moves[rng() % moves.size()]);
        }
    }
}

TEST_F(NnueFeatureTest, StartingPositionHasPiecesTurnTimeAndCastling) {
    Board board;
    // 64 pieces, one turn flag per board, one time flag, 8 castling rights.
    EXPECT_EQ(sorted_features(board, Stockfish::WHITE, true).size(), 75u);
}

// The training data stores only the side-to-move features and derives the
// other team's list with mirror_feature(), so the two must agree everywhere.
TEST_F(NnueFeatureTest, OtherTeamFeaturesAreTheMirrorAcrossRandomGames) {
    std::mt19937_64 rng(12345);
    int checked = 0;
    for (int game = 0; game < 40; ++game) {
        Board board;
        for (int ply = 0; ply < 160; ++ply) {
            for (Stockfish::Color team : {Stockfish::WHITE, Stockfish::BLACK}) {
                for (bool advantage : {false, true}) {
                    const std::vector<int> own = sorted_features(board, team, advantage);
                    ASSERT_LE(own.size(), static_cast<size_t>(nnue::MAX_ACTIVE_FEATURES));
                    for (int f : own) {
                        ASSERT_LT(f, nnue::NUM_FEATURES);
                    }
                    std::vector<int> mirrored;
                    for (int f : own) {
                        mirrored.push_back(nnue::mirror_feature(f));
                    }
                    std::sort(mirrored.begin(), mirrored.end());
                    ASSERT_EQ(mirrored, sorted_features(board, ~team, !advantage))
                        << board.fen(BOARD_A) << " | " << board.fen(BOARD_B);
                    ++checked;
                }
            }
            const int boardNumber = static_cast<int>(rng() & 1ULL);
            std::vector<Stockfish::Move> moves = board.legal_moves(boardNumber);
            if (moves.empty()) {
                moves = board.legal_moves(1 - boardNumber);
                if (moves.empty()) {
                    break;
                }
                board.push_move(1 - boardNumber, moves[rng() % moves.size()]);
            } else {
                board.push_move(boardNumber, moves[rng() % moves.size()]);
            }
        }
    }
    EXPECT_GT(checked, 1000);
}

#include <cstdio>
#include <filesystem>
#include <fstream>

#include "nnue/network.h"
#include "search/alphabeta.h"

namespace {

/// Writes a small random network in the HMNNUE02 format and returns its path.
/// With four buckets the table splits kings by rank and file.
std::string write_random_network(int hidden, int l1, int l2, uint64_t seed, int buckets = 1) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> weight(-60, 60);
    std::uniform_real_distribution<float> head(-0.3f, 0.3f);
    const std::filesystem::path path = std::filesystem::temp_directory_path()
        / ("hivemind_test_" + std::to_string(seed) + ".nnue");
    std::ofstream stream(path, std::ios::binary);
    stream.write("HMNNUE02", 8);
    const int features = 2 * buckets * nnue::PIECE_FEATURES_PER_BOARD
        + (nnue::NUM_FEATURES - nnue::PIECE_FEATURES);
    const uint32_t header[6] = {static_cast<uint32_t>(features),
                                static_cast<uint32_t>(hidden), static_cast<uint32_t>(l1),
                                static_cast<uint32_t>(l2), 255, static_cast<uint32_t>(buckets)};
    stream.write(reinterpret_cast<const char*>(header), sizeof(header));
    std::array<uint8_t, 64> table{};
    for (int square = 0; square < 64; ++square) {
        table[square] = static_cast<uint8_t>(buckets == 1 ? 0
            : ((square / 8 >= 2 ? 2 : 0) + (square % 8 >= 4 ? 1 : 0)) % buckets);
    }
    stream.write(reinterpret_cast<const char*>(table.data()), table.size());
    auto int16s = [&](size_t count) {
        for (size_t i = 0; i < count; ++i) {
            const int16_t value = static_cast<int16_t>(weight(rng));
            stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
        }
    };
    auto floats = [&](size_t count) {
        for (size_t i = 0; i < count; ++i) {
            const float value = head(rng);
            stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
        }
    };
    int16s(static_cast<size_t>(features) * hidden);
    int16s(hidden);
    floats(static_cast<size_t>(l1) * 2 * hidden);
    floats(l1);
    floats(static_cast<size_t>(l2) * l1);
    floats(l2);
    floats(l2);
    floats(1);
    return path.string();
}

}  // namespace

TEST_F(NnueFeatureTest, IncrementalUpdateMatchesRefreshAcrossRandomGames) {
  for (int buckets : {1, 4}) {
    nnue::Network network;
    std::string error;
    ASSERT_TRUE(network.load(write_random_network(64, 8, 8, 7 + buckets, buckets), &error)) << error;
    ASSERT_EQ(network.king_buckets(), buckets);
    std::mt19937_64 rng(99);
    auto parent = std::make_unique<nnue::Accumulator>();
    auto child = std::make_unique<nnue::Accumulator>();
    auto fresh = std::make_unique<nnue::Accumulator>();
    for (int game = 0; game < 10; ++game) {
        Board board;
        const bool whiteAdvantage = game % 2 == 0;
        network.refresh(board, whiteAdvantage, *parent);
        for (int ply = 0; ply < 150; ++ply) {
            const int boardNumber = static_cast<int>(rng() & 1ULL);
            std::vector<Stockfish::Move> moves = board.legal_moves(boardNumber);
            if (moves.empty()) {
                break;
            }
            board.push_move(boardNumber, moves[rng() % moves.size()]);
            network.update(board, whiteAdvantage, *parent, *child);
            network.refresh(board, whiteAdvantage, *fresh);
            for (int team = 0; team < 2; ++team) {
                ASSERT_TRUE(std::equal(child->values[team].begin(),
                                       child->values[team].begin() + network.hidden(),
                                       fresh->values[team].begin()))
                    << "game " << game << " ply " << ply;
            }
            std::swap(parent, child);
        }
    }
  }
}

TEST_F(NnueFeatureTest, AlphaBetaFindsBackRankMateInOne) {
    nnue::Network network;
    std::string error;
    ASSERT_TRUE(network.load(write_random_network(64, 8, 8, 11), &error)) << error;
    Board board;
    // White mates with Ra8 on board A. Board B has Black (our partner) to
    // move, so the black side on A cannot be handed a blocking piece in time.
    board.set("6k1/5ppp/8/8/8/8/8/R5K1[] w - - 0 1|"
              "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] b KQkq - 0 1");
    ab::Searcher searcher(network, 4);
    ab::Limits limits;
    limits.depth = 3;
    const ab::Result result = searcher.search(board, Stockfish::WHITE, true, limits);
    ASSERT_TRUE(result.hasMove);
    EXPECT_EQ(board.uci_move(BOARD_A, result.best.a), "a1a8");
    EXPECT_EQ(result.score, ab::SCORE_MATE - 1);
}

TEST_F(NnueFeatureTest, AlphaBetaReturnsLegalJointMovesAndRestoresTheBoard) {
    nnue::Network network;
    std::string error;
    ASSERT_TRUE(network.load(write_random_network(64, 8, 8, 13), &error)) << error;
    ab::Searcher searcher(network, 4);
    std::mt19937_64 rng(5);
    Board board;
    Stockfish::Color team = Stockfish::WHITE;
    bool advantage = false;
    for (int turn = 0; turn < 30; ++turn) {
        const std::string before = board.fen(BOARD_A) + "|" + board.fen(BOARD_B);
        ab::Limits limits;
        limits.depth = 3;
        const ab::Result result = searcher.search(board, team, advantage, limits);
        EXPECT_EQ(board.fen(BOARD_A) + "|" + board.fen(BOARD_B), before);
        if (!result.hasMove) {
            break;
        }
        if (result.best.a != Stockfish::MOVE_NONE) {
            ASSERT_EQ(board.side_to_move(BOARD_A), team);
            ASSERT_TRUE(board.is_legal_move(BOARD_A, result.best.a));
        }
        if (result.best.b != Stockfish::MOVE_NONE) {
            ASSERT_EQ(board.side_to_move(BOARD_B), ~team);
            ASSERT_TRUE(board.is_legal_move(BOARD_B, result.best.b));
        }
        if (result.best.a == Stockfish::MOVE_NONE && result.best.b == Stockfish::MOVE_NONE) {
            ASSERT_TRUE(advantage) << "double sit without the time advantage";
        }
        board.make_moves(result.best.a, result.best.b);
        if (board.is_checkmate(~team, !advantage) || board.is_draw()) {
            break;
        }
        team = ~team;
        advantage = !advantage;
    }
}

TEST_F(NnueFeatureTest, LazySmpSearchReturnsALegalMoveAndRestoresTheBoard) {
    nnue::Network network;
    std::string error;
    ASSERT_TRUE(network.load(write_random_network(64, 8, 8, 17), &error)) << error;
    ab::Searcher searcher(network, 8);
    searcher.options.threads = 4;
    Board board;
    board.set("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R[] w KQkq - 2 3|"
              "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R[] b KQkq - 1 1");
    const std::string before = board.fen(BOARD_A) + "|" + board.fen(BOARD_B);
    for (int round = 0; round < 3; ++round) {
        ab::Limits limits;
        limits.moveTimeMs = 150;
        const ab::Result result = searcher.search(board, Stockfish::WHITE, false, limits);
        ASSERT_TRUE(result.hasMove);
        EXPECT_EQ(board.fen(BOARD_A) + "|" + board.fen(BOARD_B), before);
        if (result.best.a != Stockfish::MOVE_NONE) {
            EXPECT_TRUE(board.is_legal_move(BOARD_A, result.best.a));
        }
        if (result.best.b != Stockfish::MOVE_NONE) {
            EXPECT_TRUE(board.is_legal_move(BOARD_B, result.best.b));
        }
    }
}

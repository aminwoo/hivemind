#include <gtest/gtest.h>

#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

#include "common/utils.h"
#include "environment/planes.h"
#include "interface/uci.h"

class UCIOpeningNoiseTestPeer {
public:
    static SearchParams::RuntimeConfig current_search_config(UCI& uci) {
        return uci.current_search_config();
    }

    static void set_board(UCI& uci, const std::string& dualFen) {
        uci.board.set(dualFen);
    }

    static bool background_enabled(const UCI& uci) {
        return uci.backgroundSearchEnabled;
    }

    static bool ponder_enabled(const UCI& uci) {
        return uci.ponderEnabled;
    }

    static void use_alphabeta(UCI& uci) {
        uci.alphaBetaMode = true;
    }

    static bool has_search(const UCI& uci) {
        return uci.mainSearchThread != nullptr || uci.ongoingSearch.load();
    }

    static Board& board(UCI& uci) { return uci.board; }
};

namespace {

void initialize_engine_state() {
    static std::once_flag initialized;
    std::call_once(initialized, [] {
        init_fairy_stockfish();
        init_policy_index();
    });
}

void set_option(UCI& uci, const std::string& name, const std::string& value) {
    std::istringstream input("name " + name + " value " + value);
    uci.setoption(input);
}

class UCIOpeningNoiseTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { initialize_engine_state(); }
};

TEST_F(UCIOpeningNoiseTest, IsDisabledByDefault) {
    UCI uci;
    const SearchParams::RuntimeConfig config =
        UCIOpeningNoiseTestPeer::current_search_config(uci);

    EXPECT_EQ(
        config.internalMateProbeMode,
        SearchParams::InternalMateProbeMode::OFF);
    EXPECT_FLOAT_EQ(config.rootDirichletAlpha, 0.0f);
    EXPECT_FLOAT_EQ(config.rootDirichletEpsilon, 0.0f);
    EXPECT_EQ(config.rootNoiseSeed, 0U);
}

TEST_F(UCIOpeningNoiseTest, SelectsInternalMateProbeExperimentMode) {
    UCI uci;
    set_option(uci, "InternalMateProbe", "telemetry");

    const SearchParams::RuntimeConfig telemetry =
        UCIOpeningNoiseTestPeer::current_search_config(uci);
    EXPECT_EQ(
        telemetry.internalMateProbeMode,
        SearchParams::InternalMateProbeMode::TELEMETRY);

    set_option(uci, "InternalMateProbe", "bias");
    const SearchParams::RuntimeConfig bias =
        UCIOpeningNoiseTestPeer::current_search_config(uci);
    EXPECT_EQ(
        bias.internalMateProbeMode,
        SearchParams::InternalMateProbeMode::BIAS);

    set_option(uci, "InternalMateProbe", "certify");
    const SearchParams::RuntimeConfig certify =
        UCIOpeningNoiseTestPeer::current_search_config(uci);
    EXPECT_EQ(
        certify.internalMateProbeMode,
        SearchParams::InternalMateProbeMode::CERTIFY);

    set_option(uci, "InternalMateProbe", "certifyonly");
    const SearchParams::RuntimeConfig certifyOnly =
        UCIOpeningNoiseTestPeer::current_search_config(uci);
    EXPECT_EQ(
        certifyOnly.internalMateProbeMode,
        SearchParams::InternalMateProbeMode::CERTIFY_ONLY);
}

TEST_F(UCIOpeningNoiseTest, SelectedMoveCertificationIsOnByDefault) {
    UCI uci;
    EXPECT_TRUE(UCIOpeningNoiseTestPeer::current_search_config(uci)
                    .certifySelectedMove);

    set_option(uci, "CertifySelectedMove", "false");
    EXPECT_FALSE(UCIOpeningNoiseTestPeer::current_search_config(uci)
                     .certifySelectedMove);
}

TEST_F(UCIOpeningNoiseTest, AppliesConfiguredNoiseInsideOpeningHorizon) {
    UCI uci;
    set_option(uci, "OpeningNoise", "true");
    set_option(uci, "OpeningNoisePlies", "12");
    set_option(uci, "OpeningNoiseAlphaPermille", "450");
    set_option(uci, "OpeningNoiseEpsilonPermille", "175");

    const SearchParams::RuntimeConfig first =
        UCIOpeningNoiseTestPeer::current_search_config(uci);
    const SearchParams::RuntimeConfig second =
        UCIOpeningNoiseTestPeer::current_search_config(uci);

    EXPECT_FLOAT_EQ(first.rootDirichletAlpha, 0.45f);
    EXPECT_FLOAT_EQ(first.rootDirichletEpsilon, 0.175f);
    EXPECT_NE(first.rootNoiseSeed, second.rootNoiseSeed);
}

TEST_F(UCIOpeningNoiseTest, StopsWhenEitherBoardReachesPlyLimit) {
    UCI uci;
    set_option(uci, "OpeningNoise", "true");
    set_option(uci, "OpeningNoisePlies", "16");
    UCIOpeningNoiseTestPeer::set_board(
        uci,
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 9|"
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    const SearchParams::RuntimeConfig config =
        UCIOpeningNoiseTestPeer::current_search_config(uci);

    EXPECT_FLOAT_EQ(config.rootDirichletAlpha, 0.0f);
    EXPECT_FLOAT_EQ(config.rootDirichletEpsilon, 0.0f);
    EXPECT_EQ(config.rootNoiseSeed, 0U);
}

TEST_F(UCIOpeningNoiseTest, AdvertisesOptionsInUciHandshake) {
    UCI uci;
    std::ostringstream output;
    std::streambuf* previous = std::cout.rdbuf(output.rdbuf());
    uci.send_uci_response();
    std::cout.rdbuf(previous);

    EXPECT_NE(output.str().find(
        "option name BackgroundSearch type check default true"),
        std::string::npos);
    EXPECT_NE(output.str().find(
        "option name OpeningNoise type check default false"),
        std::string::npos);
    EXPECT_NE(output.str().find(
        "option name OpeningNoisePlies type spin default 16 min 0 max 200"),
        std::string::npos);
    EXPECT_NE(output.str().find(
        "option name OpeningNoiseAlphaPermille type spin default 100"),
        std::string::npos);
    EXPECT_NE(output.str().find(
        "option name OpeningNoiseEpsilonPermille type spin default 600"),
        std::string::npos);
    EXPECT_NE(output.str().find(
        "option name InternalMateProbe type combo default off "
        "var off var telemetry var bias var certify"),
        std::string::npos);
}

TEST_F(UCIOpeningNoiseTest, BackgroundOptionIsIndependentOfPonder) {
    UCI uci;
    EXPECT_TRUE(UCIOpeningNoiseTestPeer::background_enabled(uci));

    testing::internal::CaptureStdout();
    set_option(uci, "BackgroundSearch", "false");
    const std::string output = testing::internal::GetCapturedStdout();
    EXPECT_NE(output.find("info string BackgroundSearch set to false"),
              std::string::npos);
    EXPECT_FALSE(UCIOpeningNoiseTestPeer::background_enabled(uci));
    EXPECT_TRUE(UCIOpeningNoiseTestPeer::ponder_enabled(uci));

    set_option(uci, "BackgroundSearch", "true");
    set_option(uci, "Ponder", "false");
    EXPECT_TRUE(UCIOpeningNoiseTestPeer::background_enabled(uci));
    EXPECT_FALSE(UCIOpeningNoiseTestPeer::ponder_enabled(uci));
}

TEST_F(UCIOpeningNoiseTest, DisabledBackgroundCommandDoesNotSearchOrNeedEngines) {
    UCI uci;
    set_option(uci, "BackgroundSearch", "false");
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    std::istringstream input("background");
    uci.go(input);
    const std::string output = testing::internal::GetCapturedStdout();
    const std::string errors = testing::internal::GetCapturedStderr();

    EXPECT_TRUE(output.empty());
    EXPECT_TRUE(errors.empty());
    EXPECT_FALSE(UCIOpeningNoiseTestPeer::has_search(uci));
}

TEST_F(UCIOpeningNoiseTest, BackgroundCommandDoesNotFallThroughToAlphaBeta) {
    UCI uci;
    UCIOpeningNoiseTestPeer::use_alphabeta(uci);
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    std::istringstream input("background");
    uci.go(input);
    const std::string output = testing::internal::GetCapturedStdout();
    const std::string errors = testing::internal::GetCapturedStderr();

    EXPECT_TRUE(output.empty());
    EXPECT_TRUE(errors.empty());
    EXPECT_FALSE(UCIOpeningNoiseTestPeer::has_search(uci));
}

TEST_F(UCIOpeningNoiseTest, SingleBoardPositionUsesStandardUciMoveHistory) {
    UCI uci;
    set_option(uci, "UCI_Variant", "chess");
    std::istringstream start("startpos moves e2e4 d7d5 e4d5");
    uci.position(start);
    Board& board = UCIOpeningNoiseTestPeer::board(uci);
    EXPECT_EQ(board.variant, Board::Variant::CHESS);
    EXPECT_EQ(board.game_ply(BOARD_A), 3);
    EXPECT_EQ(board.count_in_hand(BOARD_A, Stockfish::WHITE, Stockfish::PAWN), 0);

    set_option(uci, "UCI_Variant", "crazyhouse");
    std::istringstream fen("fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1 moves e2e4 d7d5 e4d5 g8f6 P@e4");
    uci.position(fen);
    EXPECT_EQ(board.game_ply(BOARD_A), 5);
    EXPECT_EQ(board.count_in_hand(BOARD_A, Stockfish::WHITE, Stockfish::PAWN), 0);
    EXPECT_EQ(Stockfish::type_of(board.last_move(BOARD_A)), Stockfish::DROP);

    set_option(uci, "UCI_Variant", "bughouse");
    std::istringstream bughouse("startpos moves 1e2e4 2d2d4");
    uci.position(bughouse);
    EXPECT_EQ(board.game_ply(BOARD_A), 1);
    EXPECT_EQ(board.game_ply(BOARD_B), 1);
}

TEST_F(UCIOpeningNoiseTest, VariantDefaultsUseZeroDrawContemptAndCorrectStartingRules) {
    UCI uci;
    for (const auto& variant : {"bughouse", "chess", "crazyhouse", "antichess", "chess960", "atomic", "3check"}) {
        set_option(uci, "UCI_Variant", variant);
        EXPECT_FLOAT_EQ(UCIOpeningNoiseTestPeer::current_search_config(uci).drawContempt, 0.0f);
        std::istringstream start("startpos moves e2e4");
        if (std::string(variant) != "bughouse") uci.position(start);
    }
    Board& board = UCIOpeningNoiseTestPeer::board(uci);
    EXPECT_EQ(board.pos[BOARD_A]->checks_remaining(Stockfish::WHITE), 3);
    set_option(uci, "UCI_Variant", "chess");
    set_option(uci, "UCI_Chess960", "true");
    std::istringstream castle("fen 4k3/8/8/8/8/8/8/RK5R w HA - 0 1 moves b1h1");
    uci.position(castle);
    EXPECT_EQ(board.variant, Board::Variant::CHESS960);
    EXPECT_EQ(board.pos[BOARD_A]->square<Stockfish::KING>(Stockfish::WHITE), Stockfish::SQ_G1);
    set_option(uci, "UCI_Chess960", "false");
    EXPECT_EQ(board.variant, Board::Variant::CHESS);
}

}  // namespace

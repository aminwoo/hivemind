#include <gtest/gtest.h>

#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>

#include "common/utils.h"
#include "environment/planes.h"
#include "interface/uci.h"
#include "search/mate_probe.h"

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

    static int report_interval_ms(const UCI& uci) {
        return uci.reportIntervalMs;
    }

    static void use_alphabeta(UCI& uci) {
        uci.alphaBetaMode = true;
    }

    static bool has_search(const UCI& uci) {
        return uci.mainSearchThread != nullptr || uci.ongoingSearch.load();
    }

    static Board& board(UCI& uci) { return uci.board; }

    static void set_network_reads_clock(UCI& uci, bool reads) {
        uci.networkReadsClock = reads;
    }

    static bool clock_model(const UCI& uci) { return uci.current_clock().has_value(); }

    static std::optional<TimeState> clock(const UCI& uci) { return uci.current_clock(); }
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

TEST_F(UCIOpeningNoiseTest, RootMoveTimeChargesOnlyTheRootAction) {
    UCI uci;
    set_option(uci, "TimeModel", "clock");
    set_option(uci, "Team", "white");
    set_option(uci, "TimeLeadMs", "3000");
    set_option(uci, "TeamMoveTimeMs", "1000");
    set_option(uci, "OpponentMoveTimeMs", "500");
    ASSERT_TRUE(UCIOpeningNoiseTestPeer::clock(uci).has_value());
    EXPECT_EQ(UCIOpeningNoiseTestPeer::clock(uci)->next_move_time_ms(Stockfish::WHITE), 1000);

    set_option(uci, "RootMoveTimeMs", "400");
    const TimeState budgeted = *UCIOpeningNoiseTestPeer::clock(uci);
    EXPECT_EQ(budgeted.move_time_ms(Stockfish::WHITE), 1000);
    EXPECT_EQ(budgeted.next_move_time_ms(Stockfish::WHITE), 400);
    EXPECT_EQ(budgeted.after(Stockfish::WHITE, true).lead_ms(Stockfish::WHITE), 2600);

    set_option(uci, "RootMoveTimeMs", "0");
    EXPECT_EQ(UCIOpeningNoiseTestPeer::clock(uci)->next_move_time_ms(Stockfish::WHITE), 1000);
}

TEST_F(UCIOpeningNoiseTest, RootMoveTimePersistsLikeTheOtherClockOptions) {
    // An option, like TeamMoveTimeMs: it holds across new positions and games
    // until the front end changes it, which is why a front end that budgets
    // only some moves must send 0 for the others.
    UCI uci;
    set_option(uci, "TimeModel", "clock");
    set_option(uci, "TimeLeadMs", "3000");
    set_option(uci, "TeamMoveTimeMs", "1000");
    set_option(uci, "RootMoveTimeMs", "400");
    std::istringstream first("startpos moves 1e2e4");
    uci.position(first);
    uci.new_game();
    std::istringstream second("startpos");
    uci.position(second);
    const auto clock = UCIOpeningNoiseTestPeer::clock(uci);
    ASSERT_TRUE(clock.has_value());
    EXPECT_EQ(clock->next_move_time_ms(Stockfish::WHITE), 400);
    EXPECT_EQ(clock->move_time_ms(Stockfish::WHITE), 1000);
    EXPECT_EQ(clock->lead_ms(Stockfish::WHITE), 3000);
}

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

TEST_F(UCIOpeningNoiseTest, ConfiguresMassWideningAndExplorationWithoutChangingDefaults) {
    UCI uci;
    const auto defaults = UCIOpeningNoiseTestPeer::current_search_config(uci);
    EXPECT_FLOAT_EQ(defaults.pwMassStart, 0.0f);
    EXPECT_FLOAT_EQ(defaults.rootPwMassStart, -1.0f);
    EXPECT_FALSE(defaults.pwMassNormalize);
    set_option(uci, "PWMassStartPermille", "350");
    set_option(uci, "RootPWMassStartPermille", "650");
    set_option(uci, "PWMassExponentPermille", "100");
    set_option(uci, "PWMassCapPermille", "1500");
    set_option(uci, "PWMassNormalize", "true");
    set_option(uci, "CPUCTInitPermille", "2000");
    const auto config = UCIOpeningNoiseTestPeer::current_search_config(uci);
    EXPECT_FLOAT_EQ(config.pwMassStart, 0.35f);
    EXPECT_FLOAT_EQ(config.rootPwMassStart, 0.65f);
    EXPECT_FLOAT_EQ(config.pwMassExponent, 0.10f);
    EXPECT_FLOAT_EQ(config.pwMassCap, 1.5f);
    EXPECT_TRUE(config.pwMassNormalize);
    EXPECT_FLOAT_EQ(config.cpuctInit, 2.0f);
    set_option(uci, "RootPWMassStartPermille", "-1");
    EXPECT_FLOAT_EQ(UCIOpeningNoiseTestPeer::current_search_config(uci).rootPwMassStart, -1.0f);
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
    EXPECT_NE(output.str().find(
        "option name MateThreads type spin default 4 min 1 max 256"),
        std::string::npos);
    EXPECT_NE(output.str().find(
        "option name ReportIntervalMs type spin default 0 min 0 max 60000"),
        std::string::npos);
}

TEST_F(UCIOpeningNoiseTest, ReportIntervalIsClampedAndConfirmed) {
    UCI uci;
    EXPECT_EQ(UCIOpeningNoiseTestPeer::report_interval_ms(uci), 0);

    testing::internal::CaptureStdout();
    set_option(uci, "ReportIntervalMs", "250");
    const std::string output = testing::internal::GetCapturedStdout();
    EXPECT_NE(output.find("info string ReportIntervalMs set to 250"),
              std::string::npos);
    EXPECT_EQ(UCIOpeningNoiseTestPeer::report_interval_ms(uci), 250);

    testing::internal::CaptureStdout();
    set_option(uci, "ReportIntervalMs", "-5");
    set_option(uci, "ReportIntervalMs", "999999");
    testing::internal::GetCapturedStdout();
    EXPECT_EQ(UCIOpeningNoiseTestPeer::report_interval_ms(uci), 60000);
}

TEST_F(UCIOpeningNoiseTest, MateThreadsSetsTheProbeWorkerCount) {
    UCI uci;
    set_option(uci, "MateThreads", "3");
    EXPECT_EQ(MateProbe::threads(), 3);
    set_option(uci, "MateThreads", "0");
    EXPECT_EQ(MateProbe::threads(), 1);
    set_option(uci, "MateThreads",
               std::to_string(MateProbe::DEFAULT_THREADS));
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

TEST_F(UCIOpeningNoiseTest, ClockAwarePlayIsTheDefaultForNetworksThatReadTheClock) {
    UCI clockNetwork;
    UCIOpeningNoiseTestPeer::set_network_reads_clock(clockNetwork, true);
    EXPECT_TRUE(UCIOpeningNoiseTestPeer::clock_model(clockNetwork));
    std::istringstream flag("name TimeModel value flag");
    clockNetwork.setoption(flag);
    EXPECT_FALSE(UCIOpeningNoiseTestPeer::clock_model(clockNetwork));

    // A network from before the clock planes keeps the flag unless asked.
    UCI legacyNetwork;
    EXPECT_FALSE(UCIOpeningNoiseTestPeer::clock_model(legacyNetwork));
    std::istringstream clock("name TimeModel value clock");
    legacyNetwork.setoption(clock);
    EXPECT_TRUE(UCIOpeningNoiseTestPeer::clock_model(legacyNetwork));
}

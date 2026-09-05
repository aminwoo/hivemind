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

    EXPECT_FLOAT_EQ(config.rootDirichletAlpha, 0.0f);
    EXPECT_FLOAT_EQ(config.rootDirichletEpsilon, 0.0f);
    EXPECT_EQ(config.rootNoiseSeed, 0U);
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
}

}  // namespace

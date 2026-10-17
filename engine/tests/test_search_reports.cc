#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "common/globals.h"
#include "environment/board.h"
#include "nn/engine.h"
#include "search/agent.h"

// What a verbose search prints while it runs, from real MCTS workers on the
// tiny uniform test model (zero value, uniform policy).
#if defined(HIVEMIND_BACKEND_ONNXRUNTIME)
namespace {

std::vector<std::string> info_depth_lines(const std::string& output) {
    std::vector<std::string> lines;
    std::istringstream in(output);
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("info depth ", 0) == 0) {
            lines.push_back(line);
        }
    }
    return lines;
}

// 0 when the line has no multipv field.
int multipv_rank(const std::string& line) {
    const auto at = line.find(" multipv ");
    return at == std::string::npos ? 0 : std::stoi(line.substr(at + 9));
}

class SearchReportsTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        init_fairy_stockfish();
        init_policy_index();
    }

    void SetUp() override {
        engine = std::make_unique<Engine>(0, 1);
        const auto model = std::filesystem::path(__FILE__).parent_path()
            / "data" / "uniform.onnx";
        ASSERT_TRUE(engine->loadNetwork(model.string(), ""));
        options = SearchOptions::uci(400, 3);
        options.enablePonder = false;
    }

    std::vector<std::string> search_and_capture() {
        Board board;
        Agent agent(2);
        testing::internal::CaptureStdout();
        agent.run_search(board, {engine.get()}, Stockfish::WHITE, false, options);
        return info_depth_lines(testing::internal::GetCapturedStdout());
    }

    std::unique_ptr<Engine> engine;
    SearchOptions options;
};

TEST_F(SearchReportsTest, IntervalPrintsWholeRankedBlocksWhileSearching) {
    options.reportIntervalMs = 50;
    const auto lines = search_and_capture();

    int blocks = 0;
    for (size_t i = 0; i < lines.size(); ++i) {
        const int rank = multipv_rank(lines[i]);
        ASSERT_GT(rank, 0) << lines[i];
        if (rank == 1) {
            ++blocks;
        } else {
            // Ranks within a block are contiguous: no other line splits it.
            ASSERT_GT(i, 0u);
            EXPECT_EQ(multipv_rank(lines[i - 1]), rank - 1) << lines[i];
        }
    }
    // Several periodic blocks, then the final one.
    EXPECT_GE(blocks, 3);
    EXPECT_EQ(multipv_rank(lines.back()), 3);
}

TEST_F(SearchReportsTest, NoIntervalPrintsOnlyTheRankOneLineUntilTheFinalBlock) {
    options.reportIntervalMs = 0;
    const auto lines = search_and_capture();

    ASSERT_GE(lines.size(), 4u);
    const size_t finalBlock = lines.size() - 3;
    for (size_t i = 0; i < finalBlock; ++i) {
        EXPECT_EQ(multipv_rank(lines[i]), 0) << lines[i];
    }
    for (size_t i = finalBlock; i < lines.size(); ++i) {
        EXPECT_EQ(multipv_rank(lines[i]), static_cast<int>(i - finalBlock + 1));
    }
}

}  // namespace
#endif

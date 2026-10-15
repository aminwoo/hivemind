#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <future>
#include <memory>
#include <thread>

#include "common/globals.h"
#include "environment/board.h"
#include "nn/engine.h"
#include "search/agent.h"
#include "search/searchthread.h"

// These integration tests exercise actual MCTS workers with a tiny ONNX model
// that returns zero value and uniform policy logits. No trained network or GPU
// is required. The model has one [1,78,8,8] input and three constant heads.
#if defined(HIVEMIND_BACKEND_ONNXRUNTIME)
namespace {

using Clock = std::chrono::steady_clock;

constexpr const char* kDropMateBoard =
    "r2q1kr1/pppbnp1p/5P1p/1B1P1p1N/N2Pp3/P1Pp2P1/2B1BP1P/R2QK2R"
    "[BNNQPqbnn] w KQ - 0 29";
constexpr const char* kDropMatePartner =
    "2r2r2/p2p2k1/1p2p2p/3b1pb1/P7/5P2/KP1p1PPP/2nR3R[P] w - - 1 41";

void set_busy_position(Board& board) {
    board.set(
        "4rk2/1pp2ppp/1pp2b2/8/1PP3B1/7P/P1P2PPP/4R1K1[] b - - 0 1|"
        "r2qnrk1/pp3ppp/2np2Bb/3NpPb1/3n1p2/3BpN1P/PPPB1PPP/R2Q1RK1"
        "[QRNNqrbn] b - - 0 1");
}

class RootMateSearchTest : public ::testing::Test {
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
        // Disable the separate Fairy probe so only the exact scan can supply
        // a root proof. Keep real MCTS and its normal solver running.
        options.search.enableMateProbe = false;
        options.targetNodes = 0;
    }

    std::unique_ptr<Engine> engine;
    SearchOptions options;
};

TEST_F(RootMateSearchTest, ProvenMateStopsMctsOnEitherBoard) {
    for (bool swapped : {false, true}) {
        Board board;
        board.set_fen(BOARD_A, swapped ? kDropMatePartner : kDropMateBoard);
        board.set_fen(BOARD_B, swapped ? kDropMateBoard : kDropMatePartner);
        const auto before = Agent::board_signature(board);
        Agent agent(2);
        options.moveTimeMs = 1500;
        const auto start = Clock::now();
        const auto action = agent.run_search(
            board, {engine.get()},
            swapped ? Stockfish::BLACK : Stockfish::WHITE, true, options);
        const auto elapsed = Clock::now() - start;
        EXPECT_LT(elapsed, std::chrono::milliseconds(750));
        EXPECT_EQ(agent.root_type(), NodeType::WIN);
        EXPECT_EQ(board.uci_move(swapped ? BOARD_B : BOARD_A,
            swapped ? action.moveB : action.moveA), "N@e6");
        EXPECT_EQ(swapped ? action.moveA : action.moveB, Stockfish::MOVE_NONE);
        EXPECT_EQ(Agent::board_signature(board), before);
        EXPECT_FALSE(agent.is_running());
    }
}

TEST_F(RootMateSearchTest, DeadlineCancelsTheScanBeforeReturningANormalMove) {
    Board board;
    set_busy_position(board);
    const auto before = Agent::board_signature(board);
    Agent agent(2);
    options.moveTimeMs = 50;
    const auto start = Clock::now();
    const auto action = agent.run_search(
        board, {engine.get()}, Stockfish::BLACK, false, options);
    EXPECT_LT(Clock::now() - start, std::chrono::milliseconds(500));
    EXPECT_TRUE(board.is_legal_move(BOARD_A, action.moveA));
    EXPECT_TRUE(board.is_legal_move(BOARD_B, action.moveB));
    EXPECT_EQ(Agent::board_signature(board), before);
    EXPECT_FALSE(agent.is_running());
}

TEST_F(RootMateSearchTest, FairyMateAlsoInterruptsTheLossScan) {
    Board board;
    board.set(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] b KQkq - 0 1|"
        "r1bq1b1r/ppp1p1pp/2n2nk1/3p2N1/3P4/8/PPP1PPPP/RNBQKB1R[Bb] w KQ - 2 2");
    const auto before = Agent::board_signature(board);
    Agent agent(2);
    options.search.enableMateProbe = true;
    options.moveTimeMs = 3000;
    const auto start = Clock::now();
    const auto action = agent.run_search(
        board, {engine.get()}, Stockfish::BLACK, true, options);
    EXPECT_LT(Clock::now() - start, std::chrono::milliseconds(2000));
    EXPECT_EQ(agent.root_type(), NodeType::WIN);
    EXPECT_EQ(action.moveA, Stockfish::MOVE_NONE);
    EXPECT_EQ(board.uci_move(BOARD_B, action.moveB), "B@f7");
    EXPECT_EQ(Agent::board_signature(board), before);
}

// Board A: 1.Kf7 Kh7 2.Rh1# (or 1.Kg6 Kg8 2.Ra8#), quiet first moves, so
// the checks-only scans cannot see them. Board B, our partner (black) on turn: bare kings in the
// inert case; in the leaky one a knight the opponents win in two moves there,
// which then blocks the h-file. White's team is 300 ms behind.
constexpr const char* kQuietMate = "7k/8/5K2/8/8/8/8/R7[] w - - 0 1";
constexpr const char* kInertPartner = "7k/8/8/8/8/8/8/K7[] b - - 0 1";
constexpr const char* kLeakyPartner = "2R5/8/8/7k/8/8/P7/n6K[] b - - 0 1";

// Behind on time the probe used not to run at all. It now finds the mate
// with board B sat, and the joint certifier, playing the real clock, proves
// it: the defenders' waits on board A only cost white moves on an inert
// board B.
TEST_F(RootMateSearchTest, CertifiedConditionalMateWinsBehindOnTime) {
    Board board;
    board.set(std::string(kQuietMate) + "|" + kInertPartner);
    const auto before = Agent::board_signature(board);
    Agent agent(2);
    options.search.enableMateProbe = true;
    options.search.conditionalMateProbe = true;
    options.verbose = true;
    options.moveTimeMs = 3000;
    options.clock = TimeState::clocked(Stockfish::WHITE, -300, 100, 100);
    testing::internal::CaptureStdout();
    const auto action = agent.run_search(
        board, {engine.get()}, Stockfish::WHITE, false, options);
    const std::string output = testing::internal::GetCapturedStdout();
    EXPECT_NE(output.find("conditional root mate: 1 found, 1 certified"),
              std::string::npos) << output;
    EXPECT_EQ(agent.root_type(), NodeType::WIN);
    // 1.Kf7 Kh7 2.Rh1# and 1.Kg6 Kg8 2.Ra8# are both mates in two.
    const std::string kingStep = board.uci_move(BOARD_A, action.moveA);
    EXPECT_TRUE(kingStep == "f6f7" || kingStep == "f6g6") << kingStep;
    EXPECT_TRUE(board.is_legal_move(BOARD_B, action.moveB));
    EXPECT_NE(action.moveB, Stockfish::MOVE_NONE);
    EXPECT_EQ(Agent::board_signature(board), before);
}

// The same probe hit with a knight to lose on board B is only conditional:
// the opponents' extra moves there feed the defence. No certificate, so no
// mate is claimed; the hit only orders the tree.
TEST_F(RootMateSearchTest, UncertifiedConditionalMateIsNoClaim) {
    Board board;
    board.set(std::string(kQuietMate) + "|" + kLeakyPartner);
    Agent agent(2);
    options.search.enableMateProbe = true;
    options.search.conditionalMateProbe = true;
    options.verbose = true;
    options.moveTimeMs = 1500;
    options.clock = TimeState::clocked(Stockfish::WHITE, -300, 100, 100);
    testing::internal::CaptureStdout();
    agent.run_search(board, {engine.get()}, Stockfish::WHITE, false, options);
    const std::string output = testing::internal::GetCapturedStdout();
    // The hint is promoted only if the tree has not expanded that action
    // yet, which is a race, so the count of promotions is not checked.
    EXPECT_NE(output.find("conditional root mate: 1 found, 0 certified, "),
              std::string::npos) << output;
    EXPECT_NE(agent.root_type(), NodeType::WIN);
    EXPECT_EQ(output.find("score mate"), std::string::npos) << output;
}

TEST_F(RootMateSearchTest, StopJoinsTheScanAndTheNextSearchCanRun) {
    Board board;
    set_busy_position(board);
    const auto before = Agent::board_signature(board);
    Agent agent(2);
    options.moveTimeMs = 10000;
    auto search = std::async(std::launch::async, [&] {
        return agent.run_search(
            board, {engine.get()}, Stockfish::BLACK, false, options);
    });
    const auto dispatchDeadline = Clock::now() + std::chrono::seconds(2);
    while (!agent.is_running() && Clock::now() < dispatchDeadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_TRUE(agent.is_running());
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    const auto stopped = Clock::now();
    agent.set_is_running(false);
    EXPECT_EQ(search.wait_for(std::chrono::milliseconds(500)),
        std::future_status::ready);
    search.get();
    EXPECT_LT(Clock::now() - stopped, std::chrono::milliseconds(500));
    EXPECT_EQ(Agent::board_signature(board), before);

    Board initial;
    board.set(initial.fen(BOARD_A) + "|" + initial.fen(BOARD_B));
    options.moveTimeMs = 30;
    const auto action = agent.run_search(
        board, {engine.get()}, Stockfish::WHITE, true, options);
    EXPECT_TRUE(board.is_legal_move(BOARD_A, action.moveA));
    EXPECT_TRUE(board.is_legal_move(BOARD_B, action.moveB));
    EXPECT_FALSE(agent.is_running());
}

TEST_F(RootMateSearchTest, NodeTargetCancelsTheScan) {
    Board board;
    set_busy_position(board);
    const auto before = Agent::board_signature(board);
    Agent agent(2);
    options.moveTimeMs = 0;
    options.targetNodes = 100;
    const auto start = Clock::now();
    agent.run_search(board, {engine.get()}, Stockfish::BLACK, false, options);
    EXPECT_LT(Clock::now() - start, std::chrono::milliseconds(500));
    EXPECT_EQ(Agent::board_signature(board), before);
    EXPECT_FALSE(agent.is_running());
}

TEST_F(RootMateSearchTest, WorkerFailureAlsoJoinsTheScan) {
    Board board;
    set_busy_position(board);
    Engine unloaded(0, 1);
    Agent agent(2);
    options.moveTimeMs = 10000;
    const auto start = Clock::now();
    EXPECT_THROW(agent.run_search(
        board, {&unloaded}, Stockfish::BLACK, false, options),
        std::runtime_error);
    EXPECT_LT(Clock::now() - start, std::chrono::milliseconds(500));
    EXPECT_FALSE(agent.is_running());
}

TEST_F(RootMateSearchTest, CancellationDoesNotSubmitAnotherInferenceBatch) {
    Board board;
    const auto before = Agent::board_signature(board);
    const auto time = TimeState::legacy(Stockfish::WHITE, true);
    auto root = std::make_shared<Node>(
        Stockfish::WHITE, board.search_hash_key(Stockfish::WHITE, time));
    root->configure_root_search(options.search);
    SearchInfo info(Clock::now(), 1000);
    SearchThread worker;
    worker.set_root_node(root);
    worker.set_search_info(&info);
    worker.set_runtime_config(options.search);
    std::atomic<bool> keepRunning{true};

    // Expand the root, then leave a real neural batch in flight.
    worker.run_iteration(board, engine.get(), time, &keepRunning);
    worker.run_iteration(board, engine.get(), time, &keepRunning);
    const int completed = info.get_nodes_searched();
    keepRunning.store(false, std::memory_order_release);
    worker.run_iteration(board, engine.get(), time, &keepRunning);
    EXPECT_GT(info.get_nodes_searched(), completed);

    Engine::HalfInferenceOutputs outputs;
    EXPECT_FALSE(engine->synchronizeInferenceHalf(outputs))
        << "a new lookahead batch was submitted after cancellation";
    EXPECT_EQ(Agent::board_signature(board), before);

    // Aborted reservations must not prevent this worker from searching again.
    keepRunning.store(true, std::memory_order_release);
    worker.run_iteration(board, engine.get(), time, &keepRunning);
    worker.finish_pending_iteration(board, engine.get(), time);
    EXPECT_GT(info.get_nodes_searched(), completed + 1);
}

}  // namespace
#endif

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
    // 1.Kf7 Kh7 2.Rh1# and 1.Kg6 Kg8 2.Ra8# mate in two on board A, but
    // 300 ms behind the defenders can also sit there, so the whole mate
    // takes six attacker actions (see ExhaustiveOracleCountsTheSitsTheLeadBuys).
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

// Qb7# is a contact mate: no drop can meet it, so the clock has no say.
// The partner board is quiet, with our partner on turn there or not.
constexpr const char* kContactMate = "k7/2K5/8/8/8/8/8/1Q6[] w - - 0 1";
constexpr const char* kQuietPartnerOpponentsOnTurn =
    "rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR[] w KQkq - 0 2";
constexpr const char* kQuietPartnerOnTurn =
    "rnbqkbnr/pppp1ppp/8/4p3/3P4/5N2/PPP1PPPP/RNBQKB1R[] b KQkq - 1 2";

// A team far behind on the clock still plays and reports its mate in 1, on
// either board and whether or not it is on turn on both.
TEST_F(RootMateSearchTest, MateInOneIsReportedBehindOnTime) {
    for (bool swapped : {false, true}) {
        for (bool onTurnOnBoth : {false, true}) {
            const char* partner = onTurnOnBoth
                ? kQuietPartnerOnTurn : kQuietPartnerOpponentsOnTurn;
            Board board;
            board.set(swapped
                ? std::string(partner) + "|" + kContactMate
                : std::string(kContactMate) + "|" + partner);
            const Stockfish::Color team = swapped
                ? Stockfish::BLACK : Stockfish::WHITE;
            const int mateBoard = swapped ? BOARD_B : BOARD_A;
            Agent agent(2);
            options.verbose = true;
            options.moveTimeMs = 1000;
            options.clock = TimeState::clocked(team, -30000, 100, 100);
            testing::internal::CaptureStdout();
            const auto action = agent.run_search(
                board, {engine.get()}, team, false, options);
            const std::string output = testing::internal::GetCapturedStdout();
            EXPECT_EQ(agent.root_type(), NodeType::WIN)
                << "swapped " << swapped << " both " << onTurnOnBoth;
            EXPECT_NE(output.find("score mate 1 "), std::string::npos) << output;
            EXPECT_EQ(output.find("score mate -"), std::string::npos) << output;
            EXPECT_EQ(output.find("score cp"), std::string::npos) << output;
            const std::string mate = board.uci_move(
                mateBoard, swapped ? action.moveB : action.moveA);
            EXPECT_TRUE(mate == "b1b7" || mate == "b1b8") << mate;
        }
    }
}

std::string mate_telemetry_line(const std::string& output) {
    const std::string prefix = "info string mate telemetry ";
    const size_t start = output.find(prefix);
    if (start == std::string::npos
        || output.find(prefix, start + 1) != std::string::npos) {
        return {};
    }
    return output.substr(start, output.find('\n', start) - start) + " ";
}

// Each verbose search reports once where its proofs came from and why the
// finders that found none stopped. Behind on time the root probe does not
// run; the conditional one is tried, and a certificate it cannot get says why.
TEST_F(RootMateSearchTest, MateTelemetryReportsTheConditionalCertificate) {
    for (const char* partner : {kInertPartner, kLeakyPartner}) {
        const bool inert = partner == kInertPartner;
        Board board;
        board.set(std::string(kQuietMate) + "|" + partner);
        Agent agent(2);
        options.search.enableMateProbe = true;
        options.search.conditionalMateProbe = true;
        options.verbose = true;
        options.moveTimeMs = inert ? 3000 : 1500;
        options.clock = TimeState::clocked(Stockfish::WHITE, -300, 100, 100);
        testing::internal::CaptureStdout();
        agent.run_search(board, {engine.get()}, Stockfish::WHITE, false, options);
        const std::string line =
            mate_telemetry_line(testing::internal::GetCapturedStdout());
        ASSERT_FALSE(line.empty()) << "expected exactly one telemetry line";
        EXPECT_NE(line.find(" lead=-300 "), std::string::npos) << line;
        EXPECT_NE(line.find(" probe=0 "), std::string::npos) << line;
        EXPECT_NE(line.find(" cond_tried=1 "), std::string::npos) << line;
        EXPECT_NE(line.find(" cond_found=1 "), std::string::npos) << line;
        // The inert line is certified when its share of the move suffices,
        // which a loaded machine can deny it; either way the line agrees
        // with itself. The leaky one never is, and says why.
        const bool certified = line.find(" cond_cert=1 ") != std::string::npos;
        EXPECT_TRUE(inert || !certified) << line;
        EXPECT_EQ(line.find(" cond_fail=none ") != std::string::npos,
                  certified) << line;
        EXPECT_EQ(line.find(" probe_win_ms=-1 ") == std::string::npos,
                  certified) << line;
    }
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

// Cancels a worker with a real batch in flight. With submitBeforeBackup the
// lookahead is submitted before the backup instead of after; either way no
// batch may be submitted once the search is told to stop.
void check_cancellation_does_not_submit(Engine& engineRef, SearchOptions options,
                                        bool submitBeforeBackup) {
    Engine* engine = &engineRef;
    options.search.submitBeforeBackup = submitBeforeBackup;
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
    worker.run_iteration(board, engine, time, &keepRunning);
    worker.run_iteration(board, engine, time, &keepRunning);
    const int completed = info.get_nodes_searched();
    keepRunning.store(false, std::memory_order_release);
    worker.run_iteration(board, engine, time, &keepRunning);
    EXPECT_GT(info.get_nodes_searched(), completed);

    Engine::HalfInferenceOutputs outputs;
    EXPECT_FALSE(engine->synchronizeInferenceHalf(outputs))
        << "a new lookahead batch was submitted after cancellation";
    EXPECT_EQ(Agent::board_signature(board), before);

    // Aborted reservations must not prevent this worker from searching again.
    keepRunning.store(true, std::memory_order_release);
    worker.run_iteration(board, engine, time, &keepRunning);
    worker.finish_pending_iteration(board, engine, time);
    EXPECT_GT(info.get_nodes_searched(), completed + 1);
}

TEST_F(RootMateSearchTest, CancellationDoesNotSubmitAnotherInferenceBatch) {
    check_cancellation_does_not_submit(*engine, options, false);
}

TEST_F(RootMateSearchTest, CancellationDoesNotSubmitAnotherBatchWhenSubmittingBeforeBackup) {
    check_cancellation_does_not_submit(*engine, options, true);
}

// The descent charges the root team's first action its budget, so every child
// of a root on turn on both boards carries the budgeted lead, and later actions
// pay the usual move time. Here the budget decides who may sit after the move:
// at the usual 1000 ms white falls 200 ms behind, at its 400 ms budget it
// still leads by 400.
TEST_F(RootMateSearchTest, TheRootActionPaysItsBudgetAndOnlyIt) {
    Board board;
    board.set(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1|"
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR[] b KQkq - 0 1");
    ASSERT_TRUE(board.on_turn_on_both_boards(Stockfish::WHITE));
    const TimeState plain = TimeState::clocked(Stockfish::WHITE, 800, 1000, 500);
    const TimeState budgeted = TimeState::clocked(Stockfish::WHITE, 800, 1000, 500, 400);
    ASSERT_TRUE(plain.after(Stockfish::WHITE, true).can_sit(Stockfish::BLACK));
    ASSERT_FALSE(budgeted.after(Stockfish::WHITE, true).can_sit(Stockfish::BLACK));
    EXPECT_NE(board.search_hash_key(Stockfish::WHITE, plain),
              board.search_hash_key(Stockfish::WHITE, budgeted));

    for (const TimeState& rootTime : {plain, budgeted}) {
        auto root = std::make_shared<Node>(
            Stockfish::WHITE, board.search_hash_key(Stockfish::WHITE, rootTime));
        root->configure_root_search(options.search);
        SearchInfo info(Clock::now(), 1000);
        SearchThread worker;
        worker.set_root_node(root);
        worker.set_search_info(&info);
        worker.set_runtime_config(options.search);
        std::atomic<bool> keepRunning{true};
        for (int i = 0; i < 40; ++i) {
            worker.run_iteration(board, engine.get(), rootTime, &keepRunning);
        }
        worker.finish_pending_iteration(board, engine.get(), rootTime);

        // Each child is keyed by its exact clock: the lead after the root
        // action at the cost this root was given.
        const TimeState childTime = rootTime.after(Stockfish::WHITE, true);
        int checked = 0;
        for (int index = 0; index < static_cast<int>(root->get_num_generated()); ++index) {
            const std::shared_ptr<Node> child = root->get_child(index);
            if (!child) {
                continue;
            }
            const JointActionCandidate action = root->get_joint_action(index);
            Board after(board);
            after.make_moves(action.moveA, action.moveB);
            EXPECT_EQ(child->get_hash(), after.search_hash_key(Stockfish::BLACK, childTime))
                << "child " << index;
            ++checked;
        }
        EXPECT_GT(checked, 5);
        // Below the root the budget is gone: white's next action costs its
        // usual time again.
        EXPECT_EQ(childTime.next_move_time_ms(Stockfish::WHITE), 1000);
        worker.release_tree_references();
    }
}

// One worker on its own is deterministic, and submitting before the backup
// changes only when the GPU sees each batch, so both orders grow the same tree.
TEST_F(RootMateSearchTest, SubmittingBeforeBackupSearchesLikeTheDefault) {
    const auto time = TimeState::legacy(Stockfish::WHITE, true);
    std::vector<std::pair<int, std::vector<int>>> results;
    for (const bool submitBeforeBackup : {false, true}) {
        Board board;
        options.search.submitBeforeBackup = submitBeforeBackup;
        auto root = std::make_shared<Node>(
            Stockfish::WHITE, board.search_hash_key(Stockfish::WHITE, time));
        root->configure_root_search(options.search);
        SearchInfo info(Clock::now(), 1000);
        SearchThread worker;
        worker.set_root_node(root);
        worker.set_search_info(&info);
        worker.set_runtime_config(options.search);
        std::atomic<bool> keepRunning{true};
        for (int i = 0; i < 20; ++i) {
            worker.run_iteration(board, engine.get(), time, &keepRunning);
        }
        worker.finish_pending_iteration(board, engine.get(), time);
        worker.release_tree_references();
        results.emplace_back(info.get_nodes_searched(), root->get_child_visits());
    }
    EXPECT_GT(results[0].first, 10);
    EXPECT_GT(results[0].second.size(), 1u);
    EXPECT_EQ(results[0], results[1]);
}

// The search reads a batch's outputs after submitting the next batch, so a
// batch's outputs must survive the next request: marks left in them are still
// there after the next synchronize, and the next request got buffers of its own.
TEST_F(RootMateSearchTest, OutputsSurviveTheNextRequest) {
    std::vector<__half> input(NB_INPUT_VALUES());
    Engine::HalfInferenceOutputs first, second;
    ASSERT_TRUE(engine->enqueueInferenceHalf(input.data()));
    ASSERT_TRUE(engine->synchronizeInferenceHalf(first));
    const __half mark = __float2half_rn(0.625f);
    auto* value = const_cast<__half*>(first.value);
    auto* policyA = const_cast<__half*>(first.policyA);
    auto* policyB = const_cast<__half*>(first.policyB);
    value[0] = mark;
    policyA[0] = mark;
    policyB[NB_POLICY_VALUES() - 1] = mark;

    ASSERT_TRUE(engine->enqueueInferenceHalf(input.data()));
    ASSERT_TRUE(engine->synchronizeInferenceHalf(second));
    EXPECT_EQ(__half2float(first.value[0]), 0.625f);
    EXPECT_EQ(__half2float(first.policyA[0]), 0.625f);
    EXPECT_EQ(__half2float(first.policyB[NB_POLICY_VALUES() - 1]), 0.625f);
    EXPECT_NE(__half2float(second.value[0]), 0.625f);
    EXPECT_NE(__half2float(second.policyA[0]), 0.625f);
}

}  // namespace
#endif

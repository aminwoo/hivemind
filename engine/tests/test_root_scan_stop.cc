#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "common/globals.h"
#include "environment/board.h"
#include "environment/constants.h"
#include "search/agent.h"
#include "search/search_params.h"

namespace {

// chess.com 183938310513/183938310515, our team black, `go nodes 20000`. The
// tree finished in about two seconds; bestmove followed 377 seconds later,
// because a node-limited search gives the root loss scan no deadline and its
// probes here cost far more than the per-node estimate its budget assumes.
// Every defense is a queen or rook drop with a full hand on both sides, so
// the scan's capture-feed proofs are as wide as they get.
constexpr const char* kStallBoardA =
    "r7/pp1kbp1p/6r1/1p6/3n1qP1/5N1P/PPP2P2/5K1R[PPqrbbnnppp] b - - 1 30";
constexpr const char* kStallBoardB =
    "Q2nk2r/p1p2ppp/2b5/8/3B4/4P1P1/PPP2KP1/4RB1R[QRBBNNNPPppp] w k - 1 30";

class RootScanStopTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        init_fairy_stockfish();
        init_policy_index();
    }
};

}  // namespace

// The MCTS workers end a node-limited search's root scans by raising the
// scan's `cancelled` flag once the tree is done. That only bounds bestmove if
// every layer of the loss scan - the per-defense joint proofs, capture feeds
// and Fairy-Stockfish probes underneath - polls the flag rather than running
// its allocation out. Pin that on the position that held a game for minutes.
TEST_F(RootScanStopTest, LossScanReturnsPromptlyOnceCancelled) {
    using Clock = std::chrono::steady_clock;

    Board board;
    board.set_fen(BOARD_A, kStallBoardA);
    board.set_fen(BOARD_B, kStallBoardB);

    std::atomic<bool> cancelled{false};
    std::atomic<bool> finished{false};
    std::vector<RootLossProof> proofs;
    bool proved = false;
    std::thread scan([&] {
        proved = Agent::find_root_loss_proofs(
            board, Stockfish::BLACK, false, proofs,
            SearchParams::MATE_SEARCH_NODE_BUDGET, {}, &cancelled);
        finished.store(true, std::memory_order_release);
    });

    // Left alone, the scan is the problem this guards against: it must still
    // be running when the flag goes up, or the test proves nothing.
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    EXPECT_FALSE(finished.load(std::memory_order_acquire))
        << "the loss scan finished before it could be cancelled; pick a "
           "position it cannot settle in 300ms";

    const auto cancelledAt = Clock::now();
    cancelled.store(true, std::memory_order_release);
    scan.join();
    const auto stopMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now() - cancelledAt).count();

    EXPECT_LT(stopMs, 1000) << "cancellation took " << stopMs << "ms";
    EXPECT_FALSE(proved) << "a cut-short scan must not claim a proof";
}

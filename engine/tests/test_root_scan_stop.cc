#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <utility>
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

namespace {

// chess.com 186252389103 ply 114, our team white and a second behind on the
// clock: the opponents mate on board B in two whatever we do.
constexpr const char* kLostBoardA =
    "r4r2/ppp2pkp/2B2pN1/6P1/3P2b1/5N1P/P1P2P1P/R3Q1K1[PPNBRQpppppbbb] b - - 0 25";
constexpr const char* kLostBoardB =
    "4r1nr/p4Rpp/4p3/1B2N1k1/3P3P/8/P1PP1PP1/4K2R[Bpnnnqq] b K - 0 33";

std::vector<std::pair<Stockfish::Move, Stockfish::Move>> proven_actions(
    const std::vector<RootLossProof>& proofs) {
    std::vector<std::pair<Stockfish::Move, Stockfish::Move>> actions;
    for (const RootLossProof& proof : proofs) {
        actions.emplace_back(proof.action.moveA, proof.action.moveB);
    }
    std::sort(actions.begin(), actions.end());
    return actions;
}

}  // namespace

// Threads sharing one loss scan take its defenses from a common queue, each
// on its own board. Between them they must prove every defense one thread
// proves, each exactly once.
TEST_F(RootScanStopTest, LossScanThreadsShareTheDefenses) {
    Board board;
    board.set_fen(BOARD_A, kLostBoardA);
    board.set_fen(BOARD_B, kLostBoardB);
    const TimeState time = TimeState::legacy(Stockfish::WHITE, false);
    constexpr uint64_t nodes = 1000000;

    std::vector<RootLossProof> alone;
    ASSERT_TRUE(Agent::find_root_loss_proofs(
        board, Stockfish::WHITE, time, alone, nodes));
    ASSERT_GT(alone.size(), 1u);

    LossScanShare share;
    share.remainingNodes.store(nodes);
    std::vector<RootLossProof> first;
    std::vector<RootLossProof> second;
    Board secondBoard(board);
    std::thread helper([&] {
        Agent::find_root_loss_proofs(
            secondBoard, Stockfish::WHITE, time, second, nodes, {}, nullptr,
            nullptr, nullptr, true, nullptr, {}, nullptr, &share);
    });
    Agent::find_root_loss_proofs(
        board, Stockfish::WHITE, time, first, nodes, {}, nullptr, nullptr,
        nullptr, true, nullptr, {}, nullptr, &share);
    helper.join();

    std::vector<RootLossProof> together = first;
    together.insert(together.end(), second.begin(), second.end());
    EXPECT_EQ(proven_actions(together), proven_actions(alone));
}

// A scan handed proofs from an earlier round keeps them without searching
// those defenses again, and with them proves the loss.
TEST_F(RootScanStopTest, LossScanKeepsKnownProofs) {
    Board board;
    board.set_fen(BOARD_A, kLostBoardA);
    board.set_fen(BOARD_B, kLostBoardB);
    const TimeState time = TimeState::legacy(Stockfish::WHITE, false);

    std::vector<RootLossProof> all;
    ASSERT_TRUE(Agent::find_root_loss_proofs(
        board, Stockfish::WHITE, time, all, 1000000));
    ASSERT_GT(all.size(), 1u);

    // Every defense known: nothing is left to search, not even one node.
    std::vector<RootLossProof> again;
    bool complete = false;
    EXPECT_TRUE(Agent::find_root_loss_proofs(
        board, Stockfish::WHITE, time, again, 1, {}, nullptr, nullptr,
        nullptr, true, &complete, {}, &all));
    EXPECT_TRUE(complete);
    EXPECT_EQ(proven_actions(again), proven_actions(all));

    // Half known: the scan proves the rest and returns them all, once each.
    const std::vector<RootLossProof> half(
        all.begin(), all.begin() + all.size() / 2);
    std::vector<RootLossProof> rest;
    EXPECT_TRUE(Agent::find_root_loss_proofs(
        board, Stockfish::WHITE, time, rest, 1000000, {}, nullptr, nullptr,
        nullptr, true, nullptr, {}, &half));
    EXPECT_EQ(proven_actions(rest), proven_actions(all));
}

namespace {

// Black on board A is on turn with a queen in hand and mates in three,
// Q@h2+ Nxh2 N@g3+ fxg3 N@f2#, whatever our team (White on A, Black on B)
// plays on board B: every check is a contact or knight check, and our
// partner on A cannot move until the opponents do.
constexpr const char* kSharedThreatBoardA =
    "r4r1k/ppp3pp/6p1/3pP1N1/5bN1/2P4P/P1P2PB1/6RK[PBBBnnq] b - - 0 1";
constexpr const char* kSharedThreatBoardB =
    "4k1n1/p1ppbpbp/4pb2/6pN/3P2P1/6PK/PPP1PP2/R2Q1R2[PNRQppnrrq] b - - 0 1";

}  // namespace

// Every quiet board-B defense leaves board A as it was, so one proof of the
// board-A mate answers all of them. A budget far too small to prove the mate
// once per defense still proves the whole root lost.
TEST_F(RootScanStopTest, LossScanSharesOneDirectThreatProof) {
    Board board;
    board.set_fen(BOARD_A, kSharedThreatBoardA);
    board.set_fen(BOARD_B, kSharedThreatBoardB);
    const TimeState time = TimeState::clocked(Stockfish::WHITE, -600);

    std::vector<RootLossProof> proofs;
    LossScanStats stats;
    EXPECT_TRUE(Agent::find_root_loss_proofs(
        board, Stockfish::WHITE, time, proofs, 100000, {}, nullptr, nullptr,
        nullptr, true, nullptr, {}, nullptr, nullptr, &stats));
    bool provedCapture = false;
    for (const RootLossProof& proof : proofs) {
        ASSERT_FALSE(proof.principalVariation.empty());
        EXPECT_EQ(proof.principalVariation.front().moveA,
                  proof.action.moveA);
        EXPECT_EQ(proof.principalVariation.front().moveB,
                  proof.action.moveB);
        // Our defense, then Q@h2+ Nxh2 N@g3+ fxg3 N@f2#.
        EXPECT_EQ(proof.plyToMate, 6);
        EXPECT_EQ(proof.principalVariation.size(), 6u);
        provedCapture = provedCapture
            || (proof.action.moveB != Stockfish::MOVE_NONE
                && board.is_capture(BOARD_B, proof.action.moveB));
    }
    // Quiet defenses share one certificate; each capture feeds our partner
    // on A a piece, changes board A's hand and gets one of its own.
    EXPECT_TRUE(provedCapture);
    EXPECT_GE(stats.certificateSearches, 2u);
    EXPECT_LT(stats.certificateSearches, 10u);
    EXPECT_GT(stats.certificateProofs, 40u);
}

// The certificate sits board B for every opponent action of the line, so their
// lead must pay for all three: at 100 ms a move that is exactly 300 ms.
TEST_F(RootScanStopTest, DirectThreatCertificateNeedsTheLeadForTheWholeLine) {
    Board board;
    board.set_fen(BOARD_A, kSharedThreatBoardA);
    board.set_fen(BOARD_B, kSharedThreatBoardB);

    const auto certificate_proofs = [&](int64_t ourLeadMs) {
        std::vector<RootLossProof> proofs;
        LossScanStats stats;
        Agent::find_root_loss_proofs(
            board, Stockfish::WHITE,
            TimeState::clocked(Stockfish::WHITE, ourLeadMs), proofs, 100000,
            {}, nullptr, nullptr, nullptr, true, nullptr, {}, nullptr,
            nullptr, &stats);
        return stats.certificateProofs;
    };
    EXPECT_GT(certificate_proofs(-300), 0u);
    EXPECT_EQ(certificate_proofs(-299), 0u);
}

// With our team on turn on board A as well, a defense can change board A, so
// no defense leaves it untouched and none is answered by a certificate.
TEST_F(RootScanStopTest, NoDirectThreatCertificateWhileWeAreOnTurnThere) {
    Board board;
    board.set_fen(BOARD_A,
        "r4r1k/ppp3pp/6p1/3pP1N1/5bN1/2P4P/P1P2PB1/6RK[PBBBnnq] w - - 0 1");
    board.set_fen(BOARD_B, kSharedThreatBoardB);

    std::vector<RootLossProof> proofs;
    LossScanStats stats;
    Agent::find_root_loss_proofs(
        board, Stockfish::WHITE, TimeState::clocked(Stockfish::WHITE, -600),
        proofs, 100000, {}, nullptr, nullptr, nullptr, true, nullptr, {},
        nullptr, nullptr, &stats);
    EXPECT_EQ(stats.certificateSearches, 0u);
    EXPECT_EQ(stats.certificateProofs, 0u);
}

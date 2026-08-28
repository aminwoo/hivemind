#include <gtest/gtest.h>

#include <vector>

#include "interface/uci.h"

namespace {

/** Builds a root edge for a distinct joint action, identified by its moves. */
RootEdgeStats edge(int moveA, int moveB, int visits, float q) {
    RootEdgeStats stats;
    stats.action.moveA = static_cast<Stockfish::Move>(moveA);
    stats.action.moveB = static_cast<Stockfish::Move>(moveB);
    stats.visits = visits;
    stats.q = q;
    return stats;
}

/** Builds one tree's contribution, deciding on its first edge by default. */
RootTreeResult tree(std::vector<RootEdgeStats> edges,
                    NodeType rootType = NodeType::UNSOLVED) {
    RootTreeResult result;
    result.edges = std::move(edges);
    result.rootType = rootType;
    if (!result.edges.empty()) {
        result.decision = result.edges.front().action;
    }
    return result;
}

bool is_action(const RootEdgeStats& stats, int moveA, int moveB) {
    return stats.action.moveA == static_cast<Stockfish::Move>(moveA)
        && stats.action.moveB == static_cast<Stockfish::Move>(moveB);
}

}  // namespace

TEST(RootMergeTest, EmptyInputYieldsNoAction) {
    const MergedRootResult merged = merge_root_results({});
    EXPECT_FALSE(merged.hasAction);
    EXPECT_TRUE(merged.edges.empty());
    EXPECT_EQ(merged.rootType, NodeType::UNSOLVED);
}

TEST(RootMergeTest, TreesWithoutEdgesYieldNoAction) {
    const MergedRootResult merged = merge_root_results({tree({}), tree({})});
    EXPECT_FALSE(merged.hasAction);
}

// The point of merging rather than voting: a move that no single tree ranks
// first can still win once the visit counts are summed.
TEST(RootMergeTest, SumsVisitsAcrossTrees) {
    const MergedRootResult merged = merge_root_results({
        tree({edge(1, 1, 50, 0.2f), edge(2, 2, 40, 0.1f)}),
        tree({edge(1, 1, 10, 0.2f), edge(2, 2, 40, 0.1f)}),
    });

    ASSERT_TRUE(merged.hasAction);
    ASSERT_EQ(merged.edges.size(), 2u);
    EXPECT_TRUE(is_action(merged.edges[0], 2, 2));
    EXPECT_EQ(merged.edges[0].visits, 80);
    EXPECT_TRUE(is_action(merged.edges[1], 1, 1));
    EXPECT_EQ(merged.edges[1].visits, 60);
    EXPECT_EQ(merged.action.moveA, static_cast<Stockfish::Move>(2));
}

TEST(RootMergeTest, AveragesQWeightedByVisits) {
    const MergedRootResult merged = merge_root_results({
        tree({edge(1, 1, 30, 0.5f)}),
        tree({edge(1, 1, 10, 0.1f)}),
    });

    ASSERT_EQ(merged.edges.size(), 1u);
    EXPECT_EQ(merged.edges[0].visits, 40);
    // (0.5 * 30 + 0.1 * 10) / 40
    EXPECT_NEAR(merged.edges[0].q, 0.4f, 1e-6f);
}

// An unvisited edge still carries unit weight, so its Q must not be discarded
// and must not divide by zero.
TEST(RootMergeTest, UnvisitedEdgesKeepUnitWeight) {
    const MergedRootResult merged = merge_root_results({
        tree({edge(1, 1, 0, 0.6f)}),
        tree({edge(1, 1, 0, 0.2f)}),
    });

    ASSERT_EQ(merged.edges.size(), 1u);
    EXPECT_EQ(merged.edges[0].visits, 0);
    EXPECT_NEAR(merged.edges[0].q, 0.4f, 1e-6f);
}

// A solver proof is sound even when only one tree finds it, so it must beat a
// far more heavily visited unsolved edge.
TEST(RootMergeTest, ProvenWinBeatsMoreVisitedEdge) {
    const MergedRootResult merged = merge_root_results({
        tree({edge(7, 7, 5, 1.0f)}, NodeType::WIN),
        tree({edge(1, 1, 500, 0.3f)}),
    });

    ASSERT_TRUE(merged.hasAction);
    EXPECT_EQ(merged.rootType, NodeType::WIN);
    EXPECT_TRUE(is_action(merged.edges[0], 7, 7));
    EXPECT_EQ(merged.action.moveA, static_cast<Stockfish::Move>(7));
}

TEST(RootMergeTest, ProvenWinOutranksProvenLoss) {
    const MergedRootResult merged = merge_root_results({
        tree({edge(1, 1, 10, -1.0f)}, NodeType::LOSS),
        tree({edge(7, 7, 5, 1.0f)}, NodeType::WIN),
    });

    EXPECT_EQ(merged.rootType, NodeType::WIN);
    EXPECT_TRUE(is_action(merged.edges[0], 7, 7));
}

TEST(RootMergeTest, ProvenLossIsAdoptedFromASingleTree) {
    const MergedRootResult merged = merge_root_results({
        tree({edge(1, 1, 10, -0.9f)}, NodeType::LOSS),
        tree({edge(2, 2, 20, 0.0f)}),
    });

    EXPECT_EQ(merged.rootType, NodeType::LOSS);
    EXPECT_TRUE(is_action(merged.edges[0], 1, 1));
}

TEST(RootMergeTest, UnsolvedTreesRankByVisitsThenQ) {
    const MergedRootResult merged = merge_root_results({
        tree({edge(1, 1, 20, 0.1f), edge(2, 2, 20, 0.9f)}),
    });

    ASSERT_EQ(merged.edges.size(), 2u);
    EXPECT_TRUE(is_action(merged.edges[0], 2, 2));
    EXPECT_TRUE(is_action(merged.edges[1], 1, 1));
}

// The representative tree supplies the ponder move and mate distance, so it has
// to be a tree that actually chose the merged action, preferring the one that
// searched it hardest.
TEST(RootMergeTest, RepresentativeTreeSearchedTheChosenActionHardest) {
    const MergedRootResult merged = merge_root_results({
        tree({edge(9, 9, 100, 0.4f)}),
        tree({edge(1, 1, 30, 0.2f)}),
        tree({edge(9, 9, 400, 0.4f)}),
    });

    EXPECT_TRUE(is_action(merged.edges[0], 9, 9));
    EXPECT_EQ(merged.representativeTree, 2u);
}

TEST(RootMergeTest, RepresentativeTreeStaysInRangeWhenNoTreeAgrees) {
    // Both trees decide on their own first edge; the merged winner is the
    // action they jointly visited most, which neither of them decided on.
    const MergedRootResult merged = merge_root_results({
        tree({edge(1, 1, 60, 0.5f), edge(3, 3, 50, 0.4f)}),
        tree({edge(2, 2, 60, 0.5f), edge(3, 3, 50, 0.4f)}),
    });

    ASSERT_TRUE(merged.hasAction);
    EXPECT_TRUE(is_action(merged.edges[0], 3, 3));
    EXPECT_LT(merged.representativeTree, 2u);
}

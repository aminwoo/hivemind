#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "search/agent.h"

inline std::vector<RootEdgeStats> selfplay_policy_edges(
    const std::vector<RootEdgeStats>& edges, NodeType rootType,
    const JointActionCandidate& choice, bool certifiedChoice) {
    if (certifiedChoice || rootType == NodeType::LOSS) {
        return {{choice, 1, 0.0f}};
    }
    const bool hasAlternative = std::any_of(edges.begin(), edges.end(), [](const RootEdgeStats& edge) {
        return edge.childType != NodeType::WIN;
    });
    std::vector<RootEdgeStats> eligible;
    for (const RootEdgeStats& edge : edges) {
        if (rootType == NodeType::WIN ? edge.childType == NodeType::LOSS
                                     : !hasAlternative || edge.childType != NodeType::WIN) {
            eligible.push_back(edge);
        }
    }
    if (eligible.empty()) {
        throw std::runtime_error("Solved root has no eligible policy edges");
    }
    if (std::none_of(eligible.begin(), eligible.end(), [](const RootEdgeStats& edge) {
            return edge.visits > 0;
        })) {
        for (RootEdgeStats& edge : eligible) {
            edge.visits = 1;
        }
    }
    return eligible;
}

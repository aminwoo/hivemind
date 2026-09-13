#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>

#include "environment/joint_action.h"

struct MateCandidateHint {
    JointActionCandidate action;
    int plyToMate = 0;
};

/**
 * Search-local Fairy-Stockfish candidates keyed by complete position identity.
 * Entries influence exploration only; exact solver state lives on Node.
 */
class MateCandidateTable {
public:
    void publish(uint64_t positionHash, const JointActionCandidate& action,
                 int plyToMate) {
        std::unique_lock lock(mutex_);
        const auto found = hints_.find(positionHash);
        if (found == hints_.end() || plyToMate < found->second.plyToMate) {
            hints_[positionHash] = {action, plyToMate};
        }
    }

    std::optional<MateCandidateHint> lookup(uint64_t positionHash) const {
        std::shared_lock lock(mutex_);
        const auto found = hints_.find(positionHash);
        return found == hints_.end()
            ? std::nullopt
            : std::optional<MateCandidateHint>(found->second);
    }

    void clear() {
        std::unique_lock lock(mutex_);
        hints_.clear();
    }

private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<uint64_t, MateCandidateHint> hints_;
};

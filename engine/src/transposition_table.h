#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <unordered_map>

class Node;

/**
 * @brief Thread-safe transposition table for Monte Carlo Graph Search (MCGS).
 *
 * Maps board hash keys to shared Node pointers, enabling:
 * - Faster convergence by aggregating statistics across transpositions
 * - Memory efficiency by avoiding duplicate nodes for the same position
 * - Better value estimates from increased effective sample size
 */
class TranspositionTable {
public:
    static constexpr size_t kDefaultMaxCapacity = 1000000;

    TranspositionTable() = default;
    ~TranspositionTable() = default;

    // Non-copyable, non-movable (table should be unique per search)
    TranspositionTable(const TranspositionTable&) = delete;
    TranspositionTable& operator=(const TranspositionTable&) = delete;
    TranspositionTable(TranspositionTable&&) = delete;
    TranspositionTable& operator=(TranspositionTable&&) = delete;

private:
    std::unordered_map<uint64_t, std::shared_ptr<Node>> table_;
    mutable std::shared_mutex mutex_;

    mutable std::atomic<size_t> hits_{0};
    mutable std::atomic<size_t> misses_{0};
    std::atomic<size_t> insertions_{0};
    mutable std::atomic<size_t> rejections_{0};

    size_t maxCapacity_{kDefaultMaxCapacity};

public:
    /**
     * @brief Lookup a node by position hash.
     * @param hash The Zobrist hash of the position
     * @return Shared pointer to the node, or nullptr if not found
     */
    std::shared_ptr<Node> lookup(uint64_t hash) const {
        std::shared_lock lock(mutex_);
        auto it = table_.find(hash);
        if (it != table_.end()) {
            hits_.fetch_add(1, std::memory_order_relaxed);
            return it->second;
        }
        misses_.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }

    /**
     * @brief Insert or retrieve a node for a position hash.
     *
     * If the hash exists, returns the existing node.
     * If at max capacity, returns the passed node without storing.
     * Otherwise, inserts and returns the new node.
     *
     * @param hash The Zobrist hash of the position
     * @param node The node to insert if hash not present
     * @return The node associated with the hash
     */
    std::shared_ptr<Node> insertOrGet(uint64_t hash, std::shared_ptr<Node> node) {
        std::unique_lock lock(mutex_);

        auto it = table_.find(hash);
        if (it != table_.end()) {
            hits_.fetch_add(1, std::memory_order_relaxed);
            return it->second;
        }

        if (table_.size() >= maxCapacity_) {
            rejections_.fetch_add(1, std::memory_order_relaxed);
            return node;
        }

        table_.emplace(hash, node);
        insertions_.fetch_add(1, std::memory_order_relaxed);
        return node;
    }

    /**
     * @brief Check if a position exists in the table.
     * @param hash The Zobrist hash of the position
     * @return true if position is in table
     */
    bool contains(uint64_t hash) const {
        std::shared_lock lock(mutex_);
        return table_.find(hash) != table_.end();
    }

    /**
     * @brief Clear all entries and reset statistics.
     */
    void clear() {
        std::unique_lock lock(mutex_);
        table_.clear();
        hits_.store(0, std::memory_order_relaxed);
        misses_.store(0, std::memory_order_relaxed);
        insertions_.store(0, std::memory_order_relaxed);
        rejections_.store(0, std::memory_order_relaxed);
    }

    /**
     * @brief Get the number of entries in the table.
     */
    size_t size() const {
        std::shared_lock lock(mutex_);
        return table_.size();
    }

    /**
     * @brief Reserve capacity for expected number of positions.
     * @param capacity Expected number of unique positions
     */
    void reserve(size_t capacity) {
        std::unique_lock lock(mutex_);
        table_.reserve(std::min(capacity, maxCapacity_));
    }

    /**
     * @brief Set maximum capacity (hard limit to prevent OOM).
     * @param capacity Maximum number of entries allowed
     */
    void setMaxCapacity(size_t capacity) {
        std::unique_lock lock(mutex_);
        maxCapacity_ = capacity;
    }

    // Statistics accessors
    size_t getHits() const { return hits_.load(std::memory_order_relaxed); }
    size_t getMisses() const { return misses_.load(std::memory_order_relaxed); }
    size_t getInsertions() const { return insertions_.load(std::memory_order_relaxed); }
    size_t getRejections() const { return rejections_.load(std::memory_order_relaxed); }
    size_t getMaxCapacity() const { return maxCapacity_; }

    bool isFull() const {
        std::shared_lock lock(mutex_);
        return table_.size() >= maxCapacity_;
    }

    /**
     * @brief Get hit rate as a percentage.
     */
    float getHitRate() const {
        size_t h = hits_.load(std::memory_order_relaxed);
        size_t m = misses_.load(std::memory_order_relaxed);
        size_t total = h + m;
        return total > 0 ? (100.0f * static_cast<float>(h) / static_cast<float>(total)) : 0.0f;
    }

    /**
     * @brief Get fullness as permille (0-1000) like UCI hashfull.
     */
    int getFullness() const {
        std::shared_lock lock(mutex_);
        if (maxCapacity_ == 0) return 0;
        return static_cast<int>((table_.size() * 1000) / maxCapacity_);
    }
};

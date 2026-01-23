#pragma once

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <atomic>

// Forward declaration
class Node;

/**
 * @file transposition_table.h
 * @brief Thread-safe transposition table for Monte Carlo Graph Search (MCGS).
 * 
 * In MCGS, positions reached through different move sequences share the same node.
 * This transposition table maps board hash keys to shared Node pointers, enabling:
 * - Faster convergence by aggregating statistics across transpositions
 * - Memory efficiency by avoiding duplicate nodes for the same position
 * - Better value estimates from increased effective sample size
 */

class TranspositionTable {
private:
    // Thread-safe hash map: position hash -> Node
    std::unordered_map<uint64_t, std::shared_ptr<Node>> table;
    
    // Reader-writer lock for concurrent access
    mutable std::shared_mutex rwMutex;
    
    // Statistics
    mutable std::atomic<size_t> hits{0};
    mutable std::atomic<size_t> misses{0};
    std::atomic<size_t> insertions{0};
    
    // Capacity for fullness calculation
    size_t reservedCapacity{100000};

public:
    TranspositionTable() = default;
    
    // Prevent copying (table should be unique per search)
    TranspositionTable(const TranspositionTable&) = delete;
    TranspositionTable& operator=(const TranspositionTable&) = delete;
    
    /**
     * @brief Lookup a node by position hash.
     * 
     * Thread-safe read operation. Multiple threads can lookup concurrently.
     * @param hash The Zobrist hash of the position
     * @return Shared pointer to the node, or nullptr if not found
     */
    std::shared_ptr<Node> lookup(uint64_t hash) const {
        std::shared_lock<std::shared_mutex> lock(rwMutex);
        auto it = table.find(hash);
        if (it != table.end()) {
            hits.fetch_add(1, std::memory_order_relaxed);
            return it->second;
        }
        misses.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    
    /**
     * @brief Insert or retrieve a node for a position hash.
     * 
     * If the hash already exists, returns the existing node (no insertion).
     * Otherwise, inserts the new node and returns it.
     * Thread-safe with exclusive lock on writes.
     * 
     * @param hash The Zobrist hash of the position
     * @param node The node to insert if hash not present
     * @return The node now associated with the hash (may be existing or new)
     */
    std::shared_ptr<Node> insertOrGet(uint64_t hash, std::shared_ptr<Node> node) {
        std::unique_lock<std::shared_mutex> lock(rwMutex);
        auto [it, inserted] = table.try_emplace(hash, node);
        if (inserted) {
            insertions.fetch_add(1, std::memory_order_relaxed);
        } else {
            hits.fetch_add(1, std::memory_order_relaxed);
        }
        return it->second;
    }
    
    /**
     * @brief Check if a position exists in the table.
     * 
     * @param hash The Zobrist hash of the position
     * @return true if position is in table
     */
    bool contains(uint64_t hash) const {
        std::shared_lock<std::shared_mutex> lock(rwMutex);
        return table.find(hash) != table.end();
    }
    
    /**
     * @brief Clear all entries from the table.
     * 
     * Called at the start of a new search.
     */
    void clear() {
        std::unique_lock<std::shared_mutex> lock(rwMutex);
        table.clear();
        hits.store(0, std::memory_order_relaxed);
        misses.store(0, std::memory_order_relaxed);
        insertions.store(0, std::memory_order_relaxed);
    }
    
    /**
     * @brief Get the number of entries in the table.
     */
    size_t size() const {
        std::shared_lock<std::shared_mutex> lock(rwMutex);
        return table.size();
    }
    
    /**
     * @brief Get transposition hit count.
     */
    size_t getHits() const {
        return hits.load(std::memory_order_relaxed);
    }
    
    /**
     * @brief Get transposition miss count.
     */
    size_t getMisses() const {
        return misses.load(std::memory_order_relaxed);
    }
    
    /**
     * @brief Get insertion count.
     */
    size_t getInsertions() const {
        return insertions.load(std::memory_order_relaxed);
    }
    
    /**
     * @brief Get hit rate as a percentage.
     */
    float getHitRate() const {
        size_t h = hits.load(std::memory_order_relaxed);
        size_t m = misses.load(std::memory_order_relaxed);
        size_t total = h + m;
        return total > 0 ? (100.0f * h / total) : 0.0f;
    }
    
    /**
     * @brief Reserve capacity for expected number of positions.
     * 
     * Call before search to reduce rehashing overhead.
     * @param capacity Expected number of unique positions
     */
    void reserve(size_t capacity) {
        std::unique_lock<std::shared_mutex> lock(rwMutex);
        reservedCapacity = capacity;
        table.reserve(capacity);
    }
    
    /**
     * @brief Get fullness as permille (0-1000) like UCI hashfull.
     */
    int getFullness() const {
        std::shared_lock<std::shared_mutex> lock(rwMutex);
        if (reservedCapacity == 0) return 0;
        size_t currentSize = table.size();
        return static_cast<int>((currentSize * 1000) / reservedCapacity);
    }
};

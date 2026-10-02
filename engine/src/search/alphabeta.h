#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "environment/board.h"
#include "nnue/network.h"
#include "Fairy-Stockfish/src/types.h"

/**
 * @brief Alpha-beta search over bughouse joint actions with an NNUE evaluation.
 *
 * A team's turn is one joint action: a move or a pass on each board. The
 * search expands it as two nested choices made by the same player, first on
 * board A and then on board B, so the minimax value is unchanged while move
 * ordering, reductions and pruning apply to each board's choice separately.
 * Board B's options come from the position before board A's move, matching
 * the simultaneous joint-action semantics used everywhere else in the engine.
 *
 * Scores are from the point of view of the team to play, in units of
 * EVAL_SCALE per network logit (tanh(score / EVAL_SCALE) ~ teacher value).
 */
namespace ab {

constexpr int SCORE_INF = 32000;
constexpr int SCORE_MATE = 31000;
constexpr int MAX_PLY = 128;
constexpr int SCORE_MATE_BOUND = SCORE_MATE - MAX_PLY;
constexpr float EVAL_SCALE = 400.0f;

struct JointMove {
    Stockfish::Move a = Stockfish::MOVE_NONE;
    Stockfish::Move b = Stockfish::MOVE_NONE;
    bool operator==(const JointMove&) const = default;
};

struct Limits {
    int depth = MAX_PLY - 1;
    uint64_t nodes = 0;      // 0 = unlimited
    int moveTimeMs = 0;      // 0 = unlimited
};

/// Search features and parameters, settable at runtime for A/B matches.
struct Options {
    bool checkExtension = false;    // extend a team turn spent in check (costs over a ply of depth)
    bool quietChecksInQsearch = true;  // first qsearch turn also tries checks
    // Pairs are searched while (rankA * rankB) <= lmpBase + lmpScale * depth^2,
    // times pvLmpScale at PV nodes and rootLmpScale at the root.
    int lmpBase = 4;
    int lmpScale = 3;
    int pvLmpScale = 3;
    int rootLmpScale = 6;
    double lmrDivisor = 2.0;
    int qsearchPlies = 8;
    int rfpMargin = 150;       // per remaining depth
    int futilityMargin = 200;  // per remaining depth
    int threads = 1;           // Lazy SMP search threads sharing the table
    bool pairOrdering = false;
    // Run the Fairy-Stockfish root mate probe beside the search when the team
    // holds the time advantage, as the MCTS search does.
    bool rootMateProbe = true;  // search pairs by rank product, not board A first (neutral in A/B)

    /// Sets one option from "name=value"; returns false for an unknown name.
    bool set(const std::string& assignment);
};

struct Result {
    JointMove best;
    bool hasMove = false;
    int score = 0;
    int depth = 0;
    uint64_t nodes = 0;
    std::vector<JointMove> pv;
};

struct IterationInfo {
    int depth;
    int selDepth;
    int score;
    uint64_t nodes;
    int64_t elapsedMs;
    const std::vector<JointMove>& pv;
};

class Searcher {
public:
    explicit Searcher(const nnue::Network& network, size_t hashMb = 64);
    ~Searcher();

    Result search(Board& board, Stockfish::Color team, bool teamHasTimeAdvantage,
                  const Limits& limits, const std::atomic<bool>* stop = nullptr,
                  const std::function<void(const IterationInfo&)>& onIteration = {});

    void clear();
    void resize(size_t hashMb);
    Options options;

    struct Stats {
        uint64_t mainNodes = 0, qNodes = 0, evasionNodes = 0, ttCuts = 0, rfpCuts = 0;
        uint64_t mainPairs = 0, evasionPairs = 0, qCaptures = 0, legalPairs = 0;
    };
    Stats stats;

    /// Static evaluation of `board` for `team`, in search units.
    int static_eval(Board& board, Stockfish::Color team, bool teamHasTimeAdvantage) const;

private:
    struct Table;
    struct BoardOptions;
    struct PlyOptions;
    struct PairCandidate;

    int negamax(int depth, int ply, int alpha, int beta, bool pvNode,
                Stockfish::Color team);
    int qsearch(int ply, int alpha, int beta, Stockfish::Color team);
    int qsearch_evasions(int ply, int alpha, int beta, Stockfish::Color team);

    bool team_advantage(Stockfish::Color team) const;
    int evaluate(int ply, Stockfish::Color team);
    void ensure_accumulator(int ply);
    bool in_check(Stockfish::Color team);
    int terminal_score(int ply, Stockfish::Color team, bool& terminal);
    uint64_t key(Stockfish::Color team) const;

    void generate(int boardNumber, bool onTurn, int ply, bool hasTTMove,
                  Stockfish::Move ttMove, BoardOptions& options);
    void play(int ply, const JointMove& move);
    void unplay(const JointMove& move);
    bool should_stop();

    struct TTData {
        bool hit = false;
        JointMove move;
        int score = 0;
        int depth = 0;
        int bound = 0;
    };
    TTData probe(uint64_t key) const;
    void store(uint64_t key, int depth, int score, int bound, const JointMove& move, int ply);

    Searcher(const nnue::Network& network, std::shared_ptr<Table> table);
    Result iterate(Board& board, Stockfish::Color team, bool teamHasTimeAdvantage,
                   const Limits& limits, const std::atomic<bool>* stop,
                   const std::function<void(const IterationInfo&)>& onIteration, int helperIndex);
    std::vector<Searcher*> all_searchers();

    const nnue::Network& network_;
    std::shared_ptr<Table> table_;
    std::vector<std::unique_ptr<Searcher>> helpers_;

    // Per-search state.
    Board* board_ = nullptr;
    Stockfish::Color rootTeam_ = Stockfish::WHITE;
    bool rootAdvantage_ = false;
    bool whiteAdvantage_ = false;
    std::array<int, 2> boardPlies_{};
    std::unique_ptr<nnue::Accumulator[]> accumulators_;
    std::array<bool, MAX_PLY + 2> accumulatorReady_{};
    std::unique_ptr<PlyOptions[]> options_;
    std::atomic<bool> mateStop_{false};
    int qsearchCheckDepth_ = 0;
    int rootDepth_ = 1;
    int qsearchPlyLimit_ = MAX_PLY;
    int qsearchEntryPly_ = 0;
    std::array<std::array<JointMove, MAX_PLY + 1>, MAX_PLY + 1> pv_{};
    std::array<int, MAX_PLY + 1> pvLength_{};
    std::array<std::array<std::array<Stockfish::Move, 2>, 2>, MAX_PLY + 1> killers_{};
    // [board][colour][from or 64 + dropped type or 71 for a pass][to]
    std::vector<int32_t> history_;
    uint64_t nodes_ = 0;
    uint64_t nodeLimit_ = 0;
    int selDepth_ = 0;
    bool stopped_ = false;
    const std::atomic<bool>* stopFlag_ = nullptr;
    std::chrono::steady_clock::time_point start_;
    std::chrono::steady_clock::time_point deadline_;
    bool hasDeadline_ = false;
};

/// "(e2e4,pass)" in the engine's joint-move notation.
std::string format_joint_move(Board& board, const JointMove& move);

}  // namespace ab

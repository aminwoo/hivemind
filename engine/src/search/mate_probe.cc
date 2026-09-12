#include "search/mate_probe.h"

#include <algorithm>
#include <chrono>
#include <deque>
#include <mutex>
#include <string>
#include <thread>

#include "common/globals.h"

#include "Fairy-Stockfish/src/movegen.h"
#include "Fairy-Stockfish/src/position.h"
#include "Fairy-Stockfish/src/search.h"
#include "Fairy-Stockfish/src/thread.h"
#include "Fairy-Stockfish/src/tt.h"
#include "Fairy-Stockfish/src/uci.h"
#include "Fairy-Stockfish/src/variant.h"

namespace MateProbe {

namespace {

// Search::Limits, ThreadPool::stop and the transposition table are process
// globals, so a second probe would corrupt the one already running.
std::mutex probeMutex;

// Index of the pool thread the probe drives. Thread::search() prints its pv
// and honours Limits.depth only on the main thread, so a worker searches
// silently and is stopped from here instead.
constexpr size_t PROBE_THREAD_INDEX = 1;

// Roughly what the probe searches in a millisecond, so a budget below this is
// polled by yielding instead of sleeping.
constexpr uint64_t NODES_PER_SLEEP = 4000;

// The bughouse variant, searched as one board on its own.
//
// Fairy-Stockfish's twoBoards flag is what lets a side drop a piece it does
// not hold (Position::allow_virtual_drop), on the assumption its partner will
// capture one in time. The probe exists for the position where the partner
// board sits while this one mates, and a partner who sits feeds nobody - so
// the flag comes off, and both sides play with the hands they actually have.
// That also retires the virtual-mate score band the flag switches on, which
// the probe would otherwise have to tell apart from a real mate.
//
// Promotions are narrowed to queen and knight. The policy head only knows
// those two, so a mate the probe can only reach by underpromoting to a rook
// or bishop is one the engine could never play or train on; searching it
// would just hand back a move with no policy index.
const Stockfish::Variant* bughouse_variant() {
    static const Stockfish::Variant* probeVariant = [] {
        const auto entry = Stockfish::variants.find("bughouse");
        if (entry == Stockfish::variants.end()) {
            return static_cast<const Stockfish::Variant*>(nullptr);
        }
        auto* narrowed = new Stockfish::Variant(*entry->second);
        narrowed->twoBoards = false;
        narrowed->promotionPieceTypes = {Stockfish::QUEEN, Stockfish::KNIGHT};
        return static_cast<const Stockfish::Variant*>(narrowed->conclude());
    }();
    return probeVariant;
}

}  // namespace

Result probe(const std::string& fen, int maxMateMoves, uint64_t nodeBudget,
             int budgetMs, const std::function<bool()>& abort) {
    Result result;
    if (maxMateMoves <= 0 || budgetMs < 0 || nodeBudget == 0) {
        return result;
    }

    init_fairy_stockfish();
    const Stockfish::Variant* variant = bughouse_variant();
    if (variant == nullptr || Stockfish::Threads.size() <= PROBE_THREAD_INDEX) {
        return result;
    }

    std::lock_guard<std::mutex> guard(probeMutex);
    Stockfish::Thread* worker = Stockfish::Threads[PROBE_THREAD_INDEX];
    worker->wait_for_search_finished();

    Stockfish::Position& position = worker->rootPos;
    position.set(variant, fen, false, &worker->rootState, worker);

    Stockfish::Search::RootMoves rootMoves;
    for (const auto& move : Stockfish::MoveList<Stockfish::LEGAL>(position)) {
        rootMoves.emplace_back(move);
    }
    if (rootMoves.empty()) {
        return result;
    }

    // Bughouse gives checkmate and stalemate the same terminal score. Keep an
    // immediate checkmate ahead of an immediate stalemate when their scores
    // tie, matching Hivemind's preference for the explicit mate.
    std::stable_partition(
        rootMoves.begin(), rootMoves.end(), [&](const auto& rootMove) {
            const Stockfish::Move move = rootMove.pv.front();
            if (!position.gives_check(move)) {
                return false;
            }
            Stockfish::StateInfo nextState;
            position.do_move(move, nextState);
            const bool checkmate = position.checkers()
                && Stockfish::MoveList<Stockfish::LEGAL>(position).size() == 0;
            position.undo_move(move);
            return checkmate;
        });

    // Thread::search() ends an iteration as soon as it holds a mate inside
    // 2 * Limits.mate plies, so this is what stops the probe on the first mate
    // worth having rather than leaving it to hunt for a shorter one.
    Stockfish::Search::LimitsType limits;
    limits.startTime = Stockfish::now();
    limits.mate = maxMateMoves;
    Stockfish::Search::Limits = limits;

    worker->nodes = worker->tbHits = worker->bestMoveChanges = 0;
    worker->nmpMinPly = 0;
    worker->rootDepth = worker->completedDepth = 0;
    worker->rootMoves = rootMoves;

    Stockfish::Threads.stop = false;
    Stockfish::Threads.abort = false;
    Stockfish::Threads.increaseDepth = true;
    Stockfish::TT.new_search();

    const auto deadline = budgetMs > 0
        ? std::chrono::steady_clock::now() + std::chrono::milliseconds(budgetMs)
        : std::chrono::steady_clock::time_point::max();
    worker->start_searching();
    while (!Stockfish::Threads.stop) {
        // A helper thread never raises Threads.stop itself: it returns to the
        // idle loop once rootDepth hits MAX_PLY, which a tiny tree reaches
        // well inside the node budget. Without this check the poll spins
        // forever when no deadline or abort is armed, as under self-play's
        // complete-probe mode.
        if (!worker->is_searching()) {
            break;
        }
        if (worker->nodes.load(std::memory_order_relaxed) >= nodeBudget
            || std::chrono::steady_clock::now() >= deadline
            || (abort && abort())) {
            Stockfish::Threads.stop = true;
            break;
        }
        // Small node budgets are spent in well under a millisecond, so yield
        // rather than sleep and keep the overshoot to the scheduler's mercy.
        if (nodeBudget <= NODES_PER_SLEEP) {
            std::this_thread::yield();
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    worker->wait_for_search_finished();

    result.depth = worker->completedDepth;
    result.nodes = worker->nodes.load(std::memory_order_relaxed);

    const Stockfish::Search::RootMove& best = worker->rootMoves.front();
    // An iteration cut off before its first move completes leaves the current
    // score at -VALUE_INFINITE, so fall back to the last completed one.
    const Stockfish::Value score = best.score == -Stockfish::VALUE_INFINITE
        ? best.previousScore : best.score;
    if (score < Stockfish::VALUE_MATE_IN_MAX_PLY || best.pv.empty()) {
        return result;
    }

    const int mateInMoves = (Stockfish::VALUE_MATE - score + 1) / 2;
    // Reached the deadline still holding only a long mate: the search never
    // stopped for it, and it is too thin a claim to hand back.
    if (mateInMoves > maxMateMoves) {
        return result;
    }

    result.found = true;
    result.mateInMoves = mateInMoves;
    result.bestMove = best.pv.front();

    // Format on a scratch position rather than the worker's, and walk it move
    // by move so a drop or a castle is named against the position it is played
    // in.
    Stockfish::Position line;
    std::deque<Stockfish::StateInfo> lineStates(1);
    line.set(variant, fen, false, &lineStates.back(), Stockfish::Threads.main());
    for (const Stockfish::Move move : best.pv) {
        if (!line.pseudo_legal(move) || !line.legal(move)) {
            break;
        }
        result.principalVariation.push_back(Stockfish::UCI::move(line, move));
        lineStates.emplace_back();
        line.do_move(move, lineStates.back());
    }
    return result;
}

}  // namespace MateProbe

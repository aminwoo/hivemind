#include "search/mate_probe.h"

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

const Stockfish::Variant* bughouse_variant() {
    const auto entry = Stockfish::variants.find("bughouse");
    return entry == Stockfish::variants.end() ? nullptr : entry->second;
}

}  // namespace

Result probe(const std::string& fen, int maxMateMoves, int budgetMs,
             const std::function<bool()>& abort) {
    Result result;
    if (maxMateMoves <= 0 || budgetMs <= 0) {
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

    // A virtual drop spends a piece the partner has not handed over yet. The
    // defender is allowed one - that is what keeps the model conservative - but
    // the attacker must mate with what it actually holds.
    Stockfish::Search::RootMoves rootMoves;
    for (const auto& move : Stockfish::MoveList<Stockfish::LEGAL>(position)) {
        if (!position.virtual_drop(move)) {
            rootMoves.emplace_back(move);
        }
    }
    if (rootMoves.empty()) {
        return result;
    }

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

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(budgetMs);
    worker->start_searching();
    while (!Stockfish::Threads.stop) {
        if (std::chrono::steady_clock::now() >= deadline
            || (abort && abort())) {
            Stockfish::Threads.stop = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    worker->wait_for_search_finished();

    const Stockfish::Search::RootMove& best = worker->rootMoves.front();
    // An iteration cut off before its first move completes leaves the current
    // score at -VALUE_INFINITE, so fall back to the last completed one.
    const Stockfish::Value score = best.score == -Stockfish::VALUE_INFINITE
        ? best.previousScore : best.score;
    if (score < Stockfish::VALUE_MATE_IN_MAX_PLY || best.pv.empty()) {
        return result;
    }

    result.found = true;
    result.mateInMoves = (Stockfish::VALUE_MATE - score + 1) / 2;
    result.bestMove = best.pv.front();

    // Format on a scratch position rather than the worker's, and walk it move
    // by move so a drop or a castle is named against the position it is played
    // in. Board cannot replay the line, so this is the only faithful rendering.
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

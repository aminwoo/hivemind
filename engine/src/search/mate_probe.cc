#include "search/mate_probe.h"

#include <algorithm>
#include <chrono>
#include <deque>
#include <exception>
#include <mutex>
#include <string>
#include <thread>

#include "common/globals.h"
#include "common/thread_priority.h"

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
// globals, so a second probe would corrupt the one already running. Resizing
// the pool takes it too, so workers are never added or retired mid-probe.
std::mutex probeMutex;

// Index of the first pool thread the probe drives; every thread from here to
// the end of the pool is a probe worker. Thread::search() prints its pv and
// honours Limits.depth only on the main thread, so the workers search
// silently and are stopped from here instead.
constexpr size_t FIRST_PROBE_THREAD_INDEX = 1;

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
    if (variant == nullptr) {
        return result;
    }

    std::lock_guard<std::mutex> guard(probeMutex);
    if (Stockfish::Threads.size() <= FIRST_PROBE_THREAD_INDEX) {
        return result;
    }
    const std::vector<Stockfish::Thread*> workers(
        Stockfish::Threads.begin() + FIRST_PROBE_THREAD_INDEX,
        Stockfish::Threads.end());
    for (Stockfish::Thread* worker : workers) {
        worker->wait_for_search_finished();
    }

    Stockfish::Thread* const firstWorker = workers.front();
    Stockfish::Position& position = firstWorker->rootPos;
    position.set(variant, fen, false, &firstWorker->rootState, firstWorker);

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

    // Every worker searches the same root as lazy SMP does: they share the
    // transposition table and diverge only through what each finds there.
    for (Stockfish::Thread* worker : workers) {
        if (worker != firstWorker) {
            worker->rootPos.set(variant, fen, false, &worker->rootState, worker);
        }
        worker->nodes = worker->tbHits = worker->bestMoveChanges = 0;
        worker->nmpMinPly = 0;
        worker->rootDepth = worker->completedDepth = 0;
        worker->rootMoves = rootMoves;
    }
    const auto nodes_searched = [&] {
        uint64_t nodes = 0;
        for (Stockfish::Thread* worker : workers) {
            nodes += worker->nodes.load(std::memory_order_relaxed);
        }
        return nodes;
    };

    Stockfish::Threads.stop = false;
    Stockfish::Threads.abort = false;
    Stockfish::Threads.increaseDepth = true;
    Stockfish::TT.new_search();

    const auto deadline = budgetMs > 0
        ? std::chrono::steady_clock::now() + std::chrono::milliseconds(budgetMs)
        : std::chrono::steady_clock::time_point::max();
    for (Stockfish::Thread* worker : workers) {
        worker->start_searching();
    }
    while (!Stockfish::Threads.stop) {
        // A helper thread never raises Threads.stop itself: it returns to the
        // idle loop once rootDepth hits MAX_PLY, which a tiny tree reaches
        // well inside the node budget. Without this check the poll spins
        // forever when no deadline or abort is armed, as under self-play's
        // complete-probe mode. The first worker back has searched as deep as
        // there is to go, so the rest are stopped with it.
        const bool workerReturned = std::any_of(
            workers.begin(), workers.end(),
            [](Stockfish::Thread* worker) { return !worker->is_searching(); });
        if (workerReturned
            || nodes_searched() >= nodeBudget
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
    for (Stockfish::Thread* worker : workers) {
        worker->wait_for_search_finished();
    }

    result.nodes = nodes_searched();

    // An iteration cut off before its first move completes leaves the current
    // score at -VALUE_INFINITE, so fall back to the last completed one.
    const auto settled_score = [](const Stockfish::Search::RootMove& rootMove) {
        return rootMove.score == -Stockfish::VALUE_INFINITE
            ? rootMove.previousScore : rootMove.score;
    };
    // The shortest mate any worker holds is the answer, the deeper search
    // breaking a tie and the lower index any tie left after that.
    Stockfish::Thread* bestWorker = nullptr;
    for (Stockfish::Thread* worker : workers) {
        result.depth = std::max<int>(result.depth, worker->completedDepth);
        const Stockfish::Search::RootMove& candidate = worker->rootMoves.front();
        if (candidate.pv.empty()) {
            continue;
        }
        if (bestWorker == nullptr) {
            bestWorker = worker;
            continue;
        }
        const Stockfish::Value candidateScore = settled_score(candidate);
        const Stockfish::Value bestScore =
            settled_score(bestWorker->rootMoves.front());
        if (candidateScore > bestScore
            || (candidateScore == bestScore
                && worker->completedDepth > bestWorker->completedDepth)) {
            bestWorker = worker;
        }
    }
    if (bestWorker == nullptr) {
        return result;
    }

    const Stockfish::Search::RootMove& best = bestWorker->rootMoves.front();
    const Stockfish::Value score = settled_score(best);
    if (score < Stockfish::VALUE_MATE_IN_MAX_PLY) {
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
    bool wholeLinePlayed = true;
    for (const Stockfish::Move move : best.pv) {
        if (!line.pseudo_legal(move) || !line.legal(move)) {
            wholeLinePlayed = false;
            break;
        }
        result.principalVariation.push_back(Stockfish::UCI::move(line, move));
        lineStates.emplace_back();
        line.do_move(move, lineStates.back());
    }

    // The search's line ends where its depth ran out: the mate score beyond
    // that came out of the table, by way of quiescence or mate-distance
    // pruning, without the moves. The table still holds each position's best
    // move from the deeper searches that scored it, so the line is carried on
    // from there, as far as the mate it claims, while the entries last. The
    // workers have stopped and the probe lock is held, so nothing writes the
    // table meanwhile.
    const Stockfish::Color attacker = line.side_to_move();
    // The defender's replies all lose, so the search keeps no best move for
    // a defender position. Its reply is read from the table one ply down
    // instead: the one whose position the table scores as the longest mate,
    // the defence the search would have found. Every child sits at the same
    // ply, so their stored mate scores compare directly.
    const auto longest_defence = [&] {
        Stockfish::Move longest = Stockfish::MOVE_NONE;
        int longestValue = Stockfish::VALUE_INFINITE;
        Stockfish::StateInfo replyState;
        for (const auto& reply : Stockfish::MoveList<Stockfish::LEGAL>(line)) {
            line.do_move(reply, replyState);
            bool found = false;
            const Stockfish::TTEntry* entry =
                Stockfish::TT.probe(line.key(), found);
            const int value = found ? entry->value() : Stockfish::VALUE_NONE;
            line.undo_move(reply);
            if (found && entry->move() != Stockfish::MOVE_NONE
                && value != Stockfish::VALUE_NONE
                && value >= Stockfish::VALUE_MATE_IN_MAX_PLY
                && value < longestValue) {
                longest = reply;
                longestValue = value;
            }
        }
        return longest;
    };
    const size_t matePlies = static_cast<size_t>(2 * mateInMoves - 1);
    while (wholeLinePlayed && result.principalVariation.size() < matePlies
           && Stockfish::MoveList<Stockfish::LEGAL>(line).size() > 0) {
        bool found = false;
        const Stockfish::TTEntry* entry = Stockfish::TT.probe(line.key(), found);
        Stockfish::Move move = found ? entry->move() : Stockfish::MOVE_NONE;
        if (move == Stockfish::MOVE_NONE && line.side_to_move() != attacker) {
            move = longest_defence();
        }
        if (move == Stockfish::MOVE_NONE || !line.pseudo_legal(move)
            || !line.legal(move)) {
            break;
        }
        result.principalVariation.push_back(Stockfish::UCI::move(line, move));
        ++result.tablePlies;
        lineStates.emplace_back();
        line.do_move(move, lineStates.back());
    }
    return result;
}

void set_threads(int count) {
    const size_t poolSize = FIRST_PROBE_THREAD_INDEX
        + static_cast<size_t>(std::clamp(count, 1, MAX_THREADS));
    init_fairy_stockfish();
    std::lock_guard<std::mutex> guard(probeMutex);
    // ThreadPool::set() would rebuild the main thread too, and every Board
    // position holds that one as its thread, so only the workers behind it are
    // started or retired here. init_fairy_stockfish() reserved room for the
    // largest pool, so this never reallocates under a Threads.main() reader.
    while (Stockfish::Threads.size() > poolSize) {
        delete Stockfish::Threads.back();
        Stockfish::Threads.pop_back();
    }
    grow_pool(poolSize);
}

void grow_pool(size_t poolSize) {
    const auto start_workers = [poolSize] {
        while (Stockfish::Threads.size() < poolSize) {
            auto* worker = new Stockfish::Thread(Stockfish::Threads.size());
            worker->clear();
            Stockfish::Threads.push_back(worker);
        }
    };
#if defined(__linux__) && !defined(__EMSCRIPTEN__)
    // A thread starts with its creator's priority, so the workers are
    // created from one that has lowered its own. A failure there is the
    // caller's, as it was when the caller created them.
    std::exception_ptr failure;
    std::thread([&] {
        try {
            lower_mate_search_thread_priority();
            start_workers();
        } catch (...) {
            failure = std::current_exception();
        }
    }).join();
    if (failure) {
        std::rethrow_exception(failure);
    }
#else
    start_workers();
#endif
}

int threads() {
    init_fairy_stockfish();
    std::lock_guard<std::mutex> guard(probeMutex);
    return static_cast<int>(
        Stockfish::Threads.size() - std::min<size_t>(
            Stockfish::Threads.size(), FIRST_PROBE_THREAD_INDEX));
}

}  // namespace MateProbe

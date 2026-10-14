#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "Fairy-Stockfish/src/types.h"

/**
 * @brief Bounded Fairy-Stockfish mate search for a single bughouse board.
 *
 * The hand-rolled root scanner in agent.cc only ever tries attacker moves that
 * give check, so a mate that needs a quiet preparing move is invisible to it no
 * matter how large its budget. Fairy-Stockfish's alpha-beta search has no such
 * restriction.
 *
 * The board is searched on its own: neither side may drop a piece it does not
 * hold, so this search assumes the partner board sits for the whole line and
 * feeds nobody. A team ahead on time can keep its own partner sitting; the
 * caller must also reject lines where the defending partner can use a capture
 * to pass and change the searched hand.
 *
 * The result is a pruned alpha-beta score rather than an exhaustive proof, so
 * callers should treat a hit as a strong mate candidate for a single board,
 * not as a certificate.
 */
namespace MateProbe {

/** Outcome of one probe, in the coordinates of the board that was probed. */
struct Result {
    bool found = false;
    /**
     * Mate distance in attacker moves, as UCI reports it. It comes from the
     * search's mate score, so it is the distance the search settled on rather
     * than a proven optimum, and mate-distance pruning can leave it one move
     * long.
     */
    int mateInMoves = 0;
    /** First move of the mate. Always a move the attacker can actually play. */
    Stockfish::Move bestMove = Stockfish::MOVE_NONE;
    /** What the probe got through, mate or not. Reported even when found is false. */
    int depth = 0;
    uint64_t nodes = 0;
    /**
     * The whole line in UCI. Every ply is a move of the probed board alone,
     * so Board can replay it. The search's own line stops where its depth
     * ran out and the mate score came from the table; the rest is read from
     * the table's best moves, and a line can still end short of the mate
     * where an entry was overwritten.
     */
    std::vector<std::string> principalVariation;
    /** How many plies of principalVariation were read from the table. */
    int tablePlies = 0;
};

/**
 * @brief Search one board for a forced mate for the side to move.
 *
 * The search stops at the first mate it proves within maxMateMoves rather than
 * spending the rest of the window shortening it: the caller only needs to know
 * the position is won. A mate longer than that is not reported at all - the
 * bound is what makes a pruned search's mate claim worth acting on.
 *
 * @param fen Bughouse FEN of the board to search.
 * @param maxMateMoves Longest mate to stop for, and to accept.
 * @param nodeBudget Node ceiling, so callers that meter their own search in
 *        nodes can bill this one in the same currency; `nodes` in the result
 *        reports what it actually spent. Enforced by polling, so it overshoots
 *        by up to a poll interval.
 * @param budgetMs Wall-clock ceiling, a backstop for the node cap. Zero uses
 *        only the node budget.
 * @param abort Polled about once a millisecond; the probe returns early when it
 *        answers true. The caller's own search runs while this one does, so
 *        without it the probe holds the whole budget after that search has
 *        already settled the position.
 *
 * Probes are serialized: Fairy-Stockfish keeps its limits, stop flag and
 * transposition table in globals, so only one can be in flight at a time.
 * Each one searches with every worker set_threads() asked for, sharing one
 * transposition table, and nodeBudget is their combined total.

 */
Result probe(const std::string& fen, int maxMateMoves, uint64_t nodeBudget,
             int budgetMs, const std::function<bool()>& abort = {});

/** Workers each probe searches with until set_threads() says otherwise. */
constexpr int DEFAULT_THREADS = 4;

/** Most workers set_threads() will start. */
constexpr int MAX_THREADS = 256;

/**
 * @brief Set how many Fairy-Stockfish workers each probe searches with.
 *
 * Clamped to [1, MAX_THREADS]. Waits for a probe in flight to finish, then
 * starts or retires workers so the next probe gets the new count.
 */
void set_threads(int count);

/** Workers each probe currently searches with. */
int threads();

/**
 * Starts Fairy-Stockfish workers until the pool, main thread included, holds
 * @p poolSize threads. They start at a lowered scheduling priority
 * (common/thread_priority.h): probes run beside the tree's workers for as
 * long as it searches, and should take only the cores it leaves idle.
 * The caller serializes pool changes.
 */
void grow_pool(size_t poolSize);

}  // namespace MateProbe

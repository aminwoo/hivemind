#pragma once

#include <string>
#include <vector>

#include "Fairy-Stockfish/src/types.h"

/**
 * @brief Bounded Fairy-Stockfish mate search for a single bughouse board.
 *
 * The hand-rolled root scanner in agent.cc only ever tries attacker moves that
 * give check, so a mate that needs a quiet preparing move is invisible to it no
 * matter how large its budget. Fairy-Stockfish's alpha-beta search has no such
 * restriction, and its bughouse variant already models the defender blocking a
 * check with a piece its partner supplies (Position::allow_virtual_drop, used
 * whenever evasions are generated). That is the same conservative assumption
 * hivemind's own partner-agnostic proofs make, so a mate reported here is a
 * mate under the model the scanner already trusts.
 *
 * Two caveats travel with the result. It is a pruned alpha-beta score rather
 * than an exhaustive proof, and at a node where the attacker is itself in check
 * the evasion generator hands the attacker virtual drops too, which the model
 * does not otherwise allow. Callers should treat a hit as a strong mate
 * candidate for a single board, not as a certificate.
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
    /**
     * The whole line in UCI, for reporting only. Later plies can include a
     * drop of a piece the defender does not hold yet, because the model lets
     * its partner supply one blocker; such a ply is legal to Fairy-Stockfish
     * and not to Board, so replay it there rather than through Board.
     */
    std::vector<std::string> principalVariation;
};

/**
 * @brief Search one board for a forced mate for the side to move.
 *
 * @param fen Bughouse FEN of the board to search.
 * @param maxMateMoves Stop as soon as a mate in this many attacker moves is
 *        found; the search keeps running while it only finds longer ones.
 * @param budgetMs Wall-clock ceiling for the probe.
 *
 * Probes are serialized: Fairy-Stockfish keeps its limits, stop flag and
 * transposition table in globals, so only one can be in flight at a time.
 */
Result probe(const std::string& fen, int maxMateMoves, int budgetMs);

}  // namespace MateProbe

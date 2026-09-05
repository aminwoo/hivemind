#include "search/supply_search.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace SupplySearch {
namespace {
using namespace Stockfish;

bool exposed(const Position& position, Color victim) {
    return position.count<KING>(victim) == 1
        && relative_rank(victim, position.square<KING>(victim)) > RANK_1;
}

// Geometric potential only: a future checking drop need not be available or
// safe. Its purpose is to decide which *real* supply moves deserve exploration.
// The ordinary search still enforces pockets, check evasions and sit rules.
float drop_potential(const Position& target, Color attacker, PieceType piece) {
    if (!exposed(target, ~attacker) || piece < PAWN || piece > QUEEN) {
        return 0.0f;
    }
    const Square king = target.square<KING>(~attacker);
    Bitboard drops = attacks_bb(~attacker, piece, king, target.pieces())
        & ~target.pieces() & target.board_bb();
    if (piece == PAWN) {
        drops &= ~(rank_bb(RANK_1) | rank_bb(RANK_8));
    }
    // Exclude undefended contact drops which the king can simply take.
    Bitboard contact = drops & attacks_bb<KING>(king);
    while (contact) {
        const Square square = pop_lsb(contact);
        if (!(target.attackers_to(square) & target.pieces(attacker))) {
            drops &= ~square_bb(square);
        }
    }
    if (!drops) {
        return 0.0f;
    }
    return piece == KNIGHT ? 3.0f : 1.0f;
}

float demand(const Position& target, Color attacker, PieceType piece) {
    return drop_potential(target, attacker, piece)
        / (1.0f + target.count_in_hand(attacker, piece));
}

PieceType supplied_piece(const Position& position, Square square) {
    // A captured promoted piece transfers as a pawn in bughouse.
    return (position.promotedPieces & square) ? PAWN
                                             : type_of(position.piece_on(square));
}

float king_pressure(const Position& target, const Position& feeder,
                    Color attacker) {
    if (!exposed(target, ~attacker)) return 0.0f;
    float availability[PIECE_TYPE_NB] = {};
    // A captured enemy piece on the feed board becomes the attacker's color
    // on the target board. Existing pocket pieces on the feed board cannot be
    // transferred directly, and are deliberately excluded.
    Bitboard victims = feeder.pieces(attacker) & ~feeder.pieces(KING);
    while (victims) {
        const Square square = pop_lsb(victims);
        const PieceType piece = supplied_piece(feeder, square);
        const bool attacked = feeder.attackers_to(square) & feeder.pieces(~attacker);
        // Pseudo-attacks can be pinned: this is supply *risk*, not a claim that
        // a legal capture exists. Unattacked pieces contribute only latent risk.
        availability[piece] = std::max(availability[piece], attacked ? 0.5f : 0.125f);
    }
    float result = 0.0f;
    for (PieceType piece : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN}) {
        if (target.count_in_hand(attacker, piece) > 0) availability[piece] = 1.0f;
        result += drop_potential(target, attacker, piece) * availability[piece];
    }
    return std::clamp(result / 7.0f, 0.0f, 1.0f);
}
} // namespace

float pressure(const Board& board, Color team) {
    float ours = 0.0f, theirs = 0.0f;
    for (int number : {BOARD_A, BOARD_B}) {
        const Color us = number == BOARD_A ? team : ~team;
        const Position& target = *board.pos[number];
        const Position& feeder = *board.pos[1 - number];
        // One king being mated loses the team game; use the more vulnerable
        // king on each team rather than summing an arbitrary number of checks.
        ours = std::max(ours, king_pressure(target, feeder, us));
        theirs = std::max(theirs, king_pressure(target, feeder, ~us));
    }
    return ours - theirs;
}

std::vector<float> weights(const Board& board, int boardNumber, Color team,
                          const std::vector<Move>& legalActions) {
    const Position& position = *board.pos[boardNumber];
    const Position& partner = *board.pos[1 - boardNumber];
    const Color mover = boardNumber == BOARD_A ? team : ~team;
    if (position.side_to_move() != mover) {
        return {};
    }
    const bool attackHere = exposed(position, ~mover);
    const bool feedPartner = exposed(partner, mover);
    Bitboard vulnerableShield = 0;
    const Square king = position.count<KING>(mover) == 1
        ? position.square<KING>(mover) : SQ_NONE;
    if (king != SQ_NONE) {
        Bitboard shield = attacks_bb<KING>(king)
            & position.pieces(mover, PAWN);
        while (shield) {
            const Square square = pop_lsb(shield);
            const Bitboard attackers = position.attackers_to(square);
            if ((attackers & position.pieces(~mover))
                && !(attackers & position.pieces(mover) & ~position.pieces(KING))) {
                vulnerableShield |= square_bb(square);
            }
        }
    }
    if (!attackHere && !feedPartner && !vulnerableShield) {
        return {};
    }

    float needs[PIECE_TYPE_NB] = {};
    if (feedPartner) {
        for (PieceType piece : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN}) {
            needs[piece] = demand(partner, ~mover, piece);
        }
    }
    std::vector<float> result(legalActions.size(), 0.0f);
    for (size_t index = 0; index < legalActions.size(); ++index) {
        const Move move = legalActions[index];
        if (move == MOVE_NONE || type_of(move) == CASTLING) {
            continue;
        }
        // Increase follow-up attention to checking drops once a feed has landed.
        if (attackHere && position.gives_check(move)) {
            result[index] += 3.0f;
        }
        if (!feedPartner && !vulnerableShield) {
            continue;
        }
        const Square to = to_sq(move);
        if (feedPartner && position.capture(move)) {
            const PieceType captured = type_of(move) == EN_PASSANT
                ? PAWN : supplied_piece(position, to);
            result[index] += 3.0f * needs[captured];
        }

        // One-move preparation: newly attack a piece whose capture would
        // supply a useful checking drop. This includes quiet pins like ...Bb4.
        // No make/unmake, extra inference, or fabricated pocket is needed.
        const bool drop = type_of(move) == DROP;
        const Square from = drop ? SQ_NONE : from_sq(move);
        const PieceType piece = type_of(move) == PROMOTION
            ? promotion_type(move)
            : drop ? dropped_piece_type(move)
                   : type_of(position.piece_on(from));
        Bitboard occupied = position.pieces();
        Bitboard previousAttacks = 0;
        if (!drop) {
            previousAttacks = position.attacks_from(mover,
                type_of(position.piece_on(from)), from);
            occupied &= ~square_bb(from);
        }
        if (type_of(move) == EN_PASSANT) {
            occupied &= ~square_bb(to - pawn_push(mover));
        }
        occupied |= square_bb(to);
        const Bitboard attacks = attacks_bb(mover, piece, to, occupied);
        // When a king-only shield pawn can be sacrificed, moving an adjacent
        // pawn can vacate a flight square beside both king and shield. This
        // remains useful after the king captures on the shield square.
        const bool opensFlight = piece == PAWN && king != SQ_NONE
            && (square_bb(from) & attacks_bb<KING>(king))
            && (attacks_bb<KING>(from) & vulnerableShield);
        if (opensFlight) {
            result[index] += 3.0f;
        }
        if (!feedPartner) {
            continue;
        }
        Bitboard victims = attacks
            & position.pieces(~mover) & ~position.pieces(KING)
            & ~previousAttacks & ~square_bb(to);
        float preparation = 0.0f;
        while (victims) {
            preparation = std::max(preparation,
                needs[supplied_piece(position, pop_lsb(victims))]);
        }
        result[index] += preparation;
    }
    return result;
}

void mix_policy(std::vector<float>& policy, const std::vector<float>& weights,
                float fraction) {
    if (weights.size() != policy.size() || !std::isfinite(fraction)
        || fraction <= 0.0f) {
        return;
    }
    const float total = std::accumulate(weights.begin(), weights.end(), 0.0f);
    if (!(total > 0.0f) || !std::isfinite(total)) {
        return;
    }
    fraction = std::clamp(fraction, 0.0f, 1.0f);
    for (size_t index = 0; index < policy.size(); ++index) {
        policy[index] = (1.0f - fraction) * policy[index]
            + fraction * weights[index] / total;
    }
}
} // namespace SupplySearch

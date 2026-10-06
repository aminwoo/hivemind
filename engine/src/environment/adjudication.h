#pragma once

#include "environment/board.h"
#include "environment/joint_action.h"

enum class GameStatus { ONGOING, WIN, LOSS, DRAW, NO_LEGAL_ACTION };

/**
 * @brief Whether the game is over for `team`, which is to move, shared by the
 * search root, self-play and tournaments so they score games the same way.
 *
 * A mate is not always terminal in bughouse. If the other team is mated but
 * this team still owes a move (it cannot wait and has a legal move), a capture
 * could hand the mated player a blocking piece, so the game goes on and the
 * search picks an action that preserves the mate. Likewise a mated team with
 * no time advantage and a legal move elsewhere keeps playing: its partner may
 * yet supply the piece that breaks the mate. A team with no real move may
 * still have the legal wait action; only with neither is it out of actions.
 */
inline GameStatus adjudicate_game(Board& board, Stockfish::Color team, bool timeAdvantage) {
    if (board.is_single_board()) {
        float value;
        if (board.single_board_terminal_value(value)) {
            if (value == 0.0f) return GameStatus::DRAW;
            const bool win = value > 0.0f;
            return win == (team == board.side_to_move(BOARD_A)) ? GameStatus::WIN : GameStatus::LOSS;
        }
        return GameStatus::ONGOING;
    }
    const bool aOnTurn = board.side_to_move(BOARD_A) == team;
    const bool bOnTurn = board.side_to_move(BOARD_B) == ~team;
    const bool canWait = is_double_sit_legal(timeAdvantage, aOnTurn, bOnTurn);
    const bool hasMove = (aOnTurn && !board.legal_moves(BOARD_A).empty())
        || (bOnTurn && !board.legal_moves(BOARD_B).empty());
    if (board.is_checkmate(~team, !timeAdvantage) && (canWait || !hasMove)) {
        return GameStatus::WIN;
    }
    if (board.is_checkmate(team, timeAdvantage) && (timeAdvantage || !hasMove)) {
        return GameStatus::LOSS;
    }
    if (board.is_draw()) {
        return GameStatus::DRAW;
    }
    return !hasMove && !canWait ? GameStatus::NO_LEGAL_ACTION : GameStatus::ONGOING;
}

inline int adjudicated_winner(GameStatus status, Stockfish::Color team) {
    return status == GameStatus::WIN ? static_cast<int>(team)
        : status == GameStatus::LOSS || status == GameStatus::NO_LEGAL_ACTION
            ? static_cast<int>(~team) : -1;
}

inline const char* adjudicated_termination(GameStatus status) {
    return status == GameStatus::DRAW ? "draw"
        : status == GameStatus::NO_LEGAL_ACTION ? "no legal action" : "checkmate";
}

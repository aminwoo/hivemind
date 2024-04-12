# Copyright 2023 The Pgx Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings

from functools import partial

import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._src.bughouse_utils import (  # type: ignore
    BETWEEN,
    CAN_MOVE,
    CAN_MOVE_ANY,
    INIT_LEGAL_ACTION_MASK,
    PLANE_MAP,
    TO_MAP,
    ZOBRIST_BOARD,
    ZOBRIST_CASTLING_KING,
    ZOBRIST_CASTLING_QUEEN,
    ZOBRIST_EN_PASSANT,
    ZOBRIST_SIDE,
)
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

INIT_ZOBRIST_HASH = jnp.uint32([1172276016, 1112364556])
MAX_TERMINATION_STEPS = 256 

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)

EMPTY = jnp.int32(0)
PAWN = jnp.int32(1)
KNIGHT = jnp.int32(2)
BISHOP = jnp.int32(3)
ROOK = jnp.int32(4)
QUEEN = jnp.int32(5)
KING = jnp.int32(6)
# OPP_PAWN = -1
# OPP_KNIGHT = -2
# OPP_BISHOP = -3
# OPP_ROOK = -4
# OPP_QUEEN = -5
# OPP_KING = -6


# board index (white view)
# 8  7 15 23 31 39 47 55 63
# 7  6 14 22 30 38 46 54 62
# 6  5 13 21 29 37 45 53 61
# 5  4 12 20 28 36 44 52 60
# 4  3 11 19 27 35 43 51 59
# 3  2 10 18 26 34 42 50 58
# 2  1  9 17 25 33 41 49 57
# 1  0  8 16 24 32 40 48 56
#    a  b  c  d  e  f  g  h
# board index (flipped black view)
# 8  0  8 16 24 32 40 48 56
# 7  1  9 17 25 33 41 49 57
# 6  2 10 18 26 34 42 50 58
# 5  3 11 19 27 35 43 51 59
# 4  4 12 20 28 36 44 52 60
# 3  5 13 21 29 37 45 53 61
# 2  6 14 22 30 38 46 54 62
# 1  7 15 23 31 39 47 55 63
#    a  b  c  d  e  f  g  h
# fmt: off
INIT_BOARD = jnp.int32([
    4, 1, 0, 0, 0, 0, -1, -4,
    2, 1, 0, 0, 0, 0, -1, -2,
    3, 1, 0, 0, 0, 0, -1, -3,
    5, 1, 0, 0, 0, 0, -1, -5,
    6, 1, 0, 0, 0, 0, -1, -6,
    3, 1, 0, 0, 0, 0, -1, -3,
    2, 1, 0, 0, 0, 0, -1, -2,
    4, 1, 0, 0, 0, 0, -1, -4
])
# fmt: on

# Action
# 0 ... 9 = underpromotions
# plane // 3 == 0: rook
# plane // 3 == 1: bishop
# plane // 3 == 2: knight
# plane % 3 == 0: forward
# plane % 3 == 1: right
# plane % 3 == 2: left
# 51                   22                   50
#    52                21                49
#       53             20             48
#          54          19          47
#             55       18       46
#                56    17    45
#                   57 16 44
# 23 24 25 26 27 28 29  X 30 31 32 33 34 35 36
#                   43 15 58
#                42    14    59
#             41       13       60
#          40          12          61
#       39             11             62
#    38                10                64
# 37                    9                   64


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE 
    legal_action_mask: Array = INIT_LEGAL_ACTION_MASK  # 2 * 64 * 78 + 1 = 9985
    observation: Array = jnp.zeros((8, 16, 32), dtype=jnp.float32)
    _step_count: Array = jnp.int32(0)
    # --- Chess specific ---
    _turn: Array = jnp.int32([0, 0])
    _board: Array = jnp.int32([INIT_BOARD, INIT_BOARD]) # From top left. like FEN
    # (curr, opp) Flips every turn
    _can_castle_queen_side: Array = jnp.ones((2, 2), dtype=jnp.bool_)
    _can_castle_king_side: Array = jnp.ones((2, 2), dtype=jnp.bool_)
    _en_passant: Array = jnp.int32([-1, -1])  # En passant target. Flips.
    # --- Bughouse specific --- 
    _pocket: Array = jnp.zeros((2, 2, 6), dtype=jnp.int32)
    _clock: Array = jnp.int32([[1200, 1200], [1200, 1200]])
    # # of moves since the last piece capture or pawn move
    _halfmove_count: Array = jnp.int32([0, 0])
    _fullmove_count: Array = jnp.int32([1, 1])  # increase every black move
    _zobrist_hash: Array = jnp.int32([INIT_ZOBRIST_HASH, INIT_ZOBRIST_HASH])
    _hash_history: Array = jnp.zeros((2, MAX_TERMINATION_STEPS + 1, 2), dtype=jnp.uint32).at[0, 0].set(INIT_ZOBRIST_HASH).at[1, 0].set(INIT_ZOBRIST_HASH)
    # index to possible piece positions for speeding up. Flips every turn.
    _promoted_pieces: Array = jnp.zeros((2, 64), dtype=jnp.bool_)

    @property
    def env_id(self) -> core.EnvId:
        return "bughouse"
    
    @staticmethod
    def _from_fen(fen: str, *args):
        from pgx.experimental.bughouse import from_fen

        return from_fen(fen)

    def _to_fen(self) -> str:
        from pgx.experimental.bughouse import to_fen

        return to_fen(self)


@dataclass
class Action:
    from_: Array = jnp.int32(-1)
    to: Array = jnp.int32(-1)
    underpromotion: Array = jnp.int32(-1)  # 0: rook, 1: bishop, 2: knight
    drop: Array = jnp.int32(-1)
    board_num: Array = jnp.int32(-1)
    sit: Array = FALSE

    @staticmethod
    def _from_label(label: Array):
        """We use AlphaZero style label with channel-last representation: (8, 8, 78)

          78 = drops (5) + queen moves (56) + knight moves (8) + underpromotions (3 * 3) 

        Note: this representation is reported as

        > We also tried using a flat distribution over moves for chess and shogi;
        > the final result was almost identical although training was slightly slower.

        Flat representation may have 1858 actions (= 1792 normal moves + (7 + 7 + 8) * 3 underpromotions)

        Also see
          - https://github.com/LeelaChessZero/lc0/issues/637
          - https://github.com/LeelaChessZero/lc0/pull/712
        """

        sit = jax.lax.select(label == 9984, TRUE, FALSE)
        board_num = jax.lax.select(label < 4992, 0, 1)
        label %= 4992
        from_, plane = label // 78, label % 78
        drop = jax.lax.select(plane < 73, jnp.int32(-1), jnp.int32(plane - 72))
        to = jax.lax.select(drop > 0, from_, TO_MAP[from_, plane])
        return Action(  # type: ignore
            from_=from_,
            to=to,  # -1 if impossible move
            underpromotion=jax.lax.select(plane >= 9, jnp.int32(-1), jnp.int32(plane // 3)),
            drop=drop,
            board_num=board_num,
            sit=sit, 
        )

    def _to_label(self):
        plane = PLANE_MAP[self.from_, self.to]
        plane = jax.lax.select(self.drop > 0, self.drop + 72, plane)
        plane = jax.lax.select(self.board_num == 0, plane, plane + 4992)
        ret = jnp.int32(self.from_) * 78 + jnp.int32(plane)
        ret = jax.lax.select(self.sit, 9984, ret)
        return ret
    
    def _to_string(self):
        from pgx.experimental.bughouse import to_string

        return to_string(self)


class Bughouse(core.Env):
    def __init__(self):
        super().__init__()

    def _init(self, key: PRNGKey) -> State:
        current_player = jnp.int32(jax.random.bernoulli(key))
        state = State(current_player=current_player)  # type: ignore
        state = _mask_moves(state)
        return state

    def _step(self, state: core.State, action: Array, key) -> State:
        assert isinstance(state, State)
        delta = jax.random.randint(key, (), 3, 10)
        state = _step_time(state, -delta)
        state = _step(state, action)
        state = jax.lax.cond(
            (MAX_TERMINATION_STEPS <= state._step_count),
            # end with tie
            lambda: state.replace(terminated=TRUE),  # type: ignore
            lambda: state,
        )
        return state  # type: ignore

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        return _observe(state, player_id)

    @property
    def id(self) -> core.EnvId:
        return "bughouse"

    @property
    def version(self) -> str:
        return "v1"

    @property
    def num_players(self) -> int:
        return 2


def _step(state: State, action: Array):
    a = Action._from_label(action)

    def _sit(state, a):
        return _set_current_player(state, 1 - state.current_player)
    
    def _move(state, a): 
        state = _update_zobrist_hash(state, a)

        hash_ = state._zobrist_hash[a.board_num]
        hash_ ^= _hash_castling_en_passant(state, a.board_num)

        state = _apply_move(state, a)
        state = _flip(state, a.board_num)

        hash_ ^= _hash_castling_en_passant(state, a.board_num)
        state = state.replace(_zobrist_hash=state._zobrist_hash.at[a.board_num].set(hash_))  # type: ignore

        state = _update_history(state, a.board_num)
        state = state.replace(legal_action_mask=_legal_action_mask(state))  # type: ignore
        state = _check_termination(state)
        return state

    return jax.lax.cond(a.sit, _sit, _move, state, a)

def _update_history(state: State, board_num: Array):
    # hash hist
    hash_hist = jnp.roll(state._hash_history[board_num], 2)
    hash_hist = hash_hist.at[0].set(state._zobrist_hash[board_num].astype(jnp.uint32))
    state = state.replace(_hash_history=state._hash_history.at[board_num].set(hash_hist))  # type: ignore
    return state

def _check_termination(state: State):
    def fn(board_num: Array):
        rep = (state._hash_history[board_num] == state._zobrist_hash[board_num]).all(axis=1).sum() - 1
        terminated = rep >= 2

        action_mask = jax.lax.select(board_num == 0, state.legal_action_mask[:4992], state.legal_action_mask[4992:9984])
        has_legal_action = action_mask.any() 
        is_checkmate = jax.lax.cond(state._pocket[board_num, 0].sum() > 0, 
                                      lambda: ~has_legal_action & _on_turn(state, board_num), 
                                      lambda: (~has_legal_action & _on_turn(state, board_num)) & (((state._turn[0] == state._turn[1]) & (_time_advantage(state) <= 0)) | ~_legal_drops(state, board_num))
                                      )
        terminated |= is_checkmate

        return terminated, is_checkmate
    
    terminated_left, is_checkmate_left = fn(0) 
    terminated_right, is_checkmate_right = fn(1) 
    
    reward = jax.lax.select(
        is_checkmate_left | is_checkmate_right,
        jnp.ones(2, dtype=jnp.float32).at[state.current_player].set(-1),
        jnp.zeros(2, dtype=jnp.float32),
    )
    return state.replace(  # type: ignore
        terminated=terminated_left | terminated_right,
        rewards=reward,
    )


def _legal_drops(state: State, board_num: Array): 
    def is_drop_legal(a: Action, board_num: Array):
        next_s = _flip(_apply_move(state, a), board_num)
        return ~_is_checking(next_s, board_num)
    
    @partial(jax.vmap, in_axes=(0, None))
    def legal_drops(from_, board_num: Array): 
        piece = state._board[board_num, from_]

        def legal_label(drop):
            a = Action(from_=from_, to=from_, drop=drop, board_num=board_num)
            return jax.lax.select(
                (from_ > 0) & (piece == 0) & is_drop_legal(a, board_num),
                TRUE,
                FALSE,
            )

        return legal_label(5)
    
    king_pos = jnp.argmin(jnp.abs(state._board[board_num] - KING))
    return legal_drops(CAN_MOVE[KING, king_pos], board_num).any() 
    


def _apply_move(state: State, a: Action):
    # apply move action
    piece = state._board[a.board_num, a.from_]
    # en passant
    is_en_passant = (state._en_passant[a.board_num] >= 0) & (piece == PAWN) & (state._en_passant[a.board_num] == a.to)
    removed_pawn_pos = a.to - 1
    state = state.replace(  # type: ignore
        _board=state._board.at[a.board_num, removed_pawn_pos].set(
            jax.lax.select(is_en_passant, EMPTY, state._board[a.board_num, removed_pawn_pos])
        ),
    )
    state = state.replace(  # type: ignore
        _en_passant=state._en_passant.at[a.board_num].set(jax.lax.select(
            (piece == PAWN) & (jnp.abs(a.to - a.from_) == 2),
            jnp.int32((a.to + a.from_) // 2),
            jnp.int32(-1),
        ))
    )
    # if capture update drops
    partner_color = jnp.int32(state._turn[a.board_num] == state._turn[1 - a.board_num])
    captured_piece = jax.lax.select(is_en_passant | state._promoted_pieces[a.board_num, a.to], PAWN, -state._board[a.board_num, a.to])
    state = state.replace( # type: ignore
        _pocket=jax.lax.select(captured_piece > 0, state._pocket.at[1 - a.board_num, partner_color, captured_piece].add(1), state._pocket)
    )
    # update counters
    captured = (state._board[a.board_num][a.to] < 0) | (is_en_passant)
    state = state.replace(  # type: ignore
        _halfmove_count=state._halfmove_count.at[a.board_num].set(jax.lax.select(captured | (piece == PAWN), 0, state._halfmove_count[a.board_num] + 1)),
        _fullmove_count=state._fullmove_count.at[a.board_num].set(state._fullmove_count[a.board_num] + jnp.int32(state._turn[a.board_num] == 1)),
    )
    # castling
    # Whether castling is possible or not is not checked here.
    # We assume that if castling is not possible, it is filtered out.
    # left

    state = state.replace(  # type: ignore
        _board=jax.lax.cond(
            (piece == KING) & (a.from_ == 32) & (a.to == 16),
            lambda: state._board.at[a.board_num, 0].set(EMPTY).at[a.board_num, 24].set(ROOK),
            lambda: state._board,
        ),
    )
    # right
    state = state.replace(  # type: ignore
        _board=jax.lax.cond(
            (piece == KING) & (a.from_ == 32) & (a.to == 48),
            lambda: state._board.at[a.board_num, 56].set(EMPTY).at[a.board_num, 40].set(ROOK),
            lambda: state._board,
        ),
    )
    # update my can_castle_xxx_side
    state = state.replace(  # type: ignore
        _can_castle_queen_side=state._can_castle_queen_side.at[a.board_num, 0].set(
            jax.lax.select(
                (a.from_ == 32) | (a.from_ == 0),
                FALSE,
                state._can_castle_queen_side[a.board_num, 0],
            )
        ),
        _can_castle_king_side=state._can_castle_king_side.at[a.board_num, 0].set(
            jax.lax.select(
                (a.from_ == 32) | (a.from_ == 56),
                FALSE,
                state._can_castle_king_side[a.board_num, 0],
            )
        ),
    )
    # update opp can_castle_xxx_side
    state = state.replace(  # type: ignore
        _can_castle_queen_side=state._can_castle_queen_side.at[a.board_num, 1].set(
            jax.lax.select(
                (a.to == 7),
                FALSE,
                state._can_castle_queen_side[a.board_num, 1],
            )
        ),
        _can_castle_king_side=state._can_castle_king_side.at[a.board_num, 1].set(
            jax.lax.select(
                (a.to == 63),
                FALSE,
                state._can_castle_king_side[a.board_num, 1],
            )
        ),
    )

    # set promotion mask 
    # if we captured promoted piece
    state = jax.lax.cond(
        state._promoted_pieces[a.board_num, a.to],
        lambda: state.replace(_promoted_pieces=state._promoted_pieces.at[a.board_num, a.to].set(FALSE)),  # type: ignore
        lambda: state,
    )
    # if we are moving promoted piece
    state = jax.lax.cond(
        state._promoted_pieces[a.board_num, a.from_],
        lambda: state.replace(_promoted_pieces=state._promoted_pieces.at[a.board_num, a.from_].set(FALSE).at[a.board_num, a.to].set(TRUE)),  # type: ignore
        lambda: state,
    )
    # if we promote new piece
    state = jax.lax.cond(
        (piece == PAWN) & (a.from_ % 8 == 6) & (a.drop < 0),
        lambda: state.replace(_promoted_pieces=state._promoted_pieces.at[a.board_num, a.to].set(TRUE)),  # type: ignore
        lambda: state,
    )

    # promotion to queen
    piece = jax.lax.select(
        (piece == PAWN) & (a.from_ % 8 == 6) & (a.underpromotion < 0) & (a.drop < 0),
        QUEEN,
        piece,
    )
    # underpromotion
    piece = jax.lax.select(
        a.underpromotion < 0,
        piece,
        jnp.int32([ROOK, BISHOP, KNIGHT])[a.underpromotion],
    )
    # drop
    piece = jax.lax.select(
        a.drop < 0, 
        piece, 
        a.drop,
    )

    # actually move
    state = state.replace(_board=state._board.at[a.board_num, a.from_].set(EMPTY).at[a.board_num, a.to].set(piece))  # type: ignore
    state = state.replace( # type: ignore
        _pocket=jax.lax.select(a.drop < 0, state._pocket, state._pocket.at[a.board_num, 0, a.drop].add(-1))
    )

    return state

def _flip_pos(x):
    """
    >>> _flip_pos(jnp.int32(34))
    Array(37, dtype=int32)
    >>> _flip_pos(jnp.int32(37))
    Array(34, dtype=int32)
    >>> _flip_pos(jnp.int32(-1))
    Array(-1, dtype=int32)
    """
    return jax.lax.select(x == -1, x, (x // 8) * 8 + (7 - (x % 8)))


def _rotate(board):
    return jnp.rot90(board, k=1)


def _flip(state: State, board_num: Array) -> State:
    return state.replace(  # type: ignore
        current_player=(state.current_player + 1) % 2,
        _board=state._board.at[board_num].set(-jnp.flip(state._board[board_num].reshape(8, 8), axis=1).flatten()),
        _turn=state._turn.at[board_num].set(1 - state._turn[board_num]),
        _en_passant=state._en_passant.at[board_num].set(_flip_pos(state._en_passant[board_num])),
        _can_castle_queen_side=state._can_castle_queen_side.at[board_num].set(state._can_castle_queen_side[board_num][::-1]),
        _can_castle_king_side=state._can_castle_king_side.at[board_num].set(state._can_castle_king_side[board_num][::-1]),
        _pocket=state._pocket.at[board_num].set(state._pocket[board_num][::-1]),
        _promoted_pieces=state._promoted_pieces.at[board_num].set(jnp.flip(state._promoted_pieces[board_num].reshape(8, 8), axis=1).flatten()),
        _clock=state._clock.at[board_num].set(state._clock[board_num][::-1])
    )


def _legal_action_mask(state: State):
    def is_legal(a: Action, board_num: Array):
        ok = _is_pseudo_legal(state, a)
        next_s = _flip(_apply_move(state, a), board_num)
        ok &= ~_is_checking(next_s, board_num)

        return ok
    
    def is_drop_legal(a: Action, board_num: Array):
        next_s = _flip(_apply_move(state, a), board_num)
        return ~_is_checking(next_s, board_num)
    
    @partial(jax.vmap, in_axes=(0, None))
    def legal_drops(from_, board_num: Array): 
        piece = state._board[board_num, from_]

        @jax.vmap
        def legal_labels(drop):
            a = Action(from_=from_, to=from_, drop=drop, board_num=board_num)
            return jax.lax.select(
                (state._pocket[board_num, 0, drop] > 0) & (piece == 0) & ~(drop == PAWN & ((from_ % 8 == 7) | (from_ % 8 == 0))) & is_drop_legal(a, board_num),
                a._to_label(),
                jnp.int32(-1),
            )

        return legal_labels(jnp.arange(1, 6))
    
    @partial(jax.vmap, in_axes=(0, None))
    def legal_norml_moves(from_, board_num: Array):
        piece = state._board[board_num, from_]

        @jax.vmap
        def legal_label(to):
            a = Action(from_=from_, to=to, board_num=board_num)
            return jax.lax.select(
                (piece > 0) & (to >= 0) & is_legal(a, board_num),
                a._to_label(),
                jnp.int32(-1),
            )

        return legal_label(CAN_MOVE[piece, from_])

    def legal_underpromotions(mask, board_num):
        # from_ = 6 14 22 30 38 46 54 62
        # plane = 0 ... 8
        @jax.vmap
        def make_labels(from_):
            return from_ * 78 + jnp.arange(9) + jax.lax.select(board_num == 0, 0, 4992)

        labels = make_labels(jnp.int32([6, 14, 22, 30, 38, 46, 54, 62])).flatten()

        @jax.vmap
        def legal_labels(label):
            a = Action._from_label(label)
            ok = (state._board[board_num, a.from_] == PAWN) & (a.to >= 0)
            ok &= mask[Action(from_=a.from_, to=a.to, board_num=board_num)._to_label()]
            return jax.lax.select(ok, label, -1)

        ok_labels = legal_labels(labels)
        return ok_labels.flatten()

    def legal_en_passants(board_num):
        to = state._en_passant[board_num]
        @jax.vmap
        def legal_labels(from_):
            ok = (
                (from_ >= 0)
                & (from_ < 64)
                & (to >= 0)
                & (state._board[board_num, from_] == PAWN)
                & (state._board[board_num, to - 1] == -PAWN)
            )
            a = Action(from_=from_, to=to, board_num=board_num)
            ok &= ~_is_checking(_flip(_apply_move(state, a), board_num), board_num)
            return jax.lax.select(ok, a._to_label(), -1)

        return legal_labels(jnp.int32([to - 9, to + 7]))

    def can_castle_king_side(board_num):
        ok = state._board[board_num][32] == KING
        ok &= state._board[board_num][56] == ROOK
        ok &= state._can_castle_king_side[board_num][0]
        ok &= state._board[board_num][40] == EMPTY
        ok &= state._board[board_num][48] == EMPTY

        @jax.vmap
        def is_ok(label):
            return ~_is_checking(_flip(_apply_move(state, Action._from_label(label)), board_num), board_num)

        ok &= ~_is_checking(_flip(state, board_num), board_num)
        squares = jax.lax.select(board_num == 0, jnp.int32([2526, 2527]), jnp.int32([7518, 7519]))
        ok &= is_ok(squares).all()

        return ok

    def can_castle_queen_side(board_num):
        ok = state._board[board_num][32] == KING
        ok &= state._board[board_num][0] == ROOK
        ok &= state._can_castle_queen_side[board_num][0]
        ok &= state._board[board_num][8] == EMPTY
        ok &= state._board[board_num][16] == EMPTY
        ok &= state._board[board_num][24] == EMPTY

        @jax.vmap
        def is_ok(label):
            return ~_is_checking(_flip(_apply_move(state, Action._from_label(label)), board_num), board_num)

        ok &= ~_is_checking(_flip(state, board_num), board_num)
        squares = jax.lax.select(board_num == 0, jnp.int32([2524, 2525]), jnp.int32([7516, 7517]))
        ok &= is_ok(squares).all()

        return ok

    def legal_moves_left(mask):
        actions = legal_norml_moves(jnp.arange(64), 0).flatten()  # include -1
        # +1 is to avoid setting True to the last element
        mask = mask.at[actions].set(TRUE)

        # castling
        mask = mask.at[2524].set(jax.lax.select(can_castle_queen_side(0), TRUE, mask[2524]))
        mask = mask.at[2527].set(jax.lax.select(can_castle_king_side(0), TRUE, mask[2527]))

        # set en passant
        actions = legal_en_passants(0)
        mask = mask.at[actions].set(TRUE)

        # set underpromotions
        actions = legal_underpromotions(mask, 0)
        mask = mask.at[actions].set(TRUE)

        # set drops
        actions = legal_drops(jnp.arange(64), 0).flatten() 
        mask = mask.at[actions].set(TRUE)

        # unset -1 action
        mask.at[4992].set(FALSE)
        return mask

    def legal_moves_right(mask):
        actions = legal_norml_moves(jnp.arange(64), 1).flatten()  # include -1
        mask = mask.at[actions].set(TRUE)

        # castling
        mask = mask.at[7516].set(jax.lax.select(can_castle_queen_side(1), TRUE, mask[7516]))
        mask = mask.at[7519].set(jax.lax.select(can_castle_king_side(1), TRUE, mask[7519]))

        # set en passant
        actions = legal_en_passants(1)
        mask = mask.at[actions].set(TRUE)

        # set underpromotions
        actions = legal_underpromotions(mask, 1)
        mask = mask.at[actions].set(TRUE)

        # set drops
        actions = legal_drops(jnp.arange(64), 1).flatten() 
        mask = mask.at[actions].set(TRUE)
        return mask 

    mask = jnp.zeros(2 * 64 * 78 + 1, dtype=jnp.bool_)
    mask = jax.lax.cond(
        _on_turn(state, 0),
        lambda: legal_moves_left(mask),  # type: ignore
        lambda: mask
    )
    mask = jax.lax.cond(
        _on_turn(state, 1),
        lambda: legal_moves_right(mask),  # type: ignore
        lambda: mask
    )

    # PASS action (if we are up time and diagonal players on turn)
    mask = mask.at[-1].set(_time_advantage(state) > 0 & (state._turn[0] == state._turn[1]))

    return mask


def _is_attacking(state: State, pos, board_num: Array):
    @jax.vmap
    def can_move(from_):
        a = Action(from_=from_, to=pos, board_num=board_num)
        return (from_ != -1) & _is_pseudo_legal(state, a)

    return can_move(CAN_MOVE_ANY[pos, :]).any()


def _is_checking(state: State, board_num: Array):
    """True if possible to capture the opponent king"""
    opp_king_pos = jnp.argmin(jnp.abs(state._board[board_num] - -KING))
    return _is_attacking(state, opp_king_pos, board_num)


def _is_pseudo_legal(state: State, a: Action):
    piece = state._board[a.board_num, a.from_]
    ok = (piece >= 0) & (state._board[a.board_num, a.to] <= 0)
    ok &= (CAN_MOVE[piece, a.from_] == a.to).any()
    between_ixs = BETWEEN[a.from_, a.to]
    ok &= ((between_ixs < 0) | (state._board[a.board_num, between_ixs] == EMPTY)).all()
    # filter pawn move
    ok &= ~((piece == PAWN) & ((a.to % 8) < (a.from_ % 8)))
    ok &= ~((piece == PAWN) & (jnp.abs(a.to - a.from_) <= 2) & (state._board[a.board_num, a.to] < 0))
    ok &= ~((piece == PAWN) & (jnp.abs(a.to - a.from_) > 2) & (state._board[a.board_num, a.to] >= 0))
    return (a.to >= 0) & ok


def _zobrist_hash(state, board_num: Array):
    """
    >>> state = State()
    >>> _zobrist_hash(state)
    Array([1172276016, 1112364556], dtype=uint32)
    """
    hash_ = jnp.zeros(2, dtype=jnp.uint32)
    hash_ = jax.lax.select(state._turn[board_num] == 0, hash_, hash_ ^ ZOBRIST_SIDE)
    board = jax.lax.select(state._turn[board_num] == 0, state._board[board_num], _flip(state, board_num)._board[board_num])

    def xor(i, h):
        # 0, ..., 12 (white pawn, ..., black king)
        piece = board[i] + 6
        return h ^ ZOBRIST_BOARD[i, piece]

    hash_ = jax.lax.fori_loop(0, 64, xor, hash_)
    hash_ ^= _hash_castling_en_passant(state, board_num)
    return hash_


def _hash_castling_en_passant(state: State, board_num: Array):
    # we don't take care side (turn) as it's already taken into account in hash
    zero = jnp.uint32([0, 0])
    hash_ = zero
    hash_ ^= jax.lax.select(state._can_castle_queen_side[board_num, 0], ZOBRIST_CASTLING_QUEEN[0], zero)
    hash_ ^= jax.lax.select(state._can_castle_queen_side[board_num, 1], ZOBRIST_CASTLING_QUEEN[1], zero)
    hash_ ^= jax.lax.select(state._can_castle_king_side[board_num, 0], ZOBRIST_CASTLING_KING[0], zero)
    hash_ ^= jax.lax.select(state._can_castle_king_side[board_num, 1], ZOBRIST_CASTLING_KING[1], zero)
    hash_ ^= ZOBRIST_EN_PASSANT[state._en_passant[board_num]]
    return hash_


def _update_zobrist_hash(state: State, a: Action):
    # do NOT take into account
    #  - en passant, and
    #  - castling
    hash_ = state._zobrist_hash[a.board_num]
    source_piece = state._board[a.board_num, a.from_]
    source_piece = jax.lax.select(state._turn[a.board_num] == 0, source_piece + 6, (source_piece * -1) + 6)
    destination_piece = state._board[a.board_num, a.to]
    destination_piece = jax.lax.select(state._turn[a.board_num] == 0, destination_piece + 6, (destination_piece * -1) + 6)
    from_ = jax.lax.select(state._turn[a.board_num] == 0, a.from_, _flip_pos(a.from_))
    to = jax.lax.select(state._turn[a.board_num] == 0, a.to, _flip_pos(a.to))

    # Only for non-drop moves
    hash_ = jax.lax.select(a.drop < 0, hash_ ^ ZOBRIST_BOARD[from_, source_piece], hash_) # Remove the piece from source
    hash_ = jax.lax.select(a.drop < 0, hash_ ^ ZOBRIST_BOARD[from_, 6], hash_) # Make source empty
    hash_ = jax.lax.select(a.drop < 0, hash_ ^ ZOBRIST_BOARD[to, destination_piece], hash_) # Remove the piece at target pos (including empty)

    # promotion to queen
    piece = state._board[a.board_num, a.from_]
    source_piece = jax.lax.select(
        (piece == PAWN) & (a.from_ % 8 == 6) & (a.underpromotion < 0),
        jax.lax.select(state._turn[a.board_num] == 0, QUEEN + 6, (QUEEN * -1) + 6),
        source_piece,
    )

    # underpromotion
    source_piece = jax.lax.select(
        a.underpromotion >= 0,
        jax.lax.select(
            state._turn[a.board_num] == 0,
            source_piece + 3 - a.underpromotion,
            source_piece - (3 - a.underpromotion),
        ),
        source_piece,
    )

    #drop 
    source_piece = jax.lax.select(
        a.drop >= 0,
        jax.lax.select(state._turn[a.board_num] == 0, a.drop + 6, (a.drop * -1) + 6),
        source_piece,
    )

    hash_ ^= ZOBRIST_BOARD[to, source_piece]  # Put the piece to the target pos

    # en_passant
    is_en_passant = (state._en_passant[a.board_num] >= 0) & (piece == PAWN) & (state._en_passant[a.board_num] == a.to)
    removed_pawn_pos = a.to - 1
    removed_pawn_pos = jax.lax.select(state._turn[a.board_num] == 0, removed_pawn_pos, _flip_pos(removed_pawn_pos))
    opp_pawn = jax.lax.select(state._turn[a.board_num] == 0, (PAWN * -1) + 6, PAWN + 6)
    hash_ ^= jax.lax.select(
        is_en_passant,
        ZOBRIST_BOARD[removed_pawn_pos, opp_pawn],
        jnp.uint32([0, 0]),
    )  # Remove the pawn
    hash_ ^= jax.lax.select(is_en_passant, ZOBRIST_BOARD[removed_pawn_pos, 6], jnp.uint32([0, 0]))  # empty

    hash_ ^= ZOBRIST_SIDE
    return state.replace(  # type: ignore
        _zobrist_hash=state._zobrist_hash.at[a.board_num].set(hash_),
    )


def _on_turn(state: State, board_num: Array):
    ret = FALSE
    ret |= ((board_num == 0) & (state.current_player == state._turn[board_num]))
    ret |= ((board_num == 1) & (state.current_player == 1 - state._turn[board_num]))
    return ret


def _time_advantage(state: State): 
    return state._clock[0, 1 - jnp.int32(_on_turn(state, 0))] - state._clock[1, jnp.int32(_on_turn(state, 1))]

def _mask_moves(state: State):
    mask = state.legal_action_mask
    mask = jax.lax.select(~_on_turn(state, 0), mask.at[:4992].set(FALSE), mask)
    mask = jax.lax.select(~_on_turn(state, 1), mask.at[4992:].set(FALSE), mask)
    mask = mask.at[-1].set(_time_advantage(state) > 0 & (state._turn[0] == state._turn[1]))
    return state.replace(legal_action_mask=mask) # type: ignore

def _observe(state: State, player_id: Array):
    ones = jnp.ones((1, 8, 8), dtype=jnp.float32)

    def board2planes(board_num): 
        board = _rotate(state._board[board_num].reshape((8, 8)))

        def piece_feat(p):
            return (board == p).astype(jnp.float32)

        my_pieces = jax.vmap(piece_feat)(jnp.arange(1, 7))
        opp_pieces = jax.vmap(piece_feat)(-jnp.arange(1, 7))

        pocket = state._pocket[board_num] / 16
        my_drops = ones * pocket[0][1:, None, None]
        opp_drops = ones * pocket[1][1:, None, None]

        h = state._hash_history[board_num, 0, :]
        rep = (state._hash_history == h).all(axis=1).sum() - 1
        rep = jax.lax.select((h == 0).all(), 0, rep)
        rep0 = ones * (rep == 0)
        rep1 = ones * (rep >= 1)

        on_turn = ones * jax.lax.select(_on_turn(state, board_num), 1, 0)

        promoted = state._promoted_pieces[board_num].reshape((1, 8, 8))

        en_passant = jnp.zeros((64), dtype=jnp.float32)
        en_passant = en_passant.at[state._en_passant[board_num]].set(1)
        en_passant = en_passant.reshape((1, 8, 8))

        my_queen_side_castling_right = ones * state._can_castle_queen_side[board_num][0]
        my_king_side_castling_right = ones * state._can_castle_king_side[board_num][0]
        opp_queen_side_castling_right = ones * state._can_castle_queen_side[board_num][1]
        opp_king_side_castling_right = ones * state._can_castle_king_side[board_num][1]

        time_advantage = ones * (0.5 + _time_advantage(state) / 300)

        return jnp.vstack(
            [
                my_pieces, 
                opp_pieces,
                rep0,
                rep1,
                my_drops,
                opp_drops,
                on_turn, 
                promoted, 
                en_passant,
                my_queen_side_castling_right,
                my_king_side_castling_right,
                opp_queen_side_castling_right,
                opp_king_side_castling_right,
                time_advantage
            ]
        )
    
    planes = jnp.concatenate( 
        [
            board2planes(0),
            board2planes(1),
        ],
        axis=2
    )
    return planes.transpose((1, 2, 0))


@jax.jit
def _set_clock(state, clock): 
    state = state.replace(_clock=clock)
    state = state.replace(observation=_observe(state, state.current_player))
    return state 


@jax.jit
def _set_current_player(state, current_player):
    state = state.replace(current_player=current_player)
    state = state.replace(legal_action_mask=_legal_action_mask(state))  # type: ignore
    state = state.replace(observation=_observe(state, current_player))
    state = _mask_moves(state)
    return state


@jax.jit
def _set_board_num(state, board_num): 
    state = state.replace(legal_action_mask=jax.lax.select(board_num == 0, state.legal_action_mask.at[4992:].set(FALSE), state.legal_action_mask.at[:4992].set(FALSE)))  # type: ignore
    return state


@jax.jit
def _is_promotion(state, action):
    a = Action._from_label(action)
    piece = state._board[a.board_num, a.from_]
    return (piece == PAWN) & (a.from_ % 8 == 6) & (a.drop < 0)


@jax.jit
def _step_time(state, delta):
    return state.replace(  # type: ignore
            _clock=state._clock.at[0, 0].add(delta).at[1, 0].add(delta)
        )


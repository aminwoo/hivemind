import jax
import jax.numpy as jnp

from pgx.bughouse import (
    State,
    Action,
    _check_termination,
    _flip_pos,
    _legal_action_mask,
    _observe,
    _update_history,
    _zobrist_hash,
    _mask_moves,
)

TRUE = jnp.bool_(True)

def to_string(action):
    """Convert action label to uci format move string 

    Args:
        action (jnp.int32): Action label

    Returns:
        move_uci: move string
    """
    
    # The do nothing action 
    if action.sit:
        return "pass"
    
    underpromotions = ["r", "b", "n"]
    drop_pieces = ["", "p", "n", "b", "r", "q", "k"]
    squares = ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", 
               "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", 
               "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", 
               "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", 
               "e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", 
               "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", 
               "g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8",
               "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", ""] # One extra square to account for -1
    
    move_uci = str(action.board_num)
    if action.drop > 0: 
        move_uci += drop_pieces[action.drop].upper() + '@' + squares[action.to]
    else:
        move_uci += squares[action.from_] + squares[action.to]
    if action.underpromotion >= 0:
        move_uci += underpromotions[action.underpromotion]
    return move_uci

def from_fen(fen: str) -> State:
    """_summary_

    Args:
        fen (str): _description_

    Returns:
        State: _description_
    """
    
    _turn = jnp.int32([0, 0])
    _board = jnp.zeros((2, 64), dtype=jnp.int32)
    _can_castle_queen_side = jnp.ones((2, 2), dtype=jnp.bool_)
    _can_castle_king_side = jnp.ones((2, 2), dtype=jnp.bool_)
    _en_passant = jnp.int32([-1, -1]) 
    _pocket = jnp.zeros((2, 2, 6), dtype=jnp.int32)
    _clock = jnp.int32([[1200, 1200], [1200, 1200]])
    _halfmove_count  = jnp.int32([0, 0])
    _fullmove_count = jnp.int32([1, 1])  

    MAP = {'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5}

    fens = fen.split("|")
    for board_num in range(2): 
        pieces, turn, castling, en_passant, halfmove_count, fullmove_count = fens[board_num].split()
        _halfmove_count = _halfmove_count.at[board_num].set(jnp.int32(halfmove_count))
        _fullmove_count = _fullmove_count.at[board_num].set(jnp.int32(fullmove_count))
        pieces = pieces.split("/")
        pocket = pieces[-1]
        for p in pocket: 
            if p.isupper():
                _pocket = _pocket.at[board_num, 0, MAP[p.lower()]].add(1)
            else:
                _pocket = _pocket.at[board_num, 1, MAP[p.lower()]].add(1)

        pieces = pieces[:-1]
        arr = []
        for line in pieces:
            for c in line:
                if str.isnumeric(c):
                    for _ in range(int(c)):
                        arr.append(0)
                else:
                    ix = "pnbrqk".index(str.lower(c)) + 1
                    if str.islower(c):
                        ix *= -1
                    arr.append(ix)
        if "Q" in castling:
            _can_castle_queen_side = _can_castle_queen_side.at[board_num, 0].set(TRUE)
        if "q" in castling:
            _can_castle_queen_side = _can_castle_queen_side.at[board_num, 1].set(TRUE)
        if "K" in castling:
            _can_castle_king_side = _can_castle_king_side.at[board_num, 0].set(TRUE)
        if "k" in castling:
            _can_castle_king_side = _can_castle_king_side.at[board_num, 1].set(TRUE)
        if turn == "b":
            _can_castle_queen_side.at[board_num].set(_can_castle_queen_side[board_num][::-1])
            _can_castle_king_side.at[board_num].set(_can_castle_king_side[board_num][::-1])

        mat = jnp.int32(arr).reshape(8, 8)
        if turn == "b":
            mat = -jnp.flip(mat, axis=0)
            _turn = _turn.at[board_num].set(1)
            _pocket = _pocket.at[board_num].set(_pocket[board_num][::-1])
        else:
            _turn = _turn.at[board_num].set(0)

        _en_passant = _en_passant.at[board_num].set(jnp.int32(-1) if en_passant == "-" else jnp.int32("abcdefgh".index(en_passant[0]) * 8 + int(en_passant[1]) - 1))
        if turn == "b" and _en_passant[board_num] >= 0:
            _en_passant = _en_passant.at[board_num].set(_flip_pos(_en_passant[board_num]))

        _board = _board.at[board_num].set(jnp.rot90(mat, k=3).flatten()) 
    
    state = State(  # type: ignore
        _board=_board,
        _turn=_turn,
        _can_castle_queen_side=_can_castle_queen_side,
        _can_castle_king_side=_can_castle_king_side,
        _en_passant=_en_passant,
        _halfmove_count=_halfmove_count,
        _fullmove_count=_fullmove_count,
        _pocket=_pocket,
        _clock=_clock,
    )
    state = state.replace(  # type: ignore
        legal_action_mask=jax.jit(_legal_action_mask)(state),
    )
    state = _mask_moves(state)
    state = state.replace(_zobrist_hash=state._zobrist_hash.at[0].set(_zobrist_hash(state, 0)))  # type: ignore
    state = state.replace(_zobrist_hash=state._zobrist_hash.at[1].set(_zobrist_hash(state, 1)))  # type: ignore
    state = _update_history(state, 0)
    state = _update_history(state, 1)
    state = jax.jit(_check_termination)(state)
    state = state.replace(observation=jax.jit(_observe)(state, state.current_player))  # type: ignore
    return state


def to_fen(state: State):

    """Convert state into FEN expression.

    - Board
        - Pawn:P Knight:N Bishop:B ROok:R Queen:Q King:K
        - The pice of th first player is capitalized
        - If empty, the number of consecutive spaces is inserted and shifted to the next piece. (e.g., P Empty Empty Empty R is P3R)
        - Starts from the upper left and looks to the right
        - When the row changes, insert /
    - Turn (w/b) comes after the board
    - Castling availability. K for King side, Q for Queen side. If both are not available, -
    - The place where en passant is possible. If the pawn moves 2 squares, record the position where the pawn passed
    - At last, the number of moves since the last pawn move or capture and the normal number of moves (fixed at 0 and 1 here)

    >>> s = State(_en_passant=jnp.int32(34))
    >>> _to_fen(s)
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq e3 0 1'
    >>> _to_fen(_from_fen("rnbqkbnr/pppppppp/8/8/8/P7/1PPPPPPP/RNBQKBNR b KQkq e3 0 1"))
    'rnbqkbnr/pppppppp/8/8/8/P7/1PPPPPPP/RNBQKBNR b KQkq e3 0 1'
    """

    def fn(board_num):
        pb = jnp.rot90(state._board[board_num].reshape(8, 8), k=1)
        if state._turn[board_num] == 1:
            pb = -jnp.flip(pb, axis=0)
        fen = ""
        # board
        for i in range(8):
            space_length = 0
            for j in range(8):
                piece = pb[i, j]
                if piece == 0:
                    space_length += 1
                elif space_length != 0:
                    fen += str(space_length)
                    space_length = 0
                if piece != 0:
                    if piece > 0:
                        fen += "PNBRQK"[piece - 1]
                    else:
                        fen += "pnbrqk"[-piece - 1]
            if space_length != 0:
                fen += str(space_length)
            if i != 7:
                fen += "/"
            else:
                fen += " "
        # turn
        fen += "w " if state._turn[board_num] == 0 else "b "
        # castling
        can_castle_queen_side = state._can_castle_queen_side[board_num]
        can_castle_king_side = state._can_castle_king_side[board_num]
        if state._turn[board_num] == 1:
            can_castle_queen_side = can_castle_queen_side[::-1]
            can_castle_king_side = can_castle_king_side[::-1]
        if not (can_castle_queen_side.any() | can_castle_king_side.any()):
            fen += "-"
        else:
            if can_castle_king_side[0]:
                fen += "K"
            if can_castle_queen_side[0]:
                fen += "Q"
            if can_castle_king_side[1]:
                fen += "k"
            if can_castle_queen_side[1]:
                fen += "q"
        fen += " "
        # en passant
        en_passant = state._en_passant[board_num]
        if state._turn[board_num] == 1:
            en_passant = _flip_pos(en_passant)
        ep = int(en_passant.item())
        if ep == -1:
            fen += "-"
        else:
            fen += "abcdefgh"[ep // 8]
            fen += str(ep % 8 + 1)
        fen += " "
        fen += str(state._halfmove_count[board_num].item())
        fen += " "
        fen += str(state._fullmove_count[board_num].item())
        return fen

    return fn(0) + "|" + fn(1)

def make_policy_labels(): 
    labels = [Action._from_label(i)._to_string() for i in range(9985)] 
    return labels
import math
import chess


def _move_dict_to_obj(move_dict):
    move_obj = chess.Move(
        from_square=chess.parse_square(move_dict["to_square"])
        if move_dict["from_square"] is None
        else chess.parse_square(move_dict["from_square"]),
        to_square=None
        if move_dict["to_square"] is None
        else chess.parse_square(move_dict["to_square"]),
        drop=None
        if move_dict["drop"] is None
        else chess.Piece.from_symbol(move_dict["drop"]).piece_type,
        promotion=None
        if move_dict["promotion"] is None
        else chess.Piece.from_symbol(move_dict["promotion"]).piece_type,
    )
    return move_obj


# tcn_decode and tcn_encode are 1:1 port of chess-tcn npm library that chess.com uses
def tcn_decode(n):
    tcn_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!?{~}(^)[_]@#$,./&-*++="
    piece_chars = "qnrbkp"
    w = len(n)
    c = []
    for i in range(0, w, 2):
        u = {
            "from_square": None,
            "to_square": None,
            "drop": None,
            "promotion": None,
        }
        o = tcn_chars.index(n[i])
        s = tcn_chars.index(n[i + 1])
        if s > 63:
            u["promotion"] = piece_chars[math.floor((s - 64) / 3)]
            s = o + (-8 if o < 16 else 8) + ((s - 1) % 3) - 1
        if o > 75:
            u["drop"] = piece_chars[o - 79]
        else:
            u["from_square"] = tcn_chars[o % 8] + str(math.floor(o / 8) + 1)
        u["to_square"] = tcn_chars[s % 8] + str(math.floor(s / 8) + 1)
        move = _move_dict_to_obj(u)
        c.append(move)
    return c


def tcn_encode(n):
    tcn_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!?{~}(^)[_]@#$,./&-*++="
    piece_chars = "qnrbkp"
    o = len(n)
    w = ""
    for i in range(o):
        if n[i]["drop"]:
            s = 79 + piece_chars.index(n[i]["drop"])
        else:
            s = tcn_chars.index(n[i]["from_square"][0]) + 8 * (
                int(n[i]["from_square"][1]) - 1
            )
        u = tcn_chars.index(n[i]["to_square"][0]) + 8 * (int(n[i]["to_square"][1]) - 1)
        if n[i]["promotion"]:
            add_u = 9 + u - s if u < s else u - s - 7
            u = 3 * piece_chars.index(n[i]["promotion"]) + 64 + add_u
        w += tcn_chars[s]
        w += tcn_chars[u]
    return w

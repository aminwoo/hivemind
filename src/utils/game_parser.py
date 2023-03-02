import chess
import numpy as np
from utils.tcn import tcn_decode
from utils.constants import BOARD_A, BOARD_B
from utils.bugboard import BughouseBoard


class Parser(object):

    def __init__(self, game):
        self.result = self.get_game_result(game)
        self.ratings = [
            [game["a"]["pgnHeaders"]["WhiteElo"],
             game["a"]["pgnHeaders"]["BlackElo"]],
            [game["b"]["pgnHeaders"]["WhiteElo"],
             game["b"]["pgnHeaders"]["BlackElo"]]
        ]
        self.time_control = self.get_game_time_control(game)
        if self.time_control not in [1800, 1200]:
            raise Exception('Bad time control!')
        if self.time_control <= 1800:
            moves = [
                self.get_board_moves(game["a"]),
                self.get_board_moves(game["b"]),
            ]
            times = [
                self.get_board_times(game["a"]),
                self.get_board_times(game["b"]),
            ]
            deltas = [
                self.get_board_deltas(times[0]),
                self.get_board_deltas(times[1]),
            ]
            move_order = self.get_naive_move_order(deltas)
            self.moves = self.verify_move_order(move_order, moves, times, deltas)

    def get_result(self, board_num, turn):
        if (board_num == BOARD_A and turn == chess.WHITE) or (board_num == BOARD_B and turn == chess.BLACK):
            return self.result
        return -self.result

    def get_rating(self, board_num, color):
        return self.ratings[board_num][not color]

    def get_board_deltas(self, board_times):
        white_times = board_times[::2]
        black_times = board_times[1::2]
        white_times.insert(0, self.time_control)
        black_times.insert(0, self.time_control)
        white_deltas = [a - b for a, b in zip(white_times[:-1], white_times[1:])]
        black_deltas = [a - b for a, b in zip(black_times[:-1], black_times[1:])]
        board_deltas = {
            chess.WHITE: white_deltas,
            chess.BLACK: black_deltas,
        }
        return board_deltas

    @staticmethod
    def get_game_result(game):
        winner_color = game["a"].get("colorOfWinner")
        if winner_color == "white":
            return 1
        elif winner_color == "black":
            return -1
        else:
            return 0

    @staticmethod
    def get_board_moves(board):
        tcn_moves = board["moveList"]
        board_moves = tcn_decode(tcn_moves)
        return board_moves

    @staticmethod
    def get_board_times(board):
        # Overwrites the first white and black move times to be equal to the time control
        board_times = [int(t) for t in board["moveTimestamps"].split(",")]
        return board_times

    @staticmethod
    def get_game_time_control(game):
        try:
            time_control = int(game["a"]["pgnHeaders"]["TimeControl"]) * 10
        except ValueError:
            time_control = 9999

        return time_control

    @staticmethod
    def get_naive_move_order(deltas):
        # get time_deltas
        a_deltas_b = np.array(deltas[BOARD_A][chess.BLACK])
        a_deltas_w = np.array(deltas[BOARD_A][chess.WHITE])
        b_deltas_b = np.array(deltas[BOARD_B][chess.BLACK])
        b_deltas_w = np.array(deltas[BOARD_B][chess.WHITE])

        # interleave_player_time_deltas
        a_deltas = np.empty(a_deltas_w.size + a_deltas_b.size)
        a_deltas[0::2] = a_deltas_w
        a_deltas[1::2] = a_deltas_b
        b_deltas = np.empty(b_deltas_w.size + b_deltas_b.size)
        b_deltas[0::2] = b_deltas_w
        b_deltas[1::2] = b_deltas_b

        # get accumulated player times
        a_times = np.cumsum(a_deltas)
        b_times = np.cumsum(b_deltas)

        all_times = np.concatenate((a_times, b_times))
        all_indices = np.argsort(all_times)
        move_order = np.digitize(all_indices, [0, a_times.shape[0]]) - 1
        return move_order

    @staticmethod
    def verify_move_order(move_order, moves, times, deltas):
        board = BughouseBoard()
        new_moves = []

        q = [i for i in range(len(move_order))]
        while q:
            stuck_board = -1
            for i in q:
                board_num = move_order[i]
                if moves[board_num]:
                    move = moves[board_num][0]
                else:
                    q.remove(i)
                    break
                pockets = board.boards[board_num].pockets
                turn = board.boards[board_num].turn
                time_left = times[board_num][0]
                move_time = deltas[board_num][turn][0]
                if board_num == stuck_board or (move.drop and pockets[turn].count(move.drop) <= 0):
                    stuck_board = board_num
                    continue
                else:
                    new_moves.append((board_num, move, time_left, move_time))
                    board.push(board_num, move)
                    moves[board_num].pop(0)
                    times[board_num].pop(0)
                    deltas[board_num][turn].pop(0)
                    q.remove(i)
                    break

        return new_moves


if __name__ == '__main__':
    parser = Parser({'a': {'canSendTrophy': False, 'changesPlayersRating': 1, 'colorOfWinner': 'black', 'id': 3044589951, 'uuid': '3904c616-cacc-11e3-8000-000000010001', 'initialSetup': '', 'isLiveGame': True, 'isAbortable': False, 'isAnalyzable': True, 'isCheckmate': True, 'isStalemate': False, 'isFinished': True, 'isRated': True, 'isResignable': False, 'lastMove': '*h', 'moveList': 'mC!Tbs0Kgv9IlBKBvBIBdB5QByZRfH6ZegWOHQZQyA-KAm=xoxQH+tHtkt+v-L*h', 'partnerGameId': 3044589950, 'plyCount': 32, 'ratingChangeWhite': -6, 'ratingChangeBlack': 6, 'resultMessage': 'AngelinaKali won by checkmate', 'endTime': 1535756817, 'turnColor': 'white', 'type': 'bughouse', 'typeName': 'Bughouse', 'allowVacation': False, 'pgnHeaders': {'Event': 'Live Chess - Bughouse', 'Site': 'Chess.com', 'Date': '2018.08.31', 'White': 'BughouseCoach', 'Black': 'AngelinaKali', 'Result': '0-1', 'ECO': 'C26', 'WhiteElo': 2196, 'BlackElo': 2227, 'TimeControl': '180', 'EndTime': '16:06:57 PDT', 'Termination': 'AngelinaKali won by checkmate', 'SetUp': '1', 'FEN': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', 'Variant': 'Bughouse'}, 'moveTimestamps': '1799,1799,1787,1797,1774,1796,1770,1782,1769,1777,1768,1769,1764,1760,1747,1751,1724,1741,1709,1729,1683,1566,1664,1539,1650,1326,1627,1233,1556,1214,1535,1149,1535', 'baseTime1': 1800, 'timeIncrement1': 0}, 'b': {'canSendTrophy': False, 'changesPlayersRating': 1, 'colorOfWinner': 'black', 'id': 5599745835, 'uuid': '5783ba8d-1118-11eb-bae7-c1085c010001', 'initialSetup': '', 'isLiveGame': True, 'isAbortable': False, 'isAnalyzable': True, 'isCheckmate': False, 'isStalemate': False, 'isFinished': True, 'isRated': True, 'isResignable': False, 'lastMove': 'EV', 'moveList': 'gvZJlB!TcD5Qmu6LfH0Sbs9zHQXQvK=ZK181-K19=V?!+1-C1!CsV292-N2!jszsefTNdN7TDMT2MV-lfg2V-0!?K1?2NV21pF-mgp+R=D-EpxEV', 'partnerGameId': 5599745836, 'plyCount': 56, 'ratingChangeWhite': -8, 'ratingChangeBlack': 8, 'resultMessage': 'agg69 won with their bughouse partner', 'endTime': 1603008367, 'turnColor': 'white', 'type': 'bughouse', 'typeName': 'Bughouse', 'allowVacation': False, 'pgnHeaders': {'Event': 'Live Chess - Bughouse', 'Site': 'Chess.com', 'Date': '2020.10.18', 'White': 'BiggerBishop', 'Black': 'agg69', 'Result': '0-1', 'ECO': 'D02', 'WhiteElo': 2022, 'BlackElo': 2152, 'TimeControl': '180', 'EndTime': '1:06:07 PDT', 'Termination': 'agg69 won with their bughouse partner', 'SetUp': '1', 'FEN': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', 'Variant': 'Bughouse'}, 'moveTimestamps': '1799,1794,1794,1786,1788,1774,1783,1766,1770,1754,1762,1744,1756,1730,1755,1719,1695,1691,1678,1684,1643,1666,1618,1625,1593,1616,1549,1563,1425,1547,1400,1526,1394,1516,1369,1501,1318,1482,1221,1441,1193,1276,1136,1262,1041,1248,1035,1245,805,1219,794,1199,769,1166,643,1160,627', 'baseTime1': 1800, 'timeIncrement1': 0}})
    board = BughouseBoard(parser.time_control)
    for board_num, move, time_left, move_time in parser.moves:
        board.update_time(board_num, time_left, move_time)

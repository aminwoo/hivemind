import datetime
import os

from src.domain.move2planes import mirrorMoveUCI
from src.domain.board import BughouseBoard

def write_bpgn(game_id, actions, times):
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=0)))
    headers = f'''[Event "engine bughouse game"]\n[Date "{now.strftime("%Y.%m.%d")}"]\n[Time "{now.strftime("%H:%M:%S")}"]\n[GameID "{game_id}"]\n[TimeControl "120+0"]\n\n'''
    game_str = '' 
    turns = [0, 0]
    ply = [0, 0] 
    num_actions = len(actions)

    board = BughouseBoard()
    for i in range(num_actions): 
        move_uci = actions[i]
        if move_uci == 'pass':
            continue

        board_num =  int(move_uci[0])
        move_uci = move_uci[1:]
        if turns[board_num] == 1:
            move_uci = mirrorMoveUCI(move_uci)

        move_san = board.to_san(board_num, move_uci)
        time_left = times[i][board_num][1]

        letter = 'A' if board_num == 0 else 'B'
        if ply[board_num] % 2:
            letter = letter.lower()
        game_str += f'{ply[board_num] // 2 + 1}{letter}. {move_san}' + '{' + str(time_left / 10) + '} '

        ply[board_num] += 1 
        turns[board_num] = 1 - turns[board_num]
        board.push_san(board_num, move_san)
        
    game_str += '{game end} ' + board.result() 

    with open(f'data/games/game_{game_id}.bpgn', 'w') as f:
        f.write(headers + game_str)


#actions = ['0e2e4', '1d2d4']
#times = [[[[1200, 1195], [1195, 1200]]], [[1198, 1195],[1200, 1193]]]
#write_bpgn(0, actions, times)
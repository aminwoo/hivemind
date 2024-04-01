import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from functools import partial
import json 
import time
import numpy as np 
import jax
import jax.numpy as jnp
import multiprocessing
import websocket
import chess 

from src.training.tcn import tcn_encode, tcn_decode
from src.types import POLICY_LABELS, BOARD_HEIGHT, BOARD_WIDTH, NUM_BUGHOUSE_CHANNELS
from src.domain.move2planes import mirrorMoveUCI
from src.domain.board2planes import board2planes
from src.domain.board import BughouseBoard
from src.architectures.azresnet import AZResnet, AZResnetConfig
from src.training.trainer import TrainerModule
from src.mcts.search import UCT_search

usernames = ['pumpkinspicedream', 'pumpkinspicefever']
phpsessids = ['44269139dadd51772f2400320d88d7ac', '793ba2c8778d467b542271659e1ab184']

trainer = TrainerModule(model_name='AZResNet', model_class=AZResnet, model_configs=AZResnetConfig(
    num_blocks=15,
    channels=256,
    policy_channels=4, 
    value_channels=8,
    num_policy_labels=len(POLICY_LABELS)
), optimizer_name='lion', optimizer_params={'learning_rate': 0.00001}, x=jnp.ones((1, BOARD_HEIGHT, 2 * BOARD_WIDTH, NUM_BUGHOUSE_CHANNELS)))
state = trainer.load_checkpoint('5')

model = AZResnet(
            AZResnetConfig(
                num_blocks=15,
                channels=256,
                policy_channels=4,
                value_channels=8,
                num_policy_labels=2185,
            )
        )

variables = {'params': state['params'], 'batch_stats': state['batch_stats']}
eval_fn = jax.jit(partial(model.apply, variables, train=False))
        
class Client: 

    def __init__(self, phpsessid: str, username: str, board_num: int):
        self.phpsessid = phpsessid
        self.username = username
        self.board_num = board_num
        self.clientId = ''
        self.ply = 0
        self.gameId = 0
        self.side = False
        self.id = 1
        self.ack = 1
        self.moves = ['', '']
        self.times = [[1200, 1200], [1200, 1200]]
        self.bot = None
        self.board = BughouseBoard() 
        self.playing = False
        self.times = [[1200, 1200], [1200, 1200]]
        self.moves = ['', ''] 

    def get_engine_move(self): 
        print(self.board.times)
        print(self.board.fen())
        team_side = self.side if self.board_num == 0 else not self.side
        move, _ = UCT_search(self.board, eval_fn, team_side, iterations=10)
        return move[self.board_num]
            
        '''planes = board2planes(self.board, self.side if self.board_num == 0 else not self.side)[None,]
        start = time.time()
        policy_logits, _ = forward(planes)
        print(time.time() - start)

        double_sit = np.argmax(policy_logits[0]) == 0 and np.argmax(policy_logits[1]) == 0
        for i in np.argsort(-policy_logits[self.board_num])[0]:
            if i == 0:
                if not double_sit:
                    return None
                else:
                    continue
            
            uci_move = POLICY_LABELS[i]
            if self.board.turn(self.board_num) == chess.BLACK:
                uci_move = mirrorMoveUCI(uci_move)

            move = chess.Move.from_uci(uci_move)
            if move in self.board.boards[self.board_num].legal_moves:
                return uci_move'''

        
    def seek_game(self, ws):
        data = [
            {
                'channel': '/service/game',
                'data': {
                    'tid': 'Challenge',
                    'uuid': '',
                    'to': None,
                    'from': self.username,
                    'gametype': 'bughouse',
                    'initpos': None,
                    'rated': False,
                    'minrating': None,
                    'maxrating': None,
                    'basetime': 1200,
                    'timeinc': 0
                },
                'id': self.id,
                'clientId': self.clientId,
            },
        ]
        ws.send(json.dumps(data))
        self.id += 1

    def send_move(self, ws, move): 
        data = [
            {
                'channel': '/service/game',
                'data': {
                    'move': {
                        'gid': self.gameId, 
                        'move': move,
                        'seq': self.ply,
                        'uid': self.username,
                    },
                    'tid': 'Move',
                },
                'id': self.id,
                'clientId': self.clientId,
            },
        ]
        ws.send(json.dumps(data))
        self.id += 1
        
    def connect(self): 
        ws = websocket.WebSocket()
        ws.connect('wss://live2.chess.com/cometd', cookie=f'PHPSESSID={self.phpsessid}') 
        data = [
            {
                'version':'1.0',
                'minimumVersion':'1.0',
                'channel':'/meta/handshake',
                'supportedConnectionTypes':['ssl-websocket'],
                'advice':{'timeout':60000,'interval':0},
                'clientFeatures':{
                    'protocolversion':'2.1',
                    'clientname':'LC6;chrome/121.0.6167/browser;Windows 10;jxk3sm4;78.0.2',
                    'skiphandshakeratings':True,
                    'adminservice':True,
                    'announceservice':True,
                    'arenas':True,
                    'chessgroups':True,
                    'clientstate':True,
                    'events':True,
                    'gameobserve':True,
                    'genericchatsupport':True,
                    'genericgamesupport':True,
                    'guessthemove':True,
                    'multiplegames':True,
                    'multiplegamesobserve':True,
                    'offlinechallenges':True,
                    'pingservice':True,
                    'playbughouse':True,
                    'playchess':True,
                    'playchess960':True,
                    'playcrazyhouse':True,
                    'playkingofthehill':True,
                    'playoddschess':True,
                    'playthreecheck':True,
                    'privatechats':True,
                    'stillthere':True,
                    'teammatches':True,
                    'tournaments':True,
                    'userservice':True},
                'serviceChannels':['/service/user'],
                'ext':{
                    'ack':True,
                    'timesync':{'tc':int(time.time()*1000),'l':0,'o':0}
                },
                'id':self.id,
                'clientId':None
            }
        ]
        ws.send(json.dumps(data))
        self.id += 1

        while True:
            message = json.loads(ws.recv())[0] 
            #print(message)

            if 'clientId' in message:
                self.clientId = message['clientId']
                self.seek_game(ws)

            if (message['channel'] == '/meta/connect' or message['channel'] == '/meta/handshake') and message['successful']:
                if message['channel'] == '/meta/connect':
                    self.ack = message['ext']['ack'] 

                data = [{'channel':'/meta/connect','connectionType':'ssl-websocket','ext':{'ack':self.ack,'timesync':{'tc':int(time.time()*1000),'l':0,'o':0}},'id':self.id,'clientId':self.clientId}]
                ws.send(json.dumps(data))
                self.id += 1

            if 'data' in message and 'game' in message['data'] and 'status' in message['data']['game']:
                if message['data']['game']['status'] == 'finished':
                    if self.playing:
                        self.playing = False
                        self.board.reset() 
                        self.times = [[1200, 1200], [1200, 1200]]
                        self.seek_game(ws)
                else:
                    if message['data']['game']['status'] == 'starting':
                        self.playing = True
                    
                    players = message['data']['game']['players']
                    user_index = -1 
                    for i in range(len(players)):
                        if players[i]['uid'].lower() == self.username.lower():
                            user_index = i 
                            break

                    times = message['data']['game']['clocks'][::-1]
                    tcn_moves = message['data']['game']['moves']
                    move = '' if not tcn_moves else tcn_decode(tcn_moves[-2:])[0]
                    if user_index != -1:
                        self.gameId = message['data']['game']['id']
                        self.ply = message['data']['game']['seq']
                        self.side = user_index == 0

                        delta = 0
                        for i in range(2):
                            delta = max(delta, self.times[self.board_num][i] - times[i])
                        self.times[self.board_num] = times
                        self.times[1 - self.board_num][self.board.turn(1 - self.board_num)] -= delta 

                        turn = self.ply % 2 == 0

                        if move and self.moves[self.board_num] != tcn_moves and turn == self.side:
                            self.moves[self.board_num] = tcn_moves
                            self.board.push(self.board_num, move)

                        time_advantage = self.board.time_advantage(self.side) if self.board_num == 0 else self.board.time_advantage(not self.side)
                        #time_advantage = 0
                        if self.board.turn(self.board_num) == self.side and not (self.board.turn(1 - self.board_num) == self.side and time_advantage < -10):
                            self.board.set_times(self.times)
                            bestmove = self.get_engine_move()
                            if bestmove:
                                #print("bestmove", bestmove, self.ply)
                                self.send_move(ws, tcn_encode([bestmove.uci()])); 
                                self.board.push(self.board_num, bestmove)

                    else:
                        delta = 0
                        for i in range(2):
                            delta = max(delta, self.times[1 - self.board_num][i] - times[i])
                        self.times[1 - self.board_num] = times
                        self.times[self.board_num][self.board.turn(self.board_num)] -= delta 
                    
                        if move and self.moves[1 - self.board_num] != tcn_moves:
                            self.moves[1 - self.board_num] = tcn_moves
                            self.board.push(1 - self.board_num, move)

                        time_advantage = self.board.time_advantage(self.side) if self.board_num == 0 else self.board.time_advantage(not self.side)
                        #time_advantage = 0
                        if self.board.turn(self.board_num) == self.side and not (self.board.turn(1 - self.board_num) == self.side and time_advantage < -10):
                            self.board.set_times(self.times)
                            bestmove = self.get_engine_move()
                            if bestmove:
                                #print("bestmove", bestmove, self.ply)
                                self.send_move(ws, tcn_encode([bestmove.uci()])); 
                                self.board.push(self.board_num, bestmove)

def main():
    client1 = Client(phpsessids[0], usernames[0], 0)
    client2 = Client(phpsessids[1], usernames[1], 1)
    clients = [client1, client2]

    for client in clients:
        multiprocessing.get_context('spawn').Process(target=client.connect).start()

if __name__ == '__main__':
    main()
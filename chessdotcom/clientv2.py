import asyncio
import json
import random
import time

import jax
import jax.numpy as jnp
import typer
import websockets
import yaml


from pgx.bughouse import (Action, Bughouse, _set_clock,
                          _set_current_player, _time_advantage, _is_promotion)
from chessdotcom.constants import labels

from chessdotcom.auth import get_session_key
from src.domain.move2planes import mirrorMoveUCI
from src.mcts.search import search
from src.utils.tcn import tcn_decode, tcn_encode

#########################################################
### For testing purposes only (use at your own risk!) ###
#########################################################

seed = 42
env = Bughouse()
init_fn = jax.jit(jax.vmap(env.init))
step_fn = jax.jit(jax.vmap(env.step))

update_clock = jax.jit(jax.vmap(_set_clock))
update_player = jax.jit(jax.vmap(_set_current_player))
time_advantage = jax.jit(jax.vmap(_time_advantage))
is_promotion = jax.jit(jax.vmap(_is_promotion))

ping = random.randint(11, 69)


class Client:
    """
    Client class to play an account
    """
    def __init__(self, config) -> None:
        self.config = config
        self.phpsessid = get_session_key(config["login"]["username"], config["login"]["password"])
        self.username = config["login"]["username"]
        self.partner = config["search"]["partner"]
        self.opponent = config["search"]["opponent"]
        self.board_num = config["engine"]["board"]
        self.clientId = ""
        self.ply = 0
        self.gameId = -1
        self.side = -1
        self.id = 1
        self.ack = 1
        self.playing = False
        self.state = None
        self.lengths = [0, 0]
        self.basetime = self.config["search"]["basetime"]
        self.times = [[self.basetime, self.basetime], [self.basetime, self.basetime]]
        self.turn = [0, 0]
        self.key = jax.random.PRNGKey(seed)
        self.thinking = False
        self.new_game()

    def new_game(self) -> None:
        self.key, subkey = jax.random.split(self.key)
        keys = jax.random.split(subkey, 1)
        self.state = init_fn(keys)
        step_fn(self.state, jnp.int32([0]), keys)
        self.lengths = [0, 0]
        self.times = [[self.basetime, self.basetime], [self.basetime, self.basetime]]
        self.turn = [0, 0]

    async def play_move(self, board_num: int, move: str, ws=None) -> None:
        move_uci = move
        if self.turn[board_num] == 1:
            move_uci = mirrorMoveUCI(move_uci)
        move_uci = str(board_num) + move_uci
        if move_uci.endswith("q"): 
            move_uci = move_uci[:-1]
        action = labels.index(move_uci)

        self.state = update_player(self.state, jnp.int32([self.turn[board_num]]) if board_num == 0 else jnp.int32([1 - self.turn[board_num]]))
        if self.state.legal_action_mask[0][action]:
            if ws:
                await self.send_move(ws, tcn_encode([move]))

            self.key, subkey = jax.random.split(self.key)
            keys = jax.random.split(subkey, 1)
            self.state = step_fn(self.state, jnp.int32([action]), keys)
            self.turn[board_num] = 1 - self.turn[board_num]
            print(action, "Move played:", move, "on board", board_num)

    async def send_partnership(self, ws) -> None:
        data = [
            {
                "channel": "/service/game",
                "data": {
                    "tid": "RequestBughousePair",
                    "to": self.partner,
                    "from": self.username,
                },
                "id": self.id,
                "clientId": self.clientId,
            },
        ]
        await ws.send(json.dumps(data))
        self.id += 1

    async def rematch(self, ws) -> None:
        data = [
            {
                "channel": "/service/game",
                "data": {
                    "tid": "Challenge",
                    "uuid": "",
                    "to": self.opponent,
                    "from": self.username,
                    "gametype": "bughouse",
                    "initpos": None,
                    "rated": True,
                    "minrating": self.config["search"]["minrating"],
                    "maxrating": self.config["search"]["maxrating"],
                    "rematchgid": self.gameId,
                    "color": 2 if self.side == 0 else 1,
                    "basetime": self.basetime,
                    "timeinc": 0
                },
                "id": self.id,
                "clientId": self.clientId,
            },
        ]
        await ws.send(json.dumps(data))
        self.id += 1

    async def seek_game(self, ws) -> None:
        data = [
            {
                "channel": "/service/game",
                "data": {
                    "tid": "Challenge",
                    "uuid": "",
                    "to": None,
                    "from": self.username,
                    "gametype": "bughouse",
                    "initpos": None,
                    "rated": self.config["search"]["rated"],
                    "minrating": self.config["search"]["minrating"],
                    "maxrating": self.config["search"]["maxrating"],
                    "basetime": self.basetime,
                    "timeinc": 0
                },
                "id": self.id,
                "clientId": self.clientId,
            },
        ]
        await ws.send(json.dumps(data))
        self.id += 1

    async def send_move(self, ws, move: str) -> None:
        data = [
            {
                "channel": "/service/game",
                "data": {
                    "move": {
                        "gid": self.gameId,
                        "move": move,
                        "seq": self.ply,
                        "uid": self.username,
                    },
                    "tid": "Move",
                },
                "id": self.id,
                "clientId": self.clientId,
            },
        ]
        await ws.send(json.dumps(data))
        self.id += 1

    def update_clock(self, board_num, times):
        delta = max(self.times[board_num][i] - times[i] for i in range(2))
        self.times[1 - board_num][self.turn[1 - board_num]] -= delta
        self.times[board_num] = times

    def update_clock_and_player(self):
        self.state = update_player(self.state, jnp.int32([self.turn[self.board_num]]) if self.board_num == 0 else jnp.int32([1 - self.turn[self.board_num]]))
        t = self.times.copy()
        if self.turn[0] != 0:
            t[0] = t[0][::-1]
        if self.turn[1] != 0:
            t[1] = t[1][::-1]
            
        self.state = update_clock(self.state, jnp.int32([t]))

    async def start(self) -> None:
        self.update_clock_and_player()
        time_advantage(self.state)
        update_player(self.state, jnp.int32([0]))
        is_promotion(self.state, jnp.int32([0]))
        search(self.state, self.config["engine"]["iterations"])

        async with websockets.connect("wss://live2.chess.com/cometd", extra_headers=[("Cookie", f"PHPSESSID={self.phpsessid}")]) as ws:
            data = [
                {
                    "version":"1.0",
                    "minimumVersion":"1.0",
                    "channel":"/meta/handshake",
                    "supportedConnectionTypes":["ssl-websocket"],
                    "advice":{"timeout":60000,"interval":0},
                    "clientFeatures":{
                        "protocolversion":"2.1",
                        "clientname":"LC6;chrome/121.0.6167/browser;Windows 10;jxk3sm4;78.0.2",
                        "skiphandshakeratings":True,
                        "adminservice":True,
                        "announceservice":True,
                        "arenas":True,
                        "chessgroups":True,
                        "clientstate":True,
                        "events":True,
                        "gameobserve":True,
                        "genericchatsupport":True,
                        "genericgamesupport":True,
                        "guessthemove":True,
                        "multiplegames":True,
                        "multiplegamesobserve":True,
                        "offlinechallenges":True,
                        "pingservice":True,
                        "playbughouse":True,
                        "playchess":True,
                        "playchess960":True,
                        "playcrazyhouse":True,
                        "playkingofthehill":True,
                        "playoddschess":True,
                        "playthreecheck":True,
                        "privatechats":True,
                        "stillthere":True,
                        "teammatches":True,
                        "tournaments":True,
                        "userservice":True},
                    "serviceChannels":["/service/user"],
                    "ext":{
                        "ack":True,
                        "timesync":{"tc":int(time.time()*1000),"l":ping,"o":0}
                    },
                    "id":self.id,
                    "clientId":None
                }
            ]
            await ws.send(json.dumps(data))
            self.id += 1

            async for message in ws:
                message = json.loads(message)[0]
                asyncio.create_task(self.handle_message(ws, message))

    async def handle_message(self, ws, message: str) -> None:
        if "clientId" in message:
            self.clientId = message["clientId"]
            if self.partner:
                await self.send_partnership(ws)
            await self.seek_game(ws)
            
        if "data" in message and "tid" in message["data"] and message["data"]["tid"] == "RequestBughousePair" and "from" in message["data"]:
            await self.send_partnership(ws)
            
        if "data" in message and "tid" in message["data"] and message["data"]["tid"] == "BughousePair":
            print(f"Partnered to {self.partner}")

        if (message["channel"] == "/meta/connect" or message["channel"] == "/meta/handshake") and message["successful"]:
            if message["channel"] == "/meta/connect":
                self.ack = message["ext"]["ack"]
            data = [{"channel":"/meta/connect","connectionType":"ssl-websocket","ext":{"ack":self.ack,"timesync":{"tc":int(time.time()*1000),"l":ping,"o":0}},"id":self.id,"clientId":self.clientId}]
            await ws.send(json.dumps(data))
            self.id += 1
            
        if "data" in message and "game" in message["data"] and "status" in message["data"]["game"]:
            if message["data"]["game"]["status"] == "finished":
                if self.playing:
                    print("Game ended")
                    self.playing = False
                    self.new_game()
                    #await self.rematch(ws)
                    time.sleep(2)
                    await self.seek_game(ws)
            else:
                if message["data"]["game"]["status"] == "starting":
                    self.playing = True

                players = message["data"]["game"]["players"]
                user_index = -1
                for i in range(len(players)):
                    if players[i]["uid"].lower() == self.username.lower():
                        user_index = i
                        break

                times = message["data"]["game"]["clocks"]
                tcn_moves = message["data"]["game"]["moves"]
                move = "" if not tcn_moves else tcn_decode(tcn_moves[-2:])[0].uci()
                if user_index != -1:
                    self.gameId = message["data"]["game"]["id"]
                    self.ply = message["data"]["game"]["seq"]
                    self.side = user_index

                    self.update_clock(self.board_num, times)

                    if move and self.lengths[self.board_num] < len(tcn_moves) and self.turn[self.board_num] != self.side:
                        self.lengths[self.board_num] = len(tcn_moves)
                        await self.play_move(self.board_num, move)
                else:
                    self.update_clock(1 - self.board_num, times)

                    if move and self.lengths[1 - self.board_num] < len(tcn_moves):
                        self.lengths[1 - self.board_num] = len(tcn_moves)
                        await self.play_move(1 - self.board_num, move)

                if not self.thinking and self.turn[self.board_num] == self.side and ~self.state.terminated.any():
                    self.thinking = True
                    self.update_clock_and_player()

                    out = search(self.state, self.config["engine"]["iterations"])
                    action = out.action
                    
                    move_uci = labels[action[0]]
                    if len(move_uci) < 6 and is_promotion(self.state, action)[0]:
                        move_uci += "q"
                    print(move_uci)
                    
                    if move_uci != "pass" and int(move_uci[0]) == self.board_num:
                        move_uci = move_uci[1:]
                        if self.turn[self.board_num] == 1:
                            move_uci = mirrorMoveUCI(move_uci)
                        await self.play_move(self.board_num, move_uci, ws)
                    self.thinking = False



with open("chessdotcom/config.yaml", "r") as file:
    config = yaml.safe_load(file)

print("Logging into " + config["login"]["username"])
client = Client(config)
asyncio.run(client.start())

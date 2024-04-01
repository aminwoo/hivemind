import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import math
import chess
from time import time
from copy import copy, deepcopy

from src.architectures.azresnet import AZResnet, AZResnetConfig
from src.domain.board2planes import board2planes
from src.domain.board import BughouseBoard
from src.types import POLICY_LABELS
from functools import partial
from src.training.trainer import TrainerModule
from src.domain.move2planes import mirrorMoveUCI


import chex
import jax
import jax.numpy as jnp

FPU = -1.0
FPU_ROOT = 0.0

class UCTNode():
    def __init__(self, board=None, team_on_turn=True, parent=None, move=None, prior=0):
        self.board = board
        self.team_on_turn = team_on_turn
        self.move = move
        self.is_expanded = False
        self.parent = parent 
        self.children_visits = []
        self.children_priors = []
        self.prior = prior  
        if parent == None:
            self.total_value = FPU_ROOT  
        else:
            self.total_value = FPU
        self.number_visits = 0  

    def Q(self):  
        return self.total_value / (1 + self.number_visits)

    def U(self): 
        return (math.sqrt(self.parent.number_visits)
                * self.prior / (1 + self.number_visits))

    def best_child(self, C):

        return jnp.argmax(self.children,
                   key=lambda node: node.Q() + C*node.U())

    def select_leaf(self, C):
        start = time() 
        current = self
        while current.is_expanded and current.children:
            current = current.best_child(C)
        #print(time() - start)

        if not current.board:
            start = time() 
            current.board = deepcopy(current.parent.board)
            #print(time() - start)
            move = current.move
            if move[0]:
                current.board.push(0, move[0])
            if move[1]:
                current.board.push(1, move[1])

        return current

    def expand(self, child_priors):
        priors = [jax.nn.softmax(child_priors[0][0]), jax.nn.softmax(child_priors[1][0])]
        if self.board.turn(0) == self.team_on_turn:
            for move in self.board.boards[0].legal_moves:
                move_uci = move.uci()
                if self.board.turn(0) == chess.BLACK:
                    move_uci = mirrorMoveUCI(move_uci)
                prior = priors[0][POLICY_LABELS.index(move_uci)]
                self.add_child((move, None), prior)


        if self.board.turn(1) != self.team_on_turn:
            for move in self.board.boards[1].legal_moves:
                move_uci = move.uci()
                if self.board.turn(1) == chess.BLACK:
                    move_uci = mirrorMoveUCI(move_uci)
            
                prior = priors[1][POLICY_LABELS.index(move_uci)]
                self.add_child((None, move), prior)
        
        self.is_expanded = True

    def add_child(self, move, prior):
        self.children.append(UCTNode(parent=self, move=move, prior=prior, team_on_turn=not self.team_on_turn))

    def backup(self, value_estimate: float):
        current = self
        turnfactor = -1
        while current.parent is not None:
            current.number_visits += 1
            current.total_value += (value_estimate *
                                    turnfactor)
            current = current.parent
            turnfactor *= -1
        current.number_visits += 1

def get_best_move(root):
    node = max(root.children, key=lambda node: (node.number_visits, node.Q()))
    return node.move, node.Q()

def UCT_search(board, eval_fn, team_on_turn, iterations=500, C=np.sqrt(2)):
    root = UCTNode(board, team_on_turn=team_on_turn)
    for _ in range(iterations):
        start = time() 
        leaf = root.select_leaf(C)
        print(time() - start)
        
        planes = board2planes(leaf.board, leaf.team_on_turn)[None,]
        child_priors, value_estimate = eval_fn(planes)

        if leaf.board.is_checkmate():
            value_estimate = -1
        else:
            leaf.expand(child_priors)
        leaf.backup(value_estimate)
        

    return get_best_move(root)

if __name__ == '__main__':
    import jax
    import jax.numpy as jnp
    from src.architectures.azresnet import AZResnet, AZResnetConfig
    from src.domain.board2planes import board2planes
    from src.domain.board import BughouseBoard
    from src.types import POLICY_LABELS
    from functools import partial
    from src.training.trainer import TrainerModule

    trainer = TrainerModule(model_name='AZResNet', model_class=AZResnet, model_configs=AZResnetConfig(
        num_blocks=15,
        channels=256,
        policy_channels=4, 
        value_channels=8,
        num_policy_labels=len(POLICY_LABELS)
    ), optimizer_name='lion', optimizer_params={'learning_rate': 0.00001}, x=jnp.ones((1, 8, 16, 32)))
    state = trainer.load_checkpoint('2')
    variables = {'params': state['params'], 'batch_stats': state['batch_stats']}
    model = AZResnet(
        AZResnetConfig(
            num_blocks=15,
            channels=256,
            policy_channels=4,
            value_channels=8,
            num_policy_labels=len(POLICY_LABELS),
        )
    )
    eval_fn = jax.jit(partial(model.apply, variables, train=False))
    board = BughouseBoard()

    board.set_times([[978, 1016], [993, 1001]])
    board.set_fen('r1bq1b1r/ppp1k1pp/3npp2/4N1B1/3Q4/2N2N2/PPP2KPP/R6R[QNPPPqnb] w - - 34 18 | r1b1k2r/ppp2ppp/2p1p3/6B1/B2nn3/2P1P3/P4PPP/R1B1K2R[Ppp] b kq - 37 19')

    move, score = UCT_search(board, eval_fn, True) 
    print(move, score)
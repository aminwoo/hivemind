import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import chess
import numpy as np 
import jax
import jax.numpy as jnp
from src.domain.board import BughouseBoard
from src.architectures.azresnet import AZResnet, AZResnetConfig
from src.domain.board2planes import board2planes
from src.types import POLICY_LABELS
from src.domain.move2planes import mirrorMoveUCI
from src.training.trainer import TrainerModule
from src.types import BOARD_HEIGHT, BOARD_WIDTH, NUM_BUGHOUSE_CHANNELS
from src.training.tcn import tcn_decode



trainer = TrainerModule(model_name='AZResNet', model_class=AZResnet, model_configs=AZResnetConfig(
    num_blocks=15,
    channels=256,
    policy_channels=4, 
    value_channels=8,
    num_policy_labels=len(POLICY_LABELS)
), optimizer_name='lion', optimizer_params={'learning_rate': 0.00001}, x=jnp.ones((1, BOARD_HEIGHT, 2 * BOARD_WIDTH, NUM_BUGHOUSE_CHANNELS)))
state = trainer.load_checkpoint('0')
variables = {'params': state['params'], 'batch_stats': state['batch_stats']}

def main():
    model = AZResnet(
        AZResnetConfig(
            num_blocks=15,
            channels=256,
            policy_channels=4,
            value_channels=8,
            num_policy_labels=2185,
        )
    )

    fen = 'r6k/ppp2pNN/2np4/4p3/2B1P1b1/3P4/PPP3PP/R2nK2R[NNPbbbppp] w KQ - 1 16 | r3k2r/ppp1qppp/2n3r1/3p1P2/3Pn3/2P2BB1/P1P1QPKP/R1B3R1[QPq] b kq - 4 17'
    board = BughouseBoard()
    #board.push(0, tcn_decode('mC')[0])
    board.set_fen(fen)
    board.set_times([[851, 1004], [920, 936]])
    #print(board.time_advantage(True))
    #print(board.times)

    x=jnp.ones((16, BOARD_HEIGHT, 2 * BOARD_WIDTH, NUM_BUGHOUSE_CHANNELS))

    import optax
    planes = board2planes(board, True)[None,]


    import time 
    forward = jax.jit(lambda p, x: model.apply(p, x, False))
    for i in range(100):
        start = time.time() 
        forward(variables, x)
    #policy, value = model.apply(variables, x, train=False)
        print(time.time() - start)
    a = (np.argmax(policy[0], axis=1) == 0)
    b = (np.argmax(policy[1], axis=1) == 0)
    a = a.at[0].set(True) 
    #b = b.at[0].set(True)
    #a = a.at[1].set(True) 
    b = b.at[1].set(True)
    c = a & b
    #policy = 0
    loss = optax.softmax_cross_entropy_with_integer_labels(
                    logits=policy[0], labels=np.array([1, 2, 3, 4])).mean()
                
    #print(loss)
    #loss = 10
    #print(loss)
    #print(np.sum(c))
    #print(policy[0].shape)
    #print(value)
    moves = [None, None] 
    for i in np.argsort(-policy[0])[0]:
        if i == 0:
            break
        uci_move = POLICY_LABELS[i]
        if board.get_turn(0) == chess.BLACK:
            uci_move = mirrorMoveUCI(uci_move)

        move = chess.Move.from_uci(uci_move)
        if move in board.boards[0].legal_moves:
            moves[0] = uci_move
            break
    for i in np.argsort(-policy[1])[0]:
        if i == 0:
            break
        uci_move = POLICY_LABELS[i]
        if board.get_turn(1) == chess.BLACK:
            uci_move = mirrorMoveUCI(uci_move)

        move = chess.Move.from_uci(uci_move)
        if move in board.boards[1].legal_moves:
            moves[1] = uci_move
            break
    #print(moves)

    '''cmd = ""
    while cmd != "quit":
        cmd = input()
        if cmd == "uci":
            print("uciok")
        if cmd == "ucinewgame":
            pass
        if cmd == "isready":
            print("readyok")
        if cmd == "stop":
            pass
        if cmd.startswith("position"):
            args = cmd.split(" ")[1:]
            if args[0] == "fen":
                board.set_fen(" ".join(args[1:]))

        if cmd.startswith("go"):
            planes = board2planes(board, True)[None,]
            policy, value = model.apply(variables, planes, train=False)
            moves = [None, None] 
            for i in np.argsort(-policy[0])[0]:
                uci_move = POLICY_LABELS[i]
                if board.get_turn(0) == chess.BLACK:
                    uci_move = mirrorMoveUCI(uci_move)

                move = chess.Move.from_uci(uci_move)
                if move in board.boards[0].legal_moves:
                    moves[0] = uci_move
                    break
            for i in np.argsort(-policy[1])[0]:
                uci_move = POLICY_LABELS[i]
                if board.get_turn(1) == chess.BLACK:
                    uci_move = mirrorMoveUCI(uci_move)

                move = chess.Move.from_uci(uci_move)
                if move in board.boards[1].legal_moves:
                    moves[1] = uci_move
                    break
            print(moves)'''


if __name__ == "__main__":
    main()
import chess
import numpy as np 
import jax
import jax.numpy as jnp
from board import BughouseBoard
from architectures.azresnet import AZResnet, AZResnetConfig
from board2planes import board2planes
from constants import POLICY_LABELS
from move2planes import mirrorMoveUCI


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
    x = jnp.ones((1, 8, 16, 32))
    

    variables = model.init(jax.random.key(0), x, train=False)

    board = BughouseBoard()

    cmd = ""
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
            for i in np.argsort(-policy[0])[0]:
                uci_move = POLICY_LABELS[i]
                if board.get_turn(0) == chess.BLACK:
                    uci_move = mirrorMoveUCI(uci_move)

                move = chess.Move.from_uci(uci_move)
                if move in board.boards[0].legal_moves:
                    print(move)
                    break
            for i in np.argsort(-policy[1])[0]:
                uci_move = POLICY_LABELS[i]
                if board.get_turn(1) == chess.BLACK:
                    uci_move = mirrorMoveUCI(uci_move)

                move = chess.Move.from_uci(uci_move)
                if move in board.boards[1].legal_moves:
                    print(move)
                    break



if __name__ == "__main__":
    main()
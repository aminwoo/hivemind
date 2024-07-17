import chess
import typer


def main():
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
            pass
        if cmd.startswith("go"):
            pass


if __name__ == "__main__":
    main()

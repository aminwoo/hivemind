import chess.pgn
import gzip
import json 
import bz2

games = []
with bz2.open('data/fics/export2005.bpgn.bz2', 'rt') as f:
    print(f.readlines()[:100])
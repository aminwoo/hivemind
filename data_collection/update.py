import glob
import os
import json
from tqdm import tqdm


def get_leaderboard():
    in_file_path = '../data/leaderboard.txt'
    in_file = open(in_file_path, 'r')
    players = in_file.readlines()
    players = [p.strip() for p in players]
    in_file.close()
    return players


blacklist = []
with open('../data/blacklist.txt') as f:
    for line in f:
        blacklist.append(line.rstrip('\n'))
print(blacklist)

players = get_leaderboard()
players = list(set(players))
min_rating = 2300
player_archive_base_glob = './data/archives/{0}/*.json'

player_candidates = dict()

num_games = 0
for player in tqdm(players):
    player_archive_glob = player_archive_base_glob.format(player)
    player_archive_paths = [os.path.normpath(p).replace(os.sep, '/') for p in glob.glob(player_archive_glob)]
    for a_path in player_archive_paths:
        a_file = open(a_path, 'rb')
        a_json = json.load(a_file)
        a_file.close()
        if 'games' not in a_json:
            continue
        raw_games = a_json['games']
        for raw_game in raw_games:
            if raw_game['rules'] == 'bughouse':
                if 'tcn' not in raw_game:
                    continue
                tcn_moves = raw_game['tcn'].encode('utf-8')
                num_half_moves = len(tcn_moves) / 2
                if num_half_moves >= 10:
                    num_games += 1
                w_rating = raw_game['white']['rating']
                b_rating = raw_game['black']['rating']
                w_name = raw_game['white']['username']
                b_name = raw_game['black']['username']
                if w_rating >= min_rating:
                    player_candidates[w_name] = player_candidates.get(w_name, 0) + 1
                if b_rating >= min_rating:
                    player_candidates[b_name] = player_candidates.get(b_name, 0) + 1

good_players = []
cnt = 0
for p in player_candidates:
    if player_candidates[p] > 100 and p not in blacklist:
        cnt += player_candidates[p]
        #print(p)
        good_players.append(p)
print(cnt)

out_file_path = '../data/leaderboard.txt'
out_file = open(out_file_path, 'w')
for player in good_players:
    out_file.write(player + '\n')
out_file.close()

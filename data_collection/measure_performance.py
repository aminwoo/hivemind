import glob
import os
import json
import glicko2
import pandas as pd
import time
import ast
import random

from tqdm import tqdm

from archive import get_leaderboard


def create_df():
    players = get_leaderboard()
    players = list(set(players))
    player_archive_base_glob = './data/archives/{0}/*.json'
    player_candidates = dict()

    num_games = 0

    data = []
    for player in tqdm(players):
        game_results = []
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
                    else:
                        continue
                    w_rating = raw_game['white']['rating']
                    b_rating = raw_game['black']['rating']
                    w_name = raw_game['white']['username']
                    b_name = raw_game['black']['username']
                    opp_name = b_name if player == w_name else w_name
                    opp_rating = b_rating if player == w_name else w_rating
                    result = raw_game['white']['result'] if player == w_name else raw_game['black']['result']
                    score = 1 if result == 'win' else (0.5 if result == 'repetition' else 0)
                    game_results.append((opp_name, opp_rating, score, raw_game['url']))

        unique_opps = set([i[0] for i in game_results])
        P1 = glicko2.Player()
        for r in game_results:
            P2 = glicko2.Player()
            P2.setRating(r[1])
            P1.update_player([P2.getRating()], [P2.getRd()], [r[2]])
        data.append((player, P1.getRating(), len(unique_opps)))
        # print(f'Player: {player}, Performance rating: {P1.getRating()}, Number of unique opponents: {len(unique_opps)}')

    df = pd.DataFrame(data, columns=['username', 'performance_rating', 'unique_opponents'])
    df.to_csv('./data/performance_rating.csv', index=False)
    # print(num_games)


def measure():
    start = time.perf_counter()
    with open('data/filtered_games.json') as f:
        games = json.load(f)
    print(time.perf_counter() - start)
    print(len(games))
    player_ratings = {}
    games_per = {}
    for k in tqdm(games):
        if not (games[k]['a']['pgnHeaders']['TimeControl'] == '120'):
            continue
        a_w_user = games[k]['a']['pgnHeaders']['White']
        a_b_user = games[k]['a']['pgnHeaders']['Black']
        b_w_user = games[k]['b']['pgnHeaders']['White']
        b_b_user = games[k]['b']['pgnHeaders']['Black']

        if a_w_user not in games_per:
            games_per[a_w_user] = 0

        if a_b_user not in games_per:
            games_per[a_b_user] = 0

        if b_w_user not in games_per:
            games_per[b_w_user] = 0

        if b_b_user not in games_per:
            games_per[b_b_user] = 0

        games_per[a_w_user] += 1
        games_per[a_b_user] += 1
        games_per[b_w_user] += 1
        games_per[b_b_user] += 1

    for i in range(100):
        players = {}
        l = list(games.items())
        random.shuffle(l)
        games = dict(l)
        for k in tqdm(games):
            if not (games[k]['a']['pgnHeaders']['TimeControl'] == '120'):
                continue
            a_w_rating = games[k]['a']['pgnHeaders']['WhiteElo']
            a_b_rating = games[k]['a']['pgnHeaders']['BlackElo']
            b_w_rating = games[k]['b']['pgnHeaders']['WhiteElo']
            b_b_rating = games[k]['b']['pgnHeaders']['BlackElo']

            a_w_user = games[k]['a']['pgnHeaders']['White']
            a_b_user = games[k]['a']['pgnHeaders']['Black']
            b_w_user = games[k]['b']['pgnHeaders']['White']
            b_b_user = games[k]['b']['pgnHeaders']['Black']

            a_result = games[k]['a']['pgnHeaders']['Result']
            b_result = games[k]['b']['pgnHeaders']['Result']

            if a_w_user not in players:
                players[a_w_user] = []

            if a_b_user not in players:
                players[a_b_user] = []

            if b_w_user not in players:
                players[b_w_user] = []

            if b_b_user not in players:
                players[b_b_user] = []

            players[a_w_user].append(
                (a_b_rating + b_w_rating - b_b_rating, 1 if a_result == '1-0' else (0.5 if a_result == '1/2-1/2' else 0)))
            players[a_b_user].append(
                (a_w_rating + b_b_rating - b_w_rating, 1 if a_result == '0-1' else (0.5 if a_result == '1/2-1/2' else 0)))
            players[b_w_user].append(
                (b_b_rating + a_w_rating - a_b_rating, 1 if b_result == '1-0' else (0.5 if b_result == '1/2-1/2' else 0)))
            players[b_b_user].append(
                (b_w_rating + a_b_rating - a_w_rating, 1 if b_result == '0-1' else (0.5 if b_result == '1/2-1/2' else 0)))
        #print(players)

        for player in tqdm(players):
            P1 = glicko2.Player()
            P1.setRating(player_ratings.get(player, 1500))
            P1.setRd(100)
            for r in players[player]:
                P2 = glicko2.Player()
                P2.setRating(r[0])
                P2.setRd(100)
                P1.update_player([P2.getRating()], [P2.getRd()], [r[1]])
            player_ratings[player] = P1.getRating()
        #print(player_ratings)

    data = []
    for player in tqdm(players):
        data.append((player, player_ratings.get(player, 1500), games_per[player]))

    df = pd.DataFrame(data, columns=['username', 'rating', 'games'])
    print(df)
    df.to_csv('data/adjusted_rating.csv', index=False)

    #df = pd.DataFrame(player_win_rate)
    #df.to_csv('data/win_rate.csv')

def win_rate_matrix():
    with open('../data/games.json') as f:
        games = json.load(f)

    data = []
    player_win_rate = {}
    players = {}
    for k in tqdm(games):
        #if not (games[k]['a']['pgnHeaders']['TimeControl'] == '120'):
        #    continue
        a_w_rating = games[k]['a']['pgnHeaders']['WhiteElo']
        a_b_rating = games[k]['a']['pgnHeaders']['BlackElo']
        b_w_rating = games[k]['b']['pgnHeaders']['WhiteElo']
        b_b_rating = games[k]['b']['pgnHeaders']['BlackElo']

        a_w_user = games[k]['a']['pgnHeaders']['White']
        a_b_user = games[k]['a']['pgnHeaders']['Black']
        b_w_user = games[k]['b']['pgnHeaders']['White']
        b_b_user = games[k]['b']['pgnHeaders']['Black']

        a_result = games[k]['a']['pgnHeaders']['Result']
        b_result = games[k]['b']['pgnHeaders']['Result']


        if a_w_user not in players:
            players[a_w_user] = []
            player_win_rate[a_w_user] = {}

        if a_b_user not in players:
            players[a_b_user] = []
            player_win_rate[a_b_user] = {}

        if b_w_user not in players:
            players[b_w_user] = []
            player_win_rate[b_w_user] = {}

        if b_b_user not in players:
            players[b_b_user] = []
            player_win_rate[b_b_user] = {}

        if b_b_user not in player_win_rate[a_w_user]:
            player_win_rate[a_w_user][b_b_user] = [0, 0, 0]
        player_win_rate[a_w_user][b_b_user] = [sum(i) for i in
                                               zip([1 if a_result == '1-0' else (0.5 if a_result == '1/2-1/2' else 0),
                                                    1], player_win_rate[a_w_user][b_b_user])] + [player_win_rate[a_w_user][b_b_user][2] + (b_w_rating + a_b_rating)/2]

        if b_w_user not in player_win_rate[a_b_user]:
            player_win_rate[a_b_user][b_w_user] = [0, 0, 0]
        player_win_rate[a_b_user][b_w_user] = [sum(i) for i in
                                               zip([1 if a_result == '0-1' else (0.5 if a_result == '1/2-1/2' else 0),
                                                    1], player_win_rate[a_b_user][b_w_user])] + [player_win_rate[a_b_user][b_w_user][2] + (b_b_rating + a_w_rating)/2]

        if a_b_user not in player_win_rate[b_w_user]:
            player_win_rate[b_w_user][a_b_user] = [0, 0, 0]
        player_win_rate[b_w_user][a_b_user] = [sum(i) for i in
                                               zip([1 if b_result == '1-0' else (0.5 if b_result == '1/2-1/2' else 0),
                                                    1], player_win_rate[b_w_user][a_b_user])] + [player_win_rate[b_w_user][a_b_user][2] + (a_w_rating + b_b_rating)/2]

        if a_w_user not in player_win_rate[b_b_user]:
            player_win_rate[b_b_user][a_w_user] = [0, 0, 0]
        player_win_rate[b_b_user][a_w_user] = [sum(i) for i in
                                               zip([1 if b_result == '0-1' else (0.5 if b_result == '1/2-1/2' else 0),
                                                    1], player_win_rate[b_b_user][a_w_user])] + [player_win_rate[b_b_user][a_w_user][2] + (a_b_rating + b_w_rating)/2]


        players[a_w_user].append(
            (a_b_rating + b_w_rating - b_b_rating, 1 if a_result == '1-0' else (0.5 if a_result == '1/2-1/2' else 0)))
        players[a_b_user].append(
            (a_w_rating + b_b_rating - b_w_rating, 1 if a_result == '0-1' else (0.5 if a_result == '1/2-1/2' else 0)))
        players[b_w_user].append(
            (b_b_rating + a_w_rating - a_b_rating, 1 if b_result == '1-0' else (0.5 if b_result == '1/2-1/2' else 0)))
        players[b_b_user].append(
            (b_w_rating + a_b_rating - a_w_rating, 1 if b_result == '0-1' else (0.5 if b_result == '1/2-1/2' else 0)))


    df = pd.DataFrame(player_win_rate)
    df.to_csv('../data/win_rate.csv')

if __name__ == '__main__':
    '''measure()
    df = pd.read_csv('data/adjusted_rating.csv')
    print(df.sort_values(['rating'], ascending=False))
    # print(df[df['performance_rating'] < 1900])

    players = []
    with open('data/players.txt') as f:
        for line in f:
            players.append(line.rstrip('\n'))
    print(df.loc[df['username'].isin(players)].sort_values(['rating'], ascending=False))'''
    #measure()
    #win_rate_matrix()
    df = pd.read_csv('../data/win_rate.csv')
    # print(df[['Unnamed: 0', 'chickencrossroad']])
    # print(df[['Unnamed: 0', 'nochewycandy']].loc[df['Unnamed: 0'] == 'MARVELandDCforLIFE'])
    user = '1e4I-0'
    col = df[['Unnamed: 0', user]].dropna()
    col[user] = col[user].apply(ast.literal_eval)
    col['win_percentage'] = col[user].map(lambda x: int(x[0]) / int(x[1]))
    col['opponent_team_rating'] = col[user].map(lambda x: int(x[2]) / int(x[1]))
    col['games'] = col[user].map(lambda x: int(x[1]))
    col = col.loc[col['games'] >= 50]
    col[user] = col[user].map(lambda x: [x[0], x[1]])
    col = col.rename(columns={'Unnamed: 0': 'partner'})
    col = col.drop(['games'], axis=1)
    results = col.sort_values(by='win_percentage', ascending=False)
    print(results[~results['partner'].isin(['WhenTheGodsCry', 'ClayTemple'])].to_string())

    # print(df.sort_values(['rating'], ascending=False))
    # df['performance_rating'].plot()
    # plt.show()
    # measure()

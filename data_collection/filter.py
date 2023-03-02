import json

filtered_games = {}

with open('data/games_0.json') as f:
    games = json.load(f)

for k in games:
    a_w_rating = games[k]['a']['pgnHeaders']['WhiteElo']
    a_b_rating = games[k]['a']['pgnHeaders']['BlackElo']
    b_w_rating = games[k]['b']['pgnHeaders']['WhiteElo']
    b_b_rating = games[k]['b']['pgnHeaders']['BlackElo']

    if a_w_rating > 2000 and a_b_rating > 2000 and b_w_rating > 2000 and b_b_rating > 2000:
        filtered_games[k] = games[k]

with open('data/games_1.json') as f:
    games = json.load(f)

for k in games:
    a_w_rating = games[k]['a']['pgnHeaders']['WhiteElo']
    a_b_rating = games[k]['a']['pgnHeaders']['BlackElo']
    b_w_rating = games[k]['b']['pgnHeaders']['WhiteElo']
    b_b_rating = games[k]['b']['pgnHeaders']['BlackElo']

    if a_w_rating > 2000 and a_b_rating > 2000 and b_w_rating > 2000 and b_b_rating > 2000:
        filtered_games[k] = games[k]


with open('data/filtered_games.json', 'w') as f:
    json.dump(filtered_games, f)

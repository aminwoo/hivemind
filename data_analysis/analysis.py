import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import glob
import json
import os
import tqdm
import statistics


def extract_game(game_path):
    game_file = open(game_path, 'r')
    game_json = json.load(game_file)
    game_file.close()
    return game_json


players = {}
ratings = []
time_controls = {}
results = {}

with open('../data/filtered_games.json') as f:
    games = json.load(f)

f = open("game_ids.txt", "a")
ids = list(games.keys())
ids.sort(key=lambda x: int(x.split("_")[0]))
for game in ids:
    f.write(game + '\n')
f.close()

num_games = len(games)

for game_id in games:
    game = games[game_id]
    a_w_rating = game['a']['pgnHeaders']['WhiteElo']
    a_b_rating = game['a']['pgnHeaders']['BlackElo']
    b_w_rating = game['b']['pgnHeaders']['WhiteElo']
    b_b_rating = game['b']['pgnHeaders']['BlackElo']
    a_w_username = game['a']['pgnHeaders']['White']
    a_b_username = game['a']['pgnHeaders']['Black']
    b_w_username = game['b']['pgnHeaders']['White']
    b_b_username = game['b']['pgnHeaders']['Black']
    players[a_w_username] = players.get(a_w_username, 0) + 1
    players[a_b_username] = players.get(a_b_username, 0) + 1
    players[b_w_username] = players.get(b_w_username, 0) + 1
    players[b_b_username] = players.get(b_b_username, 0) + 1
    ratings.append(a_w_rating)
    ratings.append(a_b_rating)
    ratings.append(b_w_rating)
    ratings.append(b_b_rating)
    time_control = game['a']['pgnHeaders']['TimeControl']
    time_controls[time_control] = time_controls.get(time_control, 0) + 1

    result = game['a']['pgnHeaders']['Result']
    results[result] = results.get(result, 0) + 1

m = statistics.mean(ratings)
sd = statistics.stdev(ratings)

print(num_games)
print((sum(len(game['a']['moveList']) + len(game['b']['moveList']) for game in
                        list(games.values())) / 2))
print(m, sd)

plt.hist(ratings, bins=100, color='c', edgecolor='k')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.axvline(m, color='k', linestyle='dashed')
plt.axvline(m + sd, color='y', linestyle='dashed')
plt.axvline(m - sd, color='y', linestyle='dashed')
plt.title('Player Rating Distribution')
plt.xlabel('Rating')
plt.xlim([1950, 3200])
#plt.grid()
plt.show()

top20 = sorted(players, key=players.get, reverse=True)[:20]

plt.rcdefaults()
fig, ax = plt.subplots()

# Example data
people = top20
y_pos = np.arange(len(people))
performance = [players[i] for i in top20]

print(sum(performance))
print(sum(performance) / num_games)

plt.barh(y_pos, performance, align='center')
ax.set_yticks(y_pos, labels=people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Number of Games')
ax.set_title(f'Top 20 Players by Number of Games \n ({num_games} Total Games)', pad=10)

ax.set_axisbelow(True)
plt.margins(y=0.01)
plt.grid()
plt.tight_layout()
plt.show()



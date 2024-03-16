import glob
import json
import os

dir_path = "data/games/"

games = []
for file in glob.glob(dir_path + "*"):
    with open(file) as f:
        obj = json.load(f)
    
    for i in range(len(obj)):
        try:
            obj[i]['a'] = {
                        'id': obj[i]['a']['id'],
                        'tcn': obj[i]['a']['moveList'], 
                        'time_stamps': obj[i]['a']['moveTimestamps'],
                        'time_control': str(obj[i]['a']['baseTime1'] // 10) + (("+" + str(obj[i]['a']['timeIncrement1'] // 10)) if (obj[i]['a']['timeIncrement1'] // 10) > 0 else ""), 
                        'winner': "" if "colorOfWinner" not in obj[i]['a'] else obj[i]['a']['colorOfWinner'],
                        'white': {'username': obj[i]['a']['pgnHeaders']['White'], 'rating': obj[i]['a']['pgnHeaders']['WhiteElo']},
                        'black': {'username': obj[i]['a']['pgnHeaders']['Black'], 'rating': obj[i]['a']['pgnHeaders']['BlackElo']},
                        }
            obj[i]['b'] = {
                'id': obj[i]['b']['id'],
                'tcn': obj[i]['b']['moveList'], 
                'time_stamps': obj[i]['b']['moveTimestamps'],
                'time_control': str(obj[i]['b']['baseTime1'] // 10) + (("+" + str(obj[i]['b']['timeIncrement1'] // 10)) if (obj[i]['b']['timeIncrement1'] // 10) > 0 else ""), 
                'winner': "" if "colorOfWinner" not in obj[i]['b'] else obj[i]['b']['colorOfWinner'],
                'white': {'username': obj[i]['b']['pgnHeaders']['White'], 'rating': obj[i]['b']['pgnHeaders']['WhiteElo']},
                'black': {'username': obj[i]['b']['pgnHeaders']['Black'], 'rating': obj[i]['b']['pgnHeaders']['BlackElo']},
            }
        except Exception as e:
            print(e)
            print(obj[i])
            exit()
    games += obj

print(len(games))
with open('data/games.json', 'w') as f:
    json.dump(games, f)
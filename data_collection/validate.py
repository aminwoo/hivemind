import glob
import json
import os

with open('data/games.json') as f:
    games = json.load(f)

print(len(games))
ids = set()

good = 0
for game in games:
    if game['a']['white']['rating'] > 2000 and game['a']['black']['rating'] > 2000 and game['b']['white']['rating'] > 2000  and game['b']['black']['rating'] > 2000:
        good += 1
print(good)


'''for i in range(len(games)):
    #if not games[i]['a']['winner']:
        #print(games[i]['a']['id'])

    if games[i]['a']['id'] > games[i]['b']['id']:
        games[i]['a'], games[i]['b'] = games[i]['b'], games[i]['a']

    if games[i]['a']['winner'] == games[i]['b']['winner']:
        games[i]['a']['winner'] = 'white' if games[i]['b']['winner'] == 'black' else 'black' 
        print('test')
        exit()

    if "+" not in games[i]['a']['time_control']:
        games[i]['a']['time_control'] = str(int(float(games[i]['a']['time_control'])))
    else:
        games[i]['a']['time_control'] = str(int(float(games[i]['a']['time_control'].split('+')[0]))) + '+' + str(int(float(games[i]['a']['time_control'].split('+')[1])))
    
    if "+" not in games[i]['b']['time_control']:
        games[i]['b']['time_control'] = str(int(float(games[i]['b']['time_control'])))
    else:
        games[i]['b']['time_control'] = str(int(float(games[i]['b']['time_control'].split('+')[0]))) + '+' + str(int(float(games[i]['b']['time_control'].split('+')[1])))
'''

for game in games:
    ids.add(game['a']['id'])
    ids.add(game['b']['id'])
print(len(ids))

exit()
with open('data/games.json', 'w') as f:
    json.dump(games, f)


dir_path = "data/archives/"

for path in glob.glob(dir_path + "*"):
    for file in glob.glob(path + "/*"):
        with open(file) as f:
            data = json.load(f)
            if "games" not in data:
                os.remove(file)
                print(file)
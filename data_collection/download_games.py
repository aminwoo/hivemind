import glob
import json
import os
import asyncio
from tqdm import tqdm
from archive import get_leaderboard
from aiocfscrape import CloudflareScraper


def retrieve_game_urls():
    players = get_leaderboard()
    player_archive_base_glob = './data/archives/{0}/*.json'
    base_games_path = './data/games/{0}.json'
    total_games = 0
    urls = []
    paths = []
    blacklist = []

    with open('../data/blacklist.txt') as f:
        for line in f:
            blacklist.append(line.rstrip('\n'))

    for player in tqdm(players):
        if player in blacklist:
            continue

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
                        w_rating = raw_game['white']['rating']
                        b_rating = raw_game['black']['rating']

                        if w_rating > 2000 and b_rating > 2000:
                            total_games += 1
                            game_id = raw_game['url'].split('/')[-1]
                            path = base_games_path.format(game_id)
                            base_url = 'https://www.chess.com/callback/live/game/{0}'.format(game_id)
                            urls.append(base_url)
                            paths.append(path)

    print(len(urls))
    with open('../data/game_urls.json', 'w') as f:
        json.dump(urls, f)


def download_games():
    with open('data/games_0.json') as f:
        games = json.load(f)
    ids = set()
    keys = games.keys()
    del games
    for key in tqdm(keys):
        s = key.split('_')
        ids.add(s[0])
        ids.add(s[1])
    games = {}

    with open('../data/game_urls.json') as f:
        urls = json.load(f)

    checkpoint_step = 1000
    for url in tqdm(urls):
        if url.split('/')[-1] in ids:
            continue
        try:
            obj = json.loads(asyncio.run(download_game(url)))
            partner_game_url = 'https://www.chess.com/callback/live/game/{0}'.format(obj['game']['partnerGameId'])
            partner_game = json.loads(asyncio.run(download_game(partner_game_url)))
            game = {'a': obj['game'], 'b': partner_game['game']}
            if game['a']['id'] > game['b']['id']:
                game['a'], game['b'] = game['b'], game['a']
            game_id = f"{game['a']['id']}_{game['b']['id']}"
            games[game_id] = game
            ids.add(game['a']['id'])
            ids.add(game['b']['id'])

            if len(games) % checkpoint_step == 0:
                with open('./data/games_1.json', 'w') as f:
                    json.dump(games, f)
        except:
            continue

    with open('data/games_1.json', 'w') as f:
        json.dump(games, f)


async def download_game(url):
    async with CloudflareScraper() as session:
        async with session.get(url) as resp:
            return await resp.text()


if __name__ == '__main__':
    download_games()
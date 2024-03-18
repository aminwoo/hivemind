import json
import os
import glob
import requests
import asyncio

from aiocfscrape import CloudflareScraper
from dagster import asset, AssetExecutionContext

from download import batch_download


headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/109.0.0.0"
}

async def download_game(url):
    async with CloudflareScraper() as session:
        async with session.get(url) as resp:
            return await resp.text()

@asset
def players() -> None: 
    r = requests.get("https://api.chess.com/pub/leaderboards", headers=headers)
    players = [user["username"] for user in r.json()["live_bughouse"]]

    with open("../data/leaderboard.txt", "w+") as f:
        f.write("\n".join(players))

@asset(deps=[players])
def archive_urls() -> None:
    with open("../data/leaderboard.txt") as f:
        players = [p.strip() for p in f.readlines()]
    
    os.makedirs("../data/archive_urls/", exist_ok=True)

    urls = [] 
    paths = [] 
    for player in players:
        urls.append(f"https://api.chess.com/pub/player/{player}/games/archives")
        paths.append(f"../data/archive_urls/{player}.json")

    batch_download(urls, paths)

@asset(deps=[archive_urls])
def archive() -> None: 
    with open("../data/leaderboard.txt") as f:
        players = [p.strip() for p in f.readlines()]

    os.makedirs("../data/archives", exist_ok=True)
    
    urls = [] 
    paths = [] 

    for player in players:
        with open(f"../data/archive_urls/{player}.json") as f:
            player_urls = json.load(f)["archives"]

        os.makedirs(f"../data/archives/{player}/", exist_ok=True)
        for url in player_urls:
            m, y, *_ = url.split('/')[::-1]
            if int(y) >= 2016: 
                urls.append(url)
                paths.append(f"../data/archives/{player}/{y}_{m}.json")
                
    batch_download(urls, paths)

@asset(deps=[archive])
def games(context: AssetExecutionContext):
    with open("../data/leaderboard.txt") as f:
        players = [p.strip() for p in f.readlines()]

    with open("../data/games.json") as f:
        games = json.load(f)

    game_ids = set()
    for game in games:
        game_ids.add(game["a"]["id"])
        game_ids.add(game["b"]["id"])

    player_archive_base_glob = "../data/archives/{0}/*.json"

    for player in players:
        player_archive_glob = player_archive_base_glob.format(player)
        player_archive_paths = [os.path.normpath(p).replace(os.sep, '/') for p in glob.glob(player_archive_glob)]
        for a_path in player_archive_paths:
            with open(a_path, 'rb') as f:
                a_json = json.load(f)
            if "games" not in a_json:
                continue
            raw_games = a_json["games"]
            for raw_game in raw_games:
                if raw_game["rules"] == "bughouse":
                    if "tcn" not in raw_game:
                        continue
                    tcn_moves = raw_game["tcn"].encode("utf-8")
                    num_half_moves = len(tcn_moves) / 2
                    if num_half_moves >= 10:
                        w_rating = raw_game["white"]["rating"]
                        b_rating = raw_game["black"]["rating"]

                        if w_rating > 2000 and b_rating > 2000:
                            game_id = raw_game["url"].split("/")[-1]
                            url = "https://www.chess.com/callback/live/game/{0}".format(game_id)
                            if game_id in game_ids:
                                continue

                            obj = json.loads(asyncio.run(download_game(url)))
                            partner_game_url = 'https://www.chess.com/callback/live/game/{0}'.format(obj['game']['partnerGameId'])
                            partner_game = json.loads(asyncio.run(download_game(partner_game_url)))
                            if partner_game["white"]["rating"] > 2000 and partner_game["black"]["rating"]  > 2000:
                                continue

                            game = {'a': obj['game'], 'b': partner_game['game']}
                            if game['a']['id'] > game['b']['id']:
                                game['a'], game['b'] = game['b'], game['a']
                                
                            game['a'] = {
                                        'id': game['a']['id'],
                                        'tcn': game['a']['moveList'], 
                                        'time_stamps': game['a']['moveTimestamps'],
                                        'time_control': str(game['a']['baseTime1'] // 10) + (("+" + str(game['a']['timeIncrement1'] // 10)) if (game['a']['timeIncrement1'] // 10) > 0 else ""), 
                                        'winner': "" if "colorOfWinner" not in game['a'] else game['a']['colorOfWinner'],
                                        'white': {'username': game['a']['pgnHeaders']['White'], 'rating': game['a']['pgnHeaders']['WhiteElo']},
                                        'black': {'username': game['a']['pgnHeaders']['Black'], 'rating': game['a']['pgnHeaders']['BlackElo']},
                                        }
                            game['b'] = {
                                'id': game['b']['id'],
                                'tcn': game['b']['moveList'], 
                                'time_stamps': game['b']['moveTimestamps'],
                                'time_control': str(game['b']['baseTime1'] // 10) + (("+" + str(game['b']['timeIncrement1'] // 10)) if (game['b']['timeIncrement1'] // 10) > 0 else ""), 
                                'winner': "" if "colorOfWinner" not in game['b'] else game['b']['colorOfWinner'],
                                'white': {'username': game['b']['pgnHeaders']['White'], 'rating': game['b']['pgnHeaders']['WhiteElo']},
                                'black': {'username': game['b']['pgnHeaders']['Black'], 'rating': game['b']['pgnHeaders']['BlackElo']},
                            }

                            games.append(game)
                            game_ids.add(game['a']['id'])
                            game_ids.add(game['b']['id'])

    context.log(f"Total games: {len(games)}")
    with open("../data/games.json", "w") as f:
        json.dump(games, f)
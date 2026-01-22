import polars as pl
import json
import os
import glob
import time

import cloudscraper
from tqdm import tqdm
from download import batch_download

headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/109.0.0.0"
}

scraper = cloudscraper.create_scraper(
    browser={
        'browser': 'chrome',
        'platform': 'windows',
        'desktop': True
    }
)


def download_game(url: str) -> str:
    """Bypasses Cloudflare using cloudscraper."""
    try:
        # cloudscraper will wait and solve the challenge if needed
        response = scraper.get(url, timeout=15)

        if response.status_code == 200:
            # Respect Chess.com's rate limits
            time.sleep(1.0)
            return response.text
        else:
            print(f"Failed with status: {response.status_code}")
            return "{}"

    except Exception as e:
        print(f"Network or Cloudflare error: {e}")
        return "{}"


def download_archive_urls() -> None:
    with open("../data/leaderboard.txt") as f:
        players = [p.strip() for p in f.readlines()]

    os.makedirs("../data/archive_urls/", exist_ok=True)

    urls = []
    paths = []
    for player in players:
        urls.append(f"https://api.chess.com/pub/player/{player}/games/archives")
        paths.append(f"../data/archive_urls/{player}.json")

    batch_download(urls, paths)


def download_archive() -> None:
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


def download_games() -> None:
    def save(new_data, path):
        new_df = pl.DataFrame(new_data)
        existing_df = pl.read_parquet(path)
        pl.concat([existing_df, new_df]).unique(subset="game_id").write_parquet(path)

    # 1. Load players from leaderboard
    with open("../data/leaderboard.txt") as f:
        players = [p.strip() for p in f.readlines()]

    # 2. Load existing Game IDs from Parquet to avoid duplicates
    parquet_path = "../data/games.parquet"
    if os.path.exists(parquet_path):
        # We only need the ID column to check for duplicates (very memory efficient)
        game_ids = set(pl.scan_parquet(parquet_path).select("game_id").collect().to_series())
    else:
        game_ids = set()

    BATCH_SIZE = 100
    new_games_list = []  # We store new games in a list of flat dicts
    player_archive_base_glob = "../data/archives/{0}/*.json"

    min_rating = 2400
    for player in players:
        player_archive_glob = player_archive_base_glob.format(player)
        player_archive_paths = [os.path.normpath(p).replace(os.sep, '/') for p in glob.glob(player_archive_glob)]

        for a_path in player_archive_paths:
            with open(a_path, 'rb') as f:
                a_json = json.load(f)
            if "games" not in a_json:
                continue

            for raw_game in tqdm(a_json["games"]):
                if raw_game["rules"] == "bughouse" and "tcn" in raw_game:
                    if len(raw_game["tcn"]) / 2 >= 10:
                        w_rating = raw_game["white"]["rating"]
                        b_rating = raw_game["black"]["rating"]

                        if w_rating >= min_rating and b_rating >= min_rating:
                            game_id = raw_game["url"].split("/")[-1]
                            if game_id in game_ids:
                                continue

                            try:
                                url = f"https://www.chess.com/callback/live/game/{game_id}"
                                obj = json.loads(download_game(url))

                                p_id = obj["game"]["partnerGameId"]
                                p_url = f"https://www.chess.com/callback/live/game/{p_id}"
                                p_game = json.loads(download_game(p_url))

                                players = p_game.get("players")
                                if players.get("top").get("color") == "white":
                                    white_player = players["top"]
                                    black_player = players["bottom"]
                                else:
                                    white_player = players["bottom"]
                                    black_player = players["top"]
                            except Exception:
                                print(f"Failed to fetch game: {game_id}")
                                continue

                            # Filter based on partner ratings
                            if white_player.get("rating") < 2200 or black_player.get("rating") < 2200:
                                continue

                            # Flattening logic into snake_case for the Parquet schema
                            for board_data_wrapper in [obj, p_game]:
                                board_data = board_data_wrapper["game"]
                                players = board_data_wrapper.get("players", {})

                                # Determine which player is White and which is Black
                                top_player = players.get("top", {})
                                bottom_player = players.get("bottom", {})

                                if top_player.get("color") == "white":
                                    white_p, black_p = top_player, bottom_player
                                else:
                                    white_p, black_p = bottom_player, top_player

                                new_games_list.append({
                                    "game_id": str(board_data["id"]),
                                    "partner_game_id": str(board_data.get("partnerGameId")),
                                    "tcn": board_data["moveList"],
                                    "timestamps": board_data["moveTimestamps"],
                                    "winner": board_data.get("colorOfWinner", ""),
                                    "white_user": white_p.get("username"),
                                    "white_rating": white_p.get("rating"),
                                    "black_user": black_p.get("username"),
                                    "black_rating": black_p.get("rating"),
                                    "time_control": str(int(board_data["baseTime1"] // 10)) + (
                                        ("+" + str(int(board_data["timeIncrement1"] // 10)))
                                        if (board_data["timeIncrement1"] // 10) > 0 else ""
                                    ),
                                })
                                game_ids.add(str(board_data["id"]))
                                if len(new_games_list) >= BATCH_SIZE:
                                    save(new_games_list, parquet_path)
                                    new_games_list = []
                                    print("Saved new games")

    # 3. Write to Parquet
    if new_games_list:
        save(new_games_list, parquet_path)


download_archive_urls()
download_archive()
download_games()

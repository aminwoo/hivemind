import glob
import os
import json
from downloader import batch_download


def get_leaderboard():
    in_file_path = 'data/leaderboard.txt'
    in_file = open(in_file_path, 'r')
    players = in_file.readlines()
    players = [p.strip() for p in players]
    in_file.close()
    return players


def get_player_archive_urls(player):
    base_dir_path = 'data/archive_urls/'
    base_player_path = base_dir_path + '{0}.json'
    player_path = base_player_path.format(player)
    player_file = open(player_path, 'r')
    player_urls = json.load(player_file).get('archives', [])
    player_file.close()
    return player_urls


def update_archives():
    players = get_leaderboard()
    archive_urls = {player: get_player_archive_urls(player) for player in players}
    init_month = 1
    init_year = 2016
    last_month_index = 12 * init_year + init_month
    base_archive_url = 'https://api.chess.com/pub/player/{0}/games/{1:04d}/{2:02d}'
    base_player_dir_path = 'data/archives/{0}/'
    base_archive_path = base_player_dir_path + '{1:04d}_{2:02d}.json'

    num_players = len(players)
    num_digits = len(str(num_players))
    for i, player in enumerate(players):
        print('Downloading player {0:0{2}}/{1:0{2}}:'.format(i + 1, num_players, num_digits), player)
        player_urls = archive_urls[player]
        base_dir_path = base_player_dir_path.format(player)

        os.makedirs(base_dir_path, exist_ok=True)
        a_urls = []
        a_paths = []
        for player_url in player_urls:
            url_parts = player_url.split('/')
            y = int(url_parts[7])
            m = int(url_parts[8])
            if m + (12 * y) >= last_month_index:
                a_url = base_archive_url.format(player, y, m)
                a_path = base_archive_path.format(player, y, m)
                a_urls.append(a_url)
                a_paths.append(a_path)
        batch_download(a_urls, a_paths)


def update_archive_urls(remove_old=False):
    base_dir_path = 'data/archive_urls/'
    base_player_url = 'https://api.chess.com/pub/player/{0}/games/archives'
    base_player_path = base_dir_path + '{0}.json'
    os.makedirs(base_dir_path, exist_ok=True)

    if remove_old:
        old_file_paths = [os.path.normpath(p).replace(os.sep, "/") for p in glob.glob(base_dir_path + "*")]
        for old_file_path in old_file_paths:
            os.remove(old_file_path)
    os.makedirs(base_dir_path, exist_ok=True)

    players = get_leaderboard()
    player_urls = [base_player_url.format(player) for player in players]
    player_paths = [base_player_path.format(player) for player in players]
    batch_download(player_urls, player_paths)


if __name__ == '__main__':
    update_archive_urls()
    update_archives()

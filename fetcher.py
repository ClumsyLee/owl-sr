from csv import DictWriter
from datetime import datetime
from typing import Dict, List

import requests

GAMES_CSV = 'owl.csv'
BASE_URL = 'https://api.overwatchleague.com/'


def fetch_games() -> List[Dict]:
    url = BASE_URL + 'matches'
    params = {'size': 1000}
    result = requests.get(url, params).json()

    games = []
    for raw_match in result['content']:
        games += parse_match(raw_match)
    games.sort(key=lambda game: game['start_time'])

    return games


def parse_match(raw_match) -> List[Dict]:
    if None in raw_match['competitors']:
        return []  # The competitors have not been decided, don't parse it.

    match_id = raw_match['id']
    stage = raw_match['bracket']['stage']['title']
    start_time = datetime.fromtimestamp(raw_match['startDate'] / 1000)
    team1_id = raw_match['competitors'][0]['id']
    team2_id = raw_match['competitors'][1]['id']
    team1_abbr = raw_match['competitors'][0]['abbreviatedName']
    team2_abbr = raw_match['competitors'][1]['abbreviatedName']

    # Fix an API bug.
    if stage == 'Split 4':
        stage = 'Stage 4'

    if 'Title Matches' in stage:
        match_format = 'title'
    else:
        match_format = 'regular'

    base_game = {
        'match_id': match_id,
        'stage': stage,
        'start_time': start_time,
        'team1': team1_abbr,
        'team2': team2_abbr,
        'match_format': match_format
    }

    if raw_match['state'] != 'CONCLUDED':
        return [base_game]  # This a unfinished match.

    games = []
    for raw_game in raw_match['games']:
        game = parse_game(raw_game, base_game, team1_id, team2_id)
        if game is not None:
            games.append(game)

    return games


def parse_game(raw_game, base_game: Dict, team1_id: str, team2_id: str):
    if raw_game['state'] != 'CONCLUDED':
        return None  # This is an unfinished match, don't parse it.

    game_id = raw_game['id']
    game_number = raw_game['number']
    map_name = raw_game['attributes']['map']
    score = raw_game['points']

    game = base_game.copy()
    game.update({
        'game_id': game_id,
        'game_number': game_number,
        'map': map_name,
        'score1': score[0],
        'score2': score[1]
    })

    n_team1 = 0
    n_team2 = 0
    for raw_player in raw_game['players']:
        team_id = raw_player['team']['id']
        name = raw_player['player']['name']

        if team_id == team1_id:
            n_team1 += 1
            name_key = f'team1_p{n_team1}'
        elif team_id == team2_id:
            n_team2 += 1
            name_key = f'team2_p{n_team2}'
        else:
            print(f'{game_id}: Invalid team id ({name}, {team_id}), skipping.')
            continue

        game[name_key] = name

    if n_team1 != 6 or n_team2 != 6:
        print(f'{game_id}: Invalid player numbers, skipping.')
        return None

    return game


def save_games(games: List[Dict], csv_filename: str=GAMES_CSV) -> None:
    fieldnames = [
        'match_id', 'game_id',
        'stage', 'start_time', 'team1', 'team2', 'match_format',
        'game_number', 'map', 'score1', 'score2',
        'team1_p1', 'team1_p2', 'team1_p3', 'team1_p4', 'team1_p5', 'team1_p6',
        'team2_p1', 'team2_p2', 'team2_p3', 'team2_p4', 'team2_p5', 'team2_p6',
    ]

    with open(csv_filename, 'w', newline='') as csvfile:
        writer = DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(games)


if __name__ == '__main__':
    games = fetch_games()
    save_games(games)

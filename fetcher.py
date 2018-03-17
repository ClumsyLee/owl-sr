from collections import defaultdict
from csv import (reader as csv_reader,
                 writer as csv_writer,
                 DictReader,
                 DictWriter)
from datetime import datetime
from typing import Dict, List, NamedTuple, Set, Tuple

from game import Game

import requests

GAMES_CSV = 'games.csv'
AVAILABILITIES_CSV = 'availabilities.csv'
RATINGS_CSV = 'ratings.csv'
BASE_URL = 'https://api.overwatchleague.com/'


Availabilities = Dict[Tuple[str, int], Dict[str, Set[str]]]


class CSVGame(NamedTuple):
    """Describe a single game in a CSV file."""
    match_id: int
    stage: str
    start_time: datetime
    team1: str
    team2: str
    match_format: str

    game_id: int = None
    game_number: int = None
    map_name: str = None
    score1: int = None
    score2: int = None
    team1_p1: str = None
    team1_p2: str = None
    team1_p3: str = None
    team1_p4: str = None
    team1_p5: str = None
    team1_p6: str = None
    team2_p1: str = None
    team2_p2: str = None
    team2_p3: str = None
    team2_p4: str = None
    team2_p5: str = None
    team2_p6: str = None


def fetch_games() -> List[CSVGame]:
    url = BASE_URL + 'matches'
    params = {'size': 1000}
    result = requests.get(url, params).json()

    games = []
    for raw_match in result['content']:
        games += parse_match(raw_match)
    games.sort(key=lambda game: game.start_time)

    return games


def parse_match(raw_match) -> List[CSVGame]:
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

    base_game = CSVGame(match_id=match_id, stage=stage, start_time=start_time,
                        team1=team1_abbr, team2=team2_abbr,
                        match_format=match_format)

    if raw_match['state'] != 'CONCLUDED':
        return [base_game]  # This a unfinished match.

    games = []
    for raw_game in raw_match['games']:
        game = parse_game(raw_game, base_game, team1_id, team2_id)
        if game is not None:
            games.append(game)

    return games


def parse_game(raw_game, base_game: CSVGame, team1_id: int,
               team2_id: int) -> CSVGame:
    if raw_game['state'] != 'CONCLUDED':
        return None  # This is an unfinished match, don't parse it.

    game_id = raw_game['id']
    game_number = raw_game['number']
    map_name = raw_game['attributes']['map']
    score = raw_game['points']

    n_team1 = 0
    n_team2 = 0
    names = {}

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

        names[name_key] = name

    if n_team1 != 6 or n_team2 != 6:
        print(f'{game_id}: Invalid player numbers, skipping.')
        return None

    game = base_game._replace(game_id=game_id, game_number=game_number,
                              map_name=map_name,
                              score1=score[0], score2=score[1], **names)
    return game


def save_games(games: List[CSVGame], csv_filename: str = GAMES_CSV) -> None:
    with open(csv_filename, 'w', newline='') as csv_file:
        writer = csv_writer(csv_file)
        writer.writerow(CSVGame._fields)  # Write headers.
        writer.writerows(games)


def load_games(csv_filename: str = GAMES_CSV) -> Tuple[List[Game], List[Game]]:
    """Load past & future games from a csv file."""
    past_games = []
    future_games = []

    with open(csv_filename, newline='') as csv_file:
        reader = csv_reader(csv_file)
        next(reader, None)  # Skip the header line.

        for csv_game in map(CSVGame._make, reader):
            match_id = int(csv_game.match_id)
            stage = csv_game.stage
            start_time = datetime.strptime(csv_game.start_time,
                                           '%Y-%m-%d %H:%M:%S')
            teams = (csv_game.team1, csv_game.team2)
            match_format = csv_game.match_format

            if csv_game.game_id:
                game_id = int(csv_game.game_id)
                game_number = int(csv_game.game_number)
                map_name = csv_game.map_name
                score = (int(csv_game.score1), int(csv_game.score2))
                rosters = (
                    (csv_game.team1_p1, csv_game.team1_p2, csv_game.team1_p3,
                     csv_game.team1_p4, csv_game.team1_p5, csv_game.team1_p6),
                    (csv_game.team2_p1, csv_game.team2_p2, csv_game.team2_p3,
                     csv_game.team2_p4, csv_game.team2_p5, csv_game.team2_p6)
                )

                game = Game(match_id=match_id, stage=stage,
                            start_time=start_time, teams=teams,
                            match_format=match_format, game_id=game_id,
                            game_number=game_number, map_name=map_name,
                            score=score, rosters=rosters)
                past_games.append(game)
            else:
                game = Game(match_id=match_id, stage=stage,
                            start_time=start_time, teams=teams,
                            match_format=match_format)
                future_games.append(game)

    return past_games, future_games


def load_availabilities(
        csv_filename: str = AVAILABILITIES_CSV) -> Availabilities:
    availabilities = {}

    with open(csv_filename, newline='') as csv_file:
        for row in DictReader(csv_file):
            stage = row.pop('stage')
            match_number = int(row.pop('match_number'))
            team_members = defaultdict(set)

            for name, team in row.items():
                if not team:
                    continue
                team_members[team].add(name)

            availabilities[(stage, match_number)] = team_members

    return availabilities


def save_ratings_history(history, mu, sigma, csv_filename: str = RATINGS_CSV):
    # Collect all unique names.
    names = set(name for ratings in history.values()
                for name in ratings.keys())

    row = {}
    for name in names:
        row[f'{name}.mu'] = round(mu)
        row[f'{name}.sigma'] = round(sigma)
    fieldnames = ['stage', 'match_number'] + sorted(row.keys())

    with open(csv_filename, 'w', newline='') as csv_file:
        writer = DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for (stage, match_number), ratings in history.items():
            row['stage'] = stage
            row['match_number'] = match_number

            for name, rating in ratings.items():
                row[f'{name}.mu'] = round(rating.mu)
                row[f'{name}.sigma'] = round(rating.sigma)

            writer.writerow(row)


if __name__ == '__main__':
    games = fetch_games()
    save_games(games)

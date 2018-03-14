from collections import defaultdict
from csv import DictReader
from enum import Enum
from typing import Dict, List, Tuple, Union

MatchFormat = Enum('MatchFormat', 'REGULAR TITLE')
Team = Enum('Team', 'BOS DAL FLA GLA HOU LDN NYE PHI SEO SFS SHD VAL')
Roster = Tuple[str, str, str, str, str, str]
PScores = Dict[Tuple[int, int], float]

DRAWABLE_MAPS = set([
    'hanamura', 'horizon-lunar-colony', 'temple-of-anubis', 'volskaya',
    'eichenwalde', 'hollywood', 'kings-row', 'numbani'
])


class Predictor(object):
    """Base class for all OWL predictors."""

    def __init__(self) -> None:
        super().__init__()

    def train(self, teams: Tuple[Team, Team],
              rosters: Tuple[Roster, Roster],
              score: Tuple[int, int],
              drawable: bool) -> float:
        """Given a game result, train the underlying model.
        Return the prediction point for this game before training."""
        raise NotImplementedError

    def predict(self, teams: Tuple[Team, Team],
                rosters: Tuple[Roster, Roster],
                drawable: bool=False) -> Tuple[float, float]:
        """Given two teams, return win/draw probabilities of them."""
        raise NotImplementedError

    def evaluate(self, teams: Tuple[Team, Team],
                 rosters: Tuple[Roster, Roster],
                 score: Tuple[int, int],
                 drawable: bool) -> float:
        """Return the prediction point for this game."""
        if score[0] == score[1]:
            return 0.0

        p_win, _ = self.predict(teams, rosters)
        win = score[0] > score[1]
        return 0.25 - (p_win - win)**2

    def train_all(self, csv_filename) -> float:
        """Given a csv file containing matches, train the underlying model.
        Return the prediction point for all the matches."""
        point = 0.0

        with open(csv_filename, newline='') as csv_file:
            reader = DictReader(csv_file)

            for row in reader:
                teams = (Team[row['team1']], Team[row['team2']])
                rosters = ((row['team1-p1'], row['team1-p2'], row['team1-p3'],
                            row['team1-p4'], row['team1-p5'], row['team1-p6']),
                           (row['team2-p1'], row['team2-p2'], row['team2-p3'],
                            row['team2-p4'], row['team2-p5'], row['team2-p6']))
                score = (int(row['score1']), int(row['score2']))
                drawable = row['map'] in DRAWABLE_MAPS

                point += self.train(teams, rosters, score, drawable=drawable)

        return point

    def predict_match_score(
            self, teams: Tuple[Team, Team],
            rosters: Tuple[Roster, Roster],
            match_format: MatchFormat = MatchFormat.REGULAR) -> PScores:
        """Predict the scores of a given match."""
        if match_format == MatchFormat.REGULAR:
            drawables = [True, False, True, False]
            return self._predict_bo_match_score(teams, rosters,
                                                drawables=drawables)
        elif match_format == MatchFormat.TITLE:
            drawables = [False, False, True, True, False]
            return self._predict_bo_match_score(teams, rosters,
                                                drawables=drawables)
        else:
            raise NotImplementedError

    def predict_match(
            self, teams: Tuple[Team, Team],
            rosters: Tuple[Roster, Roster],
            match_format: MatchFormat = MatchFormat.REGULAR) -> float:
        """Predict the win probability of a given match."""
        p_scores = self.predict_match_score(teams, rosters,
                                            match_format=match_format)
        p_win = 0.0

        for (score1, score2), p in p_scores.items():
            if score1 > score2:
                p_win += p

        return p_win

    def _predict_bo_match_score(self, teams: Tuple[Team, Team],
                                rosters: Tuple[Roster, Roster],
                                drawables: List[bool]) -> PScores:
        """Predict the scores of a given BO match."""
        p_scores = defaultdict(float)
        p_scores[(0, 0)] = 1.0

        p_undrawable = self.predict(teams, rosters)
        p_drawable = self.predict(teams, rosters, drawable=True)

        for drawable in drawables:
            p_win, p_draw = p_drawable if drawable else p_undrawable
            p_loss = 1.0 - p_win - p_draw
            new_p_scores = defaultdict(float)

            for (score1, score2), p in p_scores.items():
                new_p_scores[(score1 + 1, score2)] += p * p_win
                new_p_scores[(score1, score2 + 1)] += p * p_loss
                if drawable:
                    new_p_scores[(score1, score2)] += p * p_draw

            p_scores = new_p_scores

        # Add a tie-breaker game if needed.
        p_win, p_draw = p_undrawable
        new_p_scores = defaultdict(float)

        for (score1, score2), p in p_scores.items():
            if score1 == score2:
                new_p_scores[(score1 + 1, score2)] += p * p_win
                new_p_scores[(score1, score2 + 1)] += p * p_loss
            else:
                new_p_scores[(score1, score2)] += p

        p_scores = new_p_scores

        return p_scores

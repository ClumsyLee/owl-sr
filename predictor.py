from collections import defaultdict
from enum import Enum
from typing import Tuple, List, Dict

MatchFormat = Enum('MatchFormat', 'REGULAR TITLE')
MatchResult = Enum('MatchResult', 'WIN DRAW LOSS')
Map = Enum('Map', 'DORADO GIBRALTAR JUNKERTOWN ROUTE' +
                  'HANAMURA HORIZON TEMPLE VOLSKAYA' +
                  'ILIOS LIJIANG NEPAL OASIS' +
                  'EICHENWALDE HOLLYWOOD KINGS NUMBANI')
Team = Enum('Team', 'BOS DAL FLA GLA HOU LDN NYE PHI SEO SFS SHD VAL')
Roster = Tuple(str, str, str, str, str, str)
PScores = Dict[Tuple[int, int], float]

DRAWABLE_MAPS = set([Map.HANAMURA, Map.HORIZON, Map.TEMPLE, Map.VOLSKAYA,
                     Map.EICHENWALDE, Map.HOLLYWOOD, Map.KINGS, Map.NUMBANI])


class Predictor(object):
    """Base class for all OWL predictors."""

    def __init__(self) -> None:
        super().__init__()

    def train(self, teams: Tuple[Team, Team],
              rosters: Tuple[Roster, Roster],
              score: Tuple[int, int],
              drawable: bool) -> Tuple(float, float):
        """Given a game result, train the underlying model."""
        raise NotImplementedError

    def predict(self, teams: Tuple[Team, Team],
                rosters: Tuple[Roster, Roster],
                drawable: bool) -> Tuple(float, float):
        """Given two teams, return win/draw probabilities of them."""
        raise NotImplementedError

    def train_match(self, teams: Tuple[Team, Team],
                    rosters: Tuple[Roster, Roster],
                    maps: List[Map],
                    scores: List[Tuple[int, int]],
                    match_format: MatchFormat = MatchFormat.REGULAR) -> float:
        """Given a match result, train the underlying model,
        return the prediction point for this match."""
        # Calculate the prediction point.
        p_win = self.predict_match(teams, rosters, match_format=match_format)
        match_result = self._calculate_match_result(scores)
        if match_result == MatchResult.DRAW:
            point = 0.0
        else:
            win = match_result == MatchResult.WIN
            point = 0.25 - (p_win - win)**2

        # Train on each map.
        for map_name, score in zip(maps, scores):
            drawable = map_name in DRAWABLE_MAPS
            self.train(teams, rosters, score, drawable)

        return point

    def predict_match_scores(
            self, teams: Tuple[Team, Team],
            rosters: Tuple[Roster, Roster],
            match_format: MatchFormat = MatchFormat.REGULAR) -> PScores:
        """Predict the scores of a given match."""
        if match_format == MatchFormat.REGULAR:
            drawables = [True, False, True, False]
            return self._predict_bo_match_scores(teams, rosters,
                                                 drawables=drawables)
        elif match_format == MatchFormat.TITLE:
            drawables = [False, False, True, True, False]
            return self._predict_bo_match_scores(teams, rosters,
                                                 drawables=drawables)
        else:
            raise NotImplementedError

    def predict_match(
            self, teams: Tuple[Team, Team],
            rosters: Tuple[Roster, Roster],
            match_format: MatchFormat = MatchFormat.REGULAR) -> float:
        """Predict the win probability of a given match."""
        p_scores = self.predict_match_scores(teams, rosters,
                                             match_format=match_format)
        p_win = 0.0

        for (score1, score2), p in p_scores.items():
            if score1 > score2:
                p_win += p

        return p_win

    def _predict_bo_match_scores(self, teams: Tuple[Team, Team],
                                 rosters: Tuple[Roster, Roster],
                                 drawables: List[bool]) -> PScores:
        """Predict the scores of a given BO match."""
        p_scores = defaultdict(float)
        p_scores[(0, 0)] = 1.0

        p_undrawable = self.predict(teams, rosters, drawable=False)
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

    def _calculate_match_result(scores: List[Tuple[int, int]]) -> MatchResult:
        score1 = sum(score[0] for score in scores)
        score2 = sum(score[1] for score in scores)

        if score1 > score2:
            return MatchResult.WIN
        elif score1 == score2:
            return MatchResult.DRAW
        else:
            return MatchResult.LOSS

from collections import defaultdict
from csv import DictReader
from enum import Enum
from itertools import chain
from math import sqrt
from typing import Dict, List, Tuple

from trueskill import TrueSkill, calc_draw_margin

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
                 score: Tuple[int, int]) -> float:
        """Return the prediction point for this game.
        Assume it will not draw."""
        if score[0] == score[1]:
            return 0.0

        p_win, _ = self.predict(teams, rosters)
        win = score[0] > score[1]
        return 0.25 - (p_win - win)**2

    def train_all(self, csv_filename) -> float:
        """Given a csv file containing matches, train the underlying model.
        Return the prediction point for all the matches."""
        total_point = 0.0

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

                point = self.train(teams, rosters, score, drawable=drawable)
                total_point += point

        return total_point

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


class TrueSkillPredictor(Predictor):
    """TrueSkill predictor."""

    def __init__(self, mu: float=2500.0, sigma: float=2500.0 / 3.0,
                 beta: float=2500.0 / 6.0, tau: float=25.0 / 3.0,
                 draw_probability: float=0.03) -> None:
        super().__init__()

        self.env_drawable = TrueSkill(mu=mu, sigma=sigma, beta=beta, tau=tau,
                                      draw_probability=draw_probability)
        self.env_undrawable = TrueSkill(mu=mu, sigma=sigma, beta=beta, tau=tau,
                                        draw_probability=0.0)
        self.ratings = defaultdict(lambda: self.env_drawable.create_rating())

    def train(self, teams: Tuple[Team, Team],
              rosters: Tuple[Roster, Roster],
              score: Tuple[int, int],
              drawable: bool) -> float:
        """Given a game result, train the underlying model.
        Return the prediction point for this game before training."""
        point = self.evaluate(teams, rosters, score)

        score1, score2 = score
        if score1 > score2:
            ranks = [0, 1]  # Team 1 wins.
        elif score1 == score2:
            ranks = [0, 0]  # Draw.
        else:
            ranks = [1, 0]  # Team 2 wins.

        env = self.env_drawable if drawable else self.env_undrawable
        rosters_ratings = env.rate(self._rosters_ratings(rosters), ranks=ranks)
        self._update_rosters_ratings(rosters, rosters_ratings)

        return point

    def predict(self, teams: Tuple[Team, Team],
                rosters: Tuple[Roster, Roster],
                drawable: bool=False) -> Tuple[float, float]:
        """Given two teams, return win/draw probabilities of them."""
        env = self.env_drawable if drawable else self.env_undrawable

        team1_ratings, team2_ratings = self._rosters_ratings(rosters)

        delta_mu = (sum(r.mu for r in team1_ratings) -
                    sum(r.mu for r in team2_ratings))
        draw_margin = calc_draw_margin(env.draw_probability, 12)
        sum_sigma = sum(r.sigma**2 for r in chain(team1_ratings,
                                                  team2_ratings))
        denom = sqrt(12 * env.beta**2 + sum_sigma)

        p_win = env.cdf((delta_mu - draw_margin) / denom)
        p_not_loss = env.cdf((delta_mu + draw_margin) / denom)

        return p_win, p_not_loss - p_win

    def _rosters_ratings(self, rosters: Tuple[Roster, Roster]):
        roster1, roster2 = rosters
        roster1_ratings = [self.ratings[name] for name in roster1]
        roster2_ratings = [self.ratings[name] for name in roster2]
        return roster1_ratings, roster2_ratings

    def _update_rosters_ratings(self, rosters: Tuple[Roster, Roster],
                                rosters_ratings):
        for roster, ratings in zip(rosters, rosters_ratings):
            for name, rating in zip(roster, ratings):
                self.ratings[name] = rating


if __name__ == '__main__':
    predictor = TrueSkillPredictor()
    print(predictor.train_all('owl.csv'))
    print(predictor.ratings)

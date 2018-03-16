from collections import defaultdict
from itertools import chain
from math import sqrt
from typing import Dict, List, Sequence, Tuple

from scipy.optimize import fmin
from trueskill import TrueSkill, calc_draw_margin

from game import DRAWABLE_MAPS, Roster, Game
from fetcher import load_games

PScores = Dict[Tuple[int, int], float]


class Predictor(object):
    """Base class for all OWL predictors."""

    def __init__(self) -> None:
        super().__init__()

        self.expected_draws = 0.0
        self.real_draws = 0.0

        self.points = []

    def _train(self, teams: Tuple[str, str],
               rosters: Tuple[Roster, Roster],
               score: Tuple[int, int],
               drawable: bool) -> None:
        """Given a game result, train the underlying model."""
        raise NotImplementedError

    def predict(self, teams: Tuple[str, str],
                rosters: Tuple[Roster, Roster],
                drawable: bool) -> Tuple[float, float]:
        """Given two teams, return win/draw probabilities of them."""
        raise NotImplementedError

    def train(self, teams: Tuple[str, str],
              rosters: Tuple[Roster, Roster],
              score: Tuple[int, int],
              drawable: bool) -> float:
        """Given a game result, train the underlying model.
        Return the prediction point for this game before training."""
        # Count draws.
        if drawable:
            _, p_draw = self.predict(teams, rosters, drawable=True)
            self.expected_draws += p_draw
        if score[0] == score[1]:
            self.real_draws += 1.0

        point = self.evaluate(teams, rosters, score)
        self.points.append(point)
        self._train(teams, rosters, score, drawable=drawable)

        return point

    def evaluate(self, teams: Tuple[str, str],
                 rosters: Tuple[Roster, Roster],
                 score: Tuple[int, int]) -> float:
        """Return the prediction point for this game.
        Assume it will not draw."""
        if score[0] == score[1]:
            return 0.0

        p_win, _ = self.predict(teams, rosters, drawable=False)
        win = score[0] > score[1]
        return 0.25 - (p_win - win)**2

    def train_games(self, games: Sequence[Game]) -> float:
        """Given a sequence of games, train the underlying model.
        Return the prediction point for all the games."""
        total_point = 0.0

        for game in games:
            point = self.train(game.teams, game.rosters, game.score,
                               drawable=game.map_name in DRAWABLE_MAPS)
            total_point += point

        return total_point

    def predict_match_score(
            self, teams: Tuple[str, str],
            rosters: Tuple[Roster, Roster],
            match_format: str = 'regular') -> PScores:
        """Predict the scores of a given match."""
        if match_format == 'regular':
            drawables = [True, False, True, False]
            return self._predict_bo_match_score(teams, rosters,
                                                drawables=drawables)
        elif match_format == 'title':
            drawables = [False, False, True, True, False]
            return self._predict_bo_match_score(teams, rosters,
                                                drawables=drawables)
        else:
            raise NotImplementedError

    def predict_match(
            self, teams: Tuple[str, str],
            rosters: Tuple[Roster, Roster],
            match_format: str = 'regular') -> float:
        """Predict the win probability & diff expectation of a given match."""
        p_scores = self.predict_match_score(teams, rosters,
                                            match_format=match_format)
        p_win = 0.0
        e_diff = 0.0

        for (score1, score2), p in p_scores.items():
            if score1 > score2:
                p_win += p
            e_diff += p * (score1 - score2)

        return p_win, e_diff

    def _predict_bo_match_score(self, teams: Tuple[str, str],
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


class SimplePredictor(Predictor):
    """A simple predictor based on map differentials."""

    def __init__(self, alpha: float = 0.2, beta: float = 0.0) -> None:
        super().__init__()

        self.alpha = alpha
        self.beta = beta

        self.wins = defaultdict(int)
        self.records = defaultdict(int)

    def _train(self, teams: Tuple[str, str],
               rosters: Tuple[Roster, Roster],
               score: Tuple[int, int],
               drawable: bool) -> None:
        """Given a game result, train the underlying model.
        Return the prediction point for this game before training."""
        team1, team2 = teams
        score1, score2 = score

        if score1 > score2:
            # Team 1 wins.
            self.wins[team1] += 1
            self.wins[team2] -= 1
            self.records[(team1, team2)] += 1
            self.records[(team2, team1)] -= 1
        elif score1 == score2:
            # Draw.
            pass
        else:
            # Team 2 wins.
            self.wins[team2] += 1
            self.wins[team1] -= 1
            self.records[(team2, team1)] += 1
            self.records[(team1, team2)] -= 1

    def predict(self, teams: Tuple[str, str],
                rosters: Tuple[Roster, Roster],
                drawable: bool) -> Tuple[float, float]:
        """Given two teams, return win/draw probabilities of them."""
        team1, team2 = teams
        wins1 = self.wins[team1]
        wins2 = self.wins[team2]

        if wins1 > wins2:
            p_win = 0.5 + self.alpha
        elif wins1 == wins2:
            record = self.records[teams]
            if record > 0:
                p_win = 0.5 + self.beta
            elif record == 0:
                p_win = 0.5
            else:
                p_win = 0.5 - self.beta
        else:
            p_win = 0.5 - self.alpha

        return p_win, 0.0


class TrueSkillPredictor(Predictor):
    """TrueSkill predictor."""

    def __init__(self, mu: float = 2500.0, sigma: float = 2500.0 / 3.0,
                 beta: float = 2500.0 / 2.0, tau: float = 25.0 / 3.0,
                 draw_probability: float = 0.06) -> None:
        super().__init__()

        self.env_drawable = TrueSkill(mu=mu, sigma=sigma, beta=beta, tau=tau,
                                      draw_probability=draw_probability)
        self.env_undrawable = TrueSkill(mu=mu, sigma=sigma, beta=beta, tau=tau,
                                        draw_probability=0.0)
        self.ratings = defaultdict(lambda: self.env_drawable.create_rating())

    def _train(self, teams: Tuple[str, str],
               rosters: Tuple[Roster, Roster],
               score: Tuple[int, int],
               drawable: bool) -> None:
        """Given a game result, train the underlying model.
        Return the prediction point for this game before training."""
        score1, score2 = score
        if score1 > score2:
            ranks = [0, 1]  # Team 1 wins.
        elif score1 == score2:
            ranks = [0, 0]  # Draw.
        else:
            ranks = [1, 0]  # Team 2 wins.

        env = self.env_drawable if drawable else self.env_undrawable
        teams_ratings = env.rate(self._teams_ratings(teams, rosters),
                                 ranks=ranks)
        self._update_teams_ratings(teams, rosters, teams_ratings)

    def predict(self, teams: Tuple[str, str],
                rosters: Tuple[Roster, Roster],
                drawable: bool) -> Tuple[float, float]:
        """Given two teams, return win/draw probabilities of them."""
        env = self.env_drawable if drawable else self.env_undrawable

        team1_ratings, team2_ratings = self._teams_ratings(teams, rosters)
        size = len(team1_ratings) + len(team2_ratings)

        delta_mu = (sum(r.mu for r in team1_ratings) -
                    sum(r.mu for r in team2_ratings))
        draw_margin = calc_draw_margin(env.draw_probability, size, env=env)
        sum_sigma = sum(r.sigma**2 for r in chain(team1_ratings,
                                                  team2_ratings))
        denom = sqrt(size * env.beta**2 + sum_sigma)

        p_win = env.cdf((delta_mu - draw_margin) / denom)
        p_not_loss = env.cdf((delta_mu + draw_margin) / denom)

        return p_win, p_not_loss - p_win

    def _teams_ratings(self, teams: Tuple[str, str],
                       rosters: Tuple[Roster, Roster]):
        return [self._team_ratings(team, roster)
                for team, roster in zip(teams, rosters)]

    def _update_teams_ratings(self, teams: Tuple[str, str],
                              rosters: Tuple[Roster, Roster],
                              teams_ratings):
        for team, roster, ratings in zip(teams, rosters, teams_ratings):
            self._update_team_ratings(team, roster, ratings)

    def _team_ratings(self, team: str, roster: Roster):
        return [self.ratings[name] for name in roster]

    def _update_team_ratings(self, team: str, roster: Roster, team_ratings):
        for name, rating in zip(roster, team_ratings):
            self.ratings[name] = rating


class TeamTrueSkillPredictor(TrueSkillPredictor):
    """Team-based TrueSkill predictor."""

    def _team_ratings(self, team: str, roster: Roster):
        return [self.ratings[team]]

    def _update_team_ratings(self, team: str, roster: Roster, team_ratings):
        self.ratings[team] = team_ratings[0]


def optimize_beta(games: Sequence[Game], maxfun=100) -> None:
    def f(x):
        beta = 2500.0 / 6.0 * x[0]
        return -TrueSkillPredictor(beta=beta).train_games(games)

    args = fmin(f, [3.7], maxfun=maxfun)
    print(args, f(args))


def compare_methods(games: Sequence[Game]) -> None:
    classes = [
        SimplePredictor,
        TrueSkillPredictor,
        TeamTrueSkillPredictor
    ]

    for class_ in classes:
        predictor = class_()
        print(class_.__name__, predictor.train_games(games))


def predict_upcoming_matches(games: Sequence[Game], limit=6):
    pass


if __name__ == '__main__':
    past_games, future_games = load_games()
    predictor = TrueSkillPredictor()
    predictor.train_games(past_games)

    teams = ('NYE', 'SEO')
    rosters = (('Saebyeolbe', 'Meko', 'Jjonak', 'Ark', 'Libero', 'Mano'),
               ('Miro', 'Munchkin', 'tobi', 'ryujehong', 'ZUNBA', 'FLETA'))

    p_win, e_diff = predictor.predict_match(teams, rosters)
    win_percentage = round(p_win * 100.0)
    print(f'{teams[0]} vs. {teams[1]} ({win_percentage}%, {e_diff:+.1f})')

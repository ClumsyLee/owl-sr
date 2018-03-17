from collections import defaultdict, deque, OrderedDict
from itertools import chain
from math import sqrt
from pprint import pprint
from typing import Dict, List, Sequence, Set, Tuple

from scipy.optimize import fmin
from trueskill import calc_draw_margin, Rating, TrueSkill

from game import Roster, Game
from fetcher import (Availabilities,
                     load_availabilities,
                     load_games,
                     save_ratings_history)


PScores = Dict[Tuple[int, int], float]


class Predictor(object):
    """Base class for all OWL predictors."""

    def __init__(self, availabilities: Availabilities = None,
                 roster_queue_size: int = 10) -> None:
        super().__init__()

        # Players availabilities.
        if availabilities is None:
            availabilities = load_availabilities()
        self.availabilities = availabilities

        # Track recent used rosters.
        self.roster_queues = defaultdict(
            lambda: deque(maxlen=roster_queue_size))

        # Stage info.
        self.stage = None
        self.stage_match_ids = defaultdict(set)

        self.stage_wins = defaultdict(int)
        self.stage_map_diffs = defaultdict(int)
        self.stage_head_to_head_win_diffs = defaultdict(int)

        # Draw counts, used to adjust parameters related to draws.
        self.expected_draws = 0.0
        self.real_draws = 0.0

        # Points history, used to judge the performance of a predictor.
        self.points = []

    def _train(self, game: Game) -> None:
        """Given a game result, train the underlying model."""
        raise NotImplementedError

    def predict(self, teams: Tuple[str, str],
                rosters: Tuple[Roster, Roster] = None,
                drawable: bool = False) -> Tuple[float, float]:
        """Given two teams, return win/draw probabilities of them."""
        raise NotImplementedError

    def train(self, game: Game) -> float:
        """Given a game result, train the underlying model.
        Return the prediction point for this game before training."""
        self._update_rosters(game)
        self._update_stage_info(game)
        self._update_draws(game)

        point = self.evaluate(game)
        self.points.append(point)

        self._train(game)

        return point

    def evaluate(self, game: Game) -> float:
        """Return the prediction point for this game.
        Assume it will not draw."""
        if game.score[0] == game.score[1]:
            return 0.0

        p_win, _ = self.predict(game.teams, game.rosters, drawable=False)
        win = game.score[0] > game.score[1]
        return 0.25 - (p_win - win)**2

    def train_games(self, games: Sequence[Game]) -> float:
        """Given a sequence of games, train the underlying model.
        Return the prediction point for all the games."""
        total_point = 0.0

        for game in games:
            point = self.train(game)
            total_point += point

        return total_point

    def predict_match_score(
            self, teams: Tuple[str, str],
            rosters: Tuple[Roster, Roster] = None,
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
            rosters: Tuple[Roster, Roster] = None,
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

    def _update_rosters(self, game: Game) -> None:
        for team, roster in zip(game.teams, game.rosters):
            self.roster_queues[team].appendleft(roster)

    def _update_stage_info(self, game: Game) -> None:
        if game.stage != self.stage:
            self.stage = game.stage
            self.stage_match_ids.clear()

        for team, roster in zip(game.teams, game.rosters):
            self.stage_match_ids[team].add(game.match_id)

        # Wins & map diffs.
        team1, team2 = game.teams
        score1, score2 = game.score

        map_diff = score1 - score2
        self.stage_map_diffs[team1] += map_diff
        self.stage_map_diffs[team2] -= map_diff

        if score1 > score2:
            # Team 1 wins.
            self.stage_wins[team1] += 1
            self.stage_head_to_head_win_diffs[(team1, team2)] += 1
            self.stage_head_to_head_win_diffs[(team2, team1)] -= 1
        elif score1 == score2:
            # Draw.
            pass
        else:
            # Team 2 wins.
            self.stage_wins[team2] += 1
            self.stage_head_to_head_win_diffs[(team2, team1)] += 1
            self.stage_head_to_head_win_diffs[(team1, team2)] -= 1

    def _update_draws(self, game: Game) -> None:
        if game.drawable:
            _, p_draw = self.predict(game.teams, game.rosters, drawable=True)
            self.expected_draws += p_draw
        if game.score[0] == game.score[1]:
            self.real_draws += 1.0


class SimplePredictor(Predictor):
    """A simple predictor based on map differentials."""

    def __init__(self, alpha: float = 0.2, beta: float = 0.0, **kws) -> None:
        super().__init__(**kws)

        self.alpha = alpha
        self.beta = beta

    def _train(self, game: Game) -> None:
        """Given a game result, train the underlying model.
        Return the prediction point for this game before training."""
        pass

    def predict(self, teams: Tuple[str, str],
                rosters: Tuple[Roster, Roster] = None,
                drawable: bool = False) -> Tuple[float, float]:
        """Given two teams, return win/draw probabilities of them."""
        team1, team2 = teams
        wins1 = self.stage_wins[team1]
        wins2 = self.stage_wins[team2]

        if wins1 > wins2:
            p_win = 0.5 + self.alpha
        elif wins1 == wins2:
            record = self.stage_head_to_head_win_diffs[teams]
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
    """Team-based TrueSkill predictor."""

    def __init__(self, mu: float = 2500.0, sigma: float = 2500.0 / 3.0,
                 beta: float = 2500.0 / 2.0, tau: float = 25.0 / 3.0,
                 draw_probability: float = 0.06, **kws) -> None:
        super().__init__(**kws)

        self.env_drawable = TrueSkill(mu=mu, sigma=sigma, beta=beta, tau=tau,
                                      draw_probability=draw_probability)
        self.env_undrawable = TrueSkill(mu=mu, sigma=sigma, beta=beta, tau=tau,
                                        draw_probability=0.0)
        self.ratings = self._create_rating_jar()

    def _train(self, game: Game) -> None:
        """Given a game result, train the underlying model.
        Return the prediction point for this game before training."""
        score1, score2 = game.score
        if score1 > score2:
            ranks = [0, 1]  # Team 1 wins.
        elif score1 == score2:
            ranks = [0, 0]  # Draw.
        else:
            ranks = [1, 0]  # Team 2 wins.

        env = self.env_drawable if game.drawable else self.env_undrawable
        teams_ratings = env.rate(self._teams_ratings(game.teams, game.rosters),
                                 ranks=ranks)
        self._update_teams_ratings(game, teams_ratings)

    def predict(self, teams: Tuple[str, str],
                rosters: Tuple[Roster, Roster] = None,
                drawable: bool = False) -> Tuple[float, float]:
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
        return [self.ratings[teams[0]]], [self.ratings[teams[1]]]

    def _update_teams_ratings(self, game: Game, teams_ratings):
        for team, ratings in zip(game.teams, teams_ratings):
            self.ratings[team] = ratings[0]

    def _create_rating_jar(self):
        return defaultdict(lambda: self.env_drawable.create_rating())


class PlayerTrueSkillPredictor(TrueSkillPredictor):
    """Player-based TrueSkill predictor. Guess the rosters based on history
    when the rosters are not provided."""

    def __init__(self, **kws):
        super().__init__(**kws)

        self.best_rosters = {}
        self.ratings_history = OrderedDict()

    def save_ratings_history(self):
        save_ratings_history(self.ratings_history,
                             mu=self.env_drawable.mu,
                             sigma=self.env_drawable.sigma)

    def _teams_ratings(self, teams: Tuple[str, str],
                       rosters: Tuple[Roster, Roster]):
        if rosters is None:
            # No rosters provided, use the best roster.
            rosters = (self.best_rosters[teams[0]],
                       self.best_rosters[teams[1]])

        return ([self.ratings[name] for name in rosters[0]],
                [self.ratings[name] for name in rosters[1]])

    def _update_teams_ratings(self, game: Game, teams_ratings) -> None:
        for team, roster, ratings in zip(game.teams, game.rosters,
                                         teams_ratings):
            for name, rating in zip(roster, ratings):
                self.ratings[name] = rating

            self._record_team_ratings(team)

    def _record_team_ratings(self, team: str) -> None:
        match_number = len(self.stage_match_ids[team])
        match_key = (self.stage, match_number)

        members = self.availabilities[match_key][team]
        if match_key not in self.ratings_history:
            self.ratings_history[match_key] = {}
        ratings = self.ratings_history[match_key]

        # Record player ratings.
        for name in members:
            ratings[name] = self.ratings[name]

        # Update the best roster.
        best_roster = self._update_best_roster(team, members)

        # Record the team rating.
        ratings[team] = self._roster_rating(best_roster)

    def _update_best_roster(self, team: str, members: Set[str]):
        rosters = sorted(self.roster_queues[team],
                         key=lambda roster: self._min_roster_rating(roster),
                         reverse=True)
        best_roster = None

        for roster in rosters:
            if all(name in members for name in roster):
                best_roster = roster
                break

        if best_roster is None:
            # Just pick the best 6.
            sorted_members = sorted(members,
                                    key=lambda name: self._min_rating(name),
                                    reverse=True)
            best_roster = tuple(sorted_members[:6])

        self.best_rosters[team] = best_roster
        return best_roster

    def _roster_rating(self, roster: Roster) -> Tuple[float, float]:
        sum_mu = sum(self.ratings[name].mu for name in roster)
        sum_sigma = sqrt(sum(self.ratings[name].sigma**2 for name in roster))

        mu = sum_mu / 6.0
        sigma = sum_sigma / 6.0
        return Rating(mu=mu, sigma=sigma)

    def _min_roster_rating(self, roster: Roster) -> float:
        rating = self._roster_rating(roster)
        return rating.mu - 3.0 * rating.sigma

    def _min_rating(self, name: str) -> float:
        rating = self.ratings[name]
        return rating.mu - 3.0 * rating.sigma


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
        PlayerTrueSkillPredictor
    ]

    for class_ in classes:
        predictor = class_()
        print(class_.__name__, predictor.train_games(games))


def predict_stage(past_games: Sequence[Game],
                  future_games: Sequence[Game]) -> None:
    for game in games:
        pass


if __name__ == '__main__':
    past_games, future_games = load_games()

    compare_methods(past_games)

    predictor = PlayerTrueSkillPredictor()
    predictor.train_games(past_games)
    pprint(predictor.best_rosters)

    teams = ('VAL', 'BOS')

    p_win, e_diff = predictor.predict_match(teams)
    win_percentage = round(p_win * 100.0)
    print(f'{teams[0]} vs. {teams[1]} ({win_percentage}%, {e_diff:+.1f})')

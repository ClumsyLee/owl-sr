from collections import defaultdict
from datetime import datetime

from fetcher import load_games
from predictor import PlayerTrueSkillPredictor


TEAM_NAMES = {
    'SHD': 'Dragons',
    'SEO': 'Dynasty',
    'NYE': 'Excelsior',
    'DAL': 'Fuel',
    'PHI': 'Fusion',
    'GLA': 'Gladiator',
    'FLA': 'Mayhem',
    'HOU': 'Outlaws',
    'SFS': 'Shock',
    'LDN': 'Spitfire',
    'BOS': 'Uprising',
    'VAL': 'Valiant'
}

TEAM_FULL_NAMES = {
    'SHD': 'Shanghai Dragons',
    'SEO': 'Seoul Dynasty',
    'NYE': 'New York Excelsior',
    'DAL': 'Dallas Fuel',
    'PHI': 'Philadelphia Fusion',
    'GLA': 'Los Angeles Gladiators',
    'FLA': 'Florida Mayhem',
    'HOU': 'Houston Outlaws',
    'SFS': 'San Francisco Shock',
    'LDN': 'London Spitfire',
    'BOS': 'Boston Uprising',
    'VAL': 'Los Angeles Valiant'
}

TEAM_COLORS = {
    'SHD': ('#D22630', '#000000'),
    'SEO': ('#AA8A00', '#000000'),
    'NYE': ('#0F57EA', '#171C38'),
    'DAL': ('#0072CE', '#0C2340'),
    'PHI': ('#FF9E1B', '#000000'),
    'GLA': ('#3C1053', '#000000'),
    'FLA': ('#FEDA00', '#AF272F'),
    'HOU': ('#97D700', '#000000'),
    'SFS': ('#FC4C02', '#75787B'),
    'LDN': ('#59CBE8', '#FF8200'),
    'BOS': ('#174B97', '#F2DF00'),
    'VAL': ('#4A7729', '#E5D660')
}

RATING_CONFIDENCE = 1.64  # mu ± 1.64 * sigma -> 90% chance.


class MatchCard(object):
    def __init__(self, predictor, start_time, teams, score=None,
                 use_date=False) -> None:
        self.start_time = start_time
        self.teams = teams
        self.score = score

        # Render the HTML.
        hour = start_time.hour
        minute = '' if start_time.minute == 0 else f':{start_time.minute:02}'

        if start_time.minute >= 30:
            hour += 1

        suffix = 'a.m.' if hour < 12 else 'p.m.'
        if hour > 12:
            hour -= 12

        self.time_str = f'{hour}{minute} {suffix}'
        self.date_str = self.start_time.strftime('%A, %B %d').replace('0', '')
        self.use_date = use_date

        p_win, e_diff = predictor.predict_match(teams)
        win = round(p_win * 100.0)
        loss = 100 - win

        classes1 = ['win' if win > 50 else 'loss']
        classes2 = ['win' if win < 50 else 'loss']

        if score is not None:
            if score[0] < score[1]:
                classes1.append('lost')
            elif score[1] < score[0]:
                classes2.append('lost')

            score1 = str(score[0])
            score2 = str(score[1])
        else:
            score1 = ''
            score2 = ''

        self.html_template = f"""<div class="col-lg-4 col-md-6">
  <table class="table">
    <thead>
      <tr class="text-center">
        <th scope="col" class="pl-3 text-left align-middle text-muted" colspan="2">{{0.header}}</th>
        <th scope="col" class="compact d-none d-sm-table-cell"></th>
        <th scope="col" class="compact">win<br>prob.</th>
        <th scope="col" class="compact">map<br>+/-</th>
      </tr>
    </thead>
    <tbody>
      <tr scope="row" class="{' '.join(classes1)}">
        <th class="text-right compact">{render_team_logo(teams[0])}</th>
        <td>{render_team_link(predictor, teams[0])}</td>
        <td class="d-none d-sm-table-cell">{score1}</td>
        <td class="text-center{' low-chance' if win == 0 else ''}" style="background-color: rgba(255, 137, 0, {win / 100});">{percentage_str(win)}</td>
        <td class="text-center">{e_diff:+.1f}</td>
      </tr>
      <tr scope="row" class="{' '.join(classes2)}">
        <th class="text-right compact">{render_team_logo(teams[1])}</th>
        <td>{render_team_link(predictor, teams[1])}</td>
        <td class="d-none d-sm-table-cell">{score2}</td>
        <td class="text-center{' low-chance' if loss == 0 else ''}" style="background-color: rgba(255, 137, 0, {loss / 100});">{percentage_str(loss)}</td>
        <td class="text-center">{-e_diff:+.1f}</td>
      </tr>
    </tbody>
  </table>
</div>"""

    @property
    def header(self):
        return self.date_str if self.use_date else self.time_str

    @property
    def html(self):
        return self.html_template.format(self)

    @staticmethod
    def group_by_date(cards):
        card_groups = defaultdict(list)

        for card in cards:
            date = card.start_time.replace(hour=0, minute=0, second=0,
                                           microsecond=0)
            card_groups[date].append(card)

        return card_groups

    @staticmethod
    def group_by_team(cards):
        card_groups = defaultdict(list)

        for card in cards:
            for team in card.teams:
                card_groups[team].append(card)

        return card_groups


def render_team_logo(team, width=30) -> str:
    name = TEAM_NAMES[team]
    return f'<img src="imgs/{name}.png" alt="{name} Logo" width="{width}">'


def render_team_link(predictor, team) -> str:
    name = TEAM_NAMES[team]
    rating = predictor.ratings[team]
    title = f'{round(rating.mu)} ± {round(rating.sigma * RATING_CONFIDENCE)}'

    return f'<a href="/{name}" class="team" data-toggle="tooltip" data-placement="right" title="{title}">{name}</a>'


def percentage_str(percent):
    if percent == 0:
        return '&lt;1%'
    elif percent == 100:
        return '>99%'
    else:
        return f'{percent}%'


def render_page(endpoint: str, title: str, content: str) -> None:
    html = f"""<!doctype html>
<html lang="en">
  <head>
    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">

    <!-- Custom CSS -->
    <link rel="stylesheet" href="owl.css">

    <title>{title} | OWL Ratings</title>
  </head>
  <body>
    <nav class="navbar navbar-expand-lg navbar-light bg-light">
      <a class="navbar-brand" href="/">Overwatch League Ratings</a>
      <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNavDropdown" aria-controls="navbarNavDropdown" aria-expanded="false" aria-label="Toggle navigation">
        <span class="navbar-toggler-icon"></span>
      </button>
      <div class="collapse navbar-collapse" id="navbarNavDropdown">
        <ul class="navbar-nav">
          <li class="nav-item">
            <a class="nav-link{' active' if endpoint == 'index' else ''}" href="/">Standings</a>
          </li>
          <li class="nav-item">
            <a class="nav-link{' active' if endpoint == 'matches' else ''}" href="/matches">Matches</a>
          </li>
          <li class="nav-item">
            <a class="nav-link{' active' if endpoint == 'about' else ''}" href="/about">About</a>
          </li>
        </ul>
      </div>
    </nav>
    <div class="container">
      {content}
    </div>

    <hr class="mt-4 mb-2">
    <footer class="text-center pb-2">Created by <a href="https://github.com/ThomasLee969">ClumsyLi</a></footer>

    <!-- Optional JavaScript -->
    <!-- jQuery first, then Popper.js, then Bootstrap JS -->
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    <script>
      $(function () {{
        $('[data-toggle="tooltip"]').tooltip()
      }})
    </script>
  </body>
</html>"""

    with open(f'docs/{endpoint}.html', 'w') as file:
        print(html, file=file)


def render_index(predictor, future_games) -> None:
    content = ''

    p_stage = predictor.predict_stage(future_games)
    wins = predictor.stage_wins
    losses = predictor.stage_losses
    map_diffs = predictor.stage_map_diffs

    teams = sorted(p_stage.keys(),
                   key=lambda team: (round(p_stage[team][0] * 100),
                                     round(p_stage[team][1] * 100),
                                     wins[team],
                                     map_diffs[team]),
                   reverse=True)

    content += """<div class="row pt-4">
  <div class="col-lg-8 col-md-10 col-sm-12 mx-auto">
    <table class="table">
      <thead>
        <tr class="text-center">
          <th scope="col" class="compact"></th>
          <th scope="col"></th>
          <th scope="col" class="compacter">win</th>
          <th scope="col" class="compacter d-none d-sm-table-cell">loss</th>
          <th scope="col" class="compacter d-none d-sm-table-cell">map +/-</th>
          <th scope="col" class="compact">top 3<br>prob.</th>
          <th scope="col" class="compact">top 1<br>prob.</th>
        </tr>
      </thead>
      <tbody>"""

    for i, team in enumerate(teams):
        win = wins[team]
        loss = losses[team]
        map_diff = map_diffs[team]

        p_top3, p_top1 = p_stage[team]
        top3 = round(p_top3 * 100)
        top1 = round(p_top1 * 100)

        top1_str = percentage_str(top1)
        if isinstance(p_top3, bool):
            if p_top3:
                top3_str = '✓'
            else:
                top3_str = '-'
                top1_str = '-'
        else:
            top3_str = percentage_str(top3)

        content += f"""<tr scope="row" class="{'win' if i < 3 else 'loss'}">
  <th class="text-right">{render_team_logo(team)}</th>
  <td>{render_team_link(predictor, team)}</td>
  <td class="text-center">{win}</td>
  <td class="text-center d-none d-sm-table-cell">{loss}</td>
  <td class="text-center d-none d-sm-table-cell">{map_diff:+}</td>
  <td class="text-center{' low-chance' if top3 == 0 else ''}" style="background-color: rgba(255, 137, 0, {top3 / 100});">{top3_str}</td>
  <td class="text-center{' low-chance' if top1 == 0 else ''}" style="background-color: rgba(255, 137, 0, {top1 / 100});">{top1_str}</td>
</tr>"""

    content += """</tbody>
    </table>
  </div>
</div>"""
    render_page('index', f'OWL {predictor.stage} Standings', content)


def render_matches(past_games, future_games):
    predictor = PlayerTrueSkillPredictor()
    now = datetime.now()
    recent_matches = defaultdict(list)

    for game in past_games:
        if (now - game.start_time).days >= 1:
            predictor.train(game)
        else:
            recent_matches[game.match_id].append(game)

    # Render match cards.
    match_cards = []

    for match_id, games in recent_matches.items():
        start_time = games[0].start_time
        teams = games[0].teams
        score = [0, 0]

        for game in games:
            if game.score[0] > game.score[1]:
                score[0] += 1
            elif game.score[1] > game.score[0]:
                score[1] += 1

        match_cards.append(MatchCard(predictor=predictor,
                                     start_time=start_time,
                                     teams=teams,
                                     score=score))
        predictor.train_games(games)

    for game in future_games:
        match_cards.append(MatchCard(predictor=predictor,
                                     start_time=game.start_time,
                                     teams=game.teams))

    content = ''
    card_groups = MatchCard.group_by_date(match_cards)

    for date, cards in card_groups.items():
        date_str = cards[0].date_str

        content += f"""<h6 class="pt-4">{date_str}</h6>
<hr>
<div class="row">
  {''.join([card.html for card in cards])}
</div>"""

    render_page('matches', f'OWL {predictor.stage} Matches', content)
    return match_cards


def render_teams(predictor, match_cards):
    # Prepare the data for plots.
    ratings = predictor._create_rating_jar()

    last_stage = None
    stages = []
    mus = {team: [] for team in TEAM_NAMES}
    lower_bounds = []
    upper_bounds = []

    for (stage, match_number), row in predictor.ratings_history.items():
        if stage != last_stage:
            stages.append(stage)
            last_stage = stage
        else:
            stages.append('')

        lower_bound = 5000
        upper_bound = 0

        for team in TEAM_NAMES:
            if team in row:
                ratings[team] = row[team]
            mu = round(ratings[team].mu)

            mus[team].append(mu)
            lower_bound = min(lower_bound, mu)
            upper_bound = max(upper_bound, mu)

        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)

    # Prepare match cards.
    card_groups = MatchCard.group_by_team(match_cards)
    for cards in card_groups.values():
        for card in cards:
            card.use_date = True

    # Render the team pages.
    for team, name in TEAM_NAMES.items():
        full_name = TEAM_FULL_NAMES[team]
        color = TEAM_COLORS[team]

        content = f"""<h4 class="py-3 text-center">{render_team_logo(team, 40)} {full_name}</h5>
<div class="row">
  <div class="col-lg-8 col-md-10 col-sm-12 mx-auto">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.7.2/Chart.bundle.js"></script>
    <canvas id="myChart"></canvas>
  </div>
</div>
<h5 class="pt-4">Upcoming Matches</h5>
<hr>
<div class="row">
  {''.join([card.html for card in card_groups[team]])}
</div>
<script>
  var ctx = document.getElementById('myChart');
  ctx.height = 280;
  var chart = new Chart(ctx.getContext('2d'), {{
    // The type of chart we want to create
    type: 'line',

    // The data for our dataset
    data: {{
      labels: {stages},
      datasets: [{{
        backgroundColor: '{color[1]}',
        borderColor: '{color[0]}',
        data: {mus[team]},
        pointRadius: 4,
        fill: false
      }}, {{
        backgroundColor: 'rgba(0, 0, 0, 0.1)',
        borderColor: 'rgba(0, 0, 0, 0)',
        data: {lower_bounds},
        pointRadius: 0,
        pointHoverRadius: 0,
        pointBorderWidth: 0,
        fill: '+1'
      }}, {{
        backgroundColor: 'rgba(0, 0, 0, 0.1)',
        borderColor: 'rgba(0, 0, 0, 0)',
        data: {upper_bounds},
        pointRadius: 0,
        pointHoverRadius: 0,
        pointBorderWidth: 0,
        fill: false
      }}]
    }},

    // Configuration options go here
    options: {{
      animation: false,
      legend: {{
        display: false
      }},
      scales: {{
        yAxes: [{{
          ticks: {{
            stepSize: 500
          }}
        }}]
      }}
    }}
  }});
</script>"""

        render_page(name, full_name, content)


def render_about():
    content = """<div class="row pt-4">
  <div class="col-lg-8 col-md-10 col-sm-12 mx-auto">
    <h4 class="pt-4">How Did You Compute These Numbers?</h4>
    <p>I used <a href="https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/">TrueSkill</a> to keep track of skill ratings of individual players and estimate the win probabilities based on the ratings.</p>

    <h4 class="pt-4">What Is TrueSkill?</h4>
    <p>It is a ranking system designed by <a href="https://www.microsoft.com/en-us/research">Microsoft Research</a>:</p>
    <blockquote cite="https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/">
      <p>"The TrueSkill ranking system is a skill based ranking system for Xbox Live developed at Microsoft Research. The purpose of a ranking system is to both identify and track the skills of gamers in a game (mode) in order to be able to match them into competitive matches. The TrueSkill ranking system only uses the final standings of all teams in a game in order to update the skill estimates (ranks) of all gamers playing in this game. Ranking systems have been proposed for many sports but possibly the most prominent ranking system in use today is ELO."</p>
    </blockquote>

    <h4 class="pt-4">Why Not ELO/Glicko?</h4>
    <p>ELO and Glicko are designed for single-player games, which means every team will have a single rating. As a result, benching/transferring will not be handled properly.</p>

    <h4 class="pt-4">Did You Consider Draws/Bans/Transfers/Underages?</h4>
    <p>Yes.</p>

    <h4 class="pt-4">Did You Consider Tie-breakers/BO5 for the Title Matches?</h4>
    <p>Yes.</p>

    <h4 class="pt-4">But How Did You Determine the Rosters?</h4>
    <p>For a given team, sort all the rosters of it during its last 10 games based on their skill ratings, then pick the best one available. If there are no such rosters (e.g. a key member has gone), pick 6 players with the highest ratings.</p>

    <h4 class="pt-4">What Parameters Did You Use?</h4>
    <ul>
      <li><var>mu</var> = 2500</li>
      <li><var>sigma</var> = 2500 / 3</li>
      <li><var>beta</var> = 2500 / 2</li>
      <li><var>tau</var> = 25 / 3</li>
      <li><var>draw_probability</var> = 0.06 / 0 (based on the map)</li>
    </ul>
    <p>All these parameters are tuned on matches from preseason, stage 1, and stage 2.</p>

    <h4 class="pt-4">May I See Your Source Code?</h4>
    <p><a href="https://github.com/ThomasLee969/owl-sr">Sure</a>.</p>
  </div>
</div>"""

    render_page('about', f'About', content)


def render_all():
    past_games, future_games = load_games()

    predictor = PlayerTrueSkillPredictor()
    predictor.train_games(past_games)

    # Only predict the current stage (including title matches).
    future_games = [game for game in future_games
                    if predictor.stage in game.stage]

    render_index(predictor, future_games)
    match_cards = render_matches(past_games, future_games)
    render_teams(predictor, match_cards)
    render_about()


if __name__ == '__main__':
    render_all()

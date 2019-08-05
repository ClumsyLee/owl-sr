from collections import defaultdict, OrderedDict
from datetime import datetime

from fetcher import load_games
from game import Game, TEAM_DIVISIONS
from predictor import PlayerTrueSkillPredictor, Predictor


TEAM_NAMES = {
    'ATL': 'Reign',
    'BOS': 'Uprising',
    'CDH': 'Hunters',
    'DAL': 'Fuel',
    'FLA': 'Mayhem',
    'GLA': 'Gladiator',
    'GZC': 'Charge',
    'HOU': 'Outlaws',
    'HZS': 'Spark',
    'LDN': 'Spitfire',
    'NYE': 'Excelsior',
    'PAR': 'Eternal',
    'PHI': 'Fusion',
    'SEO': 'Dynasty',
    'SFS': 'Shock',
    'SHD': 'Dragons',
    'TOR': 'Defiant',
    'VAL': 'Valiant',
    'VAN': 'Titans',
    'WAS': 'Justice'
}

TEAM_FULL_NAMES = {
    'ATL': 'Atlanta Reign',
    'BOS': 'Boston Uprising',
    'CDH': 'Chengdu Hunters',
    'DAL': 'Dallas Fuel',
    'FLA': 'Florida Mayhem',
    'GLA': 'Los Angeles Gladiators',
    'GZC': 'Guangzhou Charge',
    'HOU': 'Houston Outlaws',
    'HZS': 'Hangzhou Spark',
    'LDN': 'London Spitfire',
    'NYE': 'New York Excelsior',
    'PAR': 'Paris Eternal',
    'PHI': 'Philadelphia Fusion',
    'SEO': 'Seoul Dynasty',
    'SFS': 'San Francisco Shock',
    'SHD': 'Shanghai Dragons',
    'TOR': 'Toronto Defiant',
    'VAL': 'Los Angeles Valiant',
    'VAN': 'Vancouver Titans',
    'WAS': 'Washington Justice'
}

TEAM_COLORS = {
    'ATL': ('#C4C4C4', '#910F1B'),
    'BOS': ('#174B97', '#F2DF00'),
    'CDH': ('#FFA000', '#B4926A'),
    'DAL': ('#0072CE', '#0C2340'),
    'FLA': ('#FEDA00', '#AF272F'),
    'GLA': ('#3C1053', '#000000'),
    'GZC': ('#67A2B2', '#122C42'),
    'HOU': ('#97D700', '#000000'),
    'HZS': ('#FB7299', '#5788CE'),
    'LDN': ('#59CBE8', '#FF8200'),
    'NYE': ('#0F57EA', '#171C38'),
    'PAR': ('#8D042D', '#303D56'),
    'PHI': ('#FF9E1B', '#000000'),
    'SEO': ('#AA8A00', '#000000'),
    'SFS': ('#FC4C02', '#75787B'),
    'SHD': ('#D22630', '#000000'),
    'TOR': ('#000000', '#C10021'),
    'VAL': ('#4A7729', '#E5D660'),
    'VAN': ('#09226B', '#2FB228'),
    'WAS': ('#990034', '#003768')
}

RATING_CONFIDENCE = 1.64  # mu ± 1.64 * sigma -> 90% chance.


def p_to_sort_key(p):
    if p is True:
        return 101
    elif p is False:
        return -1

    return round(p * 100)


class MatchCard(object):
    def __init__(self, predictor: Predictor, match: Game, use_date=False,
                 first_team=None) -> None:

        self.match = match
        self.use_date = use_date
        self.first_team = match.teams[0] if first_team is None else first_team

        # Render the HTML.
        hour = match.start_time.hour
        minute = ('' if match.start_time.minute == 0
                  else f':{match.start_time.minute:02}')

        suffix = 'a.m.' if hour < 12 else 'p.m.'
        if hour > 12:
            hour -= 12

        self.time_str = f'{hour}{minute} {suffix}'
        self.date_str = match.start_time.strftime('%A, %B %d').replace(' 0',
                                                                       ' ')

        p_win, e_diff = predictor.predict_match(match)
        win = round(p_win * 100)

        classes1 = ['win' if win > 50 else 'loss']
        classes2 = ['win' if win < 50 else 'loss']

        if match.score is not None:
            if match.score[0] < match.score[1]:
                classes1.append('lost')
            elif match.score[1] < match.score[0]:
                classes2.append('lost')

            score1 = str(match.score[0])
            score2 = str(match.score[1])
        else:
            score1 = ''
            score2 = ''

        self.rows = [
            f"""<tr scope="row" class="{' '.join(classes1)}">
  <th class="text-right compact">{render_team_logo(match.teams[0])}</th>
  <td class="pl-0">{render_team_link(predictor, match.teams[0], match.full_rosters[0])}</td>
  <td>{score1}</td>
  {render_chance_cell(p_win)}
  <td class="text-center">{e_diff:+.1f}</td>
</tr>""",
            f"""<tr scope="row" class="{' '.join(classes2)}">
  <th class="text-right compact">{render_team_logo(match.teams[1])}</th>
  <td class="pl-0">{render_team_link(predictor, match.teams[1], match.full_rosters[1])}</td>
  <td>{score2}</td>
  {render_chance_cell(1 - p_win)}
  <td class="text-center">{-e_diff:+.1f}</td>
</tr>"""]

        self.html_template = f"""<div class="col-lg-4 col-md-6">
  <table class="table" id="{match.match_id}">
    <thead>
      <tr class="text-center">
        <th scope="col" class="pl-3 text-left align-middle text-muted" colspan="2">{{0.header}}</th>
        <th scope="col" class="compacter"></th>
        <th scope="col" class="compact">win<br>prob.</th>
        <th scope="col" class="compacter">map<br>+/-</th>
      </tr>
    </thead>
    <tbody>
      {{0.row1}}
      {{0.row2}}
    </tbody>
  </table>
</div>"""

    @property
    def header(self):
        return self.date_str if self.use_date else self.time_str

    @property
    def row1(self):
        return self.rows[0 if self.match.teams[0] == self.first_team else 1]

    @property
    def row2(self):
        return self.rows[1 if self.match.teams[0] == self.first_team else 0]

    @property
    def html(self):
        return self.html_template.format(self)

    @staticmethod
    def group_by_date(cards):
        card_groups = defaultdict(list)

        for card in cards:
            date = without_time(card.match.start_time)
            card_groups[date].append(card)

        return card_groups

    @staticmethod
    def group_by_team(cards):
        card_groups = defaultdict(list)

        for card in cards:
            for team in card.match.teams:
                card_groups[team].append(card)

        return card_groups

    @staticmethod
    def group_by_stage(cards):
        card_groups = OrderedDict()

        for card in cards:
            if card.match.stage not in card_groups:
                card_groups[card.match.stage] = []
            card_groups[card.match.stage].append(card)

        return card_groups


def without_time(date):
    return date.replace(hour=0, minute=0, second=0, microsecond=0)


def render_team_logo(team, width=30) -> str:
    name = TEAM_NAMES[team]
    return f'<img src="imgs/{name}.png" alt="{name} Logo" width="{width}">'


def render_team_link(predictor, team, full_roster=None) -> str:
    if full_roster is None:
        rating = predictor.ratings[team]
    else:
        roster = predictor._best_roster(team, full_roster)
        rating = predictor._roster_rating(roster)

    name = TEAM_NAMES[team]
    title = f'{round(rating.mu)} ± {round(rating.sigma * RATING_CONFIDENCE)}'

    return f'<a href="/{name}" class="team" data-toggle="tooltip" data-placement="right" title="{title}">{name}</a>'


def render_chance_cell(p_win, extra_classes = []):
    percent = round(p_win * 100)

    if isinstance(p_win, bool):
        p_str = '✓' if p_win else '-'
    else:
        if percent == 0:
            p_str = '&lt;1%'
        elif percent == 100:
            p_str = '>99%'
        else:
            p_str = f'{percent}%'

    classes = set(extra_classes)
    classes.add('text-center')
    if percent == 0:
        classes.add('low-chance')

    return f'<td class="{" ".join(sorted(classes))}" style="background-color: rgba(255, 137, 0, {percent / 100});">{p_str}</td>'


def render_page(endpoint: str, title: str, content: str) -> None:
    html = f"""<!doctype html>
<html lang="en">
  <head>
    <!-- Global site tag (gtag.js) - Google Analytics -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=UA-49504108-8"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){{dataLayer.push(arguments);}}
      gtag('js', new Date());

      gtag('config', 'UA-49504108-8');
    </script>

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


def render_index(predictor, future_matches) -> None:
    content = ''

    p_stage = predictor.predict_stage(future_matches)
    # p_season = predictor.predict_season(future_matches)

    wins = predictor.stage_wins
    losses = predictor.stage_losses
    map_diffs = predictor.stage_map_diffs

    teams = sorted(p_stage.keys(),
                   key=lambda team: (p_to_sort_key(p_stage[team][0]),
                                     p_to_sort_key(p_stage[team][1]),
                                     predictor.stage_title_wins[team],
                                     wins[team],
                                     map_diffs[team]),
                   reverse=True)
    rows = []

    for i, team in enumerate(teams):
        division = TEAM_DIVISIONS[team].lower()

        win = wins[team]
        loss = losses[team]
        map_diff = map_diffs[team]

        season_win = predictor.wins[team]
        season_loss = predictor.losses[team]
        season_map_diff = predictor.map_diffs[team]

        p_title, p_top1 = p_stage[team]
        # p_playoff, p_champion = p_season[team]

        classes = set()
        classes.add(division + '-division')
        classes.add('win' if i < 8 else 'loss')

        rows.append(f"""<tr scope="row" class="{' '.join(sorted(classes))}">
  <th class="text-right">{render_team_logo(team)}</th>
  <td class="pl-0">{render_team_link(predictor, team)}</td>
  <td class="text-center">{win}</td>
  <td class="text-center d-none d-sm-table-cell">{loss}</td>
  <td class="text-center d-none d-sm-table-cell">{map_diff:+}</td>
  <td class="text-center d-none d-sm-table-cell">{season_win}</td>
  <td class="text-center d-none d-sm-table-cell">{season_loss}</td>
  <td class="text-center d-none d-sm-table-cell">{season_map_diff:+}</td>
</tr>""")

    title = f'{predictor.base_stage} Standings'
    content = f"""<h4 class="py-3 text-center">{title}</h4>
<div class="row">
  <div class="col-xl-6 col-lg-7 col-md-9 col-sm-12 mx-auto">
    <table class="table">
      <thead>
        <tr class="text-center">
          <th scope="col" class="compact"></th>
          <th scope="col"></th>
          <th scope="col" class="compacter">stage<br>win</th>
          <th scope="col" class="compacter d-none d-sm-table-cell">loss</th>
          <th scope="col" class="compacter d-none d-sm-table-cell">map +/-</th>
          <th scope="col" class="compacter d-none d-sm-table-cell">season<br>win</th>
          <th scope="col" class="compacter d-none d-sm-table-cell">loss</th>
          <th scope="col" class="compacter d-none d-sm-table-cell">map +/-</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </div>
</div>"""
    render_page('index', title, content)


def render_match_cards(predictor, past_games, future_matches):
    past_matches = defaultdict(list)

    for game in past_games:
        past_matches[game.match_id].append(game)

    # Only predict the current stage and its title matches.
    next_stage = None if len(future_matches) == 0 else future_matches[0].stage
    future_matches = [game for game in future_matches
                      if next_stage in game.stage]

    # Render match cards.
    match_cards = []

    for match_id, games in past_matches.items():
        score = [0, 0]
        for game in games:
            if game.score[0] > game.score[1]:
                score[0] += 1
            elif game.score[1] > game.score[0]:
                score[1] += 1
        # Add score and hide roster to simulate predictions beforehand.
        match = games[0]._replace(score=score, rosters=None)

        match_cards.append(MatchCard(predictor=predictor,
                                     match=match))
        predictor.train_games(games)

    for match in future_matches:
        match_cards.append(MatchCard(predictor=predictor,
                                     match=match))

    return match_cards


def render_matches(match_cards):
    card_groups = MatchCard.group_by_date(match_cards)
    now = datetime.now()
    dates = [date for date in card_groups.keys() if (now - date).days <= 0]
    sections = []

    for date in dates:
        cards = card_groups[date]
        date_str = cards[0].date_str

        sections += f"""<h6 class="pt-4">{date_str}</h6>
<hr>
<div class="row">
  {''.join([card.html for card in cards])}
</div>"""

    content = ''.join(sections)
    render_page('matches', 'Matches', content)
    return match_cards


def render_future_matches(future_cards) -> str:
    return f"""<h5 class="pt-4">Upcoming Matches</h5>
<hr>
<div class="row">
  {''.join([card.html for card in future_cards])}
</div>"""


def render_past_matches(past_cards) -> str:
    card_groups = MatchCard.group_by_stage(reversed(past_cards))
    sections = []

    for stage, cards in card_groups.items():
        sections += f"""<h5 class="pt-4">{stage}</h5>
<hr>
<div class="row">
  {''.join([card.html for card in cards])}
</div>"""

    return ''.join(sections)


def render_team(team, labels, match_info, mus, lower_bounds, upper_bounds,
                cards) -> None:
    name = TEAM_NAMES[team]
    full_name = TEAM_FULL_NAMES[team]
    color = TEAM_COLORS[team]

    past_cards = []
    future_cards = []

    for card in cards:
        card.use_date = True
        card.first_team = team

        if card.match.score is None:
            future_cards.append(card)
        else:
            past_cards.append(card)

    content = f"""<h4 class="py-3 text-center">
  {render_team_logo(team, 40)}
  <span class="align-middle pl-1">{full_name}</span>
</h4>
<div class="row">
  <div class="col-lg-8 col-md-10 col-sm-12 mx-auto">
    <canvas id="myChart"></canvas>
  </div>
</div>
{render_future_matches(future_cards)}
{render_past_matches(past_cards)}
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.7.2/Chart.bundle.js"></script>
<script>
var matchIds = {['' if info is None else info[0] for info in match_info]};
var opponents = {['' if info is None else TEAM_NAMES[info[1]] for info in match_info]};
var scores1 = {['' if info is None else info[2][0] for info in match_info]};
var scores2 = {['' if info is None else info[2][1] for info in match_info]};

function gotoHash(hash) {{
    window.location.hash = '#';
    window.location.hash = hash;

    $('tr').removeClass('highlight');
    $(hash + ' tr').addClass('highlight');
}}

var ctx = document.getElementById('myChart');
ctx.height = 220;
var chart = new Chart(ctx.getContext('2d'), {{
  // The type of chart we want to create
  type: 'line',

  // The data for our dataset
  data: {{
    labels: {labels},
    datasets: [{{
      backgroundColor: '{color[1]}',
      borderColor: '{color[0]}',
      data: {mus},
      pointRadius: {[0 if info is None else 4 for info in match_info]},
      pointHitRadius: {[0 if info is None else 6 for info in match_info]},
      pointHoverRadius: {[0 if info is None else 6 for info in match_info]},
      fill: false
    }}, {{
      backgroundColor: 'rgba(0, 0, 0, 0.1)',
      borderColor: 'rgba(0, 0, 0, 0)',
      data: {lower_bounds},
      pointRadius: 0,
      pointHitRadius: 0,
      pointHoverRadius: 0,
      pointBorderWidth: 0,
      fill: '+1'
    }}, {{
      backgroundColor: 'rgba(0, 0, 0, 0.1)',
      borderColor: 'rgba(0, 0, 0, 0)',
      data: {upper_bounds},
      pointRadius: 0,
      pointHitRadius: 0,
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
      xAxes: [{{
        display: false
      }}],
      yAxes: [{{
        ticks: {{
          stepSize: 500
        }}
      }}]
    }},
    tooltips: {{
      callbacks: {{
        footer: function(tooltipItems) {{
          var i = tooltipItems[0].index;
          if (opponents[i] != '') {{
            return scores1[i] + ':' + scores2[i] + ' ' + opponents[i];
          }}
        }}
      }}
    }}
  }}
}});

var isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints;
var lastMatchId = null;

ctx.onclick = function(event) {{
  var elements = chart.getElementAtEvent(event);
  if (elements.length == 0) {{
    lastMatchId = null;
    return;
  }}

  var match_id = matchIds[chart.getElementAtEvent(event)[0]._index];
  if (isTouchDevice && match_id != lastMatchId) {{
    lastMatchId = match_id;
    return;
  }}

  lastMatchId = match_id;
  gotoHash('#' + match_id);
}};
</script>"""

    render_page(name, full_name, content)


def render_teams(predictor, match_cards) -> None:
    # Prepare the data for plots.
    ratings = predictor._create_rating_jar()

    labels = []
    match_infos = defaultdict(list)
    mus = defaultdict(list)
    lower_bounds = []
    upper_bounds = []

    for (stage, match_number), row in predictor.ratings_history.items():
        labels.append(f'{stage}, Match {match_number}')

        lower_bound = 5000
        upper_bound = 0

        for team in TEAM_NAMES:
            ids = predictor.match_history[stage][team]

            if match_number <= len(ids):
                match_id = ids[match_number - 1]
                score = [0, 0]
                for t, s in predictor.scores[match_id].items():
                    if t == team:
                        score[0] = s
                    else:
                        opponent = t
                        score[1] = s

                info = (match_id, opponent, score)
            else:
                info = None

            match_infos[team].append(info)

            if team in row:
                ratings[team] = row[team]
            mu = round(ratings[team].mu)

            mus[team].append(mu)
            lower_bound = min(lower_bound, mu)
            upper_bound = max(upper_bound, mu)

        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)

    # Render the match cards.
    card_groups = MatchCard.group_by_team(match_cards)

    # Render the team pages.
    for team in TEAM_NAMES.keys():

        render_team(team, labels, match_infos[team], mus[team], lower_bounds,
                    upper_bounds, card_groups[team])


def render_about(predictor):
    content = f"""<div class="row pt-4">
  <div class="col-lg-8 col-md-10 col-sm-12 mx-auto">
    <h4 class="pt-4">Some Columns Are Missing on Mobiles?</h4>
    <p>Try to turn your phone sideways.</p>

    <h4 class="pt-4">How Did You Compute These Numbers?</h4>
    <p>I used <a href="https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/">TrueSkill</a> to keep track of skill ratings of individual players and estimate the win probabilities based on the ratings.</p>

    <h4 class="pt-4">What Is TrueSkill?</h4>
    <p>It is a ranking system designed by <a href="https://www.microsoft.com/en-us/research">Microsoft Research</a>:</p>
    <blockquote cite="https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/">
      <p>"The TrueSkill ranking system is a skill based ranking system for Xbox Live developed at Microsoft Research. The purpose of a ranking system is to both identify and track the skills of gamers in a game (mode) in order to be able to match them into competitive matches. The TrueSkill ranking system only uses the final standings of all teams in a game in order to update the skill estimates (ranks) of all gamers playing in this game. Ranking systems have been proposed for many sports but possibly the most prominent ranking system in use today is ELO."</p>
    </blockquote>

    <h4 class="pt-4">Why Not ELO/Glicko?</h4>
    <p>ELO and Glicko are designed for single-player games, which means every team will have a single rating. As a result, benching/transferring will not be handled properly.</p>

    <h4 class="pt-4">Did You Consider Draws/Bans/Transfers?</h4>
    <p>Yes.</p>

    <h4 class="pt-4">Did You Consider Tie-breakers/BO5 for the Title Matches?</h4>
    <p>Yes.</p>

    <h4 class="pt-4">But How Did You Determine the Rosters?</h4>
    <p>For a given team, sort all the rosters of it during its last {predictor.roster_queue_size} games (maps) based on their skill ratings, then pick the best one available. If there are no such rosters (e.g. a key member has gone), pick 6 players with the highest ratings.</p>

    <h4 class="pt-4">What Parameters Did You Use?</h4>
    <ul>
      <li><var>mu</var> = {predictor.mu:g}</li>
      <li><var>sigma</var> = {predictor.sigma:g}</li>
      <li><var>beta</var> = {predictor.beta:g}</li>
      <li><var>tau</var> = {predictor.tau:g}</li>
      <li><var>draw_probability</var> = {predictor.draw_probability:g} or 0 (depending on the map)</li>
    </ul>
    <p>All these parameters are tuned on matches from preseason, stage 1, and stage 2.</p>

    <h4 class="pt-4">May I See Your Source Code?</h4>
    <p><a href="https://github.com/ThomasLee969/owl-sr">Sure</a>.</p>
  </div>
</div>"""

    render_page('about', f'About', content)


def render_all():
    past_games, future_matches = load_games()

    predictor = PlayerTrueSkillPredictor()
    match_cards = render_match_cards(predictor, past_games, future_matches)
    predictor.save_ratings_history()

    render_index(predictor, future_matches)
    render_matches(match_cards)
    render_teams(predictor, match_cards)
    render_about(predictor)


if __name__ == '__main__':
    render_all()

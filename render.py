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
    'SHD': '#D22630',
    'SEO': '#AA8A00',
    'NYE': '#0F57EA',
    'DAL': '#0072CE',
    'PHI': '#FF9E1B',
    'GLA': '#3C1053',
    'FLA': '#FEDA00',
    'HOU': '#97D700',
    'SFS': '#FC4C02',
    'LDN': '#59CBE8',
    'BOS': '#174B97',
    'VAL': '#4A7729'
}


def percentage_str(percent):
    if percent == 0:
        return '&lt;1%'
    elif percent == 100:
        return '>99%'
    else:
        return f'{percent}%'


def render_page(endpoint: str, title: str, content: str):
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

    min_win = list(sorted(wins.values(), reverse=True))[2]

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
        name = TEAM_NAMES[team]
        win = wins[team]
        loss = losses[team]
        map_diff = map_diffs[team]

        p_top3, p_top1 = p_stage[team]
        top3 = round(p_top3 * 100)
        top1 = round(p_top1 * 100)

        max_win = win + sum(int(team in game.teams) for game in future_games)
        if max_win < min_win:
            top3_str = '-'
            top1_str = '-'
        else:
            top3_str = percentage_str(top3)
            top1_str = percentage_str(top1)

        content += f"""<tr scope="row" class="{'win' if i < 3 else 'loss'}">
  <th class="text-right"><img src="imgs/{name}.png" alt="{name} Logo" width="30"></th>
  <td><a href="/{name}" class="team">{name}</a></td>
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


def render_matches(predictor, future_games) -> None:
    content = ''
    last_date = None

    for game in future_games:
        # Add a separator if needed.
        date = (game.start_time.year,
                game.start_time.month,
                game.start_time.day)
        if date != last_date:
            if last_date is not None:
                content += '</div>'  # Ending tag for the last row.

            date_str = game.start_time.strftime('%A, %B %d').replace('0', '')
            content += f'<h6 class="pt-4">{date_str}</h6><hr><div class="row">'

            last_date = date

        # Add the current match.
        if game.start_time.hour <= 12 or game.start_time.minute != 0:
            raise RuntimeError(f'Cannot handle {game.start_time}.')
        time_str = f'{game.start_time.hour - 12} p.m.'

        p_win, e_diff = predictor.predict_match(game.teams)

        win = round(p_win * 100.0)
        loss = 100 - win

        name1 = TEAM_NAMES[game.teams[0]]
        name2 = TEAM_NAMES[game.teams[1]]

        content += f"""<div class="col-lg-4 col-md-6">
          <table class="table">
            <thead>
              <tr class="text-center">
                <th scope="col" class="text-muted compact">{time_str}</th>
                <th scope="col"></th>
                <th scope="col" class="compactr d-none d-sm-table-cell"></th>
                <th scope="col" class="compact">win<br>prob.</th>
                <th scope="col" class="compact">map<br>+/-</th>
              </tr>
            </thead>
            <tbody>
              <tr scope="row" class="{'win' if win > 50 else 'loss'}">
                <th class="text-right"><img src="imgs/{name1}.png" alt="{name1} Logo" width="30"></th>
                <td><a href="/{name1}" class="team">{name1}</a></td>
                <td class="d-none d-sm-table-cell"></td>
                <td class="text-center{' low-chance' if win == 0 else ''}" style="background-color: rgba(255, 137, 0, {win / 100});">{percentage_str(win)}</td>
                <td class="text-center">{e_diff:+.1f}</td>
              </tr>
              <tr scope="row" class="{'win' if win < 50 else 'loss'}">
                <th class="text-right"><img src="imgs/{name2}.png" alt="{name2} Logo" width="30"></th>
                <td><a href="/{name2}" class="team">{name2}</a></td>
                <td class="d-none d-sm-table-cell"></td>
                <td class="text-center{' low-chance' if loss == 0 else ''}" style="background-color: rgba(255, 137, 0, {loss / 100});">{percentage_str(loss)}</td>
                <td class="text-center">{-e_diff:+.1f}</td>
              </tr>
            </tbody>
          </table>
        </div>"""

    if last_date is not None:
        content += '</div>'  # Ending tag for the last row.

    render_page('matches', f'OWL {predictor.stage} Matches', content)


def render_teams(predictor):
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

    for team, name in TEAM_NAMES.items():
        full_name = TEAM_FULL_NAMES[team]
        color = TEAM_COLORS[team]

        content = f"""<h4 class="py-3 text-center">{full_name}</h5>
      <div class="row">
        <div class="col-lg-8 col-md-10 col-sm-12 mx-auto">
          <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.7.2/Chart.bundle.js"></script>
          <canvas id="myChart"></canvas>

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
                  borderColor: '{color}',
                  data: {mus[team]},
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
          </script>
        </div>
      </div>"""

        render_page(name, full_name, content)


def render_about(predictor):
    content = ''
    render_page('about', f'About', content)


def render_all():
    past_games, future_games = load_games()

    predictor = PlayerTrueSkillPredictor()
    predictor.train_games(past_games)

    # Only predict the current stage (including title matches).
    future_games = [game for game in future_games
                    if predictor.stage in game.stage]

    render_index(predictor, future_games)
    render_matches(predictor, future_games)
    render_teams(predictor)
    render_about(predictor)


if __name__ == '__main__':
    render_all()

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

    <title>{title}</title>
  </head>
  <body>
    <nav class="navbar navbar-expand-lg navbar-light bg-light">
      <a class="navbar-brand" href="#">Overwatch League Ratings</a>
      <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNavDropdown" aria-controls="navbarNavDropdown" aria-expanded="false" aria-label="Toggle navigation">
        <span class="navbar-toggler-icon"></span>
      </button>
      <div class="collapse navbar-collapse" id="navbarNavDropdown">
        <ul class="navbar-nav">
          <li class="nav-item">
            <a class="nav-link{' active' if endpoint == '' else ''}" href="/">Teams</a>
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

    if endpoint == '':
        endpoint = 'index'
    with open(f'docs/{endpoint}.html', 'w') as file:
        print(html, file=file)


def render_matches():
    past_games, future_games = load_games()

    predictor = PlayerTrueSkillPredictor()
    predictor.train_games(past_games)

    content = ''
    last_date = None

    for game in future_games:
        # Only predict the current stage (including title matches).
        if predictor.stage not in game.stage:
            break

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

        team1 = TEAM_NAMES[game.teams[0]]
        team2 = TEAM_NAMES[game.teams[1]]

        content += f"""<div class="col-lg-4 col-md-6">
          <table class="table">
            <thead>
              <tr class="text-center">
                <th scope="col" class="font-weight-light text-muted" style="width: 5%">{time_str}</th>
                <th scope="col"></th>
                <th scope="col" style="width: 5%"></th>
                <th scope="col" class="font-weight-light" style="width: 5%">win<br>prob.</th>
                <th scope="col" class="font-weight-light" style="width: 5%">map<br>+/-</th>
              </tr>
            </thead>
            <tbody>
              <tr scope="row" class="{'win' if win > 50 else 'loss'}">
                <th class="text-right"><img src="imgs/{team1}.png" alt="{team1} Logo" width="30"></th>
                <td>{team1}</td>
                <td></td>
                <td class="text-center" style="background-color: rgba(255, 137, 0, {win / 100});">{win}%</td>
                <td class="text-center">{e_diff:+.1f}</td>
              </tr>
              <tr scope="row" class="{'win' if win < 50 else 'loss'}">
                <th class="text-right"><img src="imgs/{team2}.png" alt="{team2} Logo" width="30"></th>
                <td>{team2}</td>
                <td></td>
                <td class="text-center" style="background-color: rgba(255, 137, 0, {loss / 100});">{loss}%</td>
                <td class="text-center">{-e_diff:+.1f}</td>
              </tr>
            </tbody>
          </table>
        </div>"""

    if last_date is not None:
        content += '</div>'  # Ending tag for the last row.

    render_page('matches', f'{predictor.stage} Matches | OWL Ratings', content)


def render_all():
    render_matches()


if __name__ == '__main__':
    render_all()

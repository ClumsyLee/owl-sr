from datetime import datetime
from typing import NamedTuple, Set, Tuple

DRAWABLE_MAPS = set([
    'hanamura', 'horizon-lunar-colony', 'temple-of-anubis', 'volskaya',
    'eichenwalde', 'hollywood', 'kings-row', 'numbani'
])

TEAMS = set(['BOS', 'DAL', 'FLA', 'GLA', 'HOU', 'LDN',
             'NYE', 'PHI', 'SEO', 'SFS', 'SHD', 'VAL'])

Roster = Tuple[str, str, str, str, str, str]
FullRoster = Set[str]


class Game(NamedTuple):
    """Describe a single game."""
    teams: Tuple[str, str]
    match_format: str

    match_id: int = None
    stage: str = None
    start_time: datetime = None

    game_id: int = None
    game_number: int = None
    map_name: str = None
    score: Tuple[int, int] = None
    rosters: Tuple[Roster, Roster] = None
    full_rosters: Tuple[FullRoster, FullRoster] = None

    @property
    def drawable(self):
        return self.map_name in DRAWABLE_MAPS

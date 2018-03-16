from datetime import datetime
from typing import NamedTuple, Tuple

DRAWABLE_MAPS = set([
    'hanamura', 'horizon-lunar-colony', 'temple-of-anubis', 'volskaya',
    'eichenwalde', 'hollywood', 'kings-row', 'numbani'
])

Roster = Tuple[str, str, str, str, str, str]


class Game(NamedTuple):
    """Describe a single game."""
    match_id: int
    stage: str
    start_time: datetime
    teams: Tuple[str, str]
    match_format: str

    game_id: int = None
    game_number: int = None
    map_name: str = None
    score: Tuple[int, int] = None
    rosters: Tuple[Roster, Roster] = None

    @property
    def drawable(self):
        return self.map_name in DRAWABLE_MAPS

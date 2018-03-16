from datetime import datetime
from enum import Enum
from typing import NamedTuple, Tuple

DRAWABLE_MAPS = set([
    'hanamura', 'horizon-lunar-colony', 'temple-of-anubis', 'volskaya',
    'eichenwalde', 'hollywood', 'kings-row', 'numbani'
])


Team = Enum('Team', 'BOS DAL FLA GLA HOU LDN NYE PHI SEO SFS SHD VAL')
MatchFormat = Enum('MatchFormat', 'REGULAR TITLE')
Roster = Tuple[str, str, str, str, str, str]


class Game(NamedTuple):
    """Describe a single game."""
    match_id: int
    stage: str
    start_time: datetime
    teams: Tuple[Team, Team]
    match_format: MatchFormat

    game_id: int = None
    game_number: int = None
    map_name: str = None
    score: Tuple[int, int] = None
    rosters: Tuple[Roster, Roster] = None

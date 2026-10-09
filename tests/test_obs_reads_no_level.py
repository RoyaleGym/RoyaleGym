"""No observation reads a unit's level, so a new level changes no input.

``EntityState.level`` is the level a unit plays at. The observation shows what that level
does (hitpoints, through fixed scales: hp / 1000, hp / max_hp, min(1, hp / 1000)), never
the number. So when the engine admits a level above the old maximum (royalesim 0.1.27
admits 17), every observation of levels 1 to 16 stays byte for byte what it was, and a
higher level only shows as more hitpoints under the same formulas. Here two builders run
side by side over the same battle, one shown the states as played and one shown every
entity at another level, and every key of every observation must match, as must the
action mask.

SKIPS
    Everything here needs the engine. Not a pass without it.
"""

from __future__ import annotations

import msgspec
import numpy as np
import pytest

from royalegym.action import TileActionParser
from royalegym.env import ClashParallelEnv
from royalegym.obs import SAME_UNIT_ALIASES, EntityListObsBuilder, SpatialObsBuilder
from royalegym.protocol import BLUE, RED, ShuffleMode
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, core_available
from royalegym.state_mutator import DefaultStateMutator, deck_ids

pytestmark = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))

BLUE_DECK = (["Musketeer", "Wallbreakers", "Ghost", "Miner", "GoblinDrill", "Tesla", "Mirror",
              "IceWizard"], [2, 1, 1, 0, 0, 0, 0, 0])
RED_DECK = (["GoldenKnight", "DarkPrince", "Prince", "ZapMachine", "Clone", "Vines", "Firecracker",
             "Graveyard"], [0, 2, 0, 0, 0, 0, 1, 0])
#: Every observation option, so every plane and field is built.
ALL_OPTIONS = dict(
    card_identity=True, spell_aim_after_ticks=0, spell_identity=True, effect_identity=True,
    evolutions=True, evolution_progress=True, heroes=True, unit_status=True, card_status=True,
    unit_actions=True, unit_identity=True, unit_aliases=dict(SAME_UNIT_ALIASES),
    button_index=True, enemy_queue=True,
)
BUILDERS = {
    "spatial": lambda: SpatialObsBuilder(**ALL_OPTIONS),
    "entity_list": EntityListObsBuilder,
}


def _battle(seed: int, steps: int = 220):
    eng = RustEngine()
    mutator = DefaultStateMutator(
        decks=[deck_ids(BLUE_DECK[0], eng.cards()), deck_ids(RED_DECK[0], eng.cards())],
        forms=[BLUE_DECK[1], RED_DECK[1]], shuffle=ShuffleMode.INDEPENDENT,
    )
    env = ClashParallelEnv(eng, action_parser=TileActionParser(ability_buttons=True),
                           state_mutator=mutator)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    states = [env.battle_state]
    while env.agents and len(states) < steps:
        acts = {}
        for agent in env.agents:
            legal = np.flatnonzero(env.action_masks(agent))
            legal = legal[legal != 0]
            acts[agent] = int(rng.choice(legal)) if len(legal) and rng.random() < 0.4 else 0
        env.step(acts)
        states.append(env.battle_state)
    return eng, states


def _at_level(state, level_of):
    return msgspec.structs.replace(
        state, entities=[msgspec.structs.replace(e, level=level_of(e)) for e in state.entities]
    )


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_the_observation_is_the_same_whatever_level_a_unit_reports(name):
    eng, states = _battle(7)
    parser = TileActionParser(ability_buttons=True)
    parser.bind(eng)
    shifted = [_at_level(s, lambda e: 17 if e.level < 0 else e.level + 6) for s in states]
    changed = sum(a.level != b.level for s, t in zip(states, shifted, strict=True)
                  for a, b in zip(s.entities, t.entities, strict=True))
    assert changed > 1000, changed  # vacuity: the levels really moved, on many entities
    played, other = BUILDERS[name](), BUILDERS[name]()
    for b in (played, other):
        b.bind(eng, parser)
    played.reset(states[0])
    other.reset(shifted[0])
    for i, (s, t) in enumerate(zip(states, shifted, strict=True)):
        for team in (BLUE, RED):
            mask = parser.action_mask(s, team)
            # The legal moves do not read levels either.
            assert parser.action_mask(t, team).tobytes() == mask.tobytes(), (i, team)
            got, want = other.build(t, team, mask), played.build(s, team, mask)
            assert got.keys() == want.keys()
            for k in want:
                assert got[k].dtype == want[k].dtype, (i, team, k)
                assert got[k].tobytes() == want[k].tobytes(), (i, team, k)

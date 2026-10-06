"""Each ability button's card and status by INDEX (observation requirement row 23, 2026-10-06).

The card-status button fields are keyed by card id, and the mask's tail gives button k's ready
bit with nothing naming k. A network reading the two learned "button 0 is the Musketeer" from the
deck order it trained on; dealt a deck with the Ice Golem first, the same network pressed its
Musketeer far less. ``SpatialObsBuilder(button_index=True)`` names button k's card (a one-hot laid
out like ``own_hand_cards``) and gives its available, spent and cooldown, index by index.
"""

from __future__ import annotations

import msgspec
import numpy as np
import pytest

from royalegym.action import TileActionParser
from royalegym.env import ClashParallelEnv
from royalegym.mock_engine import MockEngine
from royalegym.obs import COOLDOWN_SCALE, SpatialObsBuilder
from royalegym.protocol import BLUE, ShuffleMode, ability_row
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, core_available
from royalegym.state_mutator import DefaultStateMutator, deck_ids

needs_engine = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
FIELDS = ("own_button_cards", "own_button_available_by_index", "own_button_spent_by_index",
          "own_button_cooldown_by_index")


def _env(engine, names, forms, **builder):
    deck = deck_ids(names, engine.cards())
    return ClashParallelEnv(
        engine, action_parser=TileActionParser(ability_buttons=True),
        obs_builder=SpatialObsBuilder(button_index=True, **builder),
        state_mutator=DefaultStateMutator(decks=[deck, deck], forms=[forms, [0] * 8],
                                          shuffle=ShuffleMode.NONE),
    )


@needs_engine
def test_each_button_index_names_its_card_whatever_the_deck_order() -> None:
    eng = RustEngine()
    names_of = {c.card_id: c.name for c in eng.cards()}
    first_button = []
    for heroes in (("Musketeer", "IceGolemite"), ("IceGolemite", "Musketeer")):
        names = [*heroes, "Knight", "Archer", "Giant", "Minions", "Fireball", "Zap"]
        env = _env(eng, names, [2, 2, 0, 0, 0, 0, 0, 0])
        obs, _ = env.reset(seed=1)
        k_buttons, n = env.action_parser.n_buttons, len(eng.cards())
        off = env.obs_builder.vector_offsets()
        onehot = obs["blue"]["vector"][off["own_button_cards"]].reshape(k_buttons, n + 1)
        rows = env.battle_state.players[BLUE].abilities
        assert len(rows) == 2, rows
        for k in range(k_buttons):
            want = ability_row(rows[k]).card_id if k < len(rows) else n
            assert onehot[k].sum() == 1
            assert int(onehot[k].argmax()) == want, (heroes, k)
        first_button.append(names_of[int(onehot[0].argmax())])
    assert first_button == ["Musketeer", "IceGolemite"], "button 0 follows the deck order"


def test_the_status_by_index_reads_each_row() -> None:
    """Rows set by hand on a MockEngine state: available, spent and the cooldown, per index; an
    index with no button is all zeros and names "none"."""

    class Rows(MockEngine):
        ability_button_count = 3

    eng = Rows()
    names = ["Knight", "Archer", "Giant", "Minions", "Fireball", "Zap", "Cannon", "Musketeer"]
    env = _env(eng, names, [0] * 8)
    env.reset(seed=1)
    ids = {c.name: c.card_id for c in eng.cards()}
    state = env.battle_state
    rows = [[1, 0, 2, ids["Knight"], 0], [0, 1, 3, ids["Cannon"], 300]]
    state = msgspec.structs.replace(state, players=[
        msgspec.structs.replace(state.players[BLUE], abilities=rows), state.players[1]
    ])
    builder = env.obs_builder
    mask = env.action_parser.action_mask(state, BLUE)
    vec = builder.build(state, BLUE, mask)["vector"]
    off = builder.vector_offsets()
    n = len(eng.cards())
    onehot = vec[off["own_button_cards"]].reshape(3, n + 1)
    assert [int(r.argmax()) for r in onehot] == [ids["Knight"], ids["Cannon"], n]
    assert list(vec[off["own_button_available_by_index"]]) == [1, 0, 0]
    assert list(vec[off["own_button_spent_by_index"]]) == [0, 1, 0]
    cooldown = vec[off["own_button_cooldown_by_index"]]
    assert cooldown[0] == 0
    assert cooldown[1] == np.float32(300 / COOLDOWN_SCALE)
    assert cooldown[2] == 0


def test_off_by_default_and_appended_after_the_fair_fields() -> None:
    eng = MockEngine()
    names = ["Knight", "Archer", "Giant", "Minions", "Fireball", "Zap", "Cannon", "Musketeer"]
    plain = ClashParallelEnv(eng, action_parser=TileActionParser(ability_buttons=True),
                             obs_builder=SpatialObsBuilder())
    assert not set(FIELDS) & set(plain.obs_builder.vector_offsets())
    assert "button_index" not in plain.obs_builder.config()
    env = _env(MockEngine(), names, [0] * 8)
    off = env.obs_builder.vector_offsets()
    plain_off = plain.obs_builder.vector_offsets()
    assert all(off[k] == v for k, v in plain_off.items()), "no existing field moved"
    k_buttons, n = env.action_parser.n_buttons, len(eng.cards())
    sizes = [off[f].stop - off[f].start for f in FIELDS]
    assert sizes == [k_buttons * (n + 1), k_buttons, k_buttons, k_buttons]
    assert off[FIELDS[0]].start == max(s.stop for s in plain_off.values())
    assert env.obs_builder.config()["button_index"] is True


def test_button_index_needs_buttons() -> None:
    env_kw = dict(obs_builder=SpatialObsBuilder(button_index=True))
    with pytest.raises(ValueError, match="button_index"):
        ClashParallelEnv(MockEngine(), action_parser=TileActionParser(), **env_kw)

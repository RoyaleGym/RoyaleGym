"""The open-bridge territory model (royalesim 0.1.21, arena.TERRITORY_MODEL
"enemy_tower_no_deploy_rects_open_bridge").

Before it, both engines closed the whole river band to troops, a rule with no source. The replays'
bridge plays and taps measured on client 16.402 (2026-10-07) show a troop standing on a bridge whose
lane's enemy princess has fallen. Under the open-bridge model the band has no rule of its own:
the water (every river cell but the bridges) and the alive enemy towers' NoDeploySize rects
decide, and a standing princess's rect, which reaches the far bank, still closes its lane's
bridge. The mask and MockEngine follow ``DeployRules.river_band_closed_to_troops``.

SKIPS
    The RustEngine test skips without the engine. Not a pass.
"""

from __future__ import annotations

import msgspec
import pytest

from royalegym.action import TileActionParser, mask_disagreements
from royalegym.mock_engine import MockEngine
from royalegym.protocol import (
    BLUE,
    CLOSED_RIVER_BAND,
    OPEN_BRIDGE,
    TERRITORY_MODELS,
    DeployCommand,
    DeployStatus,
    MatchSetup,
    ShuffleMode,
)
from royalegym.rust_engine import CORE_IMPORT_ERROR, core_available

needs_engine = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
DECK = ["Knight", "Archer", "Giant", "Minions", "Fireball", "Zap", "Cannon", "Musketeer"]


def _model(engine, model: str):
    """``engine`` playing ``model``: its rules, which the mask and its own checks both read."""
    engine._rules = msgspec.structs.replace(engine._rules, territory_model=model)
    return engine


def _setup(engine, princesses_up: bool) -> MatchSetup:
    ids = {c.name: c.card_id for c in engine.cards()}
    deck = [ids[n] for n in DECK]
    red = [2400, 1400, 1400] if princesses_up else [2400, 0, 0]
    return MatchSetup(decks=[deck, deck], shuffle=ShuffleMode.NONE, elixir_milli=[10000, 10000],
                      start_tick=engine.rules().deploy_lockout_ticks,
                      tower_hp=[[2400, 1400, 1400], red])


def _bridge_points(engine) -> list[tuple[int, int]]:
    """Blue taps in the middle of each bridge, on each of the river's half-rows."""
    a = engine.arena()
    lo, hi = a.water_half_rows
    return [(x, hy * a.half_size + a.half_size // 2)
            for x in a.bridge_centers_x() for hy in range(lo, hi + 1)]


def test_both_names_are_implemented_and_say_which_band_rule():
    assert TERRITORY_MODELS == (CLOSED_RIVER_BAND, OPEN_BRIDGE)
    rules = MockEngine().rules()
    closed = msgspec.structs.replace(rules, territory_model=CLOSED_RIVER_BAND)
    open_ = msgspec.structs.replace(rules, territory_model=OPEN_BRIDGE)
    assert closed.river_band_closed_to_troops
    assert not open_.river_band_closed_to_troops


@pytest.mark.parametrize("model", TERRITORY_MODELS)
@pytest.mark.parametrize("princesses_up", [True, False])
def test_the_mask_and_mockengine_agree_on_every_action(model, princesses_up):
    eng = _model(MockEngine(), model)
    eng.reset(1, _setup(eng, princesses_up))
    parser = TileActionParser()
    parser.bind(eng)
    assert mask_disagreements(eng, parser, eng.state(), BLUE) == []


@pytest.mark.parametrize("model", TERRITORY_MODELS)
def test_a_fallen_lanes_bridge_takes_a_troop_only_under_the_open_bridge_model(model):
    eng = _model(MockEngine(), model)
    for princesses_up in (True, False):
        eng.reset(1, _setup(eng, princesses_up))
        statuses = {eng.check_deploy(DeployCommand(BLUE, 0, x, y))  # slot 0: the Knight
                    for x, y in _bridge_points(eng)}
        if model == OPEN_BRIDGE and not princesses_up:
            assert statuses == {DeployStatus.OK}, statuses
        else:
            assert DeployStatus.OK not in statuses, (princesses_up, statuses)


@needs_engine
def test_on_rustengine_the_bridge_follows_the_engines_own_model():
    """Whatever model the engine runs, the mask agrees with it everywhere, and a Knight on a
    fallen lane's bridge is taken exactly when that model is the open bridge."""
    from royalegym.rust_engine import RustEngine

    eng = RustEngine()
    model = eng.rules().territory_model
    assert model in TERRITORY_MODELS
    eng.reset(1, _setup(eng, princesses_up=False))
    parser = TileActionParser()
    parser.bind(eng)
    assert mask_disagreements(eng, parser, eng.state(), BLUE) == []
    statuses = {eng.check_deploy(DeployCommand(BLUE, 0, x, y)) for x, y in _bridge_points(eng)}
    if model == OPEN_BRIDGE:
        assert statuses == {DeployStatus.OK}, statuses
    else:
        assert DeployStatus.OK not in statuses, statuses

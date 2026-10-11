"""RoyaleGym's own arms: both clamps on, where RoyaleSim ships them off (owner, 2026-10-10).

RoyaleSim ships placement.ILLEGAL_TROOP_TAP and spells.ILLEGAL_SPELL_TAP at "refuse" for good, an
API choice: a direct engine user's out-of-territory tap is refused, not quietly moved. RoyaleGym
0.1.23 turns both clamps on, as the game moves such a tap, so an older RoyaleGym never meets a
changed default. The mask refuses those taps under either arm (``action.mask_disagreements``
tells the clamps' acceptances from the engine's), so a masked policy plays exactly as before.

``rust_engine.GYM_ARMS`` names them; ``gym_arms`` picks the ones a ledger lists as a candidate and
ships another arm of. An explicit ``calibration_overrides`` entry wins, so "refuse" opts out. They
are recorded in ``config()`` as ``gym_arms``, never as a constructor parameter: RoyaleLearn's
``load_env`` refuses a record whose constructor parameters differ, so a parameter would strand
every run and bot saved before 0.1.23 (RoyaleLearn, measured 2026-10-10).

SKIPS
    The engine tests need the engine and a ledger listing both clamps. Not a pass without them.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from royalegym.action import TileActionParser, mask_disagreements
from royalegym.protocol import (
    BLUE,
    RED,
    Calibration,
    DeployCommand,
    DeployStatus,
    MatchSetup,
    ShuffleMode,
)
from royalegym.rust_engine import (
    CORE_IMPORT_ERROR,
    GYM_ARMS,
    RustEngine,
    SymmetricRustEngine,
    core_available,
    gym_arms,
)

TROOP_KEY, CLAMP = "placement.ILLEGAL_TROOP_TAP", "client15535_clamp_to_legal_edge"
SPELL_KEY, SPELL_CLAMP = "spells.ILLEGAL_SPELL_TAP", "client16402_clamp_to_legal_edge"
needs_engine = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))


def _ledger(troop="refuse", spell="refuse", troop_listed=True):
    troop_arms = ["refuse", CLAMP] if troop_listed else ["refuse"]
    return {
        "placement": {"ILLEGAL_TROOP_TAP": {"value": troop, "candidates": troop_arms}},
        "spells": {"ILLEGAL_SPELL_TAP": {"value": spell, "candidates": ["refuse", SPELL_CLAMP]}},
    }


def test_the_arms_are_the_two_clamps():
    assert GYM_ARMS == {TROOP_KEY: CLAMP, SPELL_KEY: SPELL_CLAMP}


def test_a_ledger_shipping_refuse_gets_both_and_an_explicit_override_wins():
    assert gym_arms(_ledger(), {}) == GYM_ARMS
    assert gym_arms(_ledger(), {TROOP_KEY: "refuse"}) == {SPELL_KEY: SPELL_CLAMP}
    assert gym_arms(_ledger(), {TROOP_KEY: CLAMP, SPELL_KEY: "refuse"}) == {}


def test_a_key_shipped_at_the_arm_unlisted_or_missing_is_left_out():
    assert gym_arms(_ledger(troop=CLAMP), {}) == {SPELL_KEY: SPELL_CLAMP}
    assert gym_arms(_ledger(troop_listed=False), {}) == {SPELL_KEY: SPELL_CLAMP}
    assert gym_arms({"placement": {}, "spells": {}}, {}) == {}
    assert gym_arms({}, {}) == {}


def test_they_are_never_a_constructor_parameter():
    """RoyaleLearn's rules: a recorded key that were a parameter would reach the constructor
    on a reload, and one that differs from the record refuses the reload."""
    for cls in (RustEngine,):
        params = inspect.signature(cls.__init__).parameters
        assert "gym_arms" not in params
        assert all(p.kind != p.VAR_KEYWORD for p in params.values()), "RustEngine takes **kwargs"


def _needs_both():
    raw = Calibration.load().raw
    for key, arm in GYM_ARMS.items():
        section, name = key.split(".")
        entry = raw.get(section, {}).get(name)
        if entry is None or arm not in [*entry.get("candidates", []), entry.get("value")]:
            pytest.skip(f"SKIPPED, NOT PASSED: this engine's ledger does not list {key} = {arm}")
    return raw


@needs_engine
def test_a_plain_engine_runs_both_clamps_and_says_so():
    raw = _needs_both()
    engine = RustEngine()
    rules = engine.rules()
    assert (rules.illegal_troop_tap, rules.illegal_spell_tap) == (CLAMP, SPELL_CLAMP)
    config = engine.config()
    want = gym_arms(raw, {})
    assert isinstance(config["gym_arms"], dict)
    assert config["gym_arms"] == want
    assert "calibration_overrides" not in config, "a plain engine's parameters read as before"


@needs_engine
def test_refuse_opts_out_key_by_key():
    _needs_both()
    engine = RustEngine(calibration_overrides={TROOP_KEY: "refuse"})
    rules = engine.rules()
    assert (rules.illegal_troop_tap, rules.illegal_spell_tap) == ("refuse", SPELL_CLAMP)
    config = engine.config()
    assert config["calibration_overrides"] == {TROOP_KEY: "refuse"}
    assert config["gym_arms"] == {SPELL_KEY: SPELL_CLAMP}


def _battle(engine):
    ids = {c.name: c.card_id for c in engine.cards()}
    names = ("Knight", "Log", "Archer", "Giant", "Minions", "Goblins", "Zap", "Musketeer")
    deck = [ids[n] for n in names]
    engine.reset(3, MatchSetup(
        decks=[deck, deck], shuffle=ShuffleMode.NONE, elixir_milli=[100000, 100000],
        start_tick=engine.rules().deploy_lockout_ticks))
    return engine


@needs_engine
@pytest.mark.parametrize("cls", [RustEngine, SymmetricRustEngine])
def test_the_engine_moves_the_taps_and_the_mask_still_agrees(cls):
    """A Knight and a Log tapped in the enemy half are moved, not refused, and the mask, which
    refuses them, agrees with the engine on every action of both seats."""
    _needs_both()
    engine, refusing = _battle(cls()), _battle(cls(calibration_overrides={
        TROOP_KEY: "refuse", SPELL_KEY: "refuse"}))
    a = engine.arena()
    x, y = 9 * a.subtile + a.subtile // 2, 24 * a.subtile + a.subtile // 2
    for slot in (0, 1):  # the Knight, the Log
        cmd = DeployCommand(BLUE, slot, x, y)
        assert engine.check_deploy(cmd) == DeployStatus.OK
        assert refusing.check_deploy(cmd) == DeployStatus.OUT_OF_TERRITORY
    parser = TileActionParser()
    parser.bind(engine)
    state = engine.state()
    for team in (BLUE, RED):
        assert mask_disagreements(engine, parser, state, team) == []
    assert int(np.asarray(parser.action_mask(state, BLUE)).sum()) > 1

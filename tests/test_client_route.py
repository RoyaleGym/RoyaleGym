"""One game client's measured setup through RoyaleGym: ``make_env(client="15.535.29")``.

royalesim 0.1.33 ships ``client_arms(client)``: the calibration overrides that run a client's
measured mechanics ("15.535.29", the older client; "16.402" gives {}, the ledger being the current
client's), as JSON text per key. ``rust_engine.client_overrides`` decodes them into
``calibration_overrides`` values, and ``make_env(client=...)`` builds its RustEngine with them.

One of them is ``match.TICK_ORDER``, in a section this package reads outside the engine (the
elixir law, the clock), where an override is refused; this package does not read that key, so
it goes to the core alone (``OVERRIDE_CORE_ONLY``). Every other key of those sections stays
refused.

SKIPS
    The engine tests need the engine and royalesim 0.1.33's ``client_arms``. Not a pass without.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import royalegym
from royalegym.action import TileActionParser, mask_disagreements
from royalegym.env import make_env
from royalegym.protocol import BLUE, RED
from royalegym.rust_engine import (
    CORE_IMPORT_ERROR,
    OVERRIDE_CORE_ONLY,
    RustEngine,
    client_overrides,
    core_available,
)

CLIENT = "15.535.29"


def _needs_client_arms():
    if not core_available():
        pytest.skip(f"SKIPPED, NOT PASSED: {CORE_IMPORT_ERROR}")
    import royalesim

    if not hasattr(royalesim, "client_arms"):
        pytest.skip("SKIPPED, NOT PASSED: this royalesim has no client_arms (0.1.33 adds it)")
    return royalesim


def test_the_tick_order_goes_to_the_core_alone_and_nothing_here_reads_it():
    assert frozenset({"match.TICK_ORDER"}) == OVERRIDE_CORE_ONLY
    package = Path(royalegym.__file__).parent
    readers = [p.name for p in package.glob("*.py")
               if "TICK_ORDER" in p.read_text(encoding="utf-8") and p.name != "rust_engine.py"]
    assert readers == [], f"{readers} read match.TICK_ORDER: it is not the core's alone"
    # rust_engine.py names it once, in the allow-list itself; a second mention there is a read.
    engine_text = (package / "rust_engine.py").read_text(encoding="utf-8")
    assert engine_text.count("TICK_ORDER") == 1, "rust_engine.py reads match.TICK_ORDER"


def test_the_client_overrides_are_the_cores_arms_decoded():
    royalesim = _needs_client_arms()
    raw = royalesim.client_arms(CLIENT)
    got = client_overrides(CLIENT)
    assert got == {k: json.loads(v) for k, v in raw.items()}
    assert got["placement.ILLEGAL_TROOP_TAP"] == "client15535_clamp_to_legal_edge"
    assert isinstance(got["cards.CLIENT16402_VALUES"], dict)
    assert client_overrides("16.402") == {}
    with pytest.raises(ValueError, match="known clients"):
        client_overrides("1.0")


def test_another_match_key_is_still_refused():
    _needs_client_arms()
    with pytest.raises(ValueError, match=r"match\.MAX_MANA"):
        RustEngine(calibration_overrides={"match.MAX_MANA": 12})


def test_make_env_runs_the_clients_setup():
    """The engine runs every arm of the client's setup (the core holds them all), the mask's
    rules follow the ones it reads, and the mask agrees with the engine on both seats."""
    _needs_client_arms()
    want = client_overrides(CLIENT)
    env = make_env(client=CLIENT)
    engine = env.engine
    assert engine.calibration_overrides == want
    assert engine.gym_arms == {}, "the client's own clamp arms, not RoyaleGym's"
    assert set(engine._battle.calibration_overrides()) == set(want)
    rules = engine.rules()
    assert rules.illegal_troop_tap == want["placement.ILLEGAL_TROOP_TAP"]
    assert rules.enemy_building_taps == want["placement.ENEMY_BUILDING_TAPS"]
    env.reset(seed=3)
    for _ in range(12):
        env.step({"blue": 0, "red": 0})
    parser = TileActionParser()
    parser.bind(engine)
    state = engine.state()
    for team in (BLUE, RED):
        assert mask_disagreements(engine, parser, state, team) == []
    env.close()


def test_a_plain_make_env_runs_the_current_clients_ledger():
    _needs_client_arms()
    env = make_env()
    assert env.engine.calibration_overrides == {}
    env.close()


def test_a_client_needs_the_rust_engine():
    _needs_client_arms()
    with pytest.raises(ValueError, match="client"):
        make_env(client=CLIENT, engine="mock")
    with pytest.raises(ValueError, match="known clients"):
        make_env(client="1.0")

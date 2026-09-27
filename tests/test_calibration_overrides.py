"""``RustEngine(calibration_overrides=...)`` and the symmetric vehicle's use of it.

RoyaleSim 126992a shipped ``targeting.FIRST_TOWER_PICK = client_spawn_lane``, the client's
own lane rule, measured in the arena's frame on both seats and deliberately not the seat
rotation of itself. The core has no constructor keyword for it, so ``SymmetricRustEngine``
selects ``current_x`` through the core's general ``calibration_overrides`` instead
(``rust_engine.SYMMETRIC_ARMS``). An overridden engine plays other battles on the same
binary, so ``config()`` and every trace it records carry the overrides.
"""

from __future__ import annotations

import functools
import json

import msgspec
import pytest

from royalegym.protocol import MatchSetup, ShuffleMode
from royalegym.replay import ReplayRecorder, TraceHeader, _another_engine
from royalegym.rust_engine import (
    CORE_IMPORT_ERROR,
    SYMMETRIC_ARMS,
    RustEngine,
    SymmetricRustEngine,
    core_available,
    symmetric_overrides,
)

needs_core = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
KEY = "targeting.FIRST_TOWER_PICK"


def compiled_ledger() -> dict:
    import royalesim

    return json.loads(royalesim.EMBEDDED_CALIBRATION_JSON)


def ships_another_arm() -> bool:
    return KEY in symmetric_overrides(compiled_ledger())


def test_only_a_key_the_ledger_ships_another_arm_of_is_overridden():
    arm = SYMMETRIC_ARMS[KEY]
    assert symmetric_overrides({}) == {}, "a ledger older than the key overrides nothing"
    assert symmetric_overrides({"targeting": {"FIRST_TOWER_PICK": {"value": arm}}}) == {}
    assert symmetric_overrides(
        {"targeting": {"FIRST_TOWER_PICK": {"value": "client_spawn_lane"}}}
    ) == {KEY: arm}
    assert symmetric_overrides({"targeting": "not a section"}) == {}


@needs_core
def test_an_override_reaches_the_core_and_is_in_the_config_and_the_trace():
    if "FIRST_TOWER_PICK" not in compiled_ledger().get("targeting", {}):
        pytest.skip(f"SKIPPED, NOT PASSED: this engine's ledger has no {KEY}")
    plain = RustEngine()
    over = RustEngine(calibration_overrides={KEY: "current_x"})
    assert over._battle.calibration_overrides() == {KEY: '"current_x"'}
    assert over.config()["calibration_overrides"] == {KEY: "current_x"}
    assert "calibration_overrides" not in plain.config(), "a plain engine's config moved"
    ids = {c.name: c.card_id for c in over.cards()}
    deck = [ids[n] for n in ("Knight", "Archer", "Giant", "Minions", "Fireball", "Zap",
                             "Arrows", "Musketeer")]
    rec = ReplayRecorder()
    over.reset(0, MatchSetup(decks=[deck, deck], shuffle=ShuffleMode.NONE))
    rec.begin(over, 0, MatchSetup(decks=[deck, deck], shuffle=ShuffleMode.NONE))
    header = rec.trace.header
    assert header.calibration_overrides == {KEY: '"current_x"'}
    # Verified on the plain engine, the divergence line names the override first.
    note = _another_engine(header, plain)
    assert note is not None
    assert "calibration_overrides" in note, note
    assert _another_engine(header, over) is None
    # A header written before the field existed decodes as "none".
    raw = msgspec.msgpack.decode(msgspec.msgpack.encode(header))
    del raw["calibration_overrides"]
    old = msgspec.msgpack.decode(msgspec.msgpack.encode(raw), type=TraceHeader)
    assert old.calibration_overrides == {}


@needs_core
def test_an_override_of_a_key_the_mask_reads_is_refused():
    with pytest.raises(ValueError, match=r"placement\.ILLEGAL_TAP"):
        RustEngine(calibration_overrides={"placement.ILLEGAL_TAP": "refuse"})


@needs_core
def test_the_symmetric_vehicle_selects_every_symmetric_arm_this_ledger_needs():
    want = symmetric_overrides(compiled_ledger())
    got = SymmetricRustEngine().calibration_overrides
    assert got == want


@needs_core
def test_plant_the_shipped_lane_arm_breaks_the_rotation_gate():
    """The gate that went red on 126992a, with the override put back: it must go red again,
    or the green above says nothing about this key."""
    if not ships_another_arm():
        pytest.skip(f"SKIPPED, NOT PASSED: this engine ships no other arm of {KEY}")
    from test_rust_engine import multi_unit_rotation

    shipped = compiled_ledger()["targeting"]["FIRST_TOWER_PICK"]["value"]
    planted = functools.partial(SymmetricRustEngine, calibration_overrides={KEY: shipped})
    result, _stats = multi_unit_rotation(planted, 1)
    assert result is not None, "the shipped lane arm passed the rotation gate"

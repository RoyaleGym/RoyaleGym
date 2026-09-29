"""``RustEngine(calibration_overrides=...)`` and the symmetric vehicle's use of it.

RoyaleSim 126992a shipped ``targeting.FIRST_TOWER_PICK = client_spawn_lane``, the client's
own lane rule, measured in the arena's frame on both seats and deliberately not the seat
rotation of itself. The core has no constructor keyword for it, so ``SymmetricRustEngine``
selects a symmetric arm through the core's general ``calibration_overrides`` instead
(``rust_engine.SYMMETRIC_ARMS``): ``client_spawn_lane_own_frame`` where the ledger lists
it (RoyaleSim 1d661b0), else ``current_x``. An overridden engine plays other battles on the same
binary, so ``config()`` and every trace it records carry the overrides.
"""

from __future__ import annotations

import functools
import json
import re

import msgspec
import pytest

from royalegym.protocol import TROOP_TOWER_TAPS, MatchSetup, ShuffleMode
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
    arms = SYMMETRIC_ARMS[KEY]
    assert arms[0] == "client_spawn_lane_own_frame", arms

    def ledger(value, candidates=None):
        entry = {"value": value}
        if candidates is not None:
            entry["candidates"] = candidates
        return {"targeting": {"FIRST_TOWER_PICK": entry}}

    assert symmetric_overrides({}) == {}, "a ledger older than the key overrides nothing"
    assert symmetric_overrides({"targeting": "not a section"}) == {}
    # 1d661b0: the own-frame arm is a candidate, so it is taken.
    now = ["current_x", "client_spawn_lane", "client_spawn_lane_own_frame"]
    assert symmetric_overrides(ledger("client_spawn_lane", now)) == {KEY: arms[0]}
    assert symmetric_overrides(ledger(arms[0], now)) == {}, "shipped already"
    # 126992a: no own-frame candidate, so the old x rule.
    before = ["current_x", "client_spawn_lane"]
    assert symmetric_overrides(ledger("client_spawn_lane", before)) == {KEY: "current_x"}
    assert symmetric_overrides(ledger("current_x", before)) == {}
    # A ledger listing none of the symmetric arms: nothing to select.
    assert symmetric_overrides(ledger("client_spawn_lane", ["client_spawn_lane"])) == {}
    # No candidates list at all: the preferred arm.
    assert symmetric_overrides(ledger("client_spawn_lane")) == {KEY: arms[0]}
    # The death-spawn slide (parity's round 5): selected back to not_read once it ships.
    death = "spawner.DEATH_SPAWN_PUSHBACK"
    slide = {"value": "client_ring_slide", "candidates": ["not_read", "client_ring_slide"]}
    assert symmetric_overrides({"spawner": {"DEATH_SPAWN_PUSHBACK": slide}}) == {death: "not_read"}
    shipped = {"value": "not_read", "candidates": ["not_read", "client_ring_slide"]}
    assert symmetric_overrides({"spawner": {"DEATH_SPAWN_PUSHBACK": shipped}}) == {}
    # Sim's placement batch: the even box floored in the arena's frame and the building tap
    # pushed in arena coordinates, each selected back to its seat-symmetric arm once it ships.
    flipped = {
        "SNAP_EVEN_CORNER": {"value": "absolute", "candidates": ["placer_frame", "absolute"]},
        "TROOP_BUILDING_TAPS": {
            "value": "as_tower_tap", "candidates": ["not_relocated", "as_tower_tap"]
        },
    }
    assert symmetric_overrides({"placement": flipped}) == {
        "placement.SNAP_EVEN_CORNER": "placer_frame",
        "placement.TROOP_BUILDING_TAPS": "not_relocated",
    }
    kept = {
        "SNAP_EVEN_CORNER": {"value": "placer_frame", "candidates": ["placer_frame", "absolute"]},
        "TROOP_BUILDING_TAPS": {
            "value": "not_relocated", "candidates": ["not_relocated", "as_tower_tap"]
        },
    }
    assert symmetric_overrides({"placement": kept}) == {}, "shipped already"
    # RoyaleSim round 12: the roll's hit shape open on the arena's max-y edge, selected back
    # to the closed rectangle once it ships.
    rolls = ["rect_vs_circle_edge", "rect_contains_centre", "client15535_max_y_edge_open"]
    opened = {"value": "client15535_max_y_edge_open", "candidates": rolls}
    assert symmetric_overrides({"spells": {"ROLLING_HIT_SHAPE": opened}}) == {
        "spells.ROLLING_HIT_SHAPE": "rect_vs_circle_edge"
    }
    closed = {"value": "rect_vs_circle_edge", "candidates": rolls}
    assert symmetric_overrides({"spells": {"ROLLING_HIT_SHAPE": closed}}) == {}


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
def test_an_override_of_a_key_read_outside_the_engine_is_refused():
    """The elixir law and the clock read the ledger, not the engine: the core would run one
    value and the observation another."""
    for key, value in (("match.DEPLOY_LOCKOUT_TICKS", 0), ("time.TICK_MS", 50)):
        with pytest.raises(ValueError, match=re.escape(key)):
            RustEngine(calibration_overrides={key: value})


@needs_core
def test_an_override_of_a_key_the_mask_reads_and_does_not_follow_is_refused():
    """The mask implements these arms for MockEngine's model or not at all. At e817414 every
    placement key was refused and a collision or spells key constructed, with rules stating
    the shipped arm while the core ran another. Opening placement to follow TROOP_TOWER_TAPS
    let ILLEGAL_TAP = refuse through, which is a box fit in the core and a collision-circle
    test in the mask: 20,870 lattice points disagreed (review, 2026-09-27)."""
    ledger = compiled_ledger()
    cases = (
        ("placement.ILLEGAL_TAP", "refuse"),
        ("collision.BUILDING_FOOTPRINT_MODEL", "tile_size_override_box"),
        ("spells.ILLEGAL_SPELL_TAP", "client16402_clamp_to_legal_edge"),
    )
    ran = 0
    for key, value in cases:
        section, name = key.split(".", 1)
        if name not in ledger.get(section, {}):
            continue
        with pytest.raises(ValueError, match=re.escape(key)):
            RustEngine(calibration_overrides={key: value})
        ran += 1
    assert ran >= 2, f"only {ran} of the refusals had a key in this ledger"


@needs_core
def test_an_override_of_a_key_the_mask_follows_reaches_its_rules():
    """TROOP_TOWER_TAPS is followed; a key the mask never reads goes to the core alone."""
    if "TROOP_TOWER_TAPS" not in compiled_ledger().get("placement", {}):
        pytest.skip("SKIPPED, NOT PASSED: this engine's ledger has no placement.TROOP_TOWER_TAPS")
    plain = RustEngine()
    shipped = plain.rules().troop_tower_taps
    other = next(arm for arm in TROOP_TOWER_TAPS if arm != shipped)
    over = RustEngine(calibration_overrides={"placement.TROOP_TOWER_TAPS": other})
    assert over.rules().troop_tower_taps == other
    assert over.calibration.value("placement.TROOP_TOWER_TAPS") == other
    assert plain.rules().troop_tower_taps == shipped, "the override leaked into the ledger"
    if "FIRST_TOWER_PICK" in compiled_ledger().get("targeting", {}):
        lane = RustEngine(calibration_overrides={KEY: "current_x"})
        assert lane.calibration.value(KEY) == plain.calibration.value(KEY), "followed"
        assert lane._battle.calibration_overrides() == {KEY: '"current_x"'}


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

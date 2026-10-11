"""Placement arms by ledger key, for tests that claim a property of ONE rule.

RoyaleSim's placement batch (2026-09-28) ships five placement keys at new arms at once:
TAP_SNAP = client16402_tile_centre, SNAP_EVEN_CORNER = absolute, TROOP_BUILDING_TAPS =
as_tower_tap, SPELL_AS_DEPLOY_TAPS = troop_relocation, and the tower push. A test that
selects one key by name and inherits the rest from the ledger stops testing its claim the
day the others flip: a Mock/Rust comparison grades rules the Mock does not model, and a plant
whose landing needs a refused tap on an own building never lands once such a tap is moved.

``OLD_ARMS`` are the arms every key shipped at before the batch, and the clamps' "refuse", which
a plain RustEngine runs no more (``rust_engine.GYM_ARMS``). ``mock_arms`` are the arms MockEngine
states (it models no snap, no move and no clamp). ``pinned`` turns either into the
``calibration_overrides`` that select them, leaving out a key the ledger lacks or a plain
RustEngine already runs at that arm, so nothing is overridden for nothing. The shipped arms are
graded against the mask by the gates that use the plain engine.
"""

from __future__ import annotations

import json

from royalegym.mock_engine import MockEngine
from royalegym.rust_engine import gym_arms

#: The arm each placement key shipped at before RoyaleSim's placement batch.
OLD_ARMS: dict[str, str] = {
    "placement.SNAP_EVEN_CORNER": "placer_frame",
    "placement.TAP_SNAP": "none",
    "placement.TROOP_BUILDING_TAPS": "not_relocated",
    "placement.SPELL_AS_DEPLOY_TAPS": "spell_point",
    # The clamps: RoyaleSim ships them at refuse, and a plain RustEngine runs them on.
    "placement.ILLEGAL_TROOP_TAP": "refuse",
    "spells.ILLEGAL_SPELL_TAP": "refuse",
}

#: DeployRules field of each relocation key MockEngine states an arm for.
MOCK_FIELDS: dict[str, str] = {
    "placement.SNAP_EVEN_CORNER": "snap_even_corner",
    "placement.TAP_SNAP": "tap_snap",
    "placement.TROOP_BUILDING_TAPS": "troop_building_taps",
    "placement.SPELL_AS_DEPLOY_TAPS": "spell_as_deploy_taps",
    "placement.LIVE_BOTTLE_TAPS": "live_bottle_taps",
    # MockEngine moves nothing off an enemy building either ("ignored", its rules' default).
    "placement.ENEMY_BUILDING_TAPS": "enemy_building_taps",
    # Nor clamps a tap out of its territory ("refuse", whatever its ledger ships).
    "placement.ILLEGAL_TROOP_TAP": "illegal_troop_tap",
    "spells.ILLEGAL_SPELL_TAP": "illegal_spell_tap",
}


def mock_arms() -> dict[str, str]:
    """The arm MockEngine states for each relocation key (``mock_engine``)."""
    rules = MockEngine().rules()
    return {key: getattr(rules, field) for key, field in MOCK_FIELDS.items()}


def shipped_ledger() -> dict:
    """The compiled ledger, every section."""
    import royalesim

    return json.loads(royalesim.EMBEDDED_CALIBRATION_JSON)


def shipped_placement() -> dict:
    """The compiled ledger's placement section."""
    return shipped_ledger().get("placement", {})


def pinned(arms: dict[str, str]) -> dict[str, str]:
    """The ``calibration_overrides`` that run ``arms``: each key the ledger has where a plain
    RustEngine runs another arm (the ledger's, or RoyaleGym's own, ``GYM_ARMS``)."""
    ledger = shipped_ledger()
    runs = gym_arms(ledger, {})
    out = {}
    for key, arm in arms.items():
        section, _, name = key.partition(".")
        entry = ledger.get(section, {}).get(name)
        if isinstance(entry, dict) and runs.get(key, entry.get("value")) != arm:
            out[key] = arm
    return out


#: Both clamps off: what MockEngine runs, and what a test of a territory rule compares with it.
CLAMPS_OFF: dict[str, str] = {
    "placement.ILLEGAL_TROOP_TAP": "refuse",
    "spells.ILLEGAL_SPELL_TAP": "refuse",
}


def refusing() -> dict[str, str]:
    """The ``calibration_overrides`` that turn a plain RustEngine's clamps off (``CLAMPS_OFF``)."""
    return pinned(CLAMPS_OFF)

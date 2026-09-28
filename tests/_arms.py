"""Placement arms by ledger key, for tests that claim a property of ONE rule.

RoyaleSim's placement batch (2026-09-28) ships five placement keys at new arms at once:
TAP_SNAP = client16402_tile_centre, SNAP_EVEN_CORNER = absolute, TROOP_BUILDING_TAPS =
as_tower_tap, SPELL_AS_DEPLOY_TAPS = troop_relocation, and the tower push. A test that
selects one key by name and inherits the rest from the ledger stops testing its claim the
day the others flip: a Mock/Rust comparison grades rules the Mock does not model, and a plant
whose landing needs a refused tap on an own building never lands once such a tap is moved.

``OLD_ARMS`` are the arms every key shipped at before the batch. ``mock_arms`` are the arms
MockEngine states (it models no snap and no move). ``pinned`` turns either into the
``calibration_overrides`` that select them, leaving out a key the ledger lacks or already
ships at that arm, so nothing is overridden for nothing. The shipped arms are graded against
the mask by the gates that use the plain engine.
"""

from __future__ import annotations

import json

from royalegym.mock_engine import MockEngine

#: The arm each placement key shipped at before RoyaleSim's placement batch.
OLD_ARMS: dict[str, str] = {
    "placement.SNAP_EVEN_CORNER": "placer_frame",
    "placement.TAP_SNAP": "none",
    "placement.TROOP_BUILDING_TAPS": "not_relocated",
    "placement.SPELL_AS_DEPLOY_TAPS": "spell_point",
}

#: DeployRules field of each relocation key MockEngine states an arm for.
MOCK_FIELDS: dict[str, str] = {
    "placement.SNAP_EVEN_CORNER": "snap_even_corner",
    "placement.TAP_SNAP": "tap_snap",
    "placement.TROOP_BUILDING_TAPS": "troop_building_taps",
    "placement.SPELL_AS_DEPLOY_TAPS": "spell_as_deploy_taps",
    "placement.LIVE_BOTTLE_TAPS": "live_bottle_taps",
}


def mock_arms() -> dict[str, str]:
    """The arm MockEngine states for each relocation key (``mock_engine``)."""
    rules = MockEngine().rules()
    return {key: getattr(rules, field) for key, field in MOCK_FIELDS.items()}


def shipped_placement() -> dict:
    """The compiled ledger's placement section."""
    import royalesim

    return json.loads(royalesim.EMBEDDED_CALIBRATION_JSON).get("placement", {})


def pinned(arms: dict[str, str]) -> dict[str, str]:
    """The ``calibration_overrides`` that run ``arms``: each key the ledger has and ships at
    another arm."""
    ledger = shipped_placement()
    out = {}
    for key, arm in arms.items():
        entry = ledger.get(key.split(".", 1)[1])
        if isinstance(entry, dict) and entry.get("value") != arm:
            out[key] = arm
    return out

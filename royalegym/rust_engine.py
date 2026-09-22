"""RustEngine: ``protocol.Engine`` over the compiled Rust core (../RoyaleSim/crates/royalesim).

WHAT IT IS
    The adapter that lets ``ClashParallelEnv`` and everything else in royalegym run
    on the real engine with no change to the RL layer. The Rust side is
    ``royalesim.Battle`` (../RoyaleSim/crates/royalesim/src/py.rs). A step is one
    Rust call (validate, apply, tick N times with the GIL released); the state
    snapshot is one JSON byte string decoded straight into protocol structs.

CONVENTIONS, AND HOW EACH IS KNOWN RATHER THAN BELIEVED
    * Positions: protocol.py's engine frame IS the Rust frame (Blue defends low
      y, origin at Blue's back-left corner, integer subtiles). They pass through
      ``_engine_xy`` unchanged. tests/test_rust_engine.py cross-checks tower
      centres, passability of every half-cell and deploy legality over a grid
      against MockEngine, which reads the same arena.json independently.
    * Tower slots: protocol ``TowerSlot`` is named in the OWNER's frame under a
      180-degree seat rotation, so Red's own-LEFT princess is the engine's
      right-lane tower. The table is not typed in: ``_derive_slot_of_k`` matches
      the Rust tower centres to ``Arena.princess_centers`` / ``king_centers``
      (protocol's naming) and refuses to start if they are not a bijection.
    * Card ids: position in the catalogue (Supercell internal names from
      data/derived/cards.json, e.g. "Archer" not "Archers").
    * Deploy reasons: Rust returns indices into ``DEPLOY_REASONS``, a list of
      ``DeployStatus`` NAMES, so no protocol number is copied into Rust.
    * Troop territory: the engine and the mask run the same shipped mechanic, the
      closed NoDeploySize rect of every alive enemy crown tower. The engine reads
      its sizes from the cards.json it was built with, the mask from the one on
      disk; construction compares ``Battle.tower_no_deploy_rects()`` and
      ``TERRITORY_MODEL`` with ``DeployRules`` and refuses on any difference
      (``territory_differences``).
    * MatchSetup: ``protocol.validate_setup`` runs FIRST in ``reset``, before the
      core or this adapter touches anything, so a refused setup raises the
      Protocol's ValueError (same text as MockEngine) and the running battle is
      untouched. Checking only ``spawn_violation`` and the deck ids here would let
      start_tick -1 or 2**32, shuffle -1 and tower hp 2**31 reach PyO3 as an
      OverflowError, and a malformed tower_hp as an IndexError in the slot mapping --
      neither of which MockEngine raises. The core's
      own checks (py.rs ``Battle.reset``, state.rs ``scenario_spawn_now``) are the
      authority the protocol rules were written from; tests/test_rust_engine.py and
      tests/test_parity_hardening.py hold the two together by bypassing this check.
    * Spells: the catalogue kind code IS the deploy rule, mapped to
      ``Placement`` by ``_PLACEMENT_OF_KIND`` (0 TROOP, 1 BUILDING, 2 SPELL, 3
      ROLLING, 4 SPELL_NOT_ON_WATER). A unit a spell releases (the Goblin Barrel's
      Goblins) is not a card; the core reports it under the releasing spell's
      catalogue id. ``state()`` decodes the core's live spell objects into
      ``BattleState.spells`` and the stun / knockback timers into ``EntityState``.

STALE BUILDS ARE REFUSED
    The Rust crate compiles calibration.json and arena.json in (``include_str!``).
    A calibration edited after the last ``maturin develop`` would silently run the
    old constants, so construction compares every calibration ``value`` and the
    arena with the files on disk and raises on any difference. The same check
    refuses a ``Calibration.with_override`` the compiled engine cannot honour.

WHAT DIFFERS FROM MockEngine ON PURPOSE (engine mechanics, not adapter choices)
    Cards and towers run at the engine's unified level (``card_level``, 9 on the
    2018 rarity table) where the mock uses CSV level 1; spells travel, roll, stun and
    knock back here and resolve instantly there; ``crowns_from_destroyed_towers=False``
    is refused.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import msgspec

from .mock_engine import RAW_CARD_PACK
from .protocol import (
    BLUE,
    HAND_SIZE,
    RED,
    TEAMS,
    Arena,
    BattleState,
    Calibration,
    CardInfo,
    DeployCommand,
    DeployResult,
    DeployRules,
    DeployStatus,
    MatchSetup,
    Placement,
    TowerSlot,
    calibration_digest,
    calibration_values,
    data_dir,
    default_calibration,
    derived_cards_vintage,
    validate_setup,
)

try:  # the extension is optional: the package must import without it
    import royalesim as _core  # type: ignore[import-not-found]  # compiled, no stubs
except ImportError as _exc:  # pragma: no cover - exercised only on unbuilt trees
    _core = None
    CORE_IMPORT_ERROR: str | None = (
        f"royalesim is not built ({_exc}); run `maturin develop --release` in the "
        "sibling RoyaleSim checkout (../RoyaleSim) with the workspace venv active "
        "(README.md, Setup)"
    )
else:
    CORE_IMPORT_ERROR = None

# Rust kind codes in Battle.catalogue_json -> protocol Placement. The codes are the
# core's (py.rs ``kind_code``) and were numbered to equal Placement; the table is
# still spelled out so a new code raises at construction instead of mapping by accident.
_PLACEMENT_OF_KIND = {
    0: Placement.TROOP,
    1: Placement.BUILDING,
    2: Placement.SPELL,
    3: Placement.ROLLING,
    4: Placement.SPELL_NOT_ON_WATER,
}
_I32_MAX = 2**31 - 1


def core_available() -> bool:
    return _core is not None


def build_digest() -> str:
    """A short hash of the data the loaded extension was COMPILED with.

    The same hash ``protocol.calibration_digest`` takes of calibration.json on
    disk, over the copy compiled into the extension, plus the arena it was built
    with. A checkpoint pins this next to the on-disk digest: equal means the
    policy was trained on an engine built from the data that is there now, and
    unequal says which way to look without needing the two files side by side.
    Module-level and also reachable as ``RustEngine.build_digest`` so it can be
    read without constructing an engine -- construction is exactly what refuses
    on a stale build.
    """
    if _core is None:
        raise ImportError(CORE_IMPORT_ERROR)
    values = calibration_values(json.loads(_core.EMBEDDED_CALIBRATION_JSON))
    arena = json.loads(_core.EMBEDDED_ARENA_JSON)
    arena.pop("provenance", None)
    blob = json.dumps(
        {"calibration": values, "arena": arena}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def stale_build_differences(calibration: Calibration, arena_path: Path | None = None) -> list[str]:
    """What differs between the compiled-in data and ``calibration`` / arena.json."""
    if _core is None:
        raise ImportError(CORE_IMPORT_ERROR)
    diffs = []
    built = calibration_values(json.loads(_core.EMBEDDED_CALIBRATION_JSON))
    now = calibration_values(calibration.raw)
    for key in sorted(set(built) | set(now)):
        if built.get(key, "<absent>") != now.get(key, "<absent>"):
            diffs.append(
                f"calibration {key}: built {built.get(key, '<absent>')!r}, "
                f"now {now.get(key, '<absent>')!r}"
            )
    p = arena_path or data_dir() / "derived" / "arena.json"
    built_arena = json.loads(_core.EMBEDDED_ARENA_JSON)
    now_arena = json.loads(p.read_text(encoding="utf-8"))
    built_arena.pop("provenance", None)
    now_arena.pop("provenance", None)
    if built_arena != now_arena:
        diffs.append(f"arena.json differs from the one compiled in ({p})")
    return diffs


# CardInfo fields that are card DATA -- read from the table both engines are meant
# to be reading. ``hitpoints`` is deliberately absent: it is the engine's card LEVEL,
# a known and intended mechanics difference (tests/test_rust_engine.py's allow-list).
CARD_DATA_FIELDS = ("name", "elixir", "placement", "count", "radius", "flying")


def catalogue_vintage_split(
    rust_cards: Sequence[CardInfo], mock_cards: Sequence[CardInfo]
) -> str | None:
    """Why the two engines are reading DIFFERENT card tables, or None.

    The compiled engine carries the catalogue it was built with; MockEngine reads the
    raw CSVs on disk. In a public checkout those are the same vintage by
    construction -- the newer client packs are not redistributed, so the extractor
    can only build the tracked one -- and any difference here is a real defect. On a
    machine that HAS a newer pack and regenerated cards.json from it, the two are
    simply different tables, and every cross-engine comparison is then measuring the
    data rather than the engines.

    Returned as a ready reason string so a test can skip on it and SAY SO. A skip is
    not a pass: the comparison that skipped still has to run somewhere, which for
    this one is a checkout without the private pack.
    """
    differences: dict[str, list[str]] = {}
    if len(rust_cards) != len(mock_cards):
        differences["catalogue size"] = [f"{len(rust_cards)}/{len(mock_cards)} cards"]
    for a, b in zip(rust_cards, mock_cards, strict=False):
        for field in CARD_DATA_FIELDS:
            if getattr(a, field) != getattr(b, field):
                differences.setdefault(field, []).append(
                    f"{b.name} {getattr(a, field)}/{getattr(b, field)}"
                )
    if not differences:
        return None
    detail = "; ".join(f"{f}: {', '.join(v[:4])}" for f, v in sorted(differences.items()))
    return (
        "the two engines are reading different card tables, so this comparison would "
        "measure the DATA and not the engines. A SKIP IS NOT A PASS -- run it in a "
        "checkout without the private client pack, where both sides read the tracked "
        f"table. cards.json vintage {derived_cards_vintage()!r} vs MockEngine's "
        f"{RAW_CARD_PACK!r}. Differences (rust/mock) -- {detail}"
    )


def _derive_slot_of_k(arena: Arena) -> list[list[int]]:
    """[team][engine tower k] -> TowerSlot, by matching tower centres.

    Engine k is (king, left-lane princess, right-lane princess) for BOTH teams;
    protocol slots are own-frame. Matching positions makes the rotation
    convention a measured fact of the two codebases instead of a typed table.
    """
    assert _core is not None
    rust = _core.Battle.tower_positions()
    table = []
    for team in TEAMS:
        named = {tuple(arena.king_centers[team]): int(TowerSlot.KING)}
        for slot in (TowerSlot.LEFT, TowerSlot.RIGHT):
            named[tuple(arena.princess_centers[team][slot - 1])] = int(slot)
        row = []
        for k, pos in enumerate(rust[team]):
            if tuple(pos) not in named:
                raise RuntimeError(
                    f"Rust tower {k} of team {team} at {pos} matches no protocol tower "
                    f"centre {sorted(named)}: the engines disagree on tower geometry"
                )
            row.append(named[tuple(pos)])
        if sorted(row) != [0, 1, 2]:
            raise RuntimeError(f"tower slot table for team {team} is not a bijection: {row}")
        table.append(row)
    return table


def territory_differences(
    battle: Any, rules: DeployRules, arena: Arena, slot_of_k: list[list[int]]
) -> list[str]:
    """What differs between the compiled engine's troop territory and ``rules``.

    Reads the numbers the ENGINE holds (``Battle.tower_no_deploy_rects``, engine
    tower k order, mapped to protocol slots through ``slot_of_k``), never the
    adapter's own copy: a check that compares the Python rules with themselves
    cannot fail.
    """
    assert _core is not None
    diffs = []
    if rules.territory_model != _core.TERRITORY_MODEL:
        diffs.append(
            f"territory model: engine {_core.TERRITORY_MODEL!r}, rules {rules.territory_model!r}"
        )
    engine_rects = battle.tower_no_deploy_rects()
    for team in TEAMS:
        want = rules.tower_rects(arena, team)
        for k, rect in enumerate(engine_rects[team]):
            slot = slot_of_k[team][k]
            if tuple(rect) != want[slot]:
                diffs.append(
                    f"team {team} {TowerSlot(slot).name} NoDeploySize rect: engine "
                    f"{tuple(rect)}, rules {want[slot]}"
                )
    return diffs


class RustEngine:
    """Deterministic Rust battle engine behind ``protocol.Engine``."""

    def __init__(
        self,
        calibration: Calibration | None = None,
        card_names: Sequence[str] | None = None,
        arena_path: Path | None = None,
        path_search: str | None = None,
    ) -> None:
        """``path_search``: None = the ledger's ``pathfinding.PATH_SEARCH`` (the game's own
        search, measured on client 16.402 (RoyaleLive traces), which is NOT seat-symmetric:
        rotated twins can take different equal-cost routes, as in the real game).
        ``"trace_fitted_astar"`` selects the frame-planned arm whose routes are exact
        rotations -- and the fixed-distance knockback with it (the shipped
        ``knockback.DISPLACEMENT_LAW`` ladder, client16402, has two absolute-frame points
        like the search: the zero-vector direction and the water resolution's tie); the
        rotation-mirror tests run under it so they keep measuring the symmetry of everything
        else."""
        if _core is None:
            raise ImportError(CORE_IMPORT_ERROR)
        cal = calibration or default_calibration()
        diffs = stale_build_differences(cal, arena_path)
        if diffs:
            raise RuntimeError(
                "royalesim was built against different data; rebuild with "
                "`maturin develop --release`:\n  " + "\n  ".join(diffs)
            )
        self.calibration = cal
        self._arena = Arena.load(cal, arena_path)
        self._rules = DeployRules.load(cal)
        self.slot_of_k = _derive_slot_of_k(self._arena)
        self.path_search = path_search
        self._battle = _core.Battle(
            list(card_names) if card_names is not None else None, self.slot_of_k, path_search
        )
        terr = territory_differences(self._battle, self._rules, self._arena, self.slot_of_k)
        if terr:
            raise RuntimeError(
                "the Rust engine and the action mask disagree on troop territory; rebuild "
                "with `maturin develop --release` after regenerating cards.json:\n  "
                + "\n  ".join(terr)
            )
        self.card_level: int = self._battle.card_level()
        rows = json.loads(self._battle.catalogue_json())
        unknown = sorted({row[1] for row in rows} - set(_PLACEMENT_OF_KIND))
        if unknown:
            raise RuntimeError(
                f"the Rust catalogue uses kind codes {unknown} this adapter has no "
                "protocol Placement for (py.rs kind_code)"
            )
        self._cards = [
            CardInfo(
                card_id=cid,
                name=name,
                elixir=elixir,
                placement=int(_PLACEMENT_OF_KIND[kind]),
                count=count,
                radius=radius,
                flying=flying,
                hitpoints=hp,
            )
            for cid, (name, kind, elixir, count, radius, flying, hp) in enumerate(rows)
        ]
        self._status_of_reason: list[int | None] = [
            int(DeployStatus[n]) if n in DeployStatus.__members__ else None
            for n in _core.DEPLOY_REASONS
        ]
        self._decode_state = msgspec.json.Decoder(BattleState)
        self._reset_called = False

    # ------------------------------------------------------------ protocol

    def cards(self) -> Sequence[CardInfo]:
        return self._cards

    def arena(self) -> Arena:
        return self._arena

    def rules(self) -> DeployRules:
        return self._rules

    def reset(self, seed: int, setup: MatchSetup) -> None:
        """``protocol.validate_setup`` first -- nothing is read into the core, and no
        adapter table is indexed, until the whole setup has passed. See the module doc."""
        validate_setup(self._arena, self._cards, setup)
        tower_hp = None
        if setup.tower_hp is not None:
            tower_hp = [
                [int(setup.tower_hp[team][self.slot_of_k[team][k]]) for k in range(3)]
                for team in TEAMS
            ]
            if not setup.crowns_from_destroyed_towers and any(
                hp <= 0 for row in tower_hp for hp in row[1:]
            ):
                raise NotImplementedError(
                    "the Rust engine derives crowns from destroyed towers every tick; "
                    "crowns_from_destroyed_towers=False is not supported"
                )
        self._battle.reset(
            int(seed) & (2**64 - 1),
            [list(map(int, d)) for d in setup.decks],
            int(setup.shuffle),
            int(setup.start_tick),
            list(map(int, setup.elixir_milli)) if setup.elixir_milli is not None else None,
            tower_hp,
            [(sp.team, sp.card_id, sp.x, sp.y, sp.hp) for sp in setup.spawns],
        )
        self._reset_called = True

    def check_deploy(self, command: DeployCommand) -> int:
        return self._status(self._battle.check_deploy(*self._wire(command)))

    def step(self, commands: Sequence[DeployCommand], ticks: int) -> list[DeployResult]:
        if ticks < 0:
            raise ValueError("ticks must be >= 0")
        raw = self._battle.step([self._wire(c) for c in commands], int(ticks))
        return [
            DeployResult(c.team, c.hand_slot, card_id, self._status(reason), tick)
            for c, (card_id, reason, tick) in zip(commands, raw, strict=True)
        ]

    def state(self) -> BattleState:
        return self._decode_state.decode(self._battle.state_json())

    def save_state(self) -> bytes:
        return bytes(self._battle.save())

    def load_state(self, blob: bytes) -> None:
        self._battle.load(bytes(blob))
        self._reset_called = True

    def state_hash(self) -> int:
        return int(self._battle.state_hash())

    # ------------------------------------------------------------ internals

    def _engine_xy(self, x: int, y: int) -> tuple[int, int]:
        """protocol engine frame -> Rust frame. The identity; see the module doc."""
        return x, y

    def _wire(self, c: DeployCommand) -> tuple[int, int, int, int]:
        x, y = self._engine_xy(c.x, c.y)
        # Clamp only what cannot cross into i32/i64 fields; every clamped value is
        # still outside its valid range, so the verdict is unchanged.
        team = c.team if c.team in (BLUE, RED) else 2
        slot = min(max(c.hand_slot, -1), HAND_SIZE)
        lim_x = min(self._arena.width + 1, _I32_MAX)
        lim_y = min(self._arena.height + 1, _I32_MAX)
        return team, slot, min(max(x, -1), lim_x), min(max(y, -1), lim_y)

    def _status(self, reason: int) -> int:
        status = self._status_of_reason[reason]
        if status is None:
            raise RuntimeError(
                f"engine reported {_core.DEPLOY_REASONS[reason]} for a slot command"  # type: ignore[union-attr]
            )
        return status

    def config(self) -> dict[str, object]:
        """Constructor state, JSON-able, for ``ClashParallelEnv.config()``.

        Everything that makes two RustEngines run different battles from the same
        commands: which cards are in the catalogue, the level they run at, and which
        pathfinder arm was selected (``SymmetricRustEngine`` is a different engine
        for this purpose, and a checkpoint that does not say so is a checkpoint that
        cannot be reproduced).
        """
        return {
            "cards": [c.name for c in self._cards],
            "card_level": self.card_level,
            "path_search": self.path_search,
            "calibration_digest": calibration_digest(self.calibration),
        }

    build_digest = staticmethod(build_digest)

    def debug_nudge(self, uid: int, dx: int, dy: int) -> bool:
        """TEST-ONLY: move a live entity by (dx, dy) subtiles."""
        return bool(self._battle.debug_nudge(uid, dx, dy))


class SymmetricRustEngine(RustEngine):
    """``RustEngine`` under the frame-planned pathfinder (``path_search="trace_fitted_astar"``),
    which also selects the fixed-distance knockback (see ``RustEngine.__init__``).

    FOR ROTATION-MIRROR GATES ONLY. The shipped search is the game's own, measured on
    client 16.402 (RoyaleLive traces), and it is not seat-symmetric: its goal scan and
    neighbour order run in absolute arena coordinates, so a Red unit and its
    rotated Blue twin can publish different equal-cost routes (measured: 20 of 54 twin
    problems on the shipped arena).
    That is the real game and ``RustEngine`` reproduces it. The tests that assert a
    mirrored battle stays a rotation to the end exist to catch seat bias in the ENGINE'S
    OTHER SYSTEMS and in this adapter, so they run under this class, whose routes are
    exact rotations by construction.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("path_search", "trace_fitted_astar")
        super().__init__(*args, **kwargs)

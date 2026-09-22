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
      its sizes from the cards.json in the checkout it was built in, the mask from
      the one under ``data_dir()``; construction compares
      ``Battle.tower_no_deploy_rects()`` and ``TERRITORY_MODEL`` with ``DeployRules``
      and refuses on any difference (``territory_differences``).
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

THE CARD TABLE IS READ, NOT COMPILED IN
    Every ``royalesim.Battle`` reads data/derived/cards.json from the RoyaleSim
    checkout the engine was built in, when it is constructed (card.rs ``load_repo``).
    Re-running ``tools/extract_cards.py --vintage 2018 --out data/derived/cards.json``
    in that checkout changes the next RustEngine's card table at once: no rebuild, and
    no stale-build refusal, because there is nothing compiled to be stale.
    ``ROYALESIM_DATA_DIR`` does not move that file; it moves only what this package
    reads. So build in the checkout whose data you want.
    Which table an engine got is in its ``config()``: ``cards_json_fnv1a64`` (the hash
    RoyaleSim's replay fixtures record), ``cards_vintage`` and
    ``cards_json_hash_source`` (see ``RustEngine.card_table_stamp``).

WHAT DIFFERS FROM MockEngine ON PURPOSE (engine mechanics, not adapter choices)
    Cards and towers run at the engine's unified level (``card_level``, 9 on the
    2018 rarity table) where the mock uses CSV level 1; spells travel, roll, stun and
    knock back here and resolve instantly there; ``crowns_from_destroyed_towers=False``
    is refused.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.machinery
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import msgspec

from .mock_engine import RAW_CARD_PACK
from .protocol import (
    BLUE,
    DEFAULT_DATA_DIR,
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
    fnv1a64,
    mirror_state,
    to_engine,
    validate_setup,
)

# The section of THIS repo's README that walks the install end to end. Spelled once,
# so the message below and the test that checks the heading is really there cannot
# drift apart. It was "Setup" until 2026-09-22 and no README here ever had that
# heading -- a dead pointer in the one message a reader reads when nothing works.
INSTALL_SECTION = "Install"
INSTALL_POINTER = f'the "{INSTALL_SECTION}" section of the RoyaleGym README.md'

try:  # the extension is optional: the package must import without it
    import royalesim as _core  # type: ignore[import-not-found]  # compiled, no stubs
except ImportError as _exc:  # pragma: no cover - exercised only on unbuilt trees
    _core = None
    # The one pointer a reader gets at the moment the engine has not built, which is
    # the moment they most need it to land somewhere. It names THIS repo's README and
    # the heading it really has; tests/test_install_pointers.py checks that the heading exists,
    # because a pointer nothing reads is a pointer that rots quietly.
    CORE_IMPORT_ERROR: str | None = (
        f"royalesim is not built ({_exc}); run `maturin develop --release` in the "
        "sibling RoyaleSim checkout (../RoyaleSim) with the workspace venv active. "
        "The data has to be extracted BEFORE that build: arena.json is compiled in, "
        f"and cards.json is read each time an engine is constructed; {INSTALL_POINTER} "
        "has both steps in order."
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
    on a stale build. The card table is not in it, because it is not compiled in:
    ``RustEngine.card_table_stamp`` says which one an engine read.
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


# ---------------------------------------------------------------- the card table

#: What the compiled engine may call the FNV-1a 64 of the card table a Battle loaded,
#: looked up on the Battle in this order: a method or an attribute, giving 16 hex
#: digits or the integer. With none of them, the stamp is the file's, hashed from disk
#: at construction (``RustEngine.card_table_stamp``).
ENGINE_CARD_HASH_NAMES = ("cards_json_fnv1a64", "card_table_fnv1a64")
#: What the compiled engine may call the path of the cards.json it reads: a module
#: constant, or a static method or attribute of ``Battle``.
ENGINE_CARD_PATH_NAMES = ("CARDS_JSON_PATH", "cards_json_path")

# The core builds the path it reads a card table from as the crate's build directory
# followed by this (card.rs ``load_repo_file``). Both pieces are in the compiled
# extension as text, the directory right before the tail.
_CARD_PATH_TAIL = b"/../../data/derived/"
# Where an absolute path can start: a drive, a UNC share, or a POSIX root.
_PATH_ROOT = re.compile(rb"[A-Za-z]:[\\/]|\\\\|/")


def _extension_file() -> Path | None:
    """The compiled extension module file (``.pyd`` / ``.so``) behind ``royalesim``."""
    for module in (getattr(_core, "royalesim", None), _core):
        name = getattr(module, "__file__", None)
        if name and name.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES)):
            return Path(name)
    return None


@functools.cache
def _cards_json_in_build_checkout() -> Path | None:
    """data/derived/cards.json of the checkout the loaded extension was built in, or None.

    Read out of the extension itself, because nothing else knows: a wheel installed
    into a venv keeps no pointer back to the tree it was built from. The bytes before
    the build directory belong to whatever the linker put there, so each place an
    absolute path could start is tried, earliest first, and the first that names an
    existing file wins.
    """
    ext = _extension_file()
    if ext is None:
        return None
    blob = ext.read_bytes()
    at = blob.find(_CARD_PATH_TAIL)
    while at != -1:
        start = at
        # Back over text (printable ASCII or UTF-8), at most one long path's worth.
        while start > 0 and at - start < 4096 and 0x20 <= blob[start - 1] != 0x7F:
            start -= 1
        run = blob[start:at]
        for m in _PATH_ROOT.finditer(run):
            try:
                build_dir = run[m.start() :].decode("utf-8")
            except UnicodeDecodeError:
                continue
            candidate = Path(build_dir, "..", "..", "data", "derived", "cards.json")
            if candidate.is_file():
                return candidate.resolve()
        at = blob.find(_CARD_PATH_TAIL, at + 1)
    return None


def engine_cards_json_path() -> tuple[Path, str]:
    """(path, how it was found) of the cards.json the compiled engine reads.

    Not ``data_dir()``: ``ROYALESIM_DATA_DIR`` moves what this package reads and never
    what the engine reads. In order: the path the engine states, if it states one
    (``ENGINE_CARD_PATH_NAMES``), found by "engine"; the build directory compiled into
    the extension, "build checkout"; the sibling checkout of the documented layout,
    "sibling checkout", a fallback that is right only when the engine was built there.
    """
    if _core is None:
        raise ImportError(CORE_IMPORT_ERROR)
    for name in ENGINE_CARD_PATH_NAMES:
        for owner in (_core, _core.Battle):
            value = getattr(owner, name, None)
            if value is not None:
                return Path(value() if callable(value) else value), "engine"
    built = _cards_json_in_build_checkout()
    if built is not None:
        return built, "build checkout"
    return DEFAULT_DATA_DIR / "derived" / "cards.json", "sibling checkout"


# (path, size, mtime_ns) -> (FNV-1a 64, vintage). The hash is pure Python and costs
# about half a second on a full card table, so it is paid once per version of a file.
_CARDS_JSON_STAMPS: dict[tuple[str, int, int], tuple[str, str]] = {}


def cards_json_stamp(path: Path) -> tuple[str, str]:
    """(FNV-1a 64 of the file's bytes, its ``provenance.vintage``), hashed once per version.

    The hash is ``protocol.fnv1a64`` over the raw bytes, the one RoyaleSim's replay
    fixtures record as ``cards_json_fnv1a64``. Keyed on size and modification time, so
    a regenerated file is hashed again and an unchanged one is not.
    """
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _CARDS_JSON_STAMPS:
        _CARDS_JSON_STAMPS[key] = (fnv1a64(path.read_bytes()), derived_cards_vintage(path))
    return _CARDS_JSON_STAMPS[key]


def _card_table(fnv: str, vintage: str, source: str) -> dict[str, str]:
    return {"cards_json_fnv1a64": fnv, "cards_vintage": vintage, "cards_json_hash_source": source}


def _engine_card_hash(battle: Any) -> str | None:
    """The card-table hash the engine itself reports, or None when it reports none."""
    for name in ENGINE_CARD_HASH_NAMES:
        value = getattr(battle, name, None)
        if value is None:
            continue
        value = value() if callable(value) else value
        if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 2**64:
            return f"{value:016x}"
        if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{16}", value):
            return value.lower()
        raise RuntimeError(f"the engine's {name} gave {value!r}, not an FNV-1a 64 hash")
    return None


# CardInfo fields that are card DATA -- read from the table both engines are meant
# to be reading. ``hitpoints`` is deliberately absent: it is the engine's card LEVEL,
# a known and intended mechanics difference (tests/test_rust_engine.py's allow-list).
CARD_DATA_FIELDS = ("name", "elixir", "placement", "count", "radius", "flying")


def catalogue_vintage_split(
    rust_cards: Sequence[CardInfo], mock_cards: Sequence[CardInfo]
) -> str | None:
    """Why the two engines are reading DIFFERENT card tables, or None.

    The compiled engine reads data/derived/cards.json from the RoyaleSim checkout it was
    built in, each time an engine is constructed; MockEngine reads the raw CSVs under
    ``data_dir()``, every time. In a public checkout that is built there, the two are
    the same vintage by construction: the newer client packs are not redistributed, so
    the extractor can only produce the tracked table, and any difference here is then a
    real defect. On a machine that HAS a newer pack and regenerated cards.json from it,
    the two are simply different tables and every cross-engine comparison is measuring
    the data rather than the engines.

    THE TWO HALVES ARE READ FROM DIFFERENT PLACES, which is the part that catches people
    out. Pointing ``ROYALESIM_DATA_DIR`` at a 2018 data directory moves MockEngine's half
    and does not move the engine's at all -- measured: with a pure 2018 data dir the
    extension still reported 95 cards and Goblins at 4, because it went on reading the
    cards.json of the checkout it was built in. Regenerating THAT file moves the engine's
    half at once, with no rebuild. So build in the checkout whose data you want.
    ``stale_build_differences`` does not cover this: it compares calibration.json and
    arena.json, the two files compiled in, and the skip is what handles the catalogue.
    ``RustEngine.card_table_stamp`` names the table an engine actually read.

    THAT IS OBSERVED, NOT PREDICTED. A clean clone of the four repos, its data generated
    from the tracked 2018 tables and royalesim built in it, runs the tests this function
    guards: 96 passed, 0 skipped, so the two engines agree over the shared card set and
    the contract holds. On a machine that also holds a private client pack they skip
    instead, which is this function working rather than the contract going unchecked.

    Returned as a ready reason string so a test can skip on it and SAY SO. A skip is
    not a pass: the comparison that skipped still has to run somewhere.
    """
    if len(rust_cards) != len(mock_cards):
        # STOP HERE. The rows are compared position-wise, so two catalogues of
        # different SCOPE produce field differences that read as data corruption:
        # card 3 against card 3 gives "name: MiniPekka Pekka/MiniPekka", which is not
        # a claim about MiniPekka and has sent at least one reader hunting a data bug
        # that was not there. Different lengths mean the two engines were built over
        # different card SETS, which is a caller error, not a vintage split.
        return (
            f"the two engines hold different NUMBERS of cards, {len(rust_cards)} and "
            f"{len(mock_cards)}, so they were built over different card sets rather "
            f"than different tables. Construct both over the same names -- "
            f"RustEngine(card_names=...) and MockEngine(card_names=...) -- and this "
            f"comparison measures the engines. Comparing the two DEFAULT catalogues is "
            f"not a vintage check: MockEngine ships a thin slice and RustEngine's "
            f"default is every registered card in the table."
        )
    differences: dict[str, list[str]] = {}
    for a, b in zip(rust_cards, mock_cards, strict=True):
        for field in CARD_DATA_FIELDS:
            if getattr(a, field) != getattr(b, field):
                differences.setdefault(field, []).append(
                    f"{b.name} {getattr(a, field)}/{getattr(b, field)}"
                )
    if not differences:
        return None
    detail = "; ".join(f"{f}: {', '.join(v[:4])}" for f, v in sorted(differences.items()))
    engine_cards = engine_cards_json_path()[0] if _core is not None else None
    vintage = derived_cards_vintage(engine_cards)
    return (
        "the two engines are reading different card tables, so this comparison would "
        "measure the DATA and not the engines. A SKIP IS NOT A PASS -- to run it, the "
        "engine's cards.json has to be the 2018 table: it reads data/derived/cards.json "
        "in the RoyaleSim checkout it was built in "
        f"({engine_cards or 'royalesim is not built'}), each time one is constructed, "
        "and ROYALESIM_DATA_DIR does not move it. Regenerate that file with "
        "`python tools/extract_cards.py --vintage 2018 --out data/derived/cards.json` "
        "in that checkout; no rebuild is needed. cards.json vintage "
        f"{vintage!r} vs MockEngine's {RAW_CARD_PACK!r}. "
        f"Differences (rust/mock) -- {detail}"
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
        ground_y_clamp: str | None = None,
        ground_deploy_point: str | None = None,
    ) -> None:
        """``path_search``: None = the ledger's ``pathfinding.PATH_SEARCH`` (the game's own
        search, measured on client 16.402, which is NOT seat-symmetric:
        rotated twins can take different equal-cost routes, as in the real game).
        ``"trace_fitted_astar"`` selects the frame-planned arm whose routes are exact
        rotations -- and the fixed-distance knockback with it (the shipped
        ``knockback.DISPLACEMENT_LAW`` ladder, client16402, has two absolute-frame points
        like the search: the zero-vector direction and the water resolution's tie); the
        rotation-mirror tests run under it so they keep measuring the symmetry of everything
        else.

        ``ground_y_clamp``: None = the ledger's ``formation.GROUND_Y_CLAMP``
        (``client16402_deploy_column_range``, the game's own, measured per side and
        DELIBERATELY not the rotation of itself -- the range is a native unit tighter at
        the river and half a row shorter at the back edge). ``deploy_column_range_own_frame``
        is the seat-symmetric arm; ``none`` disables the clamp. It does NOT ride along with
        ``path_search``: the three knockback keys do, but the clamp is independent, so
        ``SymmetricRustEngine`` asks for it by name.

        ``ground_deploy_point``: None = the ledger's ``formation.GROUND_DEPLOY_POINT``
        (``client16402_one_unit``, the game's own, measured): a GROUND summon's ring is
        laid on a point one native unit off the tap -- in x when the tap is on the
        arena's LEFT half, either seat, and in y when the owner is side 1, either half.
        A FLYING summon's ring is laid on the tap itself. Two offsets keyed two
        different ways, so the shipped arm is neither seat-symmetric nor frame-symmetric.
        ``none`` lays every ring on the tap. Like the clamp it is independent of
        ``path_search`` and has to be asked for by name."""
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
        self.ground_y_clamp = ground_y_clamp
        self.ground_deploy_point = ground_deploy_point
        # The card table is read by the Battle below, from this file (module doc).
        # Stamped from disk on both sides of that read unless the engine reports it.
        self.cards_json_path, self.cards_json_found_by = engine_cards_json_path()
        from_engine = any(hasattr(_core.Battle, n) for n in ENGINE_CARD_HASH_NAMES)
        before = None if from_engine else self._disk_stamp()
        self._battle = _core.Battle(
            list(card_names) if card_names is not None else None,
            self.slot_of_k,
            path_search,
            ground_y_clamp,
            ground_deploy_point,
        )
        self._card_table = self._stamp_card_table(before)
        terr = territory_differences(self._battle, self._rules, self._arena, self.slot_of_k)
        if terr:
            raise RuntimeError(
                "the Rust engine and the action mask disagree on troop territory. The "
                "engine read data/derived/cards.json in the checkout it was built in "
                f"({self.cards_json_path}); the mask reads the one under data_dir() "
                f"({data_dir() / 'derived' / 'cards.json'}). Make them the same file: "
                "point ROYALESIM_DATA_DIR at that checkout's data/ folder, or build the "
                "engine in the checkout whose data you want:\n  " + "\n  ".join(terr)
            )
        self.card_level: int = self._battle.card_level()
        rows = json.loads(self._battle.catalogue_json())
        unknown = sorted({row[1] for row in rows} - set(_PLACEMENT_OF_KIND))
        if unknown:
            raise RuntimeError(
                f"the Rust catalogue uses kind codes {unknown} this adapter has no "
                "protocol Placement for (py.rs kind_code)"
            )
        # A catalogue row is positional and may GROW: the core appends a column rather
        # than rekeying the row, because this is decoded per battle. So the trailing
        # columns are taken by position from ``rest`` and an unknown one is ignored,
        # which is what lets an engine built with a newer column load here.
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
                footprint_tiles=rest[0] if rest else None,
            )
            for cid, (name, kind, elixir, count, radius, flying, hp, *rest) in enumerate(rows)
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
            DeployResult(c.team, c.hand_slot, card_id, self._status(reason), tick, c.x, c.y)
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
        pathfinder and deploy-clamp arms were selected. ``SymmetricRustEngine`` is a
        different engine for this purpose and a checkpoint that does not say so cannot
        be reproduced -- and the clamp has to be named separately from the pathfinder,
        because selecting one does not select the other. The card NAMES do not say
        which card table they were read from, so ``card_table_stamp`` is in here too.
        """
        return {
            "cards": [c.name for c in self._cards],
            "card_level": self.card_level,
            "path_search": self.path_search,
            "ground_y_clamp": self.ground_y_clamp,
            "ground_deploy_point": self.ground_deploy_point,
            "calibration_digest": calibration_digest(self.calibration),
            **self._card_table,
        }

    def card_table_stamp(self) -> dict[str, str]:
        """Which card table this engine read, taken when it was constructed.

        ``cards_json_fnv1a64``: FNV-1a 64 of the cards.json bytes, 16 hex digits, the
        value RoyaleSim's replay fixtures record under the same name.
        ``cards_vintage``: that file's ``provenance.vintage``.
        ``cards_json_hash_source``: ``"engine"`` when the compiled engine reported the
        hash of the table it loaded (``ENGINE_CARD_HASH_NAMES``), and then the vintage
        is "unknown" unless ``cards_json_path`` hashes to the same value;
        ``"disk_at_construction"`` when it was hashed from ``cards_json_path``, the
        file the engine reads, just before and just after the engine read it (a file
        that changed in between is refused); ``"unavailable"`` when that file could
        not be found, and then the other two are "" and "unknown".
        """
        return dict(self._card_table)

    def _disk_stamp(self) -> tuple[str, str] | None:
        path = self.cards_json_path
        return cards_json_stamp(path) if path.is_file() else None

    def _stamp_card_table(self, before: tuple[str, str] | None) -> dict[str, str]:
        engine_hash = _engine_card_hash(self._battle)
        disk = self._disk_stamp()
        if engine_hash is not None:
            # The file's vintage names the engine's table only if the file IS that table.
            vintage = disk[1] if disk is not None and disk[0] == engine_hash else "unknown"
            return _card_table(engine_hash, vintage, "engine")
        if disk != before:
            raise RuntimeError(
                f"{self.cards_json_path} changed while the engine was reading it, so "
                "which card table it holds cannot be said; construct it again"
            )
        if disk is None:
            return _card_table("", "unknown", "unavailable")
        return _card_table(disk[0], disk[1], "disk_at_construction")

    build_digest = staticmethod(build_digest)

    def debug_nudge(self, uid: int, dx: int, dy: int) -> bool:
        """TEST-ONLY: move a live entity by (dx, dy) subtiles."""
        return bool(self._battle.debug_nudge(uid, dx, dy))


#: Own-frame tile centres the symmetry probe deploys on. More than one, and spread
#: across both arena halves and the centre column, because an asymmetry keyed on the
#: ABSOLUTE half is invisible from a tile whose mirror is in the same half, and one
#: keyed on a rounding is invisible wherever the rounding happens to land even. The
#: deploy-point offset measured in September broke 209 of 255 own tiles: a probe that
#: tried a single tile would have had a one-in-six chance of reporting clean.
PROBE_TILES = ((1, 2), (4, 3), (9, 4), (13, 2), (16, 5))

#: Probe verdicts, keyed by engine build, card table and constructor arguments. The probe costs a
#: handful of resets and a tick; a test suite builds hundreds of these.
_PROBE_CACHE: dict[tuple, list[str]] = {}


def rotation_probe(engine: RustEngine, tiles: Sequence[tuple[int, int]] = PROBE_TILES) -> list[str]:
    """Deploy each multi-unit card at the same own-frame tile from both seats; report drift.

    THE DIRECT MEASUREMENT of the property a rotation gate assumes. Both seats command
    the same tile in their own frame, so after one tick the two groups must be exact
    rotations of each other. Anything else means some system under the engine is keyed
    on the seat or on the absolute arena frame.

    Multi-unit cards only, because a single unit lands on the tap and a formation is
    laid out AROUND it -- which is where a per-side offset shows up. Troops only:
    a building or a spell does not have a ring.

    This exists because the alternative failed twice. Deciding from a list of which
    keys are asymmetry sources means the list is right only until the engine gains a
    key, and both times it gained one the gates went red pointing at the wrong thing --
    at the observation builder, and at the engine. Measuring the property instead is
    immune to a key nobody has named yet.
    """
    arena = engine.arena()
    tile = arena.subtile
    problems: list[str] = []

    def key(state: BattleState) -> list[tuple[int, ...]]:
        return sorted(
            (e.team, e.kind, e.card_id, e.tower_slot, e.x, e.y, e.deploy_ticks)
            for e in state.entities
        )

    for card in engine.cards():
        if card.count < 2 or card.placement != Placement.TROOP:
            continue
        for tx, ty in tiles:
            if not (0 < tx < arena.tiles_x and 0 < ty < arena.tiles_y // 2):
                continue
            deck = [card.card_id] * HAND_SIZE * 2
            engine.reset(_PROBE_SEED, MatchSetup(decks=[deck, deck], shuffle=0))
            commands = []
            for team in TEAMS:
                ex, ey = to_engine(arena, team, tx * tile, ty * tile)
                commands.append(DeployCommand(team=team, hand_slot=0, x=ex, y=ey))
            engine.step(commands, 1)
            state = engine.state()
            mirrored = mirror_state(arena, state)
            if key(state) != key(mirrored):
                drift = [
                    (a[4] - b[4], a[5] - b[5])
                    for a, b in zip(key(state), key(mirrored), strict=True)
                    if a != b
                ]
                problems.append(
                    f"{card.name} (count {card.count}) at own tile ({tx}, {ty}): "
                    f"offsets (dx, dy) = {sorted(set(drift))}"
                )
                break  # one tile per card is enough to say the vehicle is not symmetric
    return problems


_PROBE_SEED = 7


def _rotation_probe_cached(args: tuple, kwargs: dict) -> list[str]:
    """Probe a THROWAWAY engine built the same way, so the caller's is untouched.

    Probing the instance itself would leave it holding the probe's battle, and a
    constructor that quietly resets what it just built is its own kind of trap.
    """
    # The card table too: it is read at construction, so it can change with no rebuild.
    cards = engine_cards_json_path()[0]
    table = cards_json_stamp(cards)[0] if cards.is_file() else ""
    cache_key = (RustEngine.build_digest(), table, repr(args), repr(sorted(kwargs.items())))
    if cache_key not in _PROBE_CACHE:
        _PROBE_CACHE[cache_key] = rotation_probe(RustEngine(*args, **kwargs))
    return _PROBE_CACHE[cache_key]


class SymmetricRustEngine(RustEngine):
    """``RustEngine`` under the frame-planned pathfinder (``path_search="trace_fitted_astar"``),
    which also selects the fixed-distance knockback, AND the own-frame deploy clamp
    (``ground_y_clamp="deploy_column_range_own_frame"``), which it does not (see
    ``RustEngine.__init__``).

    FOR ROTATION-MIRROR GATES ONLY. The shipped search is the game's own, measured on
    client 16.402, and it is not seat-symmetric: its goal scan and
    neighbour order run in absolute arena coordinates, so a Red unit and its
    rotated Blue twin can publish different equal-cost routes (measured: 20 of 54 twin
    problems on the shipped arena).
    That is the real game and ``RustEngine`` reproduces it. The tests that assert a
    mirrored battle stays a rotation to the end exist to catch seat bias in the ENGINE'S
    OTHER SYSTEMS and in this adapter, so they run under this class, whose routes are
    exact rotations by construction.

    THE DEPLOY CLAMP IS THE SAME KIND OF THING AND HAD TO BE ASKED FOR SEPARATELY.
    ``formation.GROUND_Y_CLAMP``'s shipped arm is measured per side and is deliberately
    not the rotation of itself, so a multi-unit card deployed at mirrored points does
    not land at mirrored positions -- measured through the engine, Skeletons over the
    230 legal mirrored tile centres broke the rotation 22 times under the shipped arm
    and 0 times under the own-frame one. Until the core exposed the key this class set
    only ``path_search``, so the rotation gates ran against the asymmetric clamp and
    correctly reported an asymmetry that is real and intended.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("path_search", "trace_fitted_astar")
        kwargs.setdefault("ground_y_clamp", "deploy_column_range_own_frame")
        kwargs.setdefault("ground_deploy_point", "none")
        self._probe_args = (args, dict(kwargs))
        super().__init__(*args, **kwargs)

    def symmetry_problems(self) -> list[str]:
        """Where this vehicle is NOT a rotation mirror, measured. Empty means it is.

        Not raised from ``__init__`` on purpose. Some callers want this class for its
        determinism rather than its symmetry -- the spawn-ordinal parity gates, for
        instance -- and refusing to construct would fail six tests whose claim has
        nothing to do with seats. The check belongs where the claim is made:
        ``rotation_divergence`` calls it before measuring anything, and one test
        asserts it is empty so the state of the vehicle is visible on its own.
        """
        args, kwargs = self._probe_args
        return list(_rotation_probe_cached(args, kwargs))

    def symmetry_report(self) -> str:
        """``symmetry_problems`` with what to do about it. Empty when there is nothing."""
        problems = self.symmetry_problems()
        if not problems:
            return ""
        _, kwargs = self._probe_args
        asked = ", ".join(f"{k}={v!r}" for k, v in sorted(kwargs.items()) if v is not None)
        return (
            "SymmetricRustEngine is not symmetric on this engine build. A multi-unit card "
            "deployed at the same OWN-FRAME tile by both seats does not land at mirrored "
            "positions:\n  " + "\n  ".join(problems) + "\n\n"
            "The engine is not wrong. It reproduces a measured property of the real game "
            "that is keyed per side or per arena half, and this class exists to select the "
            f"symmetric arm of every such key. It asks for {asked}, and that is no longer "
            "all of them.\n\n"
            "The arm is chosen at construction, so the missing key has to be a keyword "
            "RustEngine accepts and passes to the core. Compare the formation section of "
            "`royalesim.EMBEDDED_CALIBRATION_JSON` against RustEngine.__init__: a key whose "
            "candidates include a 'none' or '*_own_frame' arm is one this class has to be "
            "able to ask for. Until it can, a rotation gate run on this vehicle measures a "
            "property the real game has and reports it as a defect."
        )

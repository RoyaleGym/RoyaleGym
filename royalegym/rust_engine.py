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
      ROLLING, 4 SPELL_NOT_ON_WATER, 5 TUNNEL, 6 MIRROR). A unit a spell releases
      (the Goblin Barrel's Goblins) is not a card; the core reports it under the releasing spell's
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
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import msgspec

from .mock_engine import RAW_CARD_PACK, RAW_CARD_PACK_TABLE_VINTAGE
from .protocol import (
    ABILITY_BUTTONS,
    BLUE,
    CARD_KINDS,
    DEFAULT_DATA_DIR,
    HAND_SIZE,
    INSTALL_PAGE,
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
    EntityState,
    MatchSetup,
    Placement,
    ProjectileState,
    SpellState,
    TowerSlot,
    calibration_digest,
    calibration_values,
    card_is_spell,
    data_dir,
    default_calibration,
    derived_cards_vintage,
    fnv1a64,
    mirror_state,
    setup_asks_levels,
    to_engine,
    validate_setup,
)

# The section of THIS repo's README that walks the install end to end. Spelled once,
# so the message below and the test that checks the heading is really there cannot
# drift apart. It was "Setup" until 2026-09-22 and no README here ever had that
# heading -- a dead pointer in the one message a reader reads when nothing works.
INSTALL_SECTION = "Install"
INSTALL_POINTER = f'the "{INSTALL_SECTION}" section of the RoyaleGym README.md'
#: The page for building the engine from a RoyaleSim checkout instead of installing it.
BUILD_PAGE = "docs/site/pages/build-from-source.md"


def core_import_message(error: object) -> str:
    """What a reader is told when the engine is not installed: the install page, whose line
    works before PyPI too, then the page for a source build."""
    return (
        f"The battle engine (royalesim) is not installed ({error}). Install it with the "
        f"install line on {INSTALL_PAGE}. To build it from a RoyaleSim checkout instead, "
        f"follow {BUILD_PAGE}: the data is extracted first, then `maturin develop --release`."
    )


try:  # the extension is optional: the package must import without it
    import royalesim as _core  # type: ignore[import-not-found]  # compiled, no stubs
except ImportError as _exc:  # pragma: no cover - exercised only on unbuilt trees
    _core = None
    # The one pointer a reader gets at the moment the engine has not built, which is
    # the moment they most need it to land somewhere. It names THIS repo's README and
    # the heading it really has; tests/test_install_pointers.py checks that the heading exists,
    # because a pointer nothing reads is a pointer that rots quietly.
    CORE_IMPORT_ERROR: str | None = core_import_message(_exc)
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
    5: Placement.TUNNEL,
    6: Placement.MIRROR,
}
#: The catalogue columns this adapter reads BY POSITION, which must be the engine's first
#: ``CATALOGUE_FIELDS``. Later columns are found by name (``card_kind``).
CATALOGUE_PREFIX = (
    "name", "placement", "elixir", "count", "radius", "flying", "hitpoints", "footprint_tiles",
)
#: The champions of an engine whose catalogue has no ``champion`` column (RoyaleSim 0.1.3 and
#: before; checked card by card on 0.1.3 by which cards get an ability button with no form).
#: An engine that states the column is read from it instead, so a later champion is not missed.
CHAMPIONS_BEFORE_THE_COLUMN = frozenset({
    "MightyMiner", "SkeletonKing", "ArcherQueen", "GoldenKnight", "Monk", "LittlePrince",
    "Goblinstein", "BossBandit",
})
#: Ledger sections this package reads from the ledger OUTSIDE the engine (ElixirLaw, the
#: clock), so a ``calibration_overrides`` key in one of them would split the observation
#: from the engine: refused.
OVERRIDE_READ_HERE = frozenset({"match", "time"})
#: Keys the action mask FOLLOWS: an override of one is applied to the engine's own
#: calibration too, so ``rules()`` state the arm the core runs, and the mask implements
#: every arm ``DeployRules`` accepts for it (tests/test_troop_tower_taps.py). Any OTHER key
#: the mask's rules read (recorded as ``Arena`` and ``DeployRules`` load) is refused as an
#: override: its arms are implemented for MockEngine's model or not at all, so the mask
#: would judge one rule and the core run another (placement.ILLEGAL_TAP = refuse is a box
#: fit in the core and a collision-circle test in the mask). Keys the mask never reads go
#: to the core alone.
OVERRIDE_FOLLOWED = frozenset(
    {
        "placement.TROOP_TOWER_TAPS",
        "placement.SNAP_EVEN_CORNER",
        "placement.TAP_SNAP",
        "placement.TROOP_BUILDING_TAPS",
        "placement.SPELL_AS_DEPLOY_TAPS",
        "placement.LIVE_BOTTLE_TAPS",
    }
)


class _ReadRecorder(Calibration):
    """A calibration that records every key read from it."""

    def __init__(self, raw: dict[str, Any]) -> None:
        super().__init__(raw)
        self.read: set[str] = set()

    def entry(self, key: str) -> dict[str, Any]:
        self.read.add(key)
        return super().entry(key)

# Deploy reason names the engine exports that are NOT a DeployStatus, on purpose.
# ENGINE_ERROR marks a failure a slot command cannot produce (py.rs DEPLOY_REASONS), and
# ``_status`` raises if it ever arrives. Every other exported name must be a DeployStatus
# member: construction refuses one that is not.
_UNMAPPED_REASONS = frozenset({"ENGINE_ERROR"})
_I32_MAX = 2**31 - 1


def core_available() -> bool:
    """Whether the battle engine (``royalesim``) is installed; ``CORE_IMPORT_ERROR`` says
    why not when it is not."""
    return _core is not None


@functools.lru_cache(maxsize=1)
def engine_binary_digest() -> str:
    """A short hash of the compiled extension FILE that is loaded, or "unknown".

    ``build_digest`` identifies the DATA an engine was built with. Nothing identified
    the CODE: the extension exposes no version, and its distribution version is a
    constant, so an engine rebuilt from a modified or uncommitted Rust tree was
    indistinguishable from the one before it. Every guard here, and the ones in the
    sibling repos, compared data and would have passed.

    This does not say which commit built it, which is the engine's to publish. It says
    whether the binary is the same binary, which is the part that was invisible: two
    builds from different sources do not produce the same file. Read it beside
    ``build_digest`` -- one moving without the other is the interesting case, and a
    result recorded against a binary nobody can identify is worth less than one that
    names it.

    Cached: about 4 ms over 2.3 MB, paid once per process. "unknown" rather than raising
    when the module is not a file on disk, because a missing provenance stamp should not
    stop a battle.

    TAKE IT IN THE PROCESS THAT PRODUCED THE RESULT. This describes the binary THIS
    process loaded. Fetched later from a second process it describes whatever is on disk
    by then, which is not the same claim and is the easier one to make by accident: a
    recording, a picture or a metrics row stamped after the fact says which engine exists
    now, not which engine made it. Record it beside the result, not beside the report.
    """
    if _core is None:
        raise ImportError(CORE_IMPORT_ERROR)
    # The COMPILED submodule, not the package. ``royalesim.__file__`` is a 119-byte
    # __init__.py that re-exports it, and hashing that would give a stamp identical
    # across every rebuild: a provenance check that can never notice anything, which is
    # worse than none because it reads as evidence. Tested for, below.
    native = getattr(_core, "royalesim", _core)
    path = getattr(native, "__file__", None)
    if not path or Path(path).suffix.lower() in {".py", ".pyc"}:
        return "unknown"
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except OSError:
        return "unknown"


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
            # THE BUILD DIRECTORY MUST EXIST BEFORE ITS cards.json IS BELIEVED, and that
            # is a portability fix rather than a tightening. `Path(x, "..", ..).is_file()`
            # asks a DIFFERENT QUESTION on the two platforms: Windows normalises ".."
            # lexically and answers without the intermediate existing, while POSIX walks
            # each component and ENOENTs. Since this loop takes the FIRST candidate that is
            # a file, earliest first, the two platforms could pick DIFFERENT checkouts from
            # the same extension -- Windows accepting a short garbage candidate such as
            # "/crates/royalesim/../../data/derived/cards.json" that POSIX rejects. The
            # consequence is silent: `engine_cards_json_path` would label a card table with
            # a provenance that is confidently wrong. Requiring the directory to exist makes
            # both platforms strict and makes them agree. Found by a clean ubuntu CI runner
            # on 2026-09-23; it cannot be reproduced on Windows, which is the point.
            if not Path(build_dir).is_dir():
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


def embedded_card_table() -> str | None:
    """The card table compiled into the engine, when that is the one it reads; else None.

    A wheel built elsewhere (PyPI's) carries its tables inside and reads no cards.json at
    all: ``royalesim.card_table_source()`` says "embedded", and the table is
    ``royalesim.EMBEDDED_CARDS_JSON``. An engine built in a checkout says "file:<path>" and
    reads that file (``engine_cards_json_path``).
    """
    if _core is None:
        return None
    source = getattr(_core, "card_table_source", None)
    if not callable(source) or source() != "embedded":
        return None
    text = getattr(_core, "EMBEDDED_CARDS_JSON", None)
    return text if isinstance(text, str) and text else None


@functools.cache
def embedded_cards_stamp(text: str) -> tuple[str, str]:
    """(FNV-1a 64 of the compiled table's bytes, its ``provenance.vintage``), once per process."""
    provenance = json.loads(text).get("provenance")
    vintage = provenance.get("vintage", "unknown") if isinstance(provenance, dict) else "unknown"
    return fnv1a64(text.encode("utf-8")), str(vintage)


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
    rust_cards: Sequence[CardInfo],
    mock_cards: Sequence[CardInfo],
    engine_vintage: str | None = None,
) -> str | None:
    """Why the two engines are reading DIFFERENT card tables, or None.

    DECIDED BY THE TABLE, NOT BY THE ROWS. The engine's cards.json names the table it is
    (``provenance.vintage``, or ``engine_vintage`` when given). When that is the table
    extracted from MockEngine's own pack (``RAW_CARD_PACK_TABLE_VINTAGE``), the two read
    the same data, and any row difference is a defect in an engine or in this adapter,
    which builds the rows: None is returned and the caller's comparison FAILS on it.
    Until 2026-09-27 a row difference alone meant "different tables", so an adapter bug
    in one of these fields (a radius scaled in Python) read as a table split and skipped.

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
    compiled = embedded_card_table()
    if engine_vintage is not None:
        vintage = engine_vintage
    elif compiled is not None:
        vintage = embedded_cards_stamp(compiled)[1]
    else:
        engine_cards = engine_cards_json_path()[0] if _core is not None else None
        vintage = derived_cards_vintage(engine_cards)
    if vintage == RAW_CARD_PACK_TABLE_VINTAGE:
        return None
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
        calibration_overrides: Mapping[str, Any] | None = None,
        tap_snap: str | None = None,
        command_delay_ticks: int | tuple[int, int] = 0,
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
        laid on a point one native unit off the point the tap resolves to (its tile centre
        under placement.TAP_SNAP) -- in x when the tap is on the
        arena's LEFT half, either seat, and in y when the owner is side 1, either half.
        A FLYING summon's ring is laid on the tap itself. Two offsets keyed two
        different ways, so the shipped arm is neither seat-symmetric nor frame-symmetric.
        ``none`` lays every ring on the tap. Like the clamp it is independent of
        ``path_search`` and has to be asked for by name.

        ``tap_snap``: None = the ledger's ``placement.TAP_SNAP``. ``"none"`` judges a troop's
        body at its tap, the seat-symmetric arm (``SymmetricRustEngine`` asks for it). Passed to
        the core's own keyword, which an engine before RoyaleSim 6909b6f lacks: then it is
        refused here. The mask follows it, as it follows ``OVERRIDE_FOLLOWED``.

        ``calibration_overrides``: ledger key -> value, as ``data/calibration.json`` writes
        it (``{"targeting.FIRST_TOWER_PICK": "current_x"}``). The core runs the battle with
        those arms instead of the shipped ones and refuses a key or arm it does not have.
        ``config()`` and every trace recorded from this engine carry them, because an
        overridden engine plays other battles from the same build. A key the mask follows
        (``OVERRIDE_FOLLOWED``: the six placement keys of the relocation rule) is applied to
        this engine's
        own calibration too, so ``rules()`` -- what the mask reads -- state the arm the core
        runs. Any other key the mask's rules read is refused, and so are the ``match`` and
        ``time`` sections, which this package reads elsewhere (the elixir law, the clock):
        the core would run one value and the mask or the observation another."""
        if _core is None:
            raise ImportError(CORE_IMPORT_ERROR)
        self.calibration_overrides: dict[str, Any] = dict(calibration_overrides or {})
        read_here = sorted(
            k for k in self.calibration_overrides if k.split(".", 1)[0] in OVERRIDE_READ_HERE
        )
        if read_here:
            raise ValueError(
                f"calibration_overrides {read_here}: this package reads those keys from the "
                "ledger itself (the elixir law, the clock), so the engine would run one value "
                "and the observation another"
            )
        cal = calibration or default_calibration()
        diffs = stale_build_differences(cal, arena_path)
        if diffs:
            raise RuntimeError(
                "royalesim was built against different data; rebuild with "
                "`maturin develop --release`:\n  " + "\n  ".join(diffs)
            )
        # After the stale-build check, which compares the LEDGER with the build. A key the
        # ledger lacks is left to the core, whose refusal names it.
        for key, value in self.calibration_overrides.items():
            section, _, name = key.partition(".")
            if key in OVERRIDE_FOLLOWED and name in cal.raw.get(section, {}):
                cal = cal.with_override(key, value)
        if tap_snap is not None:
            if not battle_takes("tap_snap"):
                raise ValueError(
                    f"tap_snap={tap_snap!r}: this engine's Battle has no tap_snap keyword "
                    "(RoyaleSim 6909b6f adds it)"
                )
            if "TAP_SNAP" in cal.raw.get("placement", {}):
                cal = cal.with_override("placement.TAP_SNAP", tap_snap)
        self.calibration = cal
        recorded = _ReadRecorder(cal.raw)
        self._arena = Arena.load(recorded, arena_path)
        self._rules = DeployRules.load(recorded)
        unfollowed = sorted(
            k
            for k in self.calibration_overrides
            if k in recorded.read and k not in OVERRIDE_FOLLOWED
        )
        if unfollowed:
            raise ValueError(
                f"calibration_overrides {unfollowed}: the action mask's rules read those keys "
                f"and follow only {sorted(OVERRIDE_FOLLOWED)}, so the core would run one arm "
                "and the mask judge another"
            )
        self.slot_of_k = _derive_slot_of_k(self._arena)
        self.path_search = path_search
        self.ground_y_clamp = ground_y_clamp
        self.ground_deploy_point = ground_deploy_point
        # The card table is read by the Battle below, from this file (module doc), or from
        # the table compiled into a wheel built elsewhere, which reads no file and has none.
        # Stamped from disk on both sides of that read unless the engine reports it.
        self.embedded_cards = embedded_card_table()
        self.cards_json_path: Path | None
        if self.embedded_cards is not None:
            self.cards_json_path, self.cards_json_found_by = None, "embedded"
        else:
            self.cards_json_path, self.cards_json_found_by = engine_cards_json_path()
        from_engine = any(hasattr(_core.Battle, n) for n in ENGINE_CARD_HASH_NAMES)
        before = None if from_engine or self.embedded_cards else self._disk_stamp()
        # By keyword and only when there are any: a build older than the core's keyword
        # still constructs a battle that overrides nothing.
        overrides: dict[str, Any] = (
            {"calibration_overrides": {
                k: json.dumps(v) for k, v in self.calibration_overrides.items()
            }}
            if self.calibration_overrides
            else {}
        )
        self.tap_snap = tap_snap
        if tap_snap is not None:
            overrides["tap_snap"] = tap_snap
        battle_args = (
            list(card_names) if card_names is not None else None,
            self.slot_of_k,
            path_search,
            ground_y_clamp,
            ground_deploy_point,
        )
        self._battle = _core.Battle(*battle_args, **overrides)
        self.command_delay_ticks: tuple[int, int] = (0, 0)
        self.set_command_delay_ticks(command_delay_ticks)
        self._card_table = self._stamp_card_table(before)
        terr = territory_differences(self._battle, self._rules, self._arena, self.slot_of_k)
        if terr:
            read = (
                "the card table compiled into it"
                if self.cards_json_path is None
                else f"data/derived/cards.json in the checkout it was built in "
                f"({self.cards_json_path})"
            )
            raise RuntimeError(
                "the Rust engine and the action mask disagree on troop territory. The "
                f"engine read {read}; the mask reads the one under data_dir() "
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
        # than rekeying the row, because this is decoded per battle. The columns this
        # adapter reads by position must still be the engine's first ones, by NAME, and a
        # later column is found by its name; an unknown one is ignored, which is what lets
        # an engine built with a newer column load here.
        fields = list(getattr(_core, "CATALOGUE_FIELDS", ()))
        if fields and fields[: len(CATALOGUE_PREFIX)] != list(CATALOGUE_PREFIX):
            raise RuntimeError(
                f"the Rust catalogue's columns begin {fields[: len(CATALOGUE_PREFIX)]}, and "
                f"this adapter reads them by position as {list(CATALOGUE_PREFIX)}"
            )
        kind_at = fields.index("card_kind") if "card_kind" in fields else None
        champion_at = fields.index("champion") if "champion" in fields else None
        evo_at = fields.index("evo_cycle") if "evo_cycle" in fields else None

        def champion(row: list) -> bool:
            if champion_at is not None and len(row) > champion_at:
                return bool(row[champion_at])
            return row[0] in CHAMPIONS_BEFORE_THE_COLUMN

        self._cards = [
            CardInfo(
                card_id=cid,
                name=row[0],
                elixir=row[2],
                placement=int(_PLACEMENT_OF_KIND[row[1]]),
                count=row[3],
                radius=row[4],
                flying=row[5],
                hitpoints=row[6],
                footprint_tiles=row[7] if len(row) > 7 else None,
                card_kind=row[kind_at] if kind_at is not None and len(row) > kind_at else None,
                champion=champion(row),
                evo_cycle=int(row[evo_at]) if evo_at is not None and len(row) > evo_at else None,
            )
            for cid, row in enumerate(rows)
        ]
        bad_kinds = sorted({c.card_kind for c in self._cards} - {None, *CARD_KINDS})
        if bad_kinds:
            raise RuntimeError(
                f"the Rust catalogue states card kinds {bad_kinds} this adapter lacks"
            )
        # A REASON NAME WITH NO STATUS IS REFUSED HERE, NOT WHEN IT FIRES. Mapped to None,
        # it raises in ``_status`` on the first deploy the rule refuses, which can be hours
        # into a training run: a Mirror played before any other card is the first time an
        # engine that exports NOTHING_TO_MIRROR would say it. Refusing at construction
        # costs one failed start, and it names the member to add.
        unnamed = [
            n
            for n in _core.DEPLOY_REASONS
            if n not in DeployStatus.__members__ and n not in _UNMAPPED_REASONS
        ]
        if unnamed:
            raise RuntimeError(
                f"the engine exports deploy reasons {unnamed} that protocol.DeployStatus "
                "has no member for, so a deploy it refuses for one of them could not be "
                "reported. Add each as a DeployStatus member with the next free value "
                "(values are stored in traces: append, never renumber)."
            )
        self._status_of_reason: list[int | None] = [
            int(DeployStatus[n]) if n in DeployStatus.__members__ else None
            for n in _core.DEPLOY_REASONS
        ]
        #: Which positional layouts were checked against the engine's own export, by name.
        #: False means the engine exports no field list, so that layout is TRUSTED, not
        #: verified -- see ``check_field_order``.
        self.field_order_checked = check_field_order(_core)
        self._decode_state = msgspec.json.Decoder(BattleState)
        self._reset_called = False
        # Each side's (deck, forms) from the last reset, for an engine whose state does not
        # carry them (before royalesim 0.1.9). None after load_state: that battle's setup was
        # never seen here, and a stale deck would be a wrong answer, not a missing one.
        self._setup_cards: list[tuple[list[int], list[int]]] | None = None
        #: Whether the last step's results carried the engine's RESOLVED deploy
        #: position, or fell back to the command. None until something is stepped.
        #: A consumer that treats DeployResult.x,y as a place should read this and
        #: refuse rather than be handed a tap that looks like a landing.
        self.reports_resolved_position: bool | None = None
        #: Ability buttons per side, the engine's own count: the parser, the observation
        #: and ``_wire`` read this, so a build that adds a button (the champion's, the
        #: third) is pressable without a gym change.
        self.ability_button_count = int(getattr(_core, "ABILITY_BUTTONS", ABILITY_BUTTONS))
        # A SPELL WITH A TROOP'S DEPLOY RULE. Kind code 3 is every spell that takes a troop's
        # territory, and the core refuses some of those on a building (Heal) and not others
        # (the Log); the catalogue does not say which, so the mask offered Heal on the
        # seat's own crown towers and a training run stopped on the refusal. Asked of the
        # core, once per catalogue (``troop_ruled_spells``): each such spell gets a troop's
        # placement, which is the code RoyaleSim's next build gives it, where this finds
        # nothing to change. Its card_kind stays SPELL, so the troop-only laws skip it.
        ruled = troop_ruled_spells(self, battle_args, overrides)
        if ruled:
            self._cards = [
                msgspec.structs.replace(c, placement=int(Placement.TROOP)) if c.name in ruled else c
                for c in self._cards
            ]
        # A CARD THAT TRAVELS UNDER GROUND (the Miner, the Goblin Drill) goes down anywhere
        # on land, and RoyaleSim up to 244c893 reports it under its kind's code, 0 or 1,
        # which would mask it into a troop's or a building's territory. Asked of the core,
        # once per catalogue (``tunnelling_cards``): each gets code 5, TUNNEL, which is
        # the code RoyaleSim gives it once this package maps 5, where this finds nothing.
        tunnels = tunnelling_cards(self, battle_args, overrides)
        if tunnels:
            self._cards = [
                msgspec.structs.replace(c, placement=int(Placement.TUNNEL))
                if c.name in tunnels
                else c
                for c in self._cards
            ]

    # ------------------------------------------------------------ protocol

    def cards(self) -> Sequence[CardInfo]:
        return self._cards

    def arena(self) -> Arena:
        return self._arena

    def rules(self) -> DeployRules:
        return self._rules

    def set_command_delay_ticks(self, delay: int | tuple[int, int]) -> None:
        """THE COMMAND DELAY, in ticks, from the next ``reset`` on: an int for both seats, or
        (Blue, Red). A play or press accepted on tick T runs on T + delay, checked again in
        full then; until it runs the card stays in hand, the bar unspent, and the card or
        button refuses another command (CARD_PENDING). 0, the default, runs every command at
        once. The live client's is measured at 21-22 ticks (RoyaleSim df69520).

        A delay above 0 needs an engine that has it and can name its refusal: refused here,
        not at the first refused command."""
        pair = (delay, delay) if isinstance(delay, int) else tuple(delay)
        if len(pair) != 2 or any(not isinstance(d, int) or d < 0 for d in pair):
            raise ValueError(
                f"command_delay_ticks must be ticks >= 0, one or (blue, red): {delay!r}"
            )
        if pair != (0, 0):
            if not hasattr(self._battle, "set_command_delay_ticks"):
                raise NotImplementedError(
                    "this RoyaleSim build has no command delay (Battle.set_command_delay_ticks, "
                    "RoyaleSim df69520 on)"
                )
            if "CARD_PENDING" not in getattr(_core, "DEPLOY_REASONS", ()):
                raise NotImplementedError(
                    "this RoyaleSim build delays commands but does not name the refusal of a "
                    "card whose command is waiting (DEPLOY_REASONS lacks CARD_PENDING)"
                )
        if hasattr(self._battle, "set_command_delay_ticks"):
            self._battle.set_command_delay_ticks(*pair)
        self.command_delay_ticks = (int(pair[0]), int(pair[1]))

    def unit_hitpoints(self, card_id: int, level: int) -> list[tuple[str, str, int]]:
        """Every unit a play of ``card_id`` at unified ``level`` puts on the board, as the
        engine's (role, unit name, hitpoints) rows: the card's own row first (role "own",
        the unit its catalogue row describes), then each unit it puts down, each at the level
        it takes (roles "second_summon", "spawn", "death_spawn", "release"). The engine's
        rows as they are (RoyaleSim ``Battle.unit_hitpoints``), so a consumer can tell a
        unit's hitpoints at the level it was played at, which the catalogue row gives only
        at the catalogue's level (a Knight: 1766 at 11, 1938 at 12).

        ValueError for an unknown card id or a level the card's ladder lacks, naming it.
        NotImplementedError from an engine build before the rows (RoyaleSim 6909b6f).
        Not part of ``protocol.Engine``: MockEngine has no levels, so a consumer asks for it
        with ``getattr`` and keeps its own rule where it is absent.
        """
        rows = getattr(self._battle, "unit_hitpoints", None)
        if rows is None:
            raise NotImplementedError(
                "this RoyaleSim build does not list the units a card puts down "
                "(Battle.unit_hitpoints, RoyaleSim 6909b6f on)"
            )
        return [(str(r), str(n), int(hp)) for r, n, hp in rows(int(card_id), int(level))]

    def reset(self, seed: int, setup: MatchSetup) -> None:
        """``protocol.validate_setup`` first -- nothing is read into the core, and no
        adapter table is indexed, until the whole setup has passed. See the module doc."""
        validate_setup(self._arena, self._cards, setup)
        levels: dict[str, Any] = {}
        if setup_asks_levels(setup):
            signature = getattr(self._battle.reset, "__text_signature__", "") or ""
            if "tower_levels" not in signature:
                raise NotImplementedError(
                    "this RoyaleSim build has no per-side levels (Battle.reset(levels, "
                    "tower_levels), RoyaleSim 0.1.7 on); MatchSetup.levels and tower_levels "
                    "need it"
                )
            if setup.levels is not None:
                levels["levels"] = [list(map(int, side)) for side in setup.levels]
            if setup.tower_levels is not None:
                levels["tower_levels"] = list(map(int, setup.tower_levels))
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
            # By keyword and only when set: an engine older than the forms refuses it.
            **({"forms": [list(map(int, f)) for f in setup.forms]} if setup.forms else {}),
            # The same, for per-side levels: only when one is asked for.
            **levels,
        )
        self._reset_called = True
        self._setup_cards = [
            (
                list(map(int, setup.decks[team])),
                list(map(int, setup.forms[team])) if setup.forms else [0] * len(setup.decks[team]),
            )
            for team in TEAMS
        ]

    def check_deploy(self, command: DeployCommand) -> int:
        return self._status(self._battle.check_deploy(*self._wire(command)))

    def building_placement(
        self, team: int, card_name: str, x: int, y: int
    ) -> tuple[int, int, tuple[int, int, int, int]] | None:
        """Where a building tapped at ENGINE-frame ``(x, y)`` would end up, or None.

        ``(centre_x, centre_y, (x0, y0, x1, y1))``, engine frame, subtiles. A tap whose
        box does not fit is not refused: the engine moves the building to the nearest
        place it does fit, so the centre this returns is often not the point tapped.
        None means the tap is refused outright, which a tap outside the arena, on water,
        on a no-deploy cell or outside the card's territory still is.

        This is on the engine because WHERE a building lands is the engine's rule.
        Working it out here would be a second copy of that rule, and the two would part.
        ``action.GridActionParser`` asks it for the ``taps_where_the_building_stays``
        arm, and an engine that cannot answer refuses that arm rather than pretending.
        """
        got = self._battle.building_placement(int(team), str(card_name), int(x), int(y))
        if got is None:
            return None
        cx, cy, box = got
        return int(cx), int(cy), (int(box[0]), int(box[1]), int(box[2]), int(box[3]))

    def step_commands_run(self) -> list[tuple[int, int, str, int, int]] | None:
        """Every delayed command that ran or was dropped during the last ``step``, in run order,
        each ``(tick, team, kind, what, reason)``: kind "deploy" or "ability"; what a play's
        card id, or a press's command slot (HAND_SIZE + button); reason 0 when it ran, else the
        ``DEPLOY_REASONS`` code it was refused with when it came to run. None from a core that
        does not report them (RoyaleSim before 0.1.7)."""
        report = getattr(self._battle, "step_commands_run", None)
        if report is None:
            return None
        return [
            (int(tick), int(team), str(kind), int(what), int(reason))
            for tick, team, kind, what, reason in report()
        ]

    def step(self, commands: Sequence[DeployCommand], ticks: int) -> list[DeployResult]:
        if ticks < 0:
            raise ValueError("ticks must be >= 0")
        raw = self._battle.step([self._wire(c) for c in commands], int(ticks))
        # TOLERANT UNPACK. The core's per-command tuple grew two TRAILING elements, the
        # RESOLVED position: where an accepted play was put (snapped, moved off an own body,
        # a building relocated), and the requested point for a refusal. This used to unpack
        # exactly three and so raised `too many values to unpack` against the new core -- on
        # every engine deploy, in every repo, the moment the extension was rebuilt. The
        # trailing shape was chosen so a tolerant reader keeps working; this reader was not one.
        # It is now,
        # in both directions: an older core returning three still works, and the command's
        # own point is the fallback, which is exactly what the field used to hold.
        out = []
        for c, (card_id, reason, tick, *rest) in zip(commands, raw, strict=True):
            resolved = len(rest) >= 2
            x, y = (rest[0], rest[1]) if resolved else (c.x, c.y)
            if not resolved and self.reports_resolved_position is not False:
                # SAID OUT LOUD, ONCE, because the fallback is silently WRONG rather than
                # silently missing. Against an old core these are the tap, and a tap is a
                # full tile or more from the landing on half of accepted building taps
                # (measured: 118 of 234, no small-error tail). A caller reading
                # DeployResult.x as a position would get a plausible number with nothing
                # to distinguish it from a resolved one, and the field's name writes the
                # false sentence for them.
                warnings.warn(
                    "this royalesim core returns no resolved deploy position, so "
                    "DeployResult.x,y are the COMMAND. For a building whose footprint did "
                    "not fit, that is where it was tapped and not where it stands: half of "
                    "accepted Cannon taps relocate, by a full tile or more. Rebuild with "
                    "`maturin develop --release` for the resolved position, or read "
                    "RustEngine.reports_resolved_position before treating x,y as a place.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.reports_resolved_position = resolved
            out.append(DeployResult(c.team, c.hand_slot, card_id, self._status(reason), tick, x, y))
        return out

    def state(self) -> BattleState:
        state = self._decode_state.decode(self._battle.state_json())
        known = self._setup_cards
        if known is not None and not state.players[0].deck:
            state = msgspec.structs.replace(state, players=[
                msgspec.structs.replace(p, deck=list(deck), forms=list(forms))
                for p, (deck, forms) in zip(state.players, known, strict=True)
            ])
        return state

    def save_state(self) -> bytes:
        return bytes(self._battle.save())

    def load_state(self, blob: bytes) -> None:
        self._battle.load(bytes(blob))
        self._reset_called = True
        self._setup_cards = None

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
        slot = min(max(c.hand_slot, -1), HAND_SIZE + self.ability_button_count)
        lim_x = min(self._arena.width + 1, _I32_MAX)
        lim_y = min(self._arena.height + 1, _I32_MAX)
        return team, slot, min(max(x, -1), lim_x), min(max(y, -1), lim_y)

    def _status(self, reason: int) -> int:
        # A REASON THE ENGINE DID NOT EXPORT IS A VERSION MISMATCH, and it is worth one
        # sentence rather than an IndexError. The table is built from the engine's own
        # DEPLOY_REASONS, so a code past its end means the compiled engine grew a refusal
        # whose NAME was not added to that list -- which happened on 2026-09-23, when a
        # rebuild introduced the deploy lockout and returned reason 13 against 13 exported
        # names. Every consumer deploying before tick 90 died in `list index out of range`,
        # a message that names neither deploys nor lockouts.
        #
        # This REFUSES rather than falling back to a generic "refused". A fallback would let
        # a policy read a verdict this wrapper cannot name, and the whole point of the enum
        # is that a refusal says which rule refused it. Mapping an unknown code to a known
        # status is inventing that answer.
        if not 0 <= reason < len(self._status_of_reason):
            known = ", ".join(_core.DEPLOY_REASONS)  # type: ignore[union-attr]
            raise RuntimeError(
                f"the engine returned deploy reason {reason}, past the "
                f"{len(self._status_of_reason)} reasons it exports ({known}). The compiled "
                f"engine (binary {engine_binary_digest()}, data {build_digest()}) has a "
                "refusal this wrapper cannot name. Add the name to the engine's "
                "DEPLOY_REASONS and the matching member to protocol.DeployStatus; do not "
                "map it to an existing status, which would report a rule that did not fire."
            )
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
            # Only when set, so an engine that selects nothing reads as it always did.
            **({"tap_snap": self.tap_snap} if self.tap_snap is not None else {}),
            **(
                {"command_delay_ticks": list(self.command_delay_ticks)}
                if self.command_delay_ticks != (0, 0)
                else {}
            ),
            # Only when there are any, so a plain engine's config reads as it always did.
            **(
                {"calibration_overrides": dict(self.calibration_overrides)}
                if self.calibration_overrides
                else {}
            ),
            "calibration_digest": calibration_digest(self.calibration),
            # The three things an engine is: the data compiled in, the card table read
            # at construction, and the binary itself. The third was missing, so a
            # rebuild from a changed Rust tree left no trace anywhere.
            "build_digest": build_digest(),
            "engine_binary_sha256": engine_binary_digest(),
            **self._card_table,
        }

    def engine_identity(self) -> dict[str, str]:
        """Which engine this is: the data compiled in (``build_digest``) and the binary
        loaded in THIS process (``engine_binary_sha256``). A trace header records both,
        taken while it records, which is the moment the docstring of
        ``engine_binary_digest`` asks for."""
        return {"build_digest": build_digest(), "engine_binary_sha256": engine_binary_digest()}

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
        that changed in between is refused); ``"embedded"`` when the engine reads the
        table compiled into it (a wheel built elsewhere, ``embedded_card_table``) and it
        was hashed from that; ``"unavailable"`` when the file could not be found, and
        then the other two are "" and "unknown".
        """
        return dict(self._card_table)

    def _disk_stamp(self) -> tuple[str, str] | None:
        path = self.cards_json_path
        return cards_json_stamp(path) if path is not None and path.is_file() else None

    def _stamp_card_table(self, before: tuple[str, str] | None) -> dict[str, str]:
        engine_hash = _engine_card_hash(self._battle)
        compiled = self.embedded_cards
        disk = embedded_cards_stamp(compiled) if compiled is not None else self._disk_stamp()
        if engine_hash is not None:
            # The table's vintage names the engine's table only if it IS that table.
            vintage = disk[1] if disk is not None and disk[0] == engine_hash else "unknown"
            return _card_table(engine_hash, vintage, "engine")
        if compiled is not None:
            return _card_table(disk[0], disk[1], "embedded")
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


#: Positional structs and the engine export that names each one's columns.
POSITIONAL_LAYOUTS = (
    ("ENTITY_FIELDS", EntityState),
    ("SPELL_FIELDS", SpellState),
    ("PROJECTILE_FIELDS", ProjectileState),
)


def check_field_order(core: Any) -> dict[str, bool]:
    """Refuse an engine whose positional columns do not line up with this package's.

    WHY THIS EXISTS. EntityState, SpellState and ProjectileState are ``array_like``: the
    engine's state_json sends each row as a JSON ARRAY and it is decoded BY POSITION. A
    type mismatch raises on its own, but two adjacent fields of one type do not -- on
    2026-09-24 the engine gained ``target_uid`` and ``attack_phase``, two ints side by
    side, and a swap between sim's order and this package's would have drawn a uid as an
    attack phase without a word. The day before, DEPLOY_REASONS had failed exactly this
    way: a positional list grew in one repo and not the other.

    THE RULE. The engine's list must be a PREFIX of ours. Ours may be longer -- a newer
    royalegym reading an older engine, whose missing trailing columns decode to their
    "not reported" defaults. The engine's may not be longer, and no shared column may be
    in a different place.

    Returns, per export name, whether that layout was actually checked. An engine that
    exports no field list passes UNCHECKED, and says so, rather than passing silently.
    """
    checked: dict[str, bool] = {}
    for export, struct in POSITIONAL_LAYOUTS:
        theirs = getattr(core, export, None)
        if theirs is None:
            checked[export] = False
            continue
        theirs = [str(f) for f in theirs]
        ours = list(struct.__struct_fields__)
        if len(theirs) > len(ours):
            raise RuntimeError(
                f"the engine sends {len(theirs)} {struct.__name__} columns and this royalegym "
                f"knows {len(ours)}: {theirs[len(ours):]} are new. The engine is newer than "
                "this package; update royalegym before decoding its state."
            )
        for i, (a, b) in enumerate(zip(theirs, ours, strict=False)):
            if a != b:
                raise RuntimeError(
                    f"{struct.__name__} column {i} is {a!r} in the engine and {b!r} here. These "
                    "rows are decoded BY POSITION, so every column from here on would be read "
                    f"into the wrong field without an error.\n  engine: {theirs}\n"
                    f"  here:   {ours}"
                )
        checked[export] = True
    return checked


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
            # Start past the opening deploy lockout, or every probe command is refused
            # and the probe reports no asymmetry because it never placed anything --
            # which reads exactly like a symmetric engine.
            engine.reset(
                _PROBE_SEED,
                MatchSetup(
                    decks=[deck, deck],
                    shuffle=0,
                    start_tick=engine.rules().deploy_lockout_ticks,
                ),
            )
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
    compiled = embedded_card_table()
    if compiled is not None:
        table = embedded_cards_stamp(compiled)[0]
    else:
        cards = engine_cards_json_path()[0]
        table = cards_json_stamp(cards)[0] if cards.is_file() else ""
    cache_key = (RustEngine.build_digest(), table, repr(args), repr(sorted(kwargs.items())))
    if cache_key not in _PROBE_CACHE:
        _PROBE_CACHE[cache_key] = rotation_probe(RustEngine(*args, **kwargs))
    return _PROBE_CACHE[cache_key]


_TROOP_RULED_CACHE: dict[tuple, frozenset[str]] = {}


def troop_ruled_spells(engine: RustEngine, battle_args: tuple, overrides: dict) -> frozenset[str]:
    """The names of the catalogue's kind-3 spells (placement ROLLING) that the core judges
    by a TROOP's rule: accepted on free ground in the caster's own half, refused OCCUPIED on
    its own princess tower. Measured on a probe battle built exactly as ``engine``'s, both
    seats, and cached per build, card table, catalogue and constructor arguments. Both seats
    must agree, or the spell is left as it is and the every-card mask gate names it.
    """
    probes = [c for c in engine._cards if c.placement == Placement.ROLLING and card_is_spell(c)]
    if not probes:
        return frozenset()
    key = (
        build_digest(),
        repr(sorted(engine._card_table.items())) if isinstance(engine._card_table, dict)
        else repr(engine._card_table),
        tuple(c.name for c in engine._cards),
        repr(battle_args[2:]),
        repr(sorted(overrides.items())),
    )
    if key in _TROOP_RULED_CACHE:
        return _TROOP_RULED_CACHE[key]
    battle = _core.Battle(*battle_args, **overrides)
    a, t = engine._arena, engine._arena.subtile
    full = 1000 * engine.calibration.int("match.MAX_MANA")
    others = [c.card_id for c in engine._cards if c not in probes]
    verdicts: dict[str, set[tuple[int, int]]] = {c.name: set() for c in probes}
    for i in range(0, len(probes), HAND_SIZE):
        group = [c.card_id for c in probes[i : i + HAND_SIZE]]
        deck = (group + others + group * 8)[:8]
        battle.reset(1, [deck, deck], 0, engine._rules.deploy_lockout_ticks, [full, full], None, [])
        state = engine._decode_state.decode(battle.state_json())
        for team in TEAMS:
            tower = a.princess_centers[team][0]
            free = to_engine(a, team, 9 * t + t // 2, 10 * t + t // 2)
            for slot, cid in enumerate(state.players[team].hand):
                if cid not in group:
                    continue
                taps = [engine._wire(DeployCommand(team, slot, x, y)) for x, y in (tower, free)]
                on, off = (engine._status(battle.check_deploy(*tap)) for tap in taps)
                verdicts[engine._cards[cid].name].add((on, off))
    ruled = frozenset(
        name
        for name, seen in verdicts.items()
        if len(seen) == 1 and seen == {(int(DeployStatus.OCCUPIED), int(DeployStatus.OK))}
    )
    _TROOP_RULED_CACHE[key] = ruled
    return ruled


_TUNNEL_CACHE: dict[tuple[Any, ...], frozenset[str]] = {}


def battle_takes(keyword: str) -> bool:
    """Whether the core's ``Battle`` constructor takes ``keyword``, from its text signature
    (pyo3 ``#[pyo3(signature = ...)]``). False without a core or a signature."""
    sig = getattr(getattr(_core, "Battle", None), "__text_signature__", None) or ""
    return re.search(rf"\b{re.escape(keyword)}\s*=", sig) is not None


def tunnelling_cards(engine: RustEngine, battle_args: tuple, overrides: dict) -> frozenset[str]:
    """The names of the catalogue's troop and building cards (codes 0 and 1) that the core
    accepts in the ENEMY half, on free land well inside the enemy's tower rects, where it
    refuses every other troop and building OUT_OF_TERRITORY: the cards that travel under
    ground (placement.SPAWN_PATHFIND_TERRITORY). Measured on a probe battle built exactly
    as ``engine``'s, both seats, and cached per build, card table, catalogue and
    constructor arguments. Both seats must agree, and the same card must be accepted on
    free land in its own half, or it is left as it is and the every-card gate names it.
    A card the opening deal never puts in hand is not asked, and keeps its code.
    """
    probes = [
        c
        for c in engine._cards
        if c.placement in (Placement.TROOP, Placement.BUILDING)
        and c.card_kind in ("TROOP", "BUILDING")
    ]
    if not probes:
        return frozenset()
    key = (
        build_digest(),
        repr(sorted(engine._card_table.items())) if isinstance(engine._card_table, dict)
        else repr(engine._card_table),
        tuple(c.name for c in engine._cards),
        repr(battle_args[2:]),
        repr(sorted(overrides.items())),
    )
    if key in _TUNNEL_CACHE:
        return _TUNNEL_CACHE[key]
    battle = _core.Battle(*battle_args, **overrides)
    t = engine._arena.subtile
    full = 1000 * engine.calibration.int("match.MAX_MANA")
    verdicts: dict[str, set[tuple[int, int]]] = {c.name: set() for c in probes}
    ids = [c.card_id for c in probes]
    for i in range(0, len(ids), HAND_SIZE):
        group = ids[i : i + HAND_SIZE]
        deck = (group + [x for x in ids if x not in group] + group * 8)[:8]
        battle.reset(1, [deck, deck], 0, engine._rules.deploy_lockout_ticks, [full, full], None, [])
        state = engine._decode_state.decode(battle.state_json())
        for team in TEAMS:
            # Own-frame tile (9, 21): the enemy half, mid-lane, clear of every enemy tower's
            # body and inside the rects that close the enemy half to a troop while its
            # towers stand. Own-frame tile (9, 10): free land in the caster's own half.
            enemy = to_engine(engine._arena, team, 9 * t + t // 2, 21 * t + t // 2)
            own = to_engine(engine._arena, team, 9 * t + t // 2, 10 * t + t // 2)
            for slot, cid in enumerate(state.players[team].hand):
                if cid not in group:
                    continue
                taps = [engine._wire(DeployCommand(team, slot, x, y)) for x, y in (enemy, own)]
                far, near = (engine._status(battle.check_deploy(*tap)) for tap in taps)
                verdicts[engine._cards[cid].name].add((far, near))
    ok = int(DeployStatus.OK)
    tunnels = frozenset(name for name, seen in verdicts.items() if seen == {(ok, ok)})
    _TUNNEL_CACHE[key] = tunnels
    return tunnels


#: Ledger keys whose shipped arm is the client's own and deliberately not seat-symmetric,
#: which the core offers no constructor keyword for, and the arms that ARE symmetric, most
#: preferred first. ``SymmetricRustEngine`` selects the first one the ledger lists as a
#: candidate, through ``calibration_overrides``.
SYMMETRIC_ARMS: dict[str, tuple[str, ...]] = {
    # The client's own lane rule read in the placer's own frame (a candidate since RoyaleSim
    # 1d661b0), else the old rule by the unit's x, which is symmetric but not the client's.
    "targeting.FIRST_TOWER_PICK": ("client_spawn_lane_own_frame", "current_x"),
    # The half-open king block is half-open in the ARENA's frame, so the two seats may tap
    # different own-frame points around their kings; the closed block is the same for both.
    "placement.TROOP_TOWER_TAPS": ("closed_block",),
    # The measured slide lays a death-spawn ring (Golem, Lava Hound) in the arena's frame for
    # both seats, so a Red death is not the rotation of a Blue one; not_read is symmetric.
    "spawner.DEATH_SPAWN_PUSHBACK": ("not_read",),
    # The absolute arm floors an even box's tap in the arena's frame, so a 2x2 building is not
    # the rotation of its twin; the placer's frame is the same for both seats.
    "placement.SNAP_EVEN_CORNER": ("placer_frame",),
    # as_tower_tap moves a tap off an own building by the axis push, measured in arena
    # coordinates; not_relocated moves nothing.
    "placement.TROOP_BUILDING_TAPS": ("not_relocated",),
    # The measured roll's swept rectangle is open on its max-y edge in the ARENA's frame, so
    # a Log whose front edge exactly touches a unit misses it rolling +y and hits it rolling
    # -y; the closed rectangle against the circle's edge is the same for both seats.
    "spells.ROLLING_HIT_SHAPE": ("rect_vs_circle_edge",),
}


def symmetric_overrides(ledger: Mapping[str, Any]) -> dict[str, Any]:
    """The symmetric arm of each ``SYMMETRIC_ARMS`` key this compiled ledger ships some OTHER
    arm of: the most preferred one its ``candidates`` list (the first, if it lists none).

    A key the ledger lacks is left out, so an engine older than the key still constructs;
    one already shipping the arm is left out, so nothing is overridden for nothing; one
    listing none of the symmetric arms is left out, and the rotation probe reports it.
    """
    out: dict[str, Any] = {}
    for key, arms in SYMMETRIC_ARMS.items():
        section, name = key.split(".", 1)
        entry = ledger.get(section, {}).get(name) if isinstance(ledger.get(section), dict) else None
        if not (isinstance(entry, dict) and "value" in entry):
            continue
        listed = entry.get("candidates")
        arm = next((a for a in arms if not isinstance(listed, list) or a in listed), None)
        if arm is not None and entry["value"] != arm:
            out[key] = arm
    return out


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

    A KEY WITH NO KEYWORD IS SELECTED THROUGH ``calibration_overrides``: ``SYMMETRIC_ARMS``.
    The first was ``targeting.FIRST_TOWER_PICK`` (RoyaleSim 126992a). Its shipped arm reads
    a summon member's lane in the arena's frame, measured that way on both seats, so a
    Skeleton Army on the centre line splits one member differently for Blue and for Red.
    Three rotation gates went red on it at tick 140 while this class's one-tick probe read
    clean: a lane pick shows only when the members start walking. This class took the old
    ``current_x`` arm until RoyaleSim 1d661b0 added ``client_spawn_lane_own_frame``, the
    client's lane rule read in the placer's own frame, which it now prefers.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("path_search", "trace_fitted_astar")
        kwargs.setdefault("ground_y_clamp", "deploy_column_range_own_frame")
        kwargs.setdefault("ground_deploy_point", "none")
        # placement.TAP_SNAP's tile-centre arm snaps in the arena's frame; its seat-symmetric
        # arm is "none". By the core's keyword, and only where the core has it.
        if battle_takes("tap_snap"):
            kwargs.setdefault("tap_snap", "none")
        ledger = json.loads(getattr(_core, "EMBEDDED_CALIBRATION_JSON", "{}") or "{}")
        overrides = {**symmetric_overrides(ledger), **(kwargs.get("calibration_overrides") or {})}
        if overrides:
            kwargs["calibration_overrides"] = overrides
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
            "The arm is chosen at construction. Find the key in "
            "`royalesim.EMBEDDED_CALIBRATION_JSON`, any section: its doc says it is not the "
            "rotation of itself, and one of its candidates is the symmetric arm (often "
            "'none', 'not_read', '*_own_frame' or the pre-measurement arm). Add that key and "
            "arm to rust_engine.SYMMETRIC_ARMS, which selects it through "
            "calibration_overrides. Until then, a rotation gate run on this vehicle measures "
            "a property the real game has and reports it as a defect."
        )

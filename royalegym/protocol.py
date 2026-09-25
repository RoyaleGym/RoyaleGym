"""The engine contract: everything the RL layer needs from a battle simulator.

WHY A PROTOCOL
    The Rust engine (the sibling repo, ../RoyaleSim/crates/royalesim) and this RL
    layer are developed in parallel: the RL layer is built and tested against
    ``MockEngine``, and the Rust engine satisfies the SAME ``Engine`` protocol.
    Nothing in ``royalegym`` other than ``mock_engine.py`` knows which engine it is
    driving, so a reward or observation experiment never needs a recompile.

FRAMES AND UNITS
    * Positions are integer SUBTILES in the ENGINE frame: Blue (team 0) defends
      low y, Red (team 1) defends high y. 1 tile = ``Arena.subtile`` subtiles.
    * The OWN frame of a team is the engine frame for Blue, and the engine frame
      rotated 180 degrees for Red: ``x_own = W - x``, ``y_own = H - y`` where W and
      H are the arena size in subtiles. A rotation (not a y-mirror) is used
      because it is what the opponent literally sees across the table: "my left
      princess tower" means the same thing to both players, so one policy can
      play both seats. The shipped tilemap is exactly invariant under this
      rotation with the LEFT/RIGHT lane bits swapped (tests check it).
    * Tower slots (``TowerSlot``) are always named in the owner's own frame.
    * Elixir is reported as integer thousandths (``elixir_milli``). Engines may
      keep a finer internal fraction; affordability is ``elixir_milli >=
      cost * 1000``, which is exact because ``cost * 1000`` is an integer.

NO FLOATS
    Nothing in the contract is a float. ``arena.json`` carries convenience float
    fields (``center_x: 3.5``); this module reads only the integer half-cell
    indices and derives subtile positions from them.

WHAT IS NOT KNOWN
    Several rules the mask and the engine must AGREE on are not in
    calibration.json or arena.json yet (princess tower centre, the river band
    closed to troops, buildings own-half only). They live in ``DeployRules`` /
    ``Arena`` with an explicit ``*_status`` field, so the Rust engine reads the
    same numbers the mask does rather than both sides inventing their own. See
    ``DeployRules`` for each one. Troop territory is the shipped NoDeploySize rect
    mechanic (``DeployRules``), read from data/derived/cards.json.
"""

from __future__ import annotations

import csv
import enum
import hashlib
import json
import os
from collections.abc import Sequence
from functools import lru_cache
from math import gcd
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import msgspec

# --------------------------------------------------------------------------
# Paths and data loading
# --------------------------------------------------------------------------

# The folder holding the sibling checkouts (RoyaleSim, RoyaleGym, RoyaleViser,
# RoyaleLearn side by side; README.md, "Install"). Since 2026-09-21 each is its own
# repo, so this is the workspace, not a repo root.
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
# The engine's data lives with the engine (the sibling RoyaleSim checkout's data/:
# calibration.json, raw/, derived/), never as a copy in this package.
DEFAULT_DATA_DIR = WORKSPACE_ROOT / "RoyaleSim" / "data"
DATA_DIR_ENV = "ROYALESIM_DATA_DIR"


def data_dir() -> Path:
    """RoyaleSim's ``data/`` directory (calibration.json, raw/, derived/).

    ``ROYALESIM_DATA_DIR`` when set; otherwise the sibling checkout, ``../RoyaleSim/data``
    from this repo, which is the documented workspace layout (the five repos cloned into
    one folder). Raises FileNotFoundError naming both when neither is a directory, so a
    missing sibling is reported once here rather than as a bare path from every loader.
    """
    override = os.environ.get(DATA_DIR_ENV)
    p = Path(override) if override else DEFAULT_DATA_DIR
    if not p.is_dir():
        where = (
            f"{DATA_DIR_ENV}={override}" if override else f"{DEFAULT_DATA_DIR} (no {DATA_DIR_ENV})"
        )
        raise FileNotFoundError(
            f"RoyaleSim data directory not found at {where}: clone RoyaleSim next to this "
            f"repo ({WORKSPACE_ROOT / 'RoyaleSim'}) or set {DATA_DIR_ENV} to its data/ folder"
        )
    return p


def require_data_file(p: Path, *commands: str, generated: bool = True) -> Path:
    """Return ``p``, or raise a FileNotFoundError that says how to produce it.

    A fresh clone has calibration.json and raw/, and nothing under derived/: those files
    are generated, and RoyaleSim does not track them. So the first thing every entry
    point here did on a fresh clone was raise a bare path and stop. A path is not an
    instruction, and it arrives at the exact moment the reader has nothing else to go
    on -- they have just run the first line of the README.

    One helper rather than a message per call site, because there were four of these and
    only one had been written. That one was not even reachable: ``Arena.load`` runs
    before it in both engines, so nobody ever saw the good message about cards.json.
    """
    if p.exists():
        return p
    how = "".join(f"\n    python {c}" for c in commands)
    why = (
        "It is generated, and the generated files are not in the repository."
        if generated
        else "It ships with the repository, so this usually means the data directory is wrong."
    )
    raise FileNotFoundError(
        f"{p} is absent. {why} In the sibling RoyaleSim checkout run:{how}\n"
        f"The README's Install section has every step in order, and the data has to be "
        f"extracted BEFORE the engine is built. "
        f"(data dir: {DATA_DIR_ENV} or {DEFAULT_DATA_DIR})"
    )


class Calibration:
    """Read-only view of data/calibration.json.

    Every physics constant the RL layer or the mock uses is looked up here by
    dotted key (``"time.TICK_MS"``). A missing key raises rather than defaulting,
    because a silent default is exactly the hardcoded number invariant 3 forbids.
    """

    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw

    @classmethod
    def load(cls, path: Path | None = None) -> Calibration:
        p = path if path is not None else data_dir() / "calibration.json"
        require_data_file(p, "tools/extract_arena.py", generated=False)
        return cls(json.loads(p.read_text(encoding="utf-8")))

    def entry(self, key: str) -> dict[str, Any]:
        section, name = key.split(".", 1)
        try:
            e = self.raw[section][name]
        except KeyError as exc:
            raise KeyError(f"calibration.json has no {key!r}") from exc
        if not isinstance(e, dict) or "value" not in e:
            raise KeyError(f"calibration.json entry {key!r} has no 'value'")
        return e

    def value(self, key: str) -> Any:
        return self.entry(key)["value"]

    def int(self, key: str) -> int:
        v = self.value(key)
        if isinstance(v, bool) or not isinstance(v, int):
            raise TypeError(f"calibration {key!r} is {v!r}, expected an integer")
        return v

    def bool(self, key: str) -> bool:
        v = self.value(key)
        if not isinstance(v, bool):
            raise TypeError(f"calibration {key!r} is {v!r}, expected a boolean")
        return v

    def status(self, key: str) -> str:
        return str(self.entry(key).get("status", "unknown"))

    def with_override(self, key: str, value: Any) -> Calibration:
        """A copy with one value replaced. Used by tests to prove a constant is read, not baked."""
        raw = json.loads(json.dumps(self.raw))
        section, name = key.split(".", 1)
        raw[section][name]["value"] = value
        return Calibration(raw)


def load_globals_csv(path: Path | None = None) -> dict[str, tuple[int | None, bool | None]]:
    """globals.csv as name -> (NumberValue, BooleanValue). Datamined, 2018 vintage."""
    p = path or data_dir() / "raw" / "retroroyale-2018" / "csv_logic" / "globals.csv"
    require_data_file(p, "tools/extract_globals.py", generated=False)
    out: dict[str, tuple[int | None, bool | None]] = {}
    with p.open(encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    for r in rows[2:]:
        if not r or not r[0].strip():
            continue
        num = int(r[1]) if len(r) > 1 and r[1].strip() else None
        b = r[2].strip().lower() == "true" if len(r) > 2 and r[2].strip() else None
        out[r[0].strip()] = (num, b)
    return out


# --------------------------------------------------------------------------
# Enumerations (plain ints on the wire so a Rust engine can return u8/i32)
# --------------------------------------------------------------------------

BLUE = 0
RED = 1
TEAMS = (BLUE, RED)
HAND_SIZE = 4
DECK_SIZE = 8
EMPTY_CARD = -1


class Placement(enum.IntEnum):
    """How a card may be placed. Derived from which CSV the card lives in.

    TROOP     spells_characters.csv       outside every ALIVE enemy crown tower's
                                          NoDeploySize rect, off the river band, not
                                          water, not no-deploy, not inside a building
                                          footprint (see ``DeployRules``)
    BUILDING  spells_buildings.csv        own half only (never the opened ground), not
                                          water, not no-deploy, footprint may not
                                          overlap a building
    SPELL     spells_other.csv            anywhere strictly inside the arena
    ROLLING   spells_other.csv with       like TROOP but ignores building footprints
              SpellAsDeploy=true          (Log also ships CanPlaceOnBuildings=TRUE)
    SPELL_NOT_ON_WATER                    anywhere strictly inside the arena EXCEPT where
              spells_other.csv, a spell   the TROOP water test refuses (the point touches
              whose projectile spawns     a WATER half-cell, closed cells). No territory,
              characters (Goblin Barrel)  no no-deploy cells (the king block is a legal
                                          barrel target), no footprints. Only under
                                          calibration spells.SPAWNING_SPELL_WATER_RULE =
                                          refuse_touching_water; under "anywhere" such a
                                          card is a plain SPELL.
    The numbers are the Rust core's catalogue kind codes (py.rs ``kind_code``, from
    state.rs ``deploy_rule``, the engine's one definition of a card's deploy rule).
    """

    TROOP = 0
    BUILDING = 1
    SPELL = 2
    ROLLING = 3
    SPELL_NOT_ON_WATER = 4


class SpellMotion(enum.IntEnum):
    """What a live spell object is doing (``SpellState.motion``; py.rs ``state_json``)."""

    FLIGHT = 0  # a projectile travelling to ``aim`` (Fireball, Arrows, Goblin Barrel)
    AIRBORNE = 1  # The Log before it lands and starts rolling
    ROLLING = 2  # The Log rolling toward ``aim`` (the roll's end point)
    AREA = 3  # an area effect sitting at its centre (Zap)
    PULSING = 4  # an area effect that hits every HitSpeed until its life runs out (Poison)


class EntityKind(enum.IntEnum):
    TROOP = 0
    BUILDING = 1
    KING_TOWER = 2
    PRINCESS_TOWER = 3


class TowerSlot(enum.IntEnum):
    """Tower identity in the OWNER's own frame."""

    KING = 0
    LEFT = 1
    RIGHT = 2


class DeployStatus(enum.IntEnum):
    """Why a deploy was accepted or rejected. 0 is the only success."""

    OK = 0
    BAD_TEAM = 1
    BAD_SLOT = 2
    EMPTY_SLOT = 3
    NOT_ENOUGH_ELIXIR = 4
    OUT_OF_ARENA = 5
    WATER = 6
    NO_DEPLOY = 7
    OUT_OF_TERRITORY = 8
    OCCUPIED = 9
    GAME_OVER = 10
    DUPLICATE_TEAM = 11
    # The battle's opening deploy lockout (`match.DEPLOY_LOCKOUT_TICKS`, 90 ticks as of
    # 2026-09-23): a command before it is refused. The VALUE is not the engine's reason
    # index -- that is 13 -- because `RustEngine` maps reasons to statuses BY NAME, and
    # these values are persisted in recorded traces, so they are appended rather than
    # renumbered. Added after a rebuild returned the reason with no exported name and
    # every consumer deploying before tick 90 died in `list index out of range`.
    TOO_EARLY = 12


class Winner(enum.IntEnum):
    NONE = -1
    BLUE = 0
    RED = 1
    DRAW = 2


class ShuffleMode(enum.IntEnum):
    NONE = 0  # decks are used in the given order
    INDEPENDENT = 1  # each team shuffled from its own RNG stream
    MIRRORED = 2  # both teams get the same permutation (symmetry tests, self-play)


# --------------------------------------------------------------------------
# Wire structs
# --------------------------------------------------------------------------


class CardInfo(msgspec.Struct, frozen=True):
    card_id: int
    name: str
    elixir: int
    placement: int  # Placement
    count: int  # units summoned (0 for spells)
    radius: int  # collision radius of one summoned unit / building, subtiles (0 for spells)
    flying: bool
    hitpoints: int  # per unit, 0 for spells
    # Side of the square of TILES a building stands on, or None for a card that is not
    # a building and for an engine that states no footprint (MockEngine). Trailing and
    # defaulted, so a catalogue row without the column still builds a CardInfo.
    footprint_tiles: int | None = None


class DeployCommand(msgspec.Struct, frozen=True):
    team: int
    hand_slot: int
    x: int  # engine frame, subtiles
    y: int


class DeployResult(msgspec.Struct, frozen=True):
    """What an engine did with one command, and where.

    ``x`` and ``y`` are the ENGINE frame and are the RESOLVED position: where an
    accepted deploy actually put things, which for a building whose footprint did not
    fit is where the engine relocated it to and not where it was tapped. For anything
    else, and for a REFUSED command, they are the point the command asked for, so a
    penalty term can still say where the mask and the engine disagreed rather than
    only that they did.

    THEY USED TO BE THE COMMAND, ALWAYS. That was harmless until the engine began
    relocating a building whose box does not fit, and then it was not: measured over
    234 accepted Cannon taps at tile centres, half relocated and EVERY relocation moved
    a full tile or more, so anything reading this as a position was a tile or more wrong
    on half its building sample. There is no small-error tail. It retired a published
    placement-entropy figure that had been computed over a distribution with half the
    building mass in the wrong bin. The engine now returns the point it acted on rather
    than answering a second query, because a second query is a second answer that can
    disagree with the first.

    They are here because a reward function is handed the results and not the
    commands, so without them the archetypal shaping term for this game cannot be
    written at all: reward defending near your own tower, penalise dumping a tank in
    the enemy half, reward a spell that lands on a cluster. Diffing the entity lists
    is not a substitute -- a spell that resolves inside a tick never appears there,
    and a refused command leaves no trace at all.

    Trailing and defaulted, like ``EntityState``'s status timers, so an engine or a
    recorded trace written before they existed still decodes, with 0.
    IN THE ENGINE FRAME, so a term that scores position must put them through
    ``to_own`` first or it will reward Blue and punish Red for the same placement.
    """

    team: int
    hand_slot: int
    card_id: int  # EMPTY_CARD if the slot was bad
    status: int  # DeployStatus
    tick: int  # engine tick at which the command was evaluated
    x: int = 0  # ENGINE frame, subtiles, as commanded (see the class doc)
    y: int = 0


class EntityState(msgspec.Struct, frozen=True, array_like=True):
    uid: int  # unique for the whole battle, never reused
    team: int
    kind: int  # EntityKind
    card_id: int  # EMPTY_CARD for crown towers
    tower_slot: int  # TowerSlot for crown towers, -1 otherwise
    x: int
    y: int
    hp: int
    max_hp: int
    radius: int
    flying: bool
    deploy_ticks: int  # >0 while still deploying
    # Status timers, ticks remaining rounded up (py.rs ``state_json``). Trailing and
    # defaulted, so an engine that models no status effect (MockEngine) and a trace
    # recorded before status effects existed both decode with 0.
    stun_ticks: int = 0  # >0 while stunned (Zap): no move, no attack
    knockback_ticks: int = 0  # >0 while a knockback (slide or ladder) still moves the unit
    # The ground a building or crown tower stands on: (x0, y0, x1, y1), a CLOSED box in
    # engine-frame subtiles, as the engine reports it. None for a troop, and for every
    # entity of an engine that models no box (MockEngine: ``mock_engine.FOOTPRINT_MODEL``).
    # Trailing and defaulted like the timers, so a trace or an engine row without it
    # decodes with None. Never derived here: a box drawn or masked is the engine's own.
    footprint: tuple[int, int, int, int] | None = None
    # WHAT A UNIT IS DOING, for the viewer (asked for by the viser session, 2026-09-24):
    # who it attacks, where it faces, which part of an attack it is in, and what is on it.
    # Trailing and defaulted like everything above, so MockEngine, a RustEngine built
    # before sim exported these, and every trace recorded before them decode unchanged.
    #
    # THE ORDER HERE IS THE CONTRACT. This struct is array_like: state_json sends each
    # entity as a JSON ARRAY and it is decoded BY POSITION, so these must sit in exactly
    # sim's order. target_uid and attack_phase are adjacent ints, and a swap would raise
    # nothing -- a uid would be drawn as an attack phase. RustEngine compares this field
    # list against the engine's own ENTITY_FIELDS whenever the engine exports one.
    #
    # A DEFAULT HERE MEANS "THE ENGINE DID NOT SAY", which is why attack_phase defaults to
    # -1 and not to sim's 0. Sim's 0 means IDLE; defaulting to it would make an engine that
    # exports nothing look like every unit on the board is standing still.
    target_uid: int = -1  # uid of the entity it attacks; -1 for none, or a dead target
    attack_phase: int = -1  # sim: 0 idle, 1 windup, 2 cooldown/fire; -1 not reported
    facing: tuple[int, int] = (0, 0)  # a DIRECTION in engine-frame subtiles, any length
    shield: int = 0  # shield hp left; 0 for none
    buffs: tuple[tuple[str, int], ...] = ()  # (name, ms_left); names kept whole, "|" and all
    # D10 (2026-09-25): one int of bits, each set by the engine's own predicate, not a copy
    # of it. STATUS_UNDERGROUND: tunnelling now (untargetable, immune to every hit).
    # STATUS_INVISIBLE: invisible to enemies now (untargetable; area damage still lands).
    # STATUS_HIDDEN: a building hidden in the ground (the Tesla). Higher bits reserved.
    # -1 means the engine did not report, so read it through ``status_of``, never raw.
    status_flags: int = -1


#: ``EntityState.status_flags`` bits. Read them through ``status_of``.
STATUS_UNDERGROUND = 1
STATUS_INVISIBLE = 2
STATUS_HIDDEN = 4


def status_of(entity: EntityState) -> int | None:
    """``entity.status_flags``, or None when the engine did not report it.

    THE ONLY SAFE WAY TO READ THE BITS. The "not reported" default is -1, and in Python
    ``-1 & STATUS_UNDERGROUND`` is 1: a raw mask would read an engine that said nothing as
    every unit under ground, invisible and hidden at once.
    """
    return entity.status_flags if entity.status_flags >= 0 else None


class ProjectileState(msgspec.Struct, frozen=True, array_like=True):
    """A projectile in flight (Rust core ``combat.rs`` Projectile, via py.rs ``state_json``).

    ENGINE frame, subtiles. A projectile homes on ``target_uid``; when that target dies it
    keeps flying to ``aim``, the target's last known position, which is why both are here.
    Positional like EntityState, so the field ORDER is the contract with the engine and is
    checked against its PROJECTILE_FIELDS when the engine exports one.
    """

    team: int
    x: int
    y: int
    aim_x: int
    aim_y: int
    target_uid: int  # -1 once the target is gone; it then flies on to aim
    splash: int  # splash radius in subtiles; 0 = single-target (combat.rs Projectile.splash)
    firer_card_id: int  # catalogue id; FIRER_TOWER (-1) a crown tower; FIRER_UNKNOWN (-2)


#: ``ProjectileState.firer_card_id`` for a shot from a crown tower, and only for that: the
#: engine decides it from the firer's own card (RoyaleSim 0468c60). Any other firer
#: carries the catalogue id of the card that put it on the board, so a unit summoned by
#: another card fires under its summoner's id (a Rascal Girl's shot names Rascals).
FIRER_TOWER = -1
#: ``ProjectileState.firer_card_id`` when the engine cannot name the card: the firing card
#: is not in this catalogue, or the projectile was restored from a snapshot older than the
#: field. Kept apart from FIRER_TOWER on purpose -- sim's words -- so that "unknown" never
#: reads as "a tower fired this".
FIRER_UNKNOWN = -2


class SpellState(msgspec.Struct, frozen=True, array_like=True):
    """A live spell object: cast and not yet finished (Rust core, py.rs ``state_json``).

    ENGINE frame, subtiles. ``aim`` is the landing point (FLIGHT, AIRBORNE), the roll's
    END point (ROLLING) or the centre itself (AREA, PULSING). A spell that resolves inside the
    tick it materialises never appears here -- MockEngine's spells all do, so its
    ``BattleState.spells`` is always empty (mock_engine.py WHAT IT IS NOT).
    """

    team: int
    card_id: int  # catalogue id of the spell card
    motion: int  # SpellMotion
    x: int  # current centre
    y: int
    aim_x: int
    aim_y: int
    delay_ticks: int  # FLIGHT: ticks before it starts moving; PULSING: ticks of life left
    travelled: int  # ROLLING: subtiles rolled so far (0 otherwise)
    length: int  # ROLLING: total roll length in subtiles (0 otherwise)
    hits: int  # ROLLING: units hit so far (0 otherwise)


class PlayerState(msgspec.Struct, frozen=True):
    team: int
    elixir_milli: int
    hand: list[int]  # HAND_SIZE card ids, EMPTY_CARD for an empty slot
    next_card: int
    crowns: int
    tower_hp: list[int]  # indexed by TowerSlot; 0 = destroyed
    tower_max_hp: list[int]
    king_active: bool


class BattleState(msgspec.Struct, frozen=True):
    tick: int
    tick_ms: int
    regular_ticks: int  # length of regulation time in ticks
    overtime_ticks: int  # length of overtime in ticks
    elixir_rate: int  # 1 or 2 (the multiplier currently in force)
    overtime: bool
    players: list[PlayerState]  # indexed by team
    entities: list[EntityState]  # includes crown towers
    game_over: bool
    winner: int  # Winner
    spells: list[SpellState] = []  # live spell objects, engine order (not seat-canonical)
    # Projectiles in flight. Keyed rather than positional (BattleState is a JSON object),
    # so this one is safe by name. Empty from an engine that exports none.
    projectiles: list[ProjectileState] = []


class SpawnSpec(msgspec.Struct, frozen=True):
    """A unit or building placed on the board at reset, bypassing hand and elixir.

    Deploy zones (territory, no-deploy, footprints) do NOT apply: a state mutator
    may put a troop anywhere a unit could stand. What does apply is one POSITION
    rule and one HP rule, ``spawn_violation``, and EVERY engine must refuse a spec
    that breaks either with ``ValueError`` naming the reason. See that function.

    THE ORDER OF ``MatchSetup.spawns`` DOES NOT MATTER. An engine spawns the specs
    in the canonical order ``spawn_order_key`` -- per team, own-frame y, then
    own-frame x, then card, then resolved hp -- so every per-team spawn ordinal
    (Rust ``team_seq``, MockEngine ``_Ent.seq``), which both engines use as the last
    word in tie-breaks, names the same own-frame unit for both seats. Specs with
    equal keys are identical units, so their relative order cannot be observed.
    The canonical order is not decoration. With list order as a hidden tie-break
    input, a rotation-mirrored setup whose Red specs are listed in reverse desyncs
    the two engines within a few ticks (stacked Knights of different hp).
    """

    team: int
    card_id: int
    x: int
    y: int
    hp: int = -1  # -1 = full; otherwise 1 <= hp <= the card's CardInfo.hitpoints


class MatchSetup(msgspec.Struct, frozen=True):
    """Everything needed to start a battle. StateMutators produce these.

    WHAT IS REFUSED, WHAT IS CLAMPED (``setup_violation`` is the rule; every engine
    raises ``ValueError`` with its reason and LEAVES THE RUNNING BATTLE UNTOUCHED):
      decks          exactly two lists of DECK_SIZE catalogue ids -- refused otherwise.
      shuffle        a ShuffleMode value (0, 1, 2) -- refused otherwise (Rust
                     "unknown shuffle mode 7").
      start_tick     0 <= start_tick <= 2**32 - 1 (the Rust core's u32) -- refused
                     otherwise; -1 is refused, not clamped.
      elixir_milli   None, or exactly two integers fitting i64 -- a list of any other
                     length is refused (``[]`` is NOT "no override"). Each value is
                     CLAMPED to [0, MAX_MANA * 1000]: -1 starts at 0, 99999 at the cap.
      tower_hp       None, or [team][3] integers fitting i32 -- refused otherwise. A
                     princess hp <= 0 starts destroyed; a king hp <= 0 is refused. A
                     positive hp above the tower's max is accepted by both engines.
      spawns         ``spawn_violation`` per spec; list order irrelevant (SpawnSpec).
    Measured against the Rust core. ``RustEngine.reset`` runs ``validate_setup``
    before the core or the adapter touches anything, so an out-of-range integer never
    reaches PyO3: both engines refuse with this rule's ValueError and text
    (tests/test_parity_hardening.py, with an adapter plant).
    ``crowns_from_destroyed_towers=False`` is a documented RustEngine
    NotImplementedError, not a rule. When one setup breaks several rules the FIRST
    reason named may differ between engines; whether it is refused does not.
    """

    decks: list[list[int]]  # [blue 8 ids, red 8 ids]
    shuffle: int = ShuffleMode.INDEPENDENT
    start_tick: int = 0  # start mid-game on the clock
    elixir_milli: list[int] | None = None  # override starting elixir per team
    tower_hp: list[list[int]] | None = None  # [team][TowerSlot]; 0 = start destroyed
    spawns: list[SpawnSpec] = []
    crowns_from_destroyed_towers: bool = True


# --------------------------------------------------------------------------
# Arena geometry (integers only, derived from data/derived/arena.json)
# --------------------------------------------------------------------------

# Bits of the shipped tilemap. Duplicated from ../RoyaleSim/tools/extract_arena.py on
# purpose:
# Arena.load() asserts they match the "bits" block written into arena.json, so a
# drift between the two fails loudly instead of silently misreading water.
BIT_LANE_LEFT = 1
BIT_LANE_RIGHT = 2
BIT_NO_DEPLOY = 16
BIT_WATER = 32


class Arena(msgspec.Struct, frozen=True):
    subtile: int  # subtiles per tile
    half: int  # half-cells per tile (2)
    tiles_x: int
    tiles_y: int
    grid: list[list[int]]  # [hy][hx] bitmask, hy in [0, tiles_y*half)
    water_half_rows: tuple[int, int]  # inclusive
    bridges_half_cols: list[tuple[int, int]]  # inclusive, sorted by x
    king_centers: list[tuple[int, int]]  # [team] engine-frame subtiles
    princess_centers: list[list[tuple[int, int]]]  # [team][slot-1] engine frame
    princess_center_status: str

    @property
    def width(self) -> int:
        return self.tiles_x * self.subtile

    @property
    def height(self) -> int:
        return self.tiles_y * self.subtile

    @property
    def half_size(self) -> int:
        return self.subtile // self.half

    @property
    def hx(self) -> int:
        return self.tiles_x * self.half

    @property
    def hy(self) -> int:
        return self.tiles_y * self.half

    def bridge_centers_x(self) -> list[int]:
        h = self.half_size
        return [(a + b + 1) * h // 2 for a, b in self.bridges_half_cols]

    @classmethod
    def load(cls, calibration: Calibration, path: Path | None = None) -> Arena:
        p = path or data_dir() / "derived" / "arena.json"
        # THE FIRST file every entry point touches, so this is the message a fresh
        # clone actually gets -- ahead of the cards.json one, which nobody reached.
        require_data_file(p, "tools/extract_arena.py")
        a = json.loads(p.read_text(encoding="utf-8"))
        expected_bits = {
            "LANE_LEFT": BIT_LANE_LEFT,
            "LANE_RIGHT": BIT_LANE_RIGHT,
            "NO_DEPLOY": BIT_NO_DEPLOY,
            "WATER": BIT_WATER,
        }
        if a["bits"] != expected_bits:
            raise ValueError(f"arena.json bits {a['bits']} != protocol bits {expected_bits}")
        subtile = calibration.int("representation.SUBTILE_PER_TILE")
        half = int(a["half_tiles_per_tile"])
        if subtile % (2 * half) != 0:
            # Half-cell centres must be integer subtiles, or the half-tile action
            # parser would need rounding.
            raise ValueError("SUBTILE_PER_TILE must be divisible by 2*half")
        tiles_x, tiles_y = (int(v) for v in a["tiles"])
        hs = subtile // half
        kings = sorted(a["king_blocks"], key=lambda k: k["half_rows"][0])
        if len(kings) != 2:
            raise ValueError("arena.json must have exactly two king blocks")
        king_centers = [
            (
                (k["half_cols"][0] + k["half_cols"][1] + 1) * hs // 2,
                (k["half_rows"][0] + k["half_rows"][1] + 1) * hs // 2,
            )
            for k in kings
        ]
        bridges = sorted((int(b["half_cols"][0]), int(b["half_cols"][1])) for b in a["bridges"])
        width, height = tiles_x * subtile, tiles_y * subtile
        bridge_x = [(lo + hi + 1) * hs // 2 for lo, hi in bridges]
        if PRINCESS_CENTRES_KEY in a:
            blue, red = princess_centres_from_arena(a[PRINCESS_CENTRES_KEY], width, height)
            princess_status = f"arena.json {PRINCESS_CENTRES_KEY}"
        else:
            # FALLBACK until arena.json carries the centres. The static map has no
            # princess towers (calibration.json: "Princess towers are NOT in the static
            # map at all"), so x is taken as the bridge centre line (a hypothesis: they
            # visibly line up in the client) and y as the community-quoted 6.5 tiles
            # from the back edge, a GUESS. RustEngine checks these against the engine's
            # own tower positions and refuses to start on any difference.
            princess_y_own = MOCK_PRINCESS_CENTER_Y_HALF_CELLS * hs
            left_x, right_x = bridge_x[0], bridge_x[-1]
            blue = [(left_x, princess_y_own), (right_x, princess_y_own)]
            # Red's own-left is the engine-right tower, by the 180-degree rotation.
            red = [
                (width - left_x, height - princess_y_own),
                (width - right_x, height - princess_y_own),
            ]
            princess_status = "guess: x = bridge centre (hypothesis), y = 6.5 tiles (community)"
        return cls(
            subtile=subtile,
            half=half,
            tiles_x=tiles_x,
            tiles_y=tiles_y,
            grid=[list(map(int, row)) for row in a["grid"]],
            water_half_rows=(int(a["water_half_rows"][0]), int(a["water_half_rows"][1])),
            bridges_half_cols=bridges,
            king_centers=king_centers,
            princess_centers=[blue, red],
            princess_center_status=princess_status,
        )


# 6.5 tiles expressed in half-cells so no float is needed. The fallback in Arena.load,
# used only while arena.json has no PRINCESS_CENTRES_KEY.
MOCK_PRINCESS_CENTER_Y_HALF_CELLS = 13

# arena.json's princess tower centres, when the file carries them:
#
#     "princess_tower_centres": [[[x, y], [x, y]], [[x, y], [x, y]]]
#
# indexed [team][tower], team 0 = Blue (defends low y) and 1 = Red. Each centre is an
# integer [x, y] in engine-frame SUBTILES, the frame and unit of
# ``royalesim.Battle.tower_positions()``. The order of a team's two towers is free:
# they are named by their own-frame x, so Red's own-left is found, not assumed. The
# engine's own order (low engine x first, as ``tower_positions()[team][1:]``) is the
# natural one to write.
PRINCESS_CENTRES_KEY = "princess_tower_centres"


def princess_centres_from_arena(
    raw: Any, width: int, height: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """(Blue, Red) princess centres, each [own-left, own-right], from arena.json.

    Refuses anything that is not two teams of two integer points strictly inside the
    arena and on the owner's half, because a centre that is quietly wrong moves every
    tower the mask and both engines draw.
    """
    where = f"arena.json {PRINCESS_CENTRES_KEY}"
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError(f"{where} must hold two teams, [Blue, Red]: {raw!r}")
    named = []
    for team, towers in enumerate(raw):
        if not isinstance(towers, list) or len(towers) != 2:
            raise ValueError(f"{where}[{team}] must hold two towers: {towers!r}")
        points = []
        for point in towers:
            if (
                not isinstance(point, list)
                or len(point) != 2
                or not all(isinstance(v, int) and not isinstance(v, bool) for v in point)
            ):
                raise ValueError(f"{where}[{team}]: {point!r} is not an integer [x, y]")
            x, y = point
            own_half = 2 * y < height if team == BLUE else 2 * y > height
            if not (0 < x < width and 0 < y < height and own_half):
                raise ValueError(
                    f"{where}[{team}]: {point!r} is not inside the {width} x {height} "
                    f"arena on team {team}'s own half"
                )
            points.append((x, y))
        # Own-frame x is x for Blue and W - x for Red (the 180-degree rotation).
        own_x = [x if team == BLUE else width - x for x, _ in points]
        if own_x[0] == own_x[1]:
            raise ValueError(f"{where}[{team}]: both towers share x, so neither is the left")
        named.append(points if own_x[0] < own_x[1] else points[::-1])
    return named[0], named[1]


TERRITORY_MODELS = ("enemy_tower_no_deploy_rects",)
#: What an engine does with a building tap whose point is legal but whose box does not fit.
ILLEGAL_BUILDING_TAP = ("refuse", "relocate_first_fitting_ring")
KING_TOWER_NAME = "KingTower"
PRINCESS_TOWER_NAME = "PrincessTower"


def load_tower_no_deploy_sizes(path: Path | None = None) -> dict[str, tuple[int, int]]:
    """``towers[].no_deploy_size_tiles`` from data/derived/cards.json, by tower name.

    Reads ONE field of the derived card data, not the cards (MockEngine reads its
    stats from the raw CSVs; the Rust engine owns the card loader, card.rs). The
    numbers are never typed into Python: a regenerated cards.json moves the mask,
    MockEngine and the Rust engine together, with no rebuild, because the engine
    reads the file each time one is constructed. It reads the copy in the RoyaleSim
    checkout it was built in, which ``ROYALESIM_DATA_DIR`` does not move, so
    RustEngine refuses to start if the engine's rects differ from these
    (the family's data gate: stop copying constants -- parse them).
    """
    p = path or data_dir() / "derived" / "cards.json"
    # --vintage 2018 is not optional on a public clone: without it the extractor
    # wants a client asset pack that is not redistributed, and fails.
    require_data_file(
        p,
        "tools/extract_cards.py --vintage 2018",
        "tools/extract_cards.py --vintage 2018 --out data/derived/cards.json",
    )
    raw = json.loads(p.read_text(encoding="utf-8"))
    out: dict[str, tuple[int, int]] = {}
    for t in raw.get("towers", []):
        size = t.get("no_deploy_size_tiles")
        if size is not None:
            w, h = (int(v) for v in size)
            out[str(t["name"])] = (w, h)
    for name in (KING_TOWER_NAME, PRINCESS_TOWER_NAME):
        if name not in out:
            raise KeyError(
                f"cards.json tower {name!r} has no no_deploy_size_tiles; troop territory cannot "
                f"be decided. Regenerate it: ../RoyaleSim/tools/extract_cards.py --vintage 2018 "
                f"--out data/derived/cards.json"
            )
    return out


# FNV-1a 64: offset basis and prime.
FNV1A64_OFFSET = 0xCBF29CE484222325
FNV1A64_PRIME = 0x100000001B3


def fnv1a64(data: bytes) -> str:
    """FNV-1a 64 of ``data`` as 16 lowercase hex digits.

    The hash RoyaleSim stamps a card table with (``cards_json_fnv1a64`` in its replay
    fixtures, tools/make_replay_fixture.py), so a stamp taken here compares with one
    taken there as a plain string. Pure Python, about 0.3 s per MB, so callers hash a
    file once and keep the answer.
    """
    h = FNV1A64_OFFSET
    prime = FNV1A64_PRIME
    for byte in data:
        h = ((h ^ byte) * prime) & 0xFFFF_FFFF_FFFF_FFFF
    return f"{h:016x}"


def derived_cards_vintage(path: Path | None = None) -> str:
    """``provenance.vintage`` of data/derived/cards.json, or "unknown".

    The card table is GENERATED from a client asset pack, and WHICH pack is a
    property of the machine rather than of the repository: only the oldest one is
    tracked, and the generator defaults to a newer pack when a checkout happens to
    have it. So two catalogues built by the same tools on two machines can differ,
    and this is the field that says which one is in front of you.
    """
    p = path or data_dir() / "derived" / "cards.json"
    if not p.exists():
        return "unknown"
    raw = json.loads(p.read_text(encoding="utf-8"))
    provenance = raw.get("provenance")
    if not isinstance(provenance, dict):
        return "unknown"
    return str(provenance.get("vintage", "unknown"))


Rect = tuple[int, int, int, int]  # closed (x0, y0, x1, y1), engine frame subtiles


class DeployRules(msgspec.Struct, frozen=True):
    """Placement rules the action mask and the engine MUST share.

    territory_model
        calibration.json arena.TERRITORY_MODEL. Only "enemy_tower_no_deploy_rects"
        is implemented; anything else raises, so a registry change cannot be
        silently ignored by the mask.
    king_no_deploy_size / princess_no_deploy_size
        Full (width, height) in SUBTILES of each crown tower's NoDeploySize
        rectangle, from data/derived/cards.json ``no_deploy_size_tiles`` (2018
        buildings.csv NoDeploySizeW/H, read as tiles: 4/4 arena landmarks exact,
        ../RoyaleSim/tools/check_data.py). THE TROOP RULE: a troop may not be placed at a point
        inside the CLOSED rect, centred on the tower, of any ALIVE ENEMY crown
        tower. With every enemy tower up that is the whole enemy side; when one
        enemy princess falls, what opens on that side is the band between the far
        river bank and the enemy king rect: 8 half-rows (4 tiles).
        This mechanic replaced a rule both engines had invented: a fixed pocket of
        12 half-rows past the far bank on the fallen tower's side. That rule had no
        source and was 2 tiles deeper than the shipped rects allow. The shipped
        mechanic is used as it stands rather than the pocket re-tuned to match it.
    river_band_closed_to_troops (territory_status)
        UNSOURCED, carried from the old rule in both engines: no troop on any
        half-row of the river band, even a dry bridge cell whose lane's princess
        has fallen (the rects alone would open it). calibration.json
        arena.TERRITORY_MODEL open_question; a recording settles it.
    Buildings
        Own half only, even after a princess falls. UNSOURCED, unchanged.
    footprint_model
        calibration.json collision.BUILDING_FOOTPRINT_MODEL. Only
        "collision_radius_circle" is implemented; anything else raises so a
        calibration change cannot be silently ignored by the mask.
    """

    territory_model: str
    territory_status: str
    king_no_deploy_size: tuple[int, int]
    princess_no_deploy_size: tuple[int, int]
    footprint_model: str
    # What the engine does with a building tap whose POINT is legal but whose tile box
    # does not fit: "relocate_first_fitting_ring" moves it to the nearest place it
    # fits, so nothing on the board can make such a tap illegal, and "refuse" turns it
    # down. The mask asks this rather than assuming, because the two answers give a
    # building card different legal cells on the same board. Trailing and defaulted:
    # an engine that states nothing is the pre-relocation one.
    illegal_building_tap: str = "refuse"
    # Ticks from the start of a match during which the engine refuses EVERY deploy
    # (`match.DEPLOY_LOCKOUT_TICKS`, `DeployStatus.TOO_EARLY`). It is here because this
    # struct is the rules the mask and the engine must share, and a lockout the mask does
    # not know about is precisely a rule they do not share: the mask offers every card,
    # the engine refuses all of them, and a policy is penalised for obeying its own mask
    # on the opening steps of every battle.
    #
    # Trailing and defaulted to 0, which is BOTH "an engine that states nothing" and a
    # real arm -- calibration keeps 0 runnable as the behaviour this engine used to have,
    # so the value is read rather than assumed and 0 disables the rule honestly.
    deploy_lockout_ticks: int = 0

    @classmethod
    def load(cls, calibration: Calibration, cards_path: Path | None = None) -> DeployRules:
        model = str(calibration.value("collision.BUILDING_FOOTPRINT_MODEL"))
        if model != "collision_radius_circle":
            raise NotImplementedError(
                f"BUILDING_FOOTPRINT_MODEL={model!r}: only collision_radius_circle is implemented"
            )
        territory = str(calibration.value("arena.TERRITORY_MODEL"))
        if territory not in TERRITORY_MODELS:
            raise NotImplementedError(
                f"TERRITORY_MODEL={territory!r}: only {TERRITORY_MODELS} is implemented"
            )
        try:
            illegal_tap = str(calibration.value("placement.ILLEGAL_TAP"))
        except KeyError:
            illegal_tap = "refuse"
        if illegal_tap not in ILLEGAL_BUILDING_TAP:
            raise NotImplementedError(
                f"ILLEGAL_TAP={illegal_tap!r}: only {sorted(ILLEGAL_BUILDING_TAP)} are implemented"
            )
        try:
            lockout = int(calibration.value("match.DEPLOY_LOCKOUT_TICKS"))
        except KeyError:
            lockout = 0
        if lockout < 0:
            raise ValueError(f"DEPLOY_LOCKOUT_TICKS={lockout}: a lockout cannot be negative")
        subtile = calibration.int("representation.SUBTILE_PER_TILE")
        sizes = load_tower_no_deploy_sizes(cards_path)

        def in_subtiles(name: str) -> tuple[int, int]:
            w, h = sizes[name]
            if (w * subtile) % 2 or (h * subtile) % 2:
                raise ValueError(f"{name} NoDeploySize half-extent is not an integer subtile")
            return w * subtile, h * subtile

        return cls(
            territory_model=territory,
            territory_status=(
                f"{calibration.status('arena.TERRITORY_MODEL')}: rects from cards.json "
                "no_deploy_size_tiles; river band closed to troops is a guess"
            ),
            king_no_deploy_size=in_subtiles(KING_TOWER_NAME),
            princess_no_deploy_size=in_subtiles(PRINCESS_TOWER_NAME),
            footprint_model=model,
            illegal_building_tap=illegal_tap,
            deploy_lockout_ticks=lockout,
        )

    def no_deploy_rect(self, slot: int, cx: int, cy: int) -> Rect:
        """The closed NoDeploySize rect of a crown tower in ``slot`` centred at (cx, cy)."""
        w, h = self.king_no_deploy_size if slot == TowerSlot.KING else self.princess_no_deploy_size
        return cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2

    def tower_rects(self, arena: Arena, owner: int) -> list[Rect]:
        """[TowerSlot] -> the rect of ``owner``'s tower at its ARENA centre, alive or not."""
        rects = [self.no_deploy_rect(TowerSlot.KING, *arena.king_centers[owner])]
        for slot in (TowerSlot.LEFT, TowerSlot.RIGHT):
            rects.append(self.no_deploy_rect(slot, *arena.princess_centers[owner][slot - 1]))
        return rects


def rect_contains(r: Rect, x: int, y: int) -> bool:
    return r[0] <= x <= r[2] and r[1] <= y <= r[3]


# --------------------------------------------------------------------------
# MatchSetup spawn rule (every engine refuses the same specs, for the same reason)
# --------------------------------------------------------------------------


def touched_half_cells(arena: Arena, x: int, y: int) -> list[tuple[int, int]]:
    """(hx, hy) of every half-cell a point touches: closed cells, clamped to the grid.

    A point on a half-cell boundary touches both neighbours. This is the Rust
    engine's ``Arena::touching_bits`` (axis_span clamps to [0, n-1]), which is
    what makes a centre on the exact river bank line count as touching water.
    """
    h = arena.half_size

    def span(v: int, n: int) -> list[int]:
        i = v // h
        lo = i - 1 if v % h == 0 else i
        return list(range(max(lo, 0), min(i, n - 1) + 1))

    return [(hx, hy) for hy in span(y, arena.hy) for hx in span(x, arena.hx)]


def spawn_violation(arena: Arena, cards: Sequence[CardInfo], spec: SpawnSpec) -> str | None:
    """Why ``spec`` may not be placed at reset, or None. THE rule for every engine.

    ORDER AND MEANING (the Rust engine's ``scenario_spawn_now``, state.rs, is the
    authority; this states it so MockEngine and RustEngine refuse identically):
      1. team is BLUE or RED;
      2. card_id is in the catalogue, and is a TROOP or BUILDING card (spells and
         rolling spells are not units);
      3. out of arena: the centre outside the CLOSED arena rectangle
         [0, W] x [0, H] (Rust ``Arena::in_bounds``; deploys use the open interior,
         spawns the closed one -- the engine's choice, recorded not argued);
      4. on water: a non-flying unit (ground troop OR building) whose centre
         touches any WATER half-cell, closed cells (``touched_half_cells``). So
         the exact bank line y = 15 tiles is water off the bridges, and a bridge
         edge x = 2.5 tiles inside the river band is water;
      5. hp: -1 (full) or 1 <= hp <= the card's ``CardInfo.hitpoints`` (the per-unit
         max at the level the engine runs cards at). The Rust core itself accepts
         any hp >= 0; the adapter enforces this before the core sees it.
    Flying units may spawn over water.
    """
    if spec.team not in TEAMS:
        return f"bad spawn team {spec.team}"
    if not 0 <= spec.card_id < len(cards):
        return f"unknown card id {spec.card_id}"
    card = cards[spec.card_id]
    if card.placement not in (Placement.TROOP, Placement.BUILDING):
        return f"spawns must be unit or building cards, {card.name} is not"
    if not (0 <= spec.x <= arena.width and 0 <= spec.y <= arena.height):
        return f"spawn {card.name} at ({spec.x}, {spec.y}): out of arena"
    if not card.flying and any(
        arena.grid[hy][hx] & BIT_WATER for hx, hy in touched_half_cells(arena, spec.x, spec.y)
    ):
        return f"spawn {card.name} at ({spec.x}, {spec.y}): on water"
    if spec.hp != -1 and not 1 <= spec.hp <= card.hitpoints:
        return f"spawn {card.name} hp {spec.hp} outside -1 or [1, {card.hitpoints}]"
    return None


def validate_spawns(arena: Arena, cards: Sequence[CardInfo], spawns: Sequence[SpawnSpec]) -> None:
    """Raise ValueError for the first spec ``spawn_violation`` refuses."""
    for sp in spawns:
        why = spawn_violation(arena, cards, sp)
        if why is not None:
            raise ValueError(why)


def spawn_order_key(
    arena: Arena, cards: Sequence[CardInfo], spec: SpawnSpec
) -> tuple[int, int, int, str, int]:
    """Canonical spawn order of a valid ``MatchSetup`` spec (see ``SpawnSpec``).

    ``(team, own-frame y, own-frame x, card NAME, resolved hp)`` -- exactly the Rust
    core's key (state.rs ``scenario_spawn_batch``: ``(team, to_frame y, to_frame x,
    name, hp)``). Team first, so the order -- and with it every uid and state_hash --
    is invariant under ANY permutation of the list, not only within a team.

    THE CARD IS COMPARED BY NAME, NOT BY CATALOGUE ID, and the two are not
    interchangeable. Any total order on cards keeps ONE engine list-order-free and
    seat-symmetric, but two engines with different orders give same-team specs on one
    own-frame point with different cards different ``team_seq``, and so different
    tie-breaks. The Rust core sorts by card NAME, so this does too, and the two agree
    exactly: Python ``str`` ordering (code points) equals Rust ``String`` ordering
    (UTF-8 bytes) for every string. tests/test_parity_hardening.py holds the engines
    together with a catalogue whose id order is the reverse of its name order.
    """
    ox, oy = to_own(arena, spec.team, spec.x, spec.y)
    card = cards[spec.card_id]
    hp = card.hitpoints if spec.hp == -1 else spec.hp
    return spec.team, oy, ox, card.name, hp


I32_RANGE = (-(2**31), 2**31 - 1)
I64_RANGE = (-(2**63), 2**63 - 1)
U32_MAX = 2**32 - 1


def setup_violation(arena: Arena, cards: Sequence[CardInfo], setup: MatchSetup) -> str | None:
    """Why ``setup`` may not start a battle, or None. See ``MatchSetup`` for the table.

    Reads nothing but its arguments, so an engine that calls it BEFORE touching its
    state cannot be left half-reset by a refusal. Messages for the core-owned rules
    are the Rust core's own (py.rs ``Battle.reset``, state.rs
    ``scenario_set_tower_hp``), so one reason reads the same from either engine.
    """
    if len(setup.decks) != 2 or any(len(d) != DECK_SIZE for d in setup.decks):
        return "setup.decks must be two lists of DECK_SIZE card ids"
    for d in setup.decks:
        for cid in d:
            if not 0 <= cid < len(cards):
                return f"unknown card id {cid}"
    if setup.shuffle not in tuple(ShuffleMode):
        return f"unknown shuffle mode {setup.shuffle}"
    if not 0 <= setup.start_tick <= U32_MAX:
        return f"start_tick {setup.start_tick} outside [0, {U32_MAX}]"
    if setup.elixir_milli is not None:
        if len(setup.elixir_milli) != 2:
            return "elixir_milli must have two entries"
        for e in setup.elixir_milli:
            if not I64_RANGE[0] <= e <= I64_RANGE[1]:
                return f"elixir_milli {e} does not fit i64"
    if setup.tower_hp is not None:
        if len(setup.tower_hp) != 2 or any(len(row) != len(TowerSlot) for row in setup.tower_hp):
            return "tower_hp must be [team][3]"
        for row in setup.tower_hp:
            for hp in row:
                if not I32_RANGE[0] <= hp <= I32_RANGE[1]:
                    return f"tower hp {hp} does not fit i32"
            if row[TowerSlot.KING] <= 0:
                return "a battle cannot start with a destroyed king tower"
    for sp in setup.spawns:
        why = spawn_violation(arena, cards, sp)
        if why is not None:
            return why
    return None


def validate_setup(arena: Arena, cards: Sequence[CardInfo], setup: MatchSetup) -> None:
    """Raise ValueError with ``setup_violation``'s reason. Call before any state changes."""
    why = setup_violation(arena, cards, setup)
    if why is not None:
        raise ValueError(why)


# --------------------------------------------------------------------------
# The protocol
# --------------------------------------------------------------------------


@runtime_checkable
class Engine(Protocol):
    """A deterministic two-player battle engine.

    Contract, method by method:

    cards()        Static card catalogue. ``cards()[i].card_id == i``. Never
                   changes after construction.
    arena()        Static arena geometry in integer subtiles.
    rules()        The placement rules shared with the action mask.
    reset(seed, setup)
                   Start a new battle. Same (seed, setup) must give a bit-identical
                   battle. The engine owns its RNG; there is no global randomness.
                   A setup ``setup_violation`` refuses (spawns included) raises
                   ValueError with that reason and leaves the previous battle
                   untouched -- validate everything, then mutate. Spawn list order
                   does not matter (``SpawnSpec``).
    check_deploy(command) -> int
                   PURE query: the DeployStatus ``step`` would give this command
                   if it were the only command this step. Must not mutate state.
    step(commands, ticks) -> list[DeployResult]
                   All commands are validated against the CURRENT state (the one
                   ``state()`` returns), so both teams' commands are simultaneous.
                   At most one command per team; a second gets DUPLICATE_TEAM
                   (decided in input order). Accepted commands are PAID at once --
                   elixir spent, hand cycled, even when ``ticks == 0`` -- in canonical
                   (team, hand slot) order, so the resulting state and state_hash do
                   not depend on the list order; their units/spells materialise in
                   the first tick, whose elixir regeneration lands on the reduced
                   bar (the Rust engine's order, py.rs apply_commands; unsourced for
                   the live game). Then the engine advances ``ticks`` ticks
                   (stopping early if the game ends). Returns one result per
                   command, in input order.
    state() -> BattleState
                   Snapshot of the current state. Must be a pure function of the
                   battle so far (no wall clock, no object identity). ``spells`` and
                   the entity status timers are what the engine models: an engine
                   whose spells resolve within a tick reports no spell objects, one
                   without status effects reports 0 timers (MockEngine, both).
    save_state() -> bytes / load_state(blob)
                   Exact round trip, including RNG state: load then step must equal
                   having never saved. Used for curriculum starts and search.
                   ``load_state`` raises ValueError, leaving the running battle
                   untouched, when a hand, the cycle queue, a pending deploy or a
                   non-tower board entity names a card this engine's catalogue does
                   not have. Cards are matched by NAME, so a snapshot loads into a
                   catalogue that is a permuted superset of the one it was saved
                   from and plays the same cards. (Rust ``catalogue_violation``.
                   Without the check, a snapshot read through another catalogue
                   plays a Knight as a Valkyrie.)
    state_hash() -> int
                   64-bit digest of the full internal state (RNG included), stable
                   across processes. Replays check it periodically.
    """

    def cards(self) -> Sequence[CardInfo]: ...

    def arena(self) -> Arena: ...

    def rules(self) -> DeployRules: ...

    def reset(self, seed: int, setup: MatchSetup) -> None: ...

    def check_deploy(self, command: DeployCommand) -> int: ...

    def step(self, commands: Sequence[DeployCommand], ticks: int) -> list[DeployResult]: ...

    def state(self) -> BattleState: ...

    def save_state(self) -> bytes: ...

    def load_state(self, blob: bytes) -> None: ...

    def state_hash(self) -> int: ...


# --------------------------------------------------------------------------
# Frame helpers shared by obs / action / reward so they cannot disagree
# --------------------------------------------------------------------------


def to_own(arena: Arena, team: int, x: int, y: int) -> tuple[int, int]:
    """Engine frame -> team's own frame (an involution)."""
    if team == BLUE:
        return x, y
    return arena.width - x, arena.height - y


def to_engine(arena: Arena, team: int, x_own: int, y_own: int) -> tuple[int, int]:
    return to_own(arena, team, x_own, y_own)


def mirror_state(arena: Arena, s: BattleState) -> BattleState:
    """The same battle with the teams' seats swapped.

    Positions rotate 180 degrees and team ids swap. Tower slots are already
    own-frame, so they are unchanged. Obs/mask for Red on the mirror must equal
    obs/mask for Blue on the original, bit for bit.
    """
    w, h = arena.width, arena.height
    ents = [
        msgspec.structs.replace(
            e,
            team=1 - e.team,
            x=w - e.x,
            y=h - e.y,
            # A closed box rotates corner to corner: the far corner becomes the near one.
            footprint=(
                None
                if e.footprint is None
                else (
                    w - e.footprint[2],
                    h - e.footprint[3],
                    w - e.footprint[0],
                    h - e.footprint[1],
                )
            ),
        )
        for e in s.entities
    ]
    # Spell centres and aim points rotate like positions; delay, roll progress and
    # hit counts are frame-free.
    spells = [
        msgspec.structs.replace(
            sp,
            team=1 - sp.team,
            x=arena.width - sp.x,
            y=arena.height - sp.y,
            aim_x=arena.width - sp.aim_x,
            aim_y=arena.height - sp.aim_y,
        )
        for sp in s.spells
    ]
    players = [
        msgspec.structs.replace(s.players[RED], team=BLUE),
        msgspec.structs.replace(s.players[BLUE], team=RED),
    ]
    winner = s.winner
    if winner in (Winner.BLUE, Winner.RED):
        winner = 1 - winner
    return msgspec.structs.replace(s, entities=ents, players=players, winner=winner, spells=spells)


@lru_cache(maxsize=1)
def default_calibration() -> Calibration:
    return Calibration.load()


def calibration_values(raw: dict[str, Any]) -> dict[str, Any]:
    """Every ``section.KEY -> value`` in a calibration document (prose excluded)."""
    out: dict[str, Any] = {}
    for section, entries in raw.items():
        if not isinstance(entries, dict):
            continue
        for key, entry in entries.items():
            if isinstance(entry, dict) and "value" in entry:
                out[f"{section}.{key}"] = entry["value"]
    return out


def calibration_digest(calibration: Calibration | None = None) -> str:
    """A short hash of every calibration VALUE, for a checkpoint to pin.

    Values only, so reworded prose or a changed ``status`` does not invalidate a
    checkpoint while a changed constant does. ``rust_engine.build_digest`` is the
    same hash over the data the compiled engine was built with, so a checkpoint
    that carries both says whether the two agreed when it was written.
    """
    cal = calibration if calibration is not None else default_calibration()
    blob = json.dumps(calibration_values(cal.raw), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# The elixir bar's arithmetic, in one place
# --------------------------------------------------------------------------


class ElixirLaw(msgspec.Struct, frozen=True):
    """How an elixir bar fills, as integers, from calibration.json and globals.csv.

    WHY IT IS HERE AND NOT IN THE ENGINE
        ``obs.py`` counts the opponent's elixir the way a player does -- from the
        plays it sees and the regeneration rate everyone knows -- and the count is
        only worth having if it is EXACT against the bar the engine keeps. Two
        copies of the law would be two chances to drift, so this is the one copy;
        the engines derive their own from the same calibration keys and
        tests/test_env_obs.py holds this against a played-out MockEngine battle,
        tick by tick, including the seeding round trip.

    THE FINE UNIT
        Milli-elixir cannot carry the law: at TICK_MS 50 and MANA_REGEN_MS_1X
        28000 a tick is worth 17.857... milli, so a milli-space sum drifts within
        one match. One elixir is ``scale`` = lcm(regen 1x, regen 2x) fine units
        instead, which makes the per-tick gain an integer under both rates
        (checked at load). ``elixir_milli`` on the wire is ``fine * 1000 //
        scale``, a floor, so ``from_milli`` inverts it exactly only while
        ``scale`` is a multiple of 1000 (``seed_is_exact``).

    THE RATE
        1x below ``regular_ticks - speedup_ticks``, 2x from there on, where the
        threshold is globals.csv MANA_SPEED_UP_WHEN_REMAINING_SECONDS (not in
        calibration.json yet; MockEngine reads the same row). Overtime is always
        past the threshold under the shipped numbers, so the flag never decides
        the rate on its own; it is taken as 2x anyway rather than relying on that.
    """

    scale: int  # fine units per elixir
    gain_1x: int  # fine units gained per tick at 1x
    gain_2x: int
    cap_fine: int  # MAX_MANA in fine units
    tick_ms: int
    speedup_ticks: int  # ticks before the end of regulation at which 2x starts

    @classmethod
    def load(cls, calibration: Calibration | None = None) -> ElixirLaw:
        cal = calibration if calibration is not None else default_calibration()
        r1 = cal.int("match.MANA_REGEN_MS_1X")
        r2 = cal.int("match.MANA_REGEN_MS_2X")
        tick_ms = cal.int("time.TICK_MS")
        max_mana = cal.int("match.MAX_MANA")
        scale = r1 * r2 // gcd(r1, r2)
        gains = {}
        for rate, regen in ((1, r1), (2, r2)):
            per_tick = tick_ms * max_mana * scale // regen
            if per_tick * regen != tick_ms * max_mana * scale:
                raise ValueError(f"elixir gain per tick at {rate}x is not exact")
            gains[rate] = per_tick
        speedup_s = load_globals_csv().get("MANA_SPEED_UP_WHEN_REMAINING_SECONDS", (None, None))[0]
        if speedup_s is None:
            raise KeyError("globals.csv lacks MANA_SPEED_UP_WHEN_REMAINING_SECONDS")
        return cls(
            scale=scale,
            gain_1x=gains[1],
            gain_2x=gains[2],
            cap_fine=max_mana * scale,
            tick_ms=tick_ms,
            speedup_ticks=-(-speedup_s * 1000 // tick_ms),
        )

    @property
    def seed_is_exact(self) -> bool:
        """Whether ``to_milli(from_milli(m)) == m`` for every reachable m."""
        return self.scale % 1000 == 0

    def to_milli(self, fine: int) -> int:
        return fine * 1000 // self.scale

    def from_milli(self, milli: int) -> int:
        """The engine's own seeding: ``elixir_milli`` back into fine units."""
        return milli * self.scale // 1000

    def seed_fine(self, milli: int) -> int:
        """A reported bar back into fine units, snapped onto the reachable lattice.

        ``from_milli`` is a floor, and the floor is only exact for a bar the engine
        seeded from that same milli value. A bar that has been RUNNING is a start
        plus gains minus spends, and every one of those is a multiple of
        ``gcd(gain_1x, gain_2x)``, so it sits on a lattice with that spacing. The
        milli window for one reported value is ``scale / 1000`` fine units wide --
        28 at the shipped numbers, against a spacing of 500 -- so the window holds
        AT MOST ONE lattice point, and the snap is unambiguous.

        Both cases are therefore exact: a running bar snaps back to itself, and a
        bar seeded off the lattice by a MatchSetup (an odd ``elixir_milli``) has no
        lattice point in its window and keeps the floor, which is what the engine
        used. Without this, resuming from a Snapshot mid-battle starts the count up
        to ``scale / 1000 - 1`` fine units low and it never recovers.
        """
        lo = self.from_milli(milli)
        hi = lo + self.scale // 1000 - 1
        spacing = gcd(self.gain_1x, self.gain_2x)
        snapped = -(-lo // spacing) * spacing
        return snapped if snapped <= hi else lo

    def rate_at(self, tick: int, regular_ticks: int, overtime: bool = False) -> int:
        """The multiplier in force during the tick that runs FROM ``tick``."""
        if overtime or tick >= regular_ticks - self.speedup_ticks:
            return 2
        return 1

    def gain(self, rate: int) -> int:
        if rate == 1:
            return self.gain_1x
        if rate == 2:
            return self.gain_2x
        raise ValueError(f"unknown elixir rate {rate}")

    def regen(self, tick_from: int, tick_to: int, regular_ticks: int, overtime: bool) -> int:
        """Fine units a bar gains over ticks [tick_from, tick_to), ignoring the cap.

        The rate is a step function of the tick, so the interval splits in two at
        the speed-up threshold instead of being walked tick by tick.
        """
        if tick_to <= tick_from:
            return 0
        if overtime:
            return (tick_to - tick_from) * self.gain_2x
        threshold = regular_ticks - self.speedup_ticks
        slow = max(0, min(tick_to, threshold) - tick_from)
        fast = (tick_to - tick_from) - slow
        return slow * self.gain_1x + fast * self.gain_2x

    def advance(
        self,
        fine: int,
        spent_elixir: int,
        tick_from: int,
        tick_to: int,
        regular_ticks: int,
        overtime: bool,
    ) -> tuple[int, int]:
        """One decision step of a bar: (new fine value, fine units lost to the cap).

        The engine pays accepted commands BEFORE the first tick (``Engine.step``),
        and regeneration is positive, so clamping once at the end is the same bar
        as clamping every tick.
        """
        raw = fine - spent_elixir * self.scale
        raw = max(0, raw) + self.regen(tick_from, tick_to, regular_ticks, overtime)
        return min(self.cap_fine, raw), max(0, raw - self.cap_fine)


@lru_cache(maxsize=1)
def default_elixir_law() -> ElixirLaw:
    return ElixirLaw.load()

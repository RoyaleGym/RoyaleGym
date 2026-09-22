"""MockEngine: a pure-Python stand-in that satisfies ``protocol.Engine``.

WHAT IT IS FOR
    Building and testing the whole RL layer before the Rust engine exists. It is
    deliberately simple, but it is not a toy in the three ways that matter to the
    RL layer: it uses the REAL arena (water, bridges, no-deploy, king block) from
    data/derived/arena.json, it reads every physics constant it uses from
    calibration.json (or globals.csv when the registry does not have it yet), and
    it obeys the three invariants the real engine must obey.

WHAT IT IS NOT
    A fidelity claim. No unit collision, no projectile travel (ranged hits are
    instant), no stun/slow, spells hit once, splash is centred on the target, all
    cards and towers at CSV base level, overtime that ends level is decided by
    calibration match.OVERTIME_TIEBREAK (the same rule table the Rust engine
    reads; the rule itself is not yet measured on a recorded match). None of this
    should be used to argue about how the real game behaves.

    ITS SPELL EFFECTS ARE NOT THE ENGINE'S. The Rust core (../RoyaleSim/crates/royalesim
    spell.rs) flies Fireball / Arrows / Goblin Barrel to their target, lands and
    rolls The Log, stuns with Zap and knocks units back. Here every spell resolves
    in the tick it materialises: circle spells and the Log's strip hit once at the
    tap, no stun, no knockback, and the Goblin Barrel's Goblins appear at the tap at
    once. So ``BattleState.spells`` is always empty and every ``stun_ticks`` /
    ``knockback_ticks`` is 0. What IS held to the engine is the part the RL layer
    must agree on regardless of mechanics: which cards are spells, each spell's
    placement class, every deploy verdict and reason code (tests/test_rust_engine.py,
    three-way at every half-cell), the catalogue row shape (count/radius/hp 0), and
    that a released Goblin reports its barrel's card id. Do not read a spell trade
    off a MockEngine battle.

THE INVARIANTS, AS THEY APPLY HERE
    1. No floats: integer subtiles, integer elixir with an exact rational scale,
       truncation-toward-zero division (``_tdiv``) so that a Red unit's step is
       exactly the negation of the mirrored Blue unit's step. Python's ``//``
       floors, and floor(-a/b) != -floor(a/b): using it here breaks the mirror
       test, which is how that test was checked (see tests/test_env_mock_engine.py).
    2. Simultaneity: every phase reads start-of-phase state and writes into
       buffers (targets, proposed moves, damage, deaths, spawns) applied in one
       pass. Every tie-break is in the ACTING team's own frame, so nothing
       prefers Blue.
    3. No baked numbers: see ``MockEngine.__init__``.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Sequence
from fractions import Fraction
from math import isqrt
from pathlib import Path

import msgspec

from .protocol import (
    BIT_NO_DEPLOY,
    BIT_WATER,
    BLUE,
    DECK_SIZE,
    EMPTY_CARD,
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
    EntityKind,
    EntityState,
    MatchSetup,
    Placement,
    PlayerState,
    ShuffleMode,
    TowerSlot,
    Winner,
    calibration_digest,
    data_dir,
    default_calibration,
    load_globals_csv,
    spawn_order_key,
    validate_setup,
)

# calibration match.OVERTIME_TIEBREAK candidates (the Rust core's OvertimeTiebreak enum).
# The raw client pack MockEngine reads its card stats from. Only this one is tracked
# in RoyaleSim; the newer packs are not redistributed, so a public checkout has this
# and nothing else, and the derived cards.json the compiled engine is built from is
# generated from it too (rust_engine.catalogue_vintage_split).
RAW_CARD_PACK = "retroroyale-2018"

OVERTIME_TIEBREAK_RULES = ("lowest_tower_hp_absolute", "lowest_tower_hp_fraction", "none_draw")

# The card subset the mock supports. A mock design choice (a spread of placement
# types, air/ground, splash, building-targeters), not a physics constant.
MOCK_CARD_NAMES: tuple[str, ...] = (
    "Knight",
    "Archer",
    "Goblins",
    "Giant",
    "MiniPekka",
    "Musketeer",
    "Skeletons",
    "Minions",
    "HogRider",
    "Valkyrie",
    "Cannon",
    "Fireball",
    "Arrows",
    "Zap",
    "Log",
    "GoblinBarrel",  # appended after the first 15 (ids 0..14 unchanged): SPELL_NOT_ON_WATER
)

_M64 = (1 << 64) - 1
_M32 = (1 << 32) - 1


class Pcg32(msgspec.Struct):
    """Python port of royalesim::Rng (lib.rs). Same stream for the same seed.

    Cross-language equality has NOT been verified against the compiled crate; a
    parity test for it belongs here once royalesim exposes Rng.
    """

    state: int = 0
    inc: int = 1

    @classmethod
    def new(cls, seed: int) -> Pcg32:
        seed &= _M64
        r = cls(0, ((seed << 1) | 1) & _M64)
        r.next_u32()
        r.state = (r.state + seed) & _M64
        r.next_u32()
        return r

    def next_u32(self) -> int:
        old = self.state
        self.state = (old * 6364136223846793005 + self.inc) & _M64
        xorshifted = (((old >> 18) ^ old) >> 27) & _M32
        rot = old >> 59
        return ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & _M32

    def below(self, n: int) -> int:
        if n <= 0:
            return 0
        threshold = ((-n) & _M32) % n
        while True:
            r = self.next_u32()
            if r >= threshold:
                return r % n


def _tdiv(a: int, b: int) -> int:
    """Integer division truncating toward zero, like Rust's ``/`` on i64."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def _ceil_div(a: int, b: int) -> int:
    return -((-a) // b)


def _csv_table(
    path: Path, required: Sequence[str] = ()
) -> dict[str, dict[str, str]]:
    """One Supercell CSV as name -> row, refusing a table this loader cannot read.

    ``required`` names the columns the caller goes on to read. A column that is
    ABSENT from the header is a different thing from a cell that is empty: an empty
    cell is ordinary (most cards have no Projectile) and reads as the default, while
    a missing column means the file is not the schema this reader was written for,
    and every lookup into it would quietly return that same default.

    That failure has been seen. Pointed at a newer client's tables, this reader did
    not raise -- it returned radius 0 for every unit and flying False for Minions,
    a catalogue of plausible wrong numbers. Anything comparing MockEngine with
    another engine would then have been measuring the parser rather than the
    engines. Loud here, rather than subtle three layers up.
    """
    with path.open(encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise ValueError(f"{path} is empty")
    header = rows[0]
    missing = [c for c in required if c not in header]
    if missing:
        raise ValueError(
            f"{path} has no column(s) {missing}: this reader is written for the "
            f"{RAW_CARD_PACK} schema and every lookup into a column that is not there "
            f"would silently read as 0 or False. Found columns: {sorted(set(header))[:12]}"
        )
    out: dict[str, dict[str, str]] = {}
    for r in rows[2:]:
        if r and r[0].strip():
            out[r[0].strip()] = dict(zip(header, r, strict=False))
    return out


# The columns ``_load_cards`` reads out of each table. Not the whole schema -- only
# what this reader touches, so a column being added or dropped elsewhere is not an
# error here.
UNIT_COLUMNS = (
    "Name",
    "Hitpoints",
    "CollisionRadius",
    "Speed",
    "Range",
    "SightRange",
    "HitSpeed",
    "LoadTime",
    "Damage",
    "AreaDamageRadius",
    "AttacksAir",
    "AttacksGround",
    "TargetOnlyBuildings",
    "FlyingHeight",
    "DeployTime",
    "LifeTime",
)
SPELL_CARD_COLUMNS = ("Name", "ManaCost", "SummonCharacter")
OTHER_SPELL_COLUMNS = ("Name", "ManaCost", "SpellAsDeploy")


def _i(row: dict[str, str], key: str, default: int = 0) -> int:
    v = row.get(key, "").strip()
    return int(v) if v else default


def _b(row: dict[str, str], key: str) -> bool:
    return row.get(key, "").strip().lower() == "true"


# --------------------------------------------------------------------------
# Static templates
# --------------------------------------------------------------------------


class _UnitTpl(msgspec.Struct, frozen=True):
    name: str
    kind: int
    hp: int
    radius: int
    speed: int  # subtiles per tick
    range_: int
    sight: int
    hit_ticks: int
    load_ticks: int
    damage: int
    area_radius: int
    attacks_air: bool
    attacks_ground: bool
    buildings_only: bool
    flying: bool
    deploy_ticks: int
    lifetime_ticks: int  # 0 = unlimited


class _SpellTpl(msgspec.Struct, frozen=True):
    name: str
    damage: int
    radius: int  # circle radius, or half-width for rolling
    reach: int  # rolling distance forward (own frame); 0 for circle spells
    crown_pct: int  # percent of damage dealt to crown towers
    hits_air: bool
    hits_ground: bool
    spawn_count: int = 0  # units the spell releases at the tap (Goblin Barrel)
    spawn_deploy_ticks: int = 0  # their deploy timer


class _CardTpl(msgspec.Struct, frozen=True):
    info: CardInfo
    unit: int  # unit template: the card's unit, or the unit a spell releases; else -1
    spell: int  # index into spell templates, -1 for units


# --------------------------------------------------------------------------
# Dynamic state (this is exactly what save_state serialises)
# --------------------------------------------------------------------------


class _Ent(msgspec.Struct, array_like=True):
    uid: int
    seq: int  # per-team spawn counter; used for tie-breaks because it mirrors
    team: int
    tpl: int
    card_id: int
    tower_slot: int
    x: int
    y: int
    hp: int
    max_hp: int
    deploy: int
    cooldown: int
    target: int  # uid or -1
    life: int  # remaining ticks, -1 = unlimited


class _Sim(msgspec.Struct):
    tick: int
    elixir: list[int]  # scaled by MockEngine.elixir_scale
    hands: list[list[int]]
    queues: list[list[int]]
    crowns: list[int]
    king_active: list[bool]
    king_timer: list[int]
    ents: list[_Ent]
    next_uid: int
    next_seq: list[int]
    overtime: bool
    game_over: bool
    winner: int
    rng: Pcg32
    # Deploys paid for at step() and not yet materialised: [team, card_id, x, y],
    # in canonical (team, hand slot) order. Consumed by the next tick. Part of the
    # saved state so a save between step(cmds, 0) and the next tick round-trips.
    pending: list[list[int]] = []
    # The catalogue every card_id above is an index into, by NAME, in card-id order.
    # load_state matches cards by name through it (``_adopt_snapshot``). Empty only in
    # a blob saved before the catalogue was part of a snapshot, which load_state
    # refuses: its ids cannot be attributed to a card.
    catalogue: list[str] = []


class MockEngine:
    """Deterministic, seeded, integer-only. Satisfies ``protocol.Engine``."""

    def __init__(
        self,
        calibration: Calibration | None = None,
        arena_path: Path | None = None,
        card_names: Sequence[str] = MOCK_CARD_NAMES,
    ) -> None:
        cal = calibration or default_calibration()
        self.calibration = cal
        self._arena = Arena.load(cal, arena_path)
        self._rules = DeployRules.load(cal)
        g = load_globals_csv()

        # ---- every constant, and where it comes from --------------------
        self.tick_ms = cal.int("time.TICK_MS")
        self.speed_mult = cal.int("time.SPEED_TO_SUBTILES_PER_TICK")
        milli_per_tile = cal.int("representation.MILLITILE_PER_TILE")
        if self._arena.subtile % milli_per_tile:
            raise ValueError("SUBTILE_PER_TILE must be a multiple of MILLITILE_PER_TILE")
        self.sub_per_milli = self._arena.subtile // milli_per_tile
        self.regen_1x = cal.int("match.MANA_REGEN_MS_1X")
        self.regen_2x = cal.int("match.MANA_REGEN_MS_2X")
        self.start_mana = cal.int("match.START_MANA")
        self.max_mana = cal.int("match.MAX_MANA")
        self.king_activate_ticks = _ceil_div(cal.int("match.KING_ACTIVATE_TIME_MS"), self.tick_ms)
        self.regular_ticks = _ceil_div(cal.int("match.REGULAR_TIME_S") * 1000, self.tick_ms)
        self.overtime_ticks = _ceil_div(cal.int("match.OVERTIME_S") * 1000, self.tick_ms)
        self.overtime_tiebreak = str(cal.value("match.OVERTIME_TIEBREAK"))
        if self.overtime_tiebreak not in OVERTIME_TIEBREAK_RULES:
            raise ValueError(
                f"calibration match.OVERTIME_TIEBREAK {self.overtime_tiebreak!r} is not a rule"
            )
        if not cal.bool("match.THREE_CROWN_INSTANT_WIN"):
            raise NotImplementedError("mock only implements THREE_CROWN_INSTANT_WIN = true")
        self.range_to_radius = cal.bool("targeting.ADD_CHARACTER_RANGE_TO_RADIUS")
        self.keep_target_ext = self._milli(
            cal.int("targeting.LOGIC_RANGE_EXTENSION_TO_KEEP_TARGET")
        )
        self.xpos_tower_targeting = cal.bool("targeting.LOGIC_XPOS_BASED_TOWER_TARGETING")
        # Which placement class a unit-releasing spell (Goblin Barrel) gets: the registry
        # key the Rust core reads (state.rs deploy_rule). A value with no mask class is
        # refused rather than run as one of these.
        self.spawning_spell_water_rule = str(cal.value("spells.SPAWNING_SPELL_WATER_RULE"))
        if self.spawning_spell_water_rule not in ("refuse_touching_water", "anywhere"):
            raise NotImplementedError(
                f"SPAWNING_SPELL_WATER_RULE={self.spawning_spell_water_rule!r} is not implemented"
            )
        # Not in calibration.json yet -> read straight from the datamined globals.
        speedup_s = g["MANA_SPEED_UP_WHEN_REMAINING_SECONDS"][0]
        if speedup_s is None:
            raise KeyError("globals.csv lacks MANA_SPEED_UP_WHEN_REMAINING_SECONDS")
        self.speedup_ticks = _ceil_div(speedup_s * 1000, self.tick_ms)
        # Exact elixir: 1 elixir = lcm(regen_1x, regen_2x) units, so the per-tick
        # gain is an integer under both rates and nothing drifts over a match.
        self.elixir_scale = self.regen_1x * self.regen_2x // _gcd(self.regen_1x, self.regen_2x)
        self.gain = {
            1: self.tick_ms * self.max_mana * self.elixir_scale // self.regen_1x,
            2: self.tick_ms * self.max_mana * self.elixir_scale // self.regen_2x,
        }
        for rate, per_tick in self.gain.items():
            if per_tick * (self.regen_1x if rate == 1 else self.regen_2x) != (
                self.tick_ms * self.max_mana * self.elixir_scale
            ):
                raise ValueError("elixir gain per tick is not exact")

        self._load_cards(card_names)
        self._s: _Sim | None = None
        self._enc = msgspec.msgpack.Encoder()
        self._dec = msgspec.msgpack.Decoder(_Sim)

    # msgspec's codecs are C objects with no pickle support, and they are the only
    # thing in a MockEngine that has none. They hold no state -- they are a
    # compiled schema -- so dropping and rebuilding them is an exact round trip,
    # and it is what makes a whole ClashParallelEnv on this engine picklable
    # (env.py, EnvFactory: a built env is still not the thing to send to a
    # subprocess worker, but it must not be a TypeError either).

    def config(self) -> dict[str, object]:
        """Constructor state, JSON-able, for ``ClashParallelEnv.config()``.

        The card NAMES, because a catalogue subset is what makes one run's card ids
        mean something different from another's, and ``card_level`` for the same
        reason the Rust adapter reports it.
        """
        return {
            "cards": [c.name for c in self.cards()],
            "calibration_digest": calibration_digest(self.calibration),
        }

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        del state["_enc"]
        del state["_dec"]
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._enc = msgspec.msgpack.Encoder()
        self._dec = msgspec.msgpack.Decoder(_Sim)

    # ------------------------------------------------------------------ data

    def _milli(self, v: int) -> int:
        return v * self.sub_per_milli

    def _ms_ticks(self, ms: int) -> int:
        return _ceil_div(ms, self.tick_ms)

    def _load_cards(self, names: Sequence[str]) -> None:
        # joinpath, not "/": the source scan in tests/test_env_protocol.py reads a
        # division of two non-literals as arithmetic, and this module may hold none.
        base = data_dir().joinpath("raw", RAW_CARD_PACK, "csv_logic")
        chars = _csv_table(base / "characters.csv", UNIT_COLUMNS)
        bldgs = _csv_table(base / "buildings.csv", UNIT_COLUMNS)
        projs = _csv_table(base / "projectiles.csv", ("Name", "Damage"))
        aoes = _csv_table(base / "area_effect_objects.csv", ("Name",))
        s_chr = _csv_table(base / "spells_characters.csv", SPELL_CARD_COLUMNS)
        s_bld = _csv_table(base / "spells_buildings.csv", SPELL_CARD_COLUMNS)
        s_oth = _csv_table(base / "spells_other.csv", OTHER_SPELL_COLUMNS)

        self._units: list[_UnitTpl] = []
        self._spells: list[_SpellTpl] = []
        self._cards: list[_CardTpl] = []

        def unit_tpl(row: dict[str, str], kind: int) -> int:
            dmg = _i(row, "Damage")
            proj = row.get("Projectile", "").strip()
            if not dmg and proj:
                dmg = _i(projs[proj], "Damage")
            self._units.append(
                _UnitTpl(
                    name=row["Name"],
                    kind=kind,
                    hp=_i(row, "Hitpoints"),
                    radius=self._milli(_i(row, "CollisionRadius")),
                    speed=_i(row, "Speed") * self.speed_mult,
                    range_=self._milli(_i(row, "Range")),
                    sight=self._milli(_i(row, "SightRange")),
                    hit_ticks=max(1, self._ms_ticks(_i(row, "HitSpeed"))),
                    load_ticks=self._ms_ticks(_i(row, "LoadTime")),
                    damage=dmg,
                    area_radius=self._milli(_i(row, "AreaDamageRadius")),
                    attacks_air=_b(row, "AttacksAir"),
                    attacks_ground=_b(row, "AttacksGround"),
                    buildings_only=_b(row, "TargetOnlyBuildings"),
                    flying=_i(row, "FlyingHeight") > 0,
                    deploy_ticks=self._ms_ticks(_i(row, "DeployTime")),
                    lifetime_ticks=self._ms_ticks(_i(row, "LifeTime")),
                )
            )
            return len(self._units) - 1

        self._king_tpl = unit_tpl(bldgs["KingTower"], EntityKind.KING_TOWER)
        self._princess_tpl = unit_tpl(bldgs["PrincessTower"], EntityKind.PRINCESS_TOWER)

        for cid, name in enumerate(names):
            if name in s_chr:
                row = s_chr[name]
                u = unit_tpl(chars[row["SummonCharacter"]], EntityKind.TROOP)
                count = max(1, _i(row, "SummonNumber", 1))
                self._add_card(cid, name, row, Placement.TROOP, u, -1, count)
            elif name in s_bld:
                row = s_bld[name]
                u = unit_tpl(bldgs[row["SummonCharacter"]], EntityKind.BUILDING)
                self._add_card(cid, name, row, Placement.BUILDING, u, -1, 1)
            elif name in s_oth:
                row = s_oth[name]
                sp = self._spell_tpl(row, projs, aoes)
                released = self._released_character(row, projs)
                if _b(row, "SpellAsDeploy"):
                    placement = Placement.ROLLING
                elif released and self.spawning_spell_water_rule == "refuse_touching_water":
                    placement = Placement.SPELL_NOT_ON_WATER
                else:
                    placement = Placement.SPELL
                unit = unit_tpl(chars[released], EntityKind.TROOP) if released else -1
                self._add_card(cid, name, row, placement, unit, sp, 0)
            else:
                raise KeyError(f"card {name!r} not found in any spells_*.csv")

    @staticmethod
    def _released_character(row: dict[str, str], projs: dict[str, dict[str, str]]) -> str:
        """The character a spell's projectile releases (projectiles.csv SpawnCharacter)."""
        proj = row.get("Projectile", "").strip()
        return projs[proj].get("SpawnCharacter", "").strip() if proj in projs else ""

    def _spell_tpl(
        self,
        row: dict[str, str],
        projs: dict[str, dict[str, str]],
        aoes: dict[str, dict[str, str]],
    ) -> int:
        name = row["Name"]
        aoe = row.get("AreaEffectObject", "").strip()
        if aoe:
            a = aoes[aoe]
            tpl = _SpellTpl(
                name=name,
                damage=_i(a, "Damage"),
                radius=self._milli(_i(a, "Radius")),
                reach=0,
                crown_pct=100 + _i(a, "CrownTowerDamagePercent"),
                hits_air=_b(a, "HitsAir"),
                hits_ground=_b(a, "HitsGround"),
            )
        else:
            proj_name = row.get("CustomFirstProjectile", "").strip() or row["Projectile"].strip()
            p = projs[proj_name]
            rolling = p.get("SpawnProjectile", "").strip()
            if rolling:
                p = projs[rolling]
                tpl = _SpellTpl(
                    name=name,
                    damage=_i(p, "Damage"),
                    radius=self._milli(_i(p, "ProjectileRadius")),
                    reach=self._milli(_i(p, "ProjectileRange")),
                    crown_pct=100 + _i(p, "CrownTowerDamagePercent"),
                    hits_air=_b(p, "AoeToAir"),
                    hits_ground=_b(p, "AoeToGround"),
                )
            else:
                radius = _i(p, "Radius") or _i(row, "Radius")
                released = p.get("SpawnCharacter", "").strip()
                tpl = _SpellTpl(
                    name=name,
                    damage=_i(p, "Damage"),
                    radius=self._milli(radius),
                    reach=0,
                    crown_pct=100 + _i(p, "CrownTowerDamagePercent"),
                    hits_air=_b(p, "AoeToAir"),
                    hits_ground=_b(p, "AoeToGround"),
                    spawn_count=max(1, _i(p, "SpawnCharacterCount", 1)) if released else 0,
                    spawn_deploy_ticks=self._ms_ticks(_i(p, "SpawnCharacterDeployTime")),
                )
        self._spells.append(tpl)
        return len(self._spells) - 1

    def _add_card(
        self,
        cid: int,
        name: str,
        row: dict[str, str],
        placement: Placement,
        unit: int,
        spell: int,
        count: int,
    ) -> None:
        # A spell's row reports count/radius/hp 0 and not flying even when it releases
        # units (the Rust catalogue's convention, py.rs catalogue_rows).
        u = self._units[unit] if unit >= 0 and spell < 0 else None
        info = CardInfo(
            card_id=cid,
            name=name,
            elixir=_i(row, "ManaCost"),
            placement=int(placement),
            count=count,
            radius=u.radius if u else 0,
            flying=u.flying if u else False,
            hitpoints=u.hp if u else 0,
        )
        self._cards.append(_CardTpl(info=info, unit=unit, spell=spell))

    # ------------------------------------------------------------ protocol

    def cards(self) -> Sequence[CardInfo]:
        return [c.info for c in self._cards]

    def arena(self) -> Arena:
        return self._arena

    def rules(self) -> DeployRules:
        return self._rules

    def reset(self, seed: int, setup: MatchSetup) -> None:
        """Validate EVERYTHING (``protocol.setup_violation``), then build the new battle
        off to the side, then commit it. A refused setup raises ValueError and the
        running battle is untouched, as in the Rust core's ``Battle.reset`` (which
        builds a local ``BattleState`` and assigns it last).

        The order matters. Validating decks and spawns, assigning ``self._s`` and
        only then refusing a destroyed king mid-construction leaves a refused reset
        with tick 0, the new elixir and hands and zeroed Red towers in place.
        """
        validate_setup(self._arena, self.cards(), setup)
        self._s = self._new_battle(seed, setup)

    def _new_battle(self, seed: int, setup: MatchSetup) -> _Sim:
        """The battle ``setup`` describes. Assumes ``validate_setup`` passed; writes
        nothing on ``self``."""
        rng = Pcg32.new(seed)
        decks = [list(setup.decks[BLUE]), list(setup.decks[RED])]
        if setup.shuffle == ShuffleMode.INDEPENDENT:
            for d in decks:
                _shuffle(d, rng)
        elif setup.shuffle == ShuffleMode.MIRRORED:
            perm = list(range(DECK_SIZE))
            _shuffle(perm, rng)
            decks = [[d[i] for i in perm] for d in decks]

        cap_milli = self.max_mana * 1000
        start = [self.start_mana * 1000] * 2 if setup.elixir_milli is None else setup.elixir_milli
        s = _Sim(
            tick=setup.start_tick,
            # Clamped to [0, MAX_MANA], floored to the internal unit: state.rs
            # scenario_set_elixir_milli.
            elixir=[min(max(e, 0), cap_milli) * self.elixir_scale // 1000 for e in start],
            hands=[d[:HAND_SIZE] for d in decks],
            queues=[d[HAND_SIZE:] for d in decks],
            crowns=[0, 0],
            king_active=[False, False],
            king_timer=[0, 0],
            ents=[],
            next_uid=0,
            next_seq=[0, 0],
            overtime=setup.start_tick >= self.regular_ticks,
            game_over=False,
            winner=Winner.NONE,
            rng=rng,
            catalogue=[c.info.name for c in self._cards],
        )
        a = self._arena
        for team in TEAMS:
            hp = setup.tower_hp[team] if setup.tower_hp else None
            kx, ky = a.king_centers[team]
            king_hp = self._units[self._king_tpl].hp if hp is None else hp[TowerSlot.KING]
            self._spawn(s, team, self._king_tpl, EMPTY_CARD, TowerSlot.KING, kx, ky, king_hp)
            for slot in (TowerSlot.LEFT, TowerSlot.RIGHT):
                php = self._units[self._princess_tpl].hp if hp is None else hp[slot]
                px, py = a.princess_centers[team][slot - 1]
                if php > 0:
                    self._spawn(s, team, self._princess_tpl, EMPTY_CARD, slot, px, py, php)
                else:
                    if setup.crowns_from_destroyed_towers:
                        s.crowns[1 - team] += 1
                    s.king_active[team] = True
        # Canonical order, NOT list order: seq (the last tie-break key) must name the
        # same own-frame unit for both seats (protocol.SpawnSpec).
        cards = self.cards()
        for sp in sorted(setup.spawns, key=lambda q: spawn_order_key(a, cards, q)):
            c = self._cards[sp.card_id]  # validate_setup: a unit or building card
            self._spawn(s, sp.team, c.unit, sp.card_id, -1, sp.x, sp.y, sp.hp, deploy=0)
        return s

    def check_deploy(self, command: DeployCommand) -> int:
        return int(self._check(command))

    def step(self, commands: Sequence[DeployCommand], ticks: int) -> list[DeployResult]:
        s = self._sim()
        results: list[DeployResult] = []
        accepted: list[tuple[DeployCommand, int]] = []
        seen: set[int] = set()
        for cmd in commands:
            status = self._check(cmd)
            if status == DeployStatus.OK and cmd.team in seen:
                status = DeployStatus.DUPLICATE_TEAM
            card = EMPTY_CARD
            if cmd.team in TEAMS and 0 <= cmd.hand_slot < HAND_SIZE:
                card = s.hands[cmd.team][cmd.hand_slot]
            results.append(DeployResult(cmd.team, cmd.hand_slot, card, int(status), s.tick))
            if status == DeployStatus.OK:
                seen.add(cmd.team)
                accepted.append((cmd, card))
        # Apply in CANONICAL (team, slot) order, so uids -- and so state_hash -- do not
        # depend on whether Blue or Red was listed first for simultaneous plays (the
        # Rust engine's py.rs apply_commands).
        accepted.sort(key=_command_order_key)
        for cmd, card in accepted:
            self._pay(cmd, card)
        for _ in range(ticks):
            if s.game_over:
                break
            self._tick()
        return results

    def _pay(self, cmd: DeployCommand, card_id: int) -> None:
        """Spend elixir and cycle the hand NOW, at step(); units appear next tick.

        ORDER IS UNSOURCED, AND COPIED FROM THE RUST ENGINE, WHICH IS THE AUTHORITY:
        py.rs ``apply_commands`` pays before the first tick, so that tick's elixir
        regeneration lands on the reduced bar. Deploying a 3-cost card at a full
        10-elixir bar then reads 7017 milli after one tick on both engines.
        Paying inside the first tick instead, after upkeep, reads 7000: that tick's
        regeneration is clipped by the cap first, losing one tick of elixir per
        full-bar deploy. Which order the live game uses is not known.
        """
        s = self._sim()
        c = self._cards[card_id]
        team = cmd.team
        s.elixir[team] -= c.info.elixir * self.elixir_scale
        hand, queue = s.hands[team], s.queues[team]
        hand[cmd.hand_slot] = queue.pop(0) if queue else EMPTY_CARD
        queue.append(card_id)
        s.pending.append([team, card_id, cmd.x, cmd.y])

    def state(self) -> BattleState:
        s = self._sim()
        players = []
        for team in TEAMS:
            hp = [0, 0, 0]
            max_hp = [
                self._units[self._king_tpl].hp,
                self._units[self._princess_tpl].hp,
                self._units[self._princess_tpl].hp,
            ]
            for e in s.ents:
                if e.team == team and e.tower_slot >= 0:
                    hp[e.tower_slot] = e.hp
                    max_hp[e.tower_slot] = e.max_hp
            nxt = s.queues[team][0] if s.queues[team] else EMPTY_CARD
            players.append(
                PlayerState(
                    team=team,
                    elixir_milli=s.elixir[team] * 1000 // self.elixir_scale,
                    hand=list(s.hands[team]),
                    next_card=nxt,
                    crowns=s.crowns[team],
                    tower_hp=hp,
                    tower_max_hp=max_hp,
                    king_active=s.king_active[team],
                )
            )
        ents = []
        for e in s.ents:
            t = self._units[e.tpl]
            ents.append(
                EntityState(
                    uid=e.uid,
                    team=e.team,
                    kind=t.kind,
                    card_id=e.card_id,
                    tower_slot=e.tower_slot,
                    x=e.x,
                    y=e.y,
                    hp=e.hp,
                    max_hp=e.max_hp,
                    radius=t.radius,
                    flying=t.flying,
                    deploy_ticks=e.deploy,
                )
            )
        return BattleState(
            tick=s.tick,
            tick_ms=self.tick_ms,
            regular_ticks=self.regular_ticks,
            overtime_ticks=self.overtime_ticks,
            elixir_rate=self._rate(),
            overtime=s.overtime,
            players=players,
            entities=ents,
            game_over=s.game_over,
            winner=s.winner,
        )

    def save_state(self) -> bytes:
        return self._enc.encode(self._sim())

    def load_state(self, blob: bytes) -> None:
        """Restore a snapshot, matching its cards to this catalogue BY NAME.

        Refuses (ValueError, running battle untouched) when a hand, a queue, a
        pending deploy or a non-tower board entity names a card this catalogue does
        not have -- the Rust ``catalogue_violation`` rule and messages. A snapshot
        from a permuted superset catalogue is re-indexed and plays the same cards.
        Decoding the blob straight into ``self._s`` reads its ids through whatever
        catalogue loaded them: a Knight snapshot plays as a Valkyrie, and a one-card
        catalogue crashes.
        """
        self._s = self._adopt_snapshot(self._dec.decode(blob))

    def state_hash(self) -> int:
        return int.from_bytes(hashlib.blake2b(self.save_state(), digest_size=8).digest(), "little")

    # ------------------------------------------------------------ internals

    def _adopt_snapshot(self, s: _Sim) -> _Sim:
        """``s`` re-indexed into this catalogue, or ValueError. Mutates only ``s``."""
        mine = [c.info.name for c in self._cards]
        why = snapshot_catalogue_violation(s, set(mine))
        if why is not None:
            raise ValueError(why)
        return _reindex_snapshot(s, mine, self._cards)

    def _sim(self) -> _Sim:
        if self._s is None:
            raise RuntimeError("MockEngine.reset() has not been called")
        return self._s

    def _rate(self) -> int:
        s = self._sim()
        if s.overtime or s.tick >= self.regular_ticks - self.speedup_ticks:
            return 2
        return 1

    def _spawn(
        self,
        s: _Sim,
        team: int,
        tpl: int,
        card_id: int,
        slot: int,
        x: int,
        y: int,
        hp: int = -1,
        deploy: int | None = None,
    ) -> _Ent:
        t = self._units[tpl]
        e = _Ent(
            uid=s.next_uid,
            seq=s.next_seq[team],
            team=team,
            tpl=tpl,
            card_id=card_id,
            tower_slot=slot,
            x=x,
            y=y,
            hp=t.hp if hp < 0 else hp,
            max_hp=t.hp,
            deploy=t.deploy_ticks if deploy is None else deploy,
            cooldown=t.load_ticks,
            target=-1,
            life=t.lifetime_ticks if t.lifetime_ticks > 0 else -1,
        )
        s.next_uid += 1
        s.next_seq[team] += 1
        s.ents.append(e)
        return e

    # --- placement (independent of action.py on purpose: the mask test compares them)

    def _own_half_cell(self, team: int, hx: int, hy: int) -> tuple[int, int]:
        a = self._arena
        if team == BLUE:
            return hx, hy
        return a.hx - 1 - hx, a.hy - 1 - hy

    def _in_territory(self, team: int, hx: int, hy: int, building: bool) -> bool:
        """The CELL half of territory: buildings own half; troops anywhere off the
        river band (the band stays closed to troops -- UNSOURCED, DeployRules)."""
        a = self._arena
        _, oy = self._own_half_cell(team, hx, hy)
        lo, hi = a.water_half_rows
        if building:
            return oy < lo
        return not lo <= oy <= hi

    def _in_enemy_tower_rect(self, team: int, x: int, y: int) -> bool:
        """The POINT half of troop territory: inside the closed NoDeploySize rect of
        an alive enemy crown tower, centred on the tower ENTITY (not on the arena's
        centre table, which is what the mask uses -- two paths on purpose)."""
        rules = self._rules
        for e in self._sim().ents:
            if e.team != team and e.tower_slot >= 0:
                x0, y0, x1, y1 = rules.no_deploy_rect(e.tower_slot, e.x, e.y)
                if x0 <= x <= x1 and y0 <= y <= y1:
                    return True
        return False

    def _check(self, cmd: DeployCommand) -> DeployStatus:
        s = self._sim()
        a = self._arena
        if s.game_over:
            return DeployStatus.GAME_OVER
        if cmd.team not in TEAMS:
            return DeployStatus.BAD_TEAM
        if not 0 <= cmd.hand_slot < HAND_SIZE:
            return DeployStatus.BAD_SLOT
        card_id = s.hands[cmd.team][cmd.hand_slot]
        if card_id == EMPTY_CARD:
            return DeployStatus.EMPTY_SLOT
        card = self._cards[card_id].info
        if s.elixir[cmd.team] < card.elixir * self.elixir_scale:
            return DeployStatus.NOT_ENOUGH_ELIXIR
        if not (0 < cmd.x < a.width and 0 < cmd.y < a.height):
            return DeployStatus.OUT_OF_ARENA
        if card.placement == Placement.SPELL:
            return DeployStatus.OK
        # A point on a half-cell boundary belongs to every cell it touches (closed
        # squares). That makes the rule invariant under the 180-degree rotation; a
        # floor-based "which cell is this point in" rule is not.
        h = a.half_size
        xs = [cmd.x // h] + ([cmd.x // h - 1] if cmd.x % h == 0 else [])
        ys = [cmd.y // h] + ([cmd.y // h - 1] if cmd.y % h == 0 else [])
        cells = [(hx, hy) for hx in xs for hy in ys]
        if any(a.grid[hy][hx] & BIT_WATER for hx, hy in cells):
            return DeployStatus.WATER
        if card.placement == Placement.SPELL_NOT_ON_WATER:
            return DeployStatus.OK  # water is its only position rule (protocol.Placement)
        if any(a.grid[hy][hx] & BIT_NO_DEPLOY for hx, hy in cells):
            return DeployStatus.NO_DEPLOY
        building = card.placement == Placement.BUILDING
        if not all(self._in_territory(cmd.team, hx, hy, building) for hx, hy in cells):
            return DeployStatus.OUT_OF_TERRITORY
        if not building and self._in_enemy_tower_rect(cmd.team, cmd.x, cmd.y):
            return DeployStatus.OUT_OF_TERRITORY
        if card.placement in (Placement.TROOP, Placement.BUILDING):
            extra = card.radius if card.placement == Placement.BUILDING else 0
            for e in s.ents:
                if self._units[e.tpl].kind == EntityKind.TROOP:
                    continue
                r = self._units[e.tpl].radius + extra
                if (cmd.x - e.x) ** 2 + (cmd.y - e.y) ** 2 <= r * r:
                    return DeployStatus.OCCUPIED
        return DeployStatus.OK

    # --- the tick

    def _tick(self) -> None:
        s = self._sim()
        units = self._units

        # UPKEEP ------------------------------------------------------------
        rate = self._rate()
        cap = self.max_mana * self.elixir_scale
        for team in TEAMS:
            s.elixir[team] = min(cap, s.elixir[team] + self.gain[rate])
            if s.king_active[team] and s.king_timer[team] > 0:
                s.king_timer[team] -= 1
        dying: set[int] = set()
        for e in s.ents:
            if e.deploy > 0:
                e.deploy -= 1
            elif e.life > 0:
                e.life -= 1
                if e.life == 0:
                    dying.add(e.uid)
        spells: list[tuple[int, int, int, int]] = []  # team, spell tpl, x, y
        # team, tpl, card, x, y, deploy ticks (None = the unit's own)
        spawn_queue: list[tuple[int, int, int, int, int, int | None]] = []
        # Deploys were paid for at step() (``_pay``); they materialise here.
        for team, card_id, x, y in s.pending:
            c = self._cards[card_id]
            if c.spell >= 0:
                spells.append((team, c.spell, x, y))
                sp = self._spells[c.spell]
                if sp.spawn_count and c.unit >= 0:
                    # Released at the tap at once: the MOCK simplification (module doc).
                    # They report the spell's card id, as the Rust core's do.
                    r = units[c.unit].radius
                    for ox, oy in self._formation(team, sp.spawn_count, r, x, y):
                        spawn_queue.append((team, c.unit, card_id, ox, oy, sp.spawn_deploy_ticks))
            else:
                for ox, oy in self._formation(team, c.info.count, units[c.unit].radius, x, y):
                    spawn_queue.append((team, c.unit, card_id, ox, oy, None))
        s.pending = []

        # SPAWN -------------------------------------------------------------
        for team, tpl, card_id, x, y, deploy in spawn_queue:
            self._spawn(s, team, tpl, card_id, -1, x, y, deploy=deploy)

        by_uid = {e.uid: e for e in s.ents}

        # TARGET (reads start-of-tick positions only) ------------------------
        new_target: dict[int, int] = {}
        for e in s.ents:
            if e.deploy > 0 or units[e.tpl].damage <= 0:
                continue
            new_target[e.uid] = self._choose_target(e, by_uid)
        for uid, target_uid in new_target.items():
            by_uid[uid].target = target_uid

        # PATH + MOVE (propose from old positions, apply together) -----------
        moves: list[tuple[_Ent, int, int]] = []
        for e in s.ents:
            t = units[e.tpl]
            if e.deploy > 0 or t.speed <= 0 or e.target < 0:
                continue
            tgt = by_uid.get(e.target)
            if tgt is None or self._in_range(e, tgt, 0):
                continue
            wx, wy = self._waypoint(e, tgt)
            moves.append((e, *_step_toward(e.x, e.y, wx, wy, t.speed)))
        for e, nx, ny in moves:
            e.x, e.y = nx, ny

        # ATTACK (post-move positions, damage into a buffer) -----------------
        damage: dict[int, int] = {}
        for e in s.ents:
            if e.cooldown > 0:
                e.cooldown -= 1
            t = units[e.tpl]
            if e.deploy > 0 or e.target < 0 or t.damage <= 0:
                continue
            if t.kind == EntityKind.KING_TOWER and (
                not s.king_active[e.team] or s.king_timer[e.team] > 0
            ):
                continue
            tgt = by_uid.get(e.target)
            if tgt is None or not self._in_range(e, tgt, 0) or e.cooldown > 0:
                continue
            e.cooldown = t.hit_ticks
            if t.area_radius > 0:
                for o in s.ents:
                    if (
                        o.team != e.team
                        and self._can_hit(t, o)
                        and self._within(tgt.x, tgt.y, o, t.area_radius)
                    ):
                        damage[o.uid] = damage.get(o.uid, 0) + t.damage
            else:
                damage[tgt.uid] = damage.get(tgt.uid, 0) + t.damage

        # PROJECTILE (spells land this tick) --------------------------------
        for team, sp_i, x, y in spells:
            sp = self._spells[sp_i]
            _, fy = self._forward(team)
            for o in s.ents:
                if o.team == team:
                    continue
                ot = units[o.tpl]
                if not (sp.hits_air if ot.flying else sp.hits_ground):
                    continue
                if sp.reach > 0:
                    # Rolling spell, modelled as an instant strip from the deploy
                    # point forward (own frame) -- the mock has no projectile travel.
                    r = sp.radius + ot.radius
                    along = (o.y - y) * fy
                    hit = abs(o.x - x) <= r and -ot.radius <= along <= sp.reach + ot.radius
                else:
                    hit = self._within(x, y, o, sp.radius)
                if hit:
                    dmg = sp.damage
                    if ot.kind in (EntityKind.KING_TOWER, EntityKind.PRINCESS_TOWER):
                        dmg = _tdiv(dmg * sp.crown_pct, 100)
                    damage[o.uid] = damage.get(o.uid, 0) + dmg

        # RESOLVE (one pass) -------------------------------------------------
        king_hit = [False, False]
        for uid, dmg in damage.items():
            o = by_uid[uid]
            o.hp -= dmg
            if dmg > 0 and o.tower_slot == TowerSlot.KING:
                king_hit[o.team] = True
            if o.hp <= 0:
                dying.add(uid)

        # REAP ----------------------------------------------------------------
        kings_dead = [False, False]
        for e in s.ents:
            if e.uid not in dying:
                continue
            if e.tower_slot == TowerSlot.KING:
                kings_dead[e.team] = True
            elif e.tower_slot >= 0:
                s.crowns[1 - e.team] += 1
                king_hit[e.team] = True
        if dying:
            s.ents = [e for e in s.ents if e.uid not in dying]
        for team in TEAMS:
            if king_hit[team] and not s.king_active[team]:
                s.king_active[team] = True
                s.king_timer[team] = self.king_activate_ticks
            if kings_dead[team]:
                s.crowns[1 - team] = 3

        # JUDGE ---------------------------------------------------------------
        s.tick += 1
        if kings_dead[BLUE] or kings_dead[RED]:
            s.game_over = True
            if kings_dead[BLUE] and kings_dead[RED]:
                s.winner = Winner.DRAW
            else:
                s.winner = Winner.RED if kings_dead[BLUE] else Winner.BLUE
            return
        lead = s.crowns[BLUE] - s.crowns[RED]
        if s.overtime and lead != 0:
            s.game_over = True
            s.winner = Winner.BLUE if lead > 0 else Winner.RED
        elif not s.overtime and s.tick >= self.regular_ticks:
            if lead != 0:
                s.game_over = True
                s.winner = Winner.BLUE if lead > 0 else Winner.RED
            else:
                s.overtime = True
        if s.overtime and not s.game_over and s.tick >= self.regular_ticks + self.overtime_ticks:
            s.game_over = True
            s.winner = self._overtime_tiebreak(s)

    def _overtime_tiebreak(self, s: _Sim) -> Winner:
        """The verdict when overtime runs out level on crowns (calibration
        match.OVERTIME_TIEBREAK, the Rust engine's state.rs overtime_tiebreak): each
        side's key is its WEAKEST standing crown tower and the weaker weakest tower
        loses; an exact tie is a draw. Absolute compares hp; fraction compares
        hp/max_hp by exact cross-multiplication. Scalars per side, so the seat
        rotation cannot enter."""
        if self.overtime_tiebreak == "none_draw":
            return Winner.DRAW
        fraction = self.overtime_tiebreak == "lowest_tower_hp_fraction"

        def weakest(team: int):
            towers = [
                (max(e.hp, 0), max(e.max_hp, 1))
                for e in s.ents
                if e.team == team and e.tower_slot >= 0
            ]
            if not towers:
                return None
            if fraction:
                return min(towers, key=lambda t: Fraction(t[0], t[1]))
            return min(towers, key=lambda t: t[0])

        b, r = weakest(BLUE), weakest(RED)
        if b is None or r is None:
            if b is None and r is None:
                return Winner.DRAW
            return Winner.RED if b is None else Winner.BLUE
        kb, kr = (Fraction(b[0], b[1]), Fraction(r[0], r[1])) if fraction else (b[0], r[0])
        if kb == kr:
            return Winner.DRAW
        return Winner.BLUE if kb > kr else Winner.RED

    # --- geometry helpers (all tie-breaks in the acting team's own frame)

    def _forward(self, team: int) -> tuple[int, int]:
        return (1, 1) if team == BLUE else (-1, -1)

    def _own_xy(self, team: int, x: int, y: int) -> tuple[int, int]:
        a = self._arena
        return (x, y) if team == BLUE else (a.width - x, a.height - y)

    def _formation(
        self, team: int, count: int, radius: int, x: int, y: int
    ) -> list[tuple[int, int]]:
        if count <= 1:
            return [(x, y)]
        a = self._arena
        fx, fy = self._forward(team)
        cols = [-1, 1] if count == 2 else [-1, 0, 1]
        out = []
        for k in range(count):
            dx = cols[k % len(cols)] * radius * fx
            dy = -(k // len(cols)) * 2 * radius * fy
            px, py = x + dx, y + dy
            if not (0 < px < a.width and 0 < py < a.height) or self._side(py) != self._side(y):
                px, py = x, y
            out.append((px, py))
        return out

    def _side(self, y: int) -> int:
        a = self._arena
        lo = a.water_half_rows[0] * a.half_size
        hi = (a.water_half_rows[1] + 1) * a.half_size
        if y < lo:
            return -1
        if y > hi:
            return 1
        return 0

    def _waypoint(self, e: _Ent, tgt: _Ent) -> tuple[int, int]:
        """Ground units cross the river only on a bridge. Flying units go direct."""
        if self._units[e.tpl].flying:
            return tgt.x, tgt.y
        a = self._arena
        su, st = self._side(e.y), self._side(tgt.y)
        lo = a.water_half_rows[0] * a.half_size
        hi = (a.water_half_rows[1] + 1) * a.half_size
        if su == st:
            if su != 0 or self._bridge_for(e.team, e.x) == self._bridge_for(e.team, tgt.x):
                return tgt.x, tgt.y
            # Both in the river band but on different bridges: step off forward first.
            _, fy = self._forward(e.team)
            return e.x, (hi + 1 if fy > 0 else lo - 1)
        if su == 0:
            return e.x, (hi + 1 if st > 0 else lo - 1)
        bx = self._bridge_for(e.team, tgt.x if st == 0 else e.x)
        return bx, (lo if su < 0 else hi)

    def _bridge_for(self, team: int, x: int) -> int:
        """Nearest bridge centre to x; ties go to the team's own-left bridge."""
        own_ref, _ = self._own_xy(team, x, 0)
        best: tuple[int, int] | None = None
        best_bx = 0
        for bx in self._arena.bridge_centers_x():
            obx, _ = self._own_xy(team, bx, 0)
            key = (abs(obx - own_ref), obx)
            if best is None or key < best:
                best, best_bx = key, bx
        return best_bx

    def _can_hit(self, t: _UnitTpl, o: _Ent) -> bool:
        ot = self._units[o.tpl]
        if t.buildings_only and ot.kind == EntityKind.TROOP:
            return False
        return t.attacks_air if ot.flying else t.attacks_ground

    def _within(self, x: int, y: int, o: _Ent, reach: int) -> bool:
        r = reach + (self._units[o.tpl].radius if self.range_to_radius else 0)
        return (x - o.x) ** 2 + (y - o.y) ** 2 <= r * r

    def _in_range(self, e: _Ent, tgt: _Ent, ext: int) -> bool:
        return self._within(e.x, e.y, tgt, self._units[e.tpl].range_ + ext)

    def _choose_target(self, e: _Ent, by_uid: dict[int, _Ent]) -> int:
        s = self._sim()
        t = self._units[e.tpl]
        cur = by_uid.get(e.target)
        if (
            cur is not None
            and self._can_hit(t, cur)
            and self._in_range(e, cur, self.keep_target_ext)
        ):
            return cur.uid
        is_mover = t.kind == EntityKind.TROOP
        reach = t.sight if is_mover else t.range_
        best: tuple[int, int, int, int] | None = None
        best_uid = -1
        for o in s.ents:
            if o.team == e.team or not self._can_hit(t, o) or not self._within(e.x, e.y, o, reach):
                continue
            ox, oy = self._own_xy(e.team, o.x, o.y)
            key = ((e.x - o.x) ** 2 + (e.y - o.y) ** 2, oy, ox, o.seq)
            if best is None or key < best:
                best, best_uid = key, o.uid
        if best_uid >= 0 or not is_mover:
            return best_uid
        return self._default_tower(e)

    def _default_tower(self, e: _Ent) -> int:
        s = self._sim()
        enemy = 1 - e.team
        towers = {o.tower_slot: o for o in s.ents if o.team == enemy and o.tower_slot >= 0}
        if self.xpos_tower_targeting:
            own_x, _ = self._own_xy(e.team, e.x, e.y)
            # The enemy's own-RIGHT princess stands on my own-left.
            slot = TowerSlot.RIGHT if own_x <= self._arena.width // 2 else TowerSlot.LEFT
            if slot in towers:
                return towers[slot].uid
            return towers[TowerSlot.KING].uid if TowerSlot.KING in towers else -1
        best: tuple[int, int, int] | None = None
        uid = -1
        for o in towers.values():
            ox, oy = self._own_xy(e.team, o.x, o.y)
            key = ((e.x - o.x) ** 2 + (e.y - o.y) ** 2, oy, ox)
            if best is None or key < best:
                best, uid = key, o.uid
        return uid


def snapshot_catalogue_violation(s: _Sim, catalogue: set[str]) -> str | None:
    """Why snapshot ``s`` cannot run behind a catalogue holding ``catalogue`` names.

    The Rust ``py.rs::catalogue_violation`` rule, in its order and with its messages:
    every card a battle can still produce -- a hand slot, the cycle queue, a pending
    deploy, a non-tower entity on the board -- must be in the catalogue. Crown
    towers are never catalogue cards. Module-level so a test can plant it away.
    """
    names = s.catalogue
    if not names:
        return "snapshot carries no card catalogue: its card ids cannot be resolved"

    def label(cid: int) -> str:
        return names[cid] if 0 <= cid < len(names) else f"id {cid}"

    def missing(cid: int) -> bool:
        return not 0 <= cid < len(names) or names[cid] not in catalogue

    for team in TEAMS:
        for cid in s.hands[team]:
            if cid != EMPTY_CARD and missing(cid):
                return f"snapshot hand card {label(cid)} not in catalogue"
        for cid in s.queues[team]:
            if missing(cid):
                return f"snapshot queue card {label(cid)} not in catalogue"
    for _, cid, _, _ in s.pending:
        if missing(cid):
            return f"snapshot pending spawn {label(cid)} not in catalogue"
    for e in s.ents:
        if e.tower_slot < 0 and missing(e.card_id):
            return f"snapshot board entity {label(e.card_id)} not in catalogue"
    return None


def _reindex_snapshot(s: _Sim, catalogue: list[str], cards: list[_CardTpl]) -> _Sim:
    """Every card id in ``s`` moved from ``s.catalogue`` to ``catalogue`` by name.

    Requires ``snapshot_catalogue_violation`` to have passed. A snapshot of the same
    catalogue is returned untouched, so save -> load -> save is byte-exact. Unit
    templates are laid out in catalogue order too, so each board entity's ``tpl``
    follows its card. Module-level so a test can plant it away.
    """
    if s.catalogue == catalogue:
        return s
    id_of_name = {name: cid for cid, name in enumerate(catalogue)}

    def remap(cid: int) -> int:
        return EMPTY_CARD if cid == EMPTY_CARD else id_of_name[s.catalogue[cid]]

    s.hands = [[remap(c) for c in hand] for hand in s.hands]
    s.queues = [[remap(c) for c in queue] for queue in s.queues]
    s.pending = [[team, remap(cid), x, y] for team, cid, x, y in s.pending]
    for e in s.ents:
        if e.tower_slot < 0:
            e.card_id = remap(e.card_id)
            e.tpl = cards[e.card_id].unit
    s.catalogue = list(catalogue)
    return s


def _command_order_key(accepted: tuple[DeployCommand, int]) -> tuple[int, int]:
    """Canonical application order of simultaneous accepted deploys: (team, slot).
    Module-level so tests/test_rust_engine.py can plant input order back in."""
    return accepted[0].team, accepted[0].hand_slot


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _shuffle(items: list[int], rng: Pcg32) -> None:
    for i in range(len(items) - 1, 0, -1):
        j = rng.below(i + 1)
        items[i], items[j] = items[j], items[i]


def _step_toward(x: int, y: int, tx: int, ty: int, amount: int) -> tuple[int, int]:
    """fixed.rs Vec2::step_toward: never overshoots, truncates the step toward zero."""
    dx, dy = tx - x, ty - y
    length = isqrt(dx * dx + dy * dy)
    if length == 0 or amount >= length:
        return tx, ty
    return x + _tdiv(dx * amount, length), y + _tdiv(dy * amount, length)

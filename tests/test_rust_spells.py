"""The RL layer on RustEngine with spells: the full 18-card thin slice, end to end.

WHY IT EXISTS
    The Rust core plays Fireball, Zap, Arrows, The Log and Goblin Barrel, which cost
    the catalogue two more kind codes and the state two more shapes (live spell
    objects, stun timers). None of that reaches a bot creator by itself: the adapter
    has to map the new kind codes, protocol.py needs a placement class for a spell
    that may not land on water, and the obs builders have to read the spell objects
    and the stun timers. This file is the evidence that each of those works THROUGH
    PYTHON, measured the same way the troop wiring is.

WHAT IT CHECKS
    a. The catalogue: every thin-slice card on RustEngine AND MockEngine with the same
       placement class, elixir and spell row shape (count / radius / hp 0); the
       default 65-card catalogue constructs.
    b. SPELL LEGALITY THREE WAYS -- the action mask (built from RustEngine's own
       catalogue), MockEngine and RustEngine -- at every half-cell centre and corner,
       both teams, all seven territory tower states, verdict AND reason, for SPELL
       (Fireball, Zap, Arrows), ROLLING (Log) and SPELL_NOT_ON_WATER (Goblin Barrel),
       with buildings on the board. Plus the mask's hot path equal to its general path.
    c. The rotation symmetry with spells: one shared policy on both seats that aims
       spells at enemy troops, so both seats cast rotated twins in the SAME step;
       every step obs[blue] == obs[red] (spell and stun channels included), rewards,
       statuses, and the engine state (spell objects and stun timers included) equal
       to its own rotation. Floors on casts, live spells, rolling Logs and stuns.
    d. A masked random rollout of the full thin slice for both seats: zero
       rejections, every spell cast, and the entity-list builder's spell rows. And the
       spell rotation run on MockEngine, which must report no spell object or stun.
    e. A per-tick replay of a spell battle re-verifies bit for bit and renders its
       spell objects.

PLANTS (each on a green baseline, aimed at the check it certifies)
    the adapter mapping kind 4 to SPELL (river offered to the barrel); the mask
    refusing a Log over a building footprint; the adapter reporting one Red spell's
    aim point one subtile off (rotation check).

WHAT IT CANNOT CATCH
    Whether any spell behaves like the live game (nothing here is JUDGED against
    a measurement). Mock spell EFFECTS: MockEngine's
    spells are instant by design and are held to the engine on legality and API shape
    only (mock_engine.py).

SKIPS
    Only when royalesim is not built. A stale build is a failure.
"""

from __future__ import annotations

import collections
import json

import msgspec
import numpy as np
import pytest

from _lockout import lockout_ticks
from royalegym import rust_engine as rust_engine_module
from royalegym.action import PlacementOracle
from royalegym.done_condition import GameOverCondition, StepLimitCondition
from royalegym.env import ClashParallelEnv
from royalegym.mock_engine import MockEngine
from royalegym.obs import EntityListObsBuilder, SpatialObsBuilder, spatial_channels
from royalegym.protocol import (
    BLUE,
    RED,
    DeployCommand,
    DeployStatus,
    EntityKind,
    MatchSetup,
    Placement,
    ShuffleMode,
    SpawnSpec,
    SpellMotion,
    data_dir,
)
from royalegym.render import build_view, extract_view, render_html
from royalegym.replay import ReplayRecorder, verify_trace
from royalegym.rust_engine import (
    CORE_IMPORT_ERROR,
    RustEngine,
    SymmetricRustEngine,
    catalogue_vintage_split,
    core_available,
)
from royalegym.selfplay import RandomLegalOpponent
from royalegym.state_mutator import DefaultStateMutator
from test_rust_engine import TERRITORY_STATES, every_half_cell_point, rotation_divergence

pytestmark = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))

SPELL_CLASSES = {
    "Fireball": Placement.SPELL,
    "Zap": Placement.SPELL,
    "Arrows": Placement.SPELL,
    "Log": Placement.ROLLING,
    "GoblinBarrel": Placement.SPELL_NOT_ON_WATER,
}


def thin_slice() -> list[str]:
    raw = json.loads((data_dir() / "derived" / "cards.json").read_text(encoding="utf-8"))
    names = list(raw["thin_slice"])
    assert len(names) == 18, names
    return names


# ---------------------------------------------------------------------------
# a. the catalogue


def test_thin_slice_catalogue_agrees_between_engines_and_the_default_catalogue_builds():
    names = thin_slice()
    rust, mock = RustEngine(card_names=names), MockEngine(card_names=names)
    split = catalogue_vintage_split(rust.cards(), mock.cards())
    if split is not None:
        pytest.skip(split)
    for r, m in zip(rust.cards(), mock.cards(), strict=True):
        assert (r.card_id, r.name, r.placement, r.elixir, r.count, r.flying) == (
            m.card_id,
            m.name,
            m.placement,
            m.elixir,
            m.count,
            m.flying,
        ), (r, m)
        if r.placement >= Placement.SPELL:
            assert (r.count, r.radius, r.hitpoints, m.radius, m.hitpoints) == (0, 0, 0, 0, 0), r
        else:
            assert r.radius == m.radius, (r, m)  # hp differs: card level (rust_engine.py)
    assert {c.name: Placement(c.placement) for c in rust.cards() if c.name in SPELL_CLASSES} == (
        SPELL_CLASSES
    )
    # An adapter that knows kind codes 0 and 1 only raises KeyError: 2 here.
    full = RustEngine()
    got = collections.Counter(Placement(c.placement).name for c in full.cards())
    print(f"default catalogue: {len(full.cards())} cards {dict(got)}")
    assert set(names) <= {c.name for c in full.cards()}
    assert got["SPELL"] >= 3, got
    assert got["ROLLING"] >= 1, got
    assert got["SPELL_NOT_ON_WATER"] >= 1, got


def test_unknown_kind_code_is_refused_at_construction(monkeypatch):
    """A kind code the adapter has no Placement for must raise, not map by accident."""
    monkeypatch.delitem(rust_engine_module._PLACEMENT_OF_KIND, 4)
    assert 4 not in rust_engine_module._PLACEMENT_OF_KIND, "plant did not land"
    with pytest.raises(RuntimeError, match=r"kind codes \[4\]"):
        RustEngine(card_names=["Knight", "GoblinBarrel"])


# ---------------------------------------------------------------------------
# b. spell legality, three ways, at every half-cell

SPELL_CAT = ("Fireball", "Log", "GoblinBarrel", "Zap", "Arrows", "Cannon", "Knight", "Tesla")
FB, LOG, BARREL, ZAP, ARROWS, CANNON, KNIGHT, TESLA = range(8)
# Blue's hand Fireball / Log / Goblin Barrel / Zap; Red's Arrows / Goblin Barrel / Log /
# Fireball (ShuffleMode.NONE: the hand is the first four).
SPELL_DECKS = [
    [FB, LOG, BARREL, ZAP, ARROWS, CANNON, KNIGHT, TESLA],
    [ARROWS, BARREL, LOG, FB, ZAP, CANNON, KNIGHT, TESLA],
]


#: Ticks a match refuses every deploy for. Battles here start past it, or Rust answers
#: TOO_EARLY where Mock answers a placement verdict and the comparison grades timing
#: instead of territory. Read from an engine because 0 is a real calibration arm.
LOCKOUT = lockout_ticks()


def spell_setup(arena, tower_hp) -> MatchSetup:
    s = arena.subtile
    w, h = arena.width, arena.height
    return MatchSetup(
        decks=SPELL_DECKS,
        shuffle=ShuffleMode.NONE,
        elixir_milli=[10000, 10000],
        tower_hp=tower_hp,
        start_tick=LOCKOUT,
        spawns=[
            # Buildings on both sides (footprints the Log ignores and troops may not use):
            # a Cannon on a tile corner on each side, a Tesla on a tile centre and on a
            # tile corner in the back half.
            SpawnSpec(BLUE, CANNON, s * 9, s * 10),
            SpawnSpec(BLUE, TESLA, s * 3 + s // 2, s * 4 + s // 2),
            SpawnSpec(RED, CANNON, w - s * 5 - s // 2, h - s * 12 - s // 2),
            SpawnSpec(RED, TESLA, w - s * 14, h - s * 7),
            SpawnSpec(RED, KNIGHT, s * 9, s * 20),  # troops never block placement
        ],
    )


def spell_legality(rust, mock, oracle, tower_hp):
    """(disagreements, per-card status counts, special-case counts)."""
    setup = spell_setup(mock.arena(), tower_hp)
    rust.reset(4, setup)
    mock.reset(4, setup)
    st = mock.state()
    a = mock.arena()
    xs, ys = every_half_cell_point(a)
    dis: collections.Counter = collections.Counter()
    seen: collections.Counter = collections.Counter()
    special: collections.Counter = collections.Counter()
    buildings = [e for e in st.entities if e.kind != EntityKind.TROOP]
    for team in (BLUE, RED):
        for slot in range(4):
            card = rust.cards()[st.players[team].hand[slot]]
            assert card.placement >= Placement.SPELL, card
            for pitch in (1, 2):
                px, py = oracle.points(pitch)
                if not np.array_equal(
                    oracle.point_grid(st, team, card, pitch),
                    oracle.legal_points(st, team, card, px, py),
                ):
                    dis[(team, card.name, "point_grid != legal_points", pitch)] += 1
            mask = oracle.legal_points(st, team, card, xs, ys)
            for x, y, m in zip(xs.tolist(), ys.tolist(), mask.tolist(), strict=True):
                c = DeployCommand(team, slot, x, y)
                ms, rs = mock.check_deploy(c), rust.check_deploy(c)
                seen[(card.name, DeployStatus(rs).name)] += 1
                if not (bool(m) == (ms == DeployStatus.OK) == (rs == DeployStatus.OK)) or ms != rs:
                    dis[
                        (
                            team,
                            card.name,
                            int(m),
                            DeployStatus(ms).name,
                            DeployStatus(rs).name,
                            x,
                            y,
                        )
                    ] += 1
                if rs == DeployStatus.OK:
                    on_building = any(
                        (x - e.x) ** 2 + (y - e.y) ** 2 <= e.radius**2 for e in buildings
                    )
                    h = a.half_size
                    cell = a.grid[min(y // h, a.hy - 1)][min(x // h, a.hx - 1)]
                    special[(card.name, "OK on a building footprint")] += int(on_building)
                    special[(card.name, "OK on a no-deploy cell")] += int(bool(cell & 16))
    return dis, seen, special


@pytest.fixture(scope="module")
def spell_engines():
    rust = RustEngine(card_names=SPELL_CAT)
    mock = MockEngine(card_names=SPELL_CAT)
    # The mask is built from RustEngine's OWN catalogue -- what the env binds to.
    oracle = PlacementOracle(rust.arena(), rust.rules(), rust.cards())
    return rust, mock, oracle


@pytest.mark.parametrize("towers", list(TERRITORY_STATES))
def test_spell_mask_mock_and_rust_agree_on_every_half_cell(spell_engines, towers):
    rust, mock, oracle = spell_engines
    dis, seen, special = spell_legality(rust, mock, oracle, TERRITORY_STATES[towers])
    assert not dis, f"{sum(dis.values())} disagreements, first: {list(dis)[:6]}"
    n = every_half_cell_point(mock.arena())[0].size
    # Vacuity, per class: every reason each class can produce, in bulk. Measured
    # 2026-09-13 on all_up, summed over both teams (Fireball, Log and Goblin Barrel
    # are in both hands, Zap and Arrows in one): Fireball OK 9018 / OUT_OF_ARENA 400;
    # Log OK 3812, OUT_OF_TERRITORY 3936, NO_DEPLOY 756, WATER 514, OUT_OF_ARENA 400;
    # Goblin Barrel OK 8504, WATER 514, OUT_OF_ARENA 400; Log OK on a building
    # footprint 130, Goblin Barrel on a footprint 440 and on a no-deploy cell 640.
    for name, reasons in (
        ("Fireball", ("OK", "OUT_OF_ARENA")),
        ("Zap", ("OK", "OUT_OF_ARENA")),
        ("Arrows", ("OK", "OUT_OF_ARENA")),
        ("Log", ("OK", "OUT_OF_TERRITORY", "NO_DEPLOY", "WATER", "OUT_OF_ARENA")),
        ("GoblinBarrel", ("OK", "WATER", "OUT_OF_ARENA")),
    ):
        for reason in reasons:
            assert seen[(name, reason)] >= 100, (name, reason, dict(seen))
        total = sum(v for (card, _), v in seen.items() if card == name)
        assert total % n == 0, (name, total, n)
        assert total >= n, (name, total, n)
    assert not any(
        r in ("NO_DEPLOY", "OUT_OF_TERRITORY", "OCCUPIED") for c, r in seen if c == "GoblinBarrel"
    )
    assert not any(r == "OCCUPIED" for c, r in seen if c == "Log"), (
        "a Log was refused on a building"
    )
    assert special[("Log", "OK on a building footprint")] >= 20, dict(special)
    assert special[("GoblinBarrel", "OK on a building footprint")] >= 20, dict(special)
    assert special[("GoblinBarrel", "OK on a no-deploy cell")] >= 100, dict(special)


def test_plant_adapter_maps_the_barrel_to_anywhere_is_caught(monkeypatch):
    """Regression plant: kind 4 read as SPELL -- the mask offers the river to a Goblin
    Barrel that both engines refuse as WATER, and nothing else changes."""
    clean = RustEngine(card_names=SPELL_CAT)
    mock = MockEngine(card_names=SPELL_CAT)
    ok_dis, _, _ = spell_legality(
        clean,
        mock,
        PlacementOracle(clean.arena(), clean.rules(), clean.cards()),
        TERRITORY_STATES["all_up"],
    )
    assert not ok_dis, "baseline must be green"
    monkeypatch.setitem(rust_engine_module._PLACEMENT_OF_KIND, 4, Placement.SPELL)
    rust = RustEngine(card_names=SPELL_CAT)
    assert rust.cards()[BARREL].placement == Placement.SPELL, "plant did not land"
    oracle = PlacementOracle(rust.arena(), rust.rules(), rust.cards())
    dis, _, _ = spell_legality(rust, mock, oracle, TERRITORY_STATES["all_up"])
    assert dis, "PLANT DID NOT LAND: a river-open barrel mask passed the three-way check"
    assert {(k[1], k[2], k[3], k[4]) for k in dis if len(k) == 7} == {
        ("GoblinBarrel", 1, "WATER", "WATER")
    }


def test_plant_mask_refusing_a_log_over_buildings_is_caught(spell_engines, monkeypatch):
    """Plant: the mask applies the TROOP footprint rule to ROLLING (Log ships
    CanPlaceOnBuildings=TRUE; both engines let it land on a building)."""
    rust, mock, _ = spell_engines
    planted = PlacementOracle(rust.arena(), rust.rules(), rust.cards())
    orig_lp, orig_pg = PlacementOracle.legal_points, PlacementOracle.point_grid

    def as_troop(card):
        return (
            msgspec.structs.replace(card, placement=Placement.TROOP)
            if card.placement == Placement.ROLLING
            else card
        )

    monkeypatch.setattr(
        PlacementOracle,
        "legal_points",
        lambda self, st, team, card, px, py: orig_lp(self, st, team, as_troop(card), px, py),
    )
    monkeypatch.setattr(
        PlacementOracle,
        "point_grid",
        lambda self, st, team, card, pitch: orig_pg(self, st, team, as_troop(card), pitch),
    )
    dis, _, _ = spell_legality(rust, mock, planted, TERRITORY_STATES["red_left_down"])
    assert dis, "PLANT DID NOT LAND: a footprint-blocked Log mask passed"
    assert {(k[1], k[2], k[3], k[4]) for k in dis if len(k) == 7} == {("Log", 0, "OK", "OK")}


# ---------------------------------------------------------------------------
# c. the rotation symmetry with spells, both seats casting in the same step

SPELL_ROTATION_DECK = (
    "Fireball",
    "Zap",
    "Log",
    "GoblinBarrel",
    "Arrows",
    "Knight",
    "Minions",
    "SkeletonArmy",
)


# Spatial channel index by name, so a reordering moves this with it.
CHANNEL = {name: i for i, (name, _) in enumerate(spatial_channels())}


def spell_aiming_policy(noop_prob: float = 0.35, aim_prob: float = 0.85):
    """ONE policy for both seats, a pure function of (obs, mask, rng): with probability
    ``aim_prob`` play a legal action on a tile holding enemy troops (on the enemy half
    that is, by the territory rule, almost always a spell), otherwise a random legal
    action. Reads only the own-frame observation (the spatial enemy-troop channels, or
    the entity-list enemy troop rows), so it is seat-symmetric by construction and any
    asymmetry is the stack's."""

    def enemy_troop_tiles(obs):
        if "spatial" in obs:
            # BY NAME. These were channels 4 and 5 until crown towers were split out
            # of the buildings planes; a hardcoded index here does not fail, it
            # quietly aims the policy at a different channel and the test goes on
            # measuring something else.
            return obs["spatial"][CHANNEL["enemy_ground_troops"]] + obs["spatial"][
                CHANNEL["enemy_air_troops"]
            ]
        grid = np.zeros((32, 18), dtype=np.float32)
        for f in obs["entities"]:
            if f[0] and f[2] and f[3]:  # present, enemy, troop
                grid[min(int(f[8] * 32), 31), min(int(f[7] * 18), 17)] += 1
        return grid

    def act(obs, mask, rng):
        legal = np.flatnonzero(mask)[1:]
        if not legal.size or rng.random() < noop_prob:
            return 0
        enemy = enemy_troop_tiles(obs)
        tile = (legal - 1) % (18 * 32)
        on_enemy = legal[enemy.reshape(-1)[tile] > 0]
        if on_enemy.size and rng.random() < aim_prob:
            return int(rng.choice(on_enemy))
        return int(rng.choice(legal))

    return act


# Floors sit below what was measured 2026-09-13 over 400 steps with the spatial
# builder: seed 1 -- 29 same-step spell casts (Zap 8, Log 8, Arrows 7, Goblin Barrel 6),
# spells live on 86 steps, a Log rolling on 56, a stun on 6; seed 2 -- 26 casts, 85, 56,
# 5. The spell / stun obs counters equalled the engine counters on both seeds.
SPELL_ROTATION_FLOORS = {
    "same_step_spell_casts": 15,
    "spell_live_steps": 50,
    "rolling_steps": 30,
    "stunned_steps": 3,
    "obs_spell_channel_steps": 50,
    "obs_stun_channel_steps": 3,
    "steps": 400,
}


@pytest.mark.parametrize(
    ("seed", "builder"), [(1, SpatialObsBuilder), (2, SpatialObsBuilder), (1, EntityListObsBuilder)]
)
def test_rotation_mirror_holds_with_spells_cast_by_both_seats_in_the_same_step(seed, builder):
    stats = collections.Counter()
    result = rotation_divergence(
        seed,
        steps=400,
        engine_cls=SymmetricRustEngine,
        deck=SPELL_ROTATION_DECK,
        act=spell_aiming_policy(),
        stats=stats,
        obs_builder_cls=builder,
    )
    print(f"spell rotation seed {seed} {builder.__name__}: {result} {dict(stats)}")
    assert result is None, result
    for k, floor in SPELL_ROTATION_FLOORS.items():
        assert stats[k] >= floor, f"vacuous: {k} = {stats[k]} < {floor} ({dict(stats)})"
    # Every placement CLASS cast by both seats in one step. Per card is not required:
    # seed 1 measured 2026-09-13 never cast the 4-elixir Fireball (Zap 8, Log 8, Arrows
    # 7, Goblin Barrel 6) -- the policy spends elixir on cheaper spells first.
    for cls in set(SPELL_CLASSES.values()):
        cast = sum(stats[f"cast:{n}"] for n, c in SPELL_CLASSES.items() if c == cls)
        assert cast >= 3, f"vacuous: {cls.name} cast {cast} times ({dict(stats)})"


def test_rotation_mirror_holds_with_spells_on_mock_engine_which_reports_no_spell_objects():
    """The same policy and deck on MockEngine: its (instant) spells keep the battle a
    rotation mirror, and it reports exactly what mock_engine.py says -- no live spell
    object and no stun, ever. Measured 2026-09-13: seed 1, 28 same-step casts (Zap 8,
    Log 8, Goblin Barrel 6, Arrows 6), every spell / stun counter 0."""
    stats = collections.Counter()
    result = rotation_divergence(
        1,
        steps=400,
        engine_cls=MockEngine,
        deck=SPELL_ROTATION_DECK,
        act=spell_aiming_policy(),
        stats=stats,
    )
    print(f"mock spell rotation: {result} {dict(stats)}")
    assert result is None, result
    assert stats["same_step_spell_casts"] >= 15, dict(stats)
    for k in ("spell_live_steps", "rolling_steps", "stunned_steps", "obs_spell_channel_steps"):
        assert stats[k] == 0, f"MockEngine reported {k} = {stats[k]}; its doc says it cannot"


class _RedSpellAimNudged(SymmetricRustEngine):
    """PLANT (adapter): every Red spell object reported with its aim point one subtile
    off in x -- a spell decided in the engine frame by the smallest amount."""

    nudged = 0

    def state(self):
        s = super().state()
        if not any(q.team == RED for q in s.spells):
            return s
        type(self).nudged += 1
        return msgspec.structs.replace(
            s,
            spells=[
                msgspec.structs.replace(q, aim_x=q.aim_x + 1) if q.team == RED else q
                for q in s.spells
            ],
        )


def test_plant_red_spell_aim_one_subtile_off_is_caught():
    _RedSpellAimNudged.nudged = 0
    result = rotation_divergence(
        1,
        steps=400,
        engine_cls=_RedSpellAimNudged,
        deck=SPELL_ROTATION_DECK,
        act=spell_aiming_policy(),
    )
    print(f"spell aim plant: {_RedSpellAimNudged.nudged} nudged states, {result}")
    assert _RedSpellAimNudged.nudged > 0, "plant did not land: no Red spell was ever live"
    assert result is not None, "PLANT DID NOT LAND: a Red aim point one subtile off passed"
    assert "rotation mirror" in result, f"PLANT DID NOT LAND on the state check: {result}"


# ---------------------------------------------------------------------------
# d. the full thin slice, both seats, masked random play


def test_full_thin_slice_masked_rollout_never_rejected_and_casts_every_spell():
    names = thin_slice()
    engine = RustEngine(card_names=names)
    ids = {c.name: c.card_id for c in engine.cards()}
    deck_a = [
        ids[n]
        for n in ("Fireball", "Zap", "Arrows", "Log", "GoblinBarrel", "Knight", "Cannon", "Minions")
    ]
    deck_b = [
        ids[n]
        for n in (
            "GoblinBarrel",
            "Log",
            "Zap",
            "Fireball",
            "Arrows",
            "Tesla",
            "Giant",
            "BabyDragon",
        )
    ]
    env = ClashParallelEnv(
        engine=engine,
        obs_builder=EntityListObsBuilder(),
        state_mutator=DefaultStateMutator(decks=[deck_a, deck_b]),
        termination_cond=GameOverCondition(),
        truncation_cond=StepLimitCondition(10_000),
        decision_ms=250,
    )
    rng = np.random.default_rng(11)
    # noop 0.85: at 0.3 a random policy spends every 2-3 elixir and never holds the 4
    # a Fireball needs (measured 2026-09-13: 0 Fireballs at 0.3, 2 at 0.7, 3 at 0.85
    # in 600 steps; 0 rejections in all three).
    opp = RandomLegalOpponent(noop_prob=0.85)
    obs, _ = env.reset(seed=11)
    rejected, casts = [], collections.Counter()
    spell_rows = 0
    for _ in range(600):
        if not env.agents:
            obs, _ = env.reset()
        prev = env.battle_state
        acts = {a: opp.act(obs[a], obs[a]["action_mask"], rng) for a in env.agents}
        obs, _, _, _, info = env.step(acts)
        spell_rows = max(spell_rows, int(obs["blue"]["spells"][:, 0].sum()) if env.agents else 0)
        for agent, act in acts.items():
            if act == 0:
                continue
            team = BLUE if agent == "blue" else RED
            if info[agent]["deploy_status"] != DeployStatus.OK:
                rejected.append(
                    (prev.tick, agent, act, DeployStatus(info[agent]["deploy_status"]).name)
                )
            else:
                casts[
                    engine.cards()[prev.players[team].hand[env.action_parser.decode(act)[0]]].name
                ] += 1
    print(f"thin-slice rollout casts {dict(casts)}, max spell rows seen {spell_rows}")
    assert rejected == []
    # Measured 2026-09-13: GoblinBarrel 8, Zap 9, Arrows 6, Log 9, Fireball 3.
    for name in SPELL_CLASSES:
        assert casts[name] >= 2, (name, dict(casts))
    assert spell_rows >= 1, "the entity-list builder never saw a live spell object"


# ---------------------------------------------------------------------------
# e. replay and viewer with spells


def test_spell_replay_on_rust_verifies_and_renders_spell_objects():
    names = thin_slice()
    engine = RustEngine(card_names=names)
    ids = {c.name: c.card_id for c in engine.cards()}
    deck = [
        ids[n]
        for n in (
            "Fireball",
            "Log",
            "Zap",
            "GoblinBarrel",
            "Arrows",
            "Knight",
            "Minions",
            "Valkyrie",
        )
    ]
    rec = ReplayRecorder(frame_every_tick=True)
    env = ClashParallelEnv(
        engine=engine,
        recorder=rec,
        state_mutator=DefaultStateMutator(decks=[deck, deck[::-1]]),
        termination_cond=GameOverCondition(),
        truncation_cond=StepLimitCondition(160),
    )
    obs, _ = env.reset(seed=7)
    rng = np.random.default_rng(7)
    act = spell_aiming_policy(noop_prob=0.3)
    while env.agents:
        obs, *_ = env.step({a: act(obs[a], obs[a]["action_mask"], rng) for a in env.agents})
    trace = rec.trace
    assert trace is not None
    frames_with = [f for f in trace.frames if f.spells]
    motions = {q.motion for f in trace.frames for q in f.spells}
    stunned = sum(1 for f in trace.frames if any(e.stun_ticks > 0 for e in f.entities))
    print(
        f"spell replay: {len(trace.frames)} frames, {len(frames_with)} with spells, "
        f"motions {sorted(motions)}, {stunned} with a stun"
    )
    assert len(frames_with) >= 20
    assert {SpellMotion.FLIGHT, SpellMotion.ROLLING} <= motions, motions
    assert verify_trace(trace, RustEngine(card_names=names)) == []
    view = extract_view(render_html(trace))
    assert view["spell_fields"] == list(trace.header.spell_fields)
    ff = view["frame_fields"]
    drawn = [f for f in view["frames"] if f[ff.index("spells")]]
    assert len(drawn) == len(frames_with)
    first = trace.frames.index(frames_with[0])
    assert view["frames"][first][ff.index("spells")] == msgspec.to_builtins(frames_with[0].spells)
    # A trace missing a spell column it has data for is refused, not drawn wrong.
    bad = msgspec.structs.replace(
        trace, header=msgspec.structs.replace(trace.header, spell_fields=["team", "card_id"])
    )
    with pytest.raises(ValueError, match=r"spell\.motion"):
        build_view(bad)

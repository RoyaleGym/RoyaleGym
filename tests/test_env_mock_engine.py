"""MockEngine behaviour: determinism, simultaneity, bridges, match flow."""

from __future__ import annotations

import contextlib

import msgspec
import numpy as np
import pytest

from royalegym import mock_engine
from royalegym.action import TileActionParser
from royalegym.env import ClashParallelEnv
from royalegym.mock_engine import MockEngine
from royalegym.protocol import (
    BLUE,
    RED,
    DeployCommand,
    DeployStatus,
    EntityKind,
    MatchSetup,
    ShuffleMode,
    SpawnSpec,
    TowerSlot,
    Winner,
    mirror_state,
)
from royalegym.selfplay import RandomLegalOpponent
from royalegym.state_setter import DefaultStateSetter
from royalegym.terminal import GameOverCondition, StepLimitCondition

DECK = list(range(8))  # Knight Archer Goblins Giant MiniPekka Musketeer Skeletons Minions
ALL_TYPES_DECK = [0, 3, 7, 9, 10, 11, 13, 14]  # troop, tank, air, splash, building, 3 spells


def random_commands(eng: MockEngine, parser: TileActionParser, rng: np.random.Generator, p: float):
    s = eng.state()
    cmds = []
    for team in (BLUE, RED):
        legal = np.flatnonzero(parser.action_mask(s, team))[1:]
        if legal.size and rng.random() < p:
            cmds.append(parser.parse(int(rng.choice(legal)), s, team))
    return cmds


def play(seed: int, ticks: int, chunk: int, deck=DECK) -> tuple[int, list[int]]:
    eng = MockEngine()
    parser = TileActionParser()
    parser.bind(eng)
    eng.reset(seed, MatchSetup(decks=[deck, deck]))
    rng = np.random.default_rng(seed)
    hashes = []
    t = 0
    while t < ticks and not eng.state().game_over:
        eng.step(random_commands(eng, parser, rng, 0.3), chunk)
        t += chunk
        hashes.append(eng.state_hash())
    return eng.state_hash(), hashes


def test_same_seed_same_battle_different_seed_different_battle():
    a, ha = play(11, 1500, 10)
    b, hb = play(11, 1500, 10)
    c, _ = play(12, 1500, 10)
    assert ha == hb
    assert a == b
    assert a != c


def test_chunked_stepping_equals_tick_by_tick():
    """The protocol says step(cmds, n) == step(cmds, 1) + step([], 1) * (n-1)."""
    eng1, eng2 = MockEngine(), MockEngine()
    parser = TileActionParser()
    parser.bind(eng1)
    setup = MatchSetup(decks=[ALL_TYPES_DECK, DECK])
    eng1.reset(5, setup)
    eng2.reset(5, setup)
    rng = np.random.default_rng(5)
    for _ in range(120):
        cmds = random_commands(eng1, parser, rng, 0.5)
        r1 = eng1.step(cmds, 7)
        r2 = eng2.step(cmds, 1)
        for _ in range(6):
            eng2.step([], 1)
        assert r1 == r2
        assert eng1.state_hash() == eng2.state_hash()


def test_save_load_round_trip_resumes_identically():
    eng = MockEngine()
    parser = TileActionParser()
    parser.bind(eng)
    eng.reset(9, MatchSetup(decks=[ALL_TYPES_DECK, ALL_TYPES_DECK]))
    rng = np.random.default_rng(9)
    for _ in range(60):
        eng.step(random_commands(eng, parser, rng, 0.5), 5)
    blob = eng.save_state()
    h0 = eng.state_hash()
    script = []
    for _ in range(60):
        cmds = random_commands(eng, parser, rng, 0.5)
        script.append(cmds)
        eng.step(cmds, 5)
    final = eng.state_hash()
    other = MockEngine()
    other.load_state(blob)
    assert other.state_hash() == h0
    for cmds in script:
        other.step(cmds, 5)
    assert other.state_hash() == final


def test_hand_cycles_through_the_deck_in_order():
    eng = MockEngine()
    eng.reset(0, MatchSetup(decks=[DECK, DECK], shuffle=ShuffleMode.NONE, elixir_milli=[10000, 0]))
    a = eng.arena()
    s = eng.state()
    assert s.players[BLUE].hand == [0, 1, 2, 3]
    assert s.players[BLUE].next_card == 4
    spot = (a.subtile * 9 // 2, a.subtile * 21 // 2)  # tile (4, 10) centre, own half
    r = eng.step([DeployCommand(BLUE, 2, *spot)], 1)
    assert r[0].status == DeployStatus.OK
    assert r[0].card_id == 2
    p = eng.state().players[BLUE]
    assert p.hand == [0, 1, 4, 3]
    assert p.next_card == 5
    # Goblins cost 2; one tick of regen was added before payment.
    assert p.elixir_milli < 10000 - 2000 + 100


def test_elixir_doubles_in_the_last_minute_and_in_overtime():
    eng = MockEngine()
    reg = eng.regular_ticks
    eng.reset(
        0, MatchSetup(decks=[DECK, DECK], start_tick=reg - eng.speedup_ticks, elixir_milli=[0, 0])
    )
    assert eng.state().elixir_rate == 2
    eng.reset(0, MatchSetup(decks=[DECK, DECK], start_tick=reg - eng.speedup_ticks - 1))
    assert eng.state().elixir_rate == 1
    eng.reset(0, MatchSetup(decks=[DECK, DECK], start_tick=reg))
    assert eng.state().overtime
    assert eng.state().elixir_rate == 2


def test_level_game_goes_to_overtime_then_draw():
    eng = MockEngine()
    eng.reset(0, MatchSetup(decks=[DECK, DECK], start_tick=eng.regular_ticks - 2))
    eng.step([], 3)
    s = eng.state()
    assert s.overtime
    assert not s.game_over
    eng.step([], eng.overtime_ticks)
    s = eng.state()
    assert s.game_over
    assert s.winner == Winner.DRAW


def test_overtime_is_sudden_death():
    eng = MockEngine()
    a = eng.arena()
    # Red's own-LEFT princess at 1 hp, Blue drops a Fireball on it in overtime.
    eng.reset(
        0,
        MatchSetup(
            decks=[[11, 0, 1, 2, 3, 4, 5, 6], DECK],
            shuffle=ShuffleMode.NONE,
            start_tick=eng.regular_ticks,
            elixir_milli=[10000, 0],
            tower_hp=[[2400, 1400, 1400], [2400, 1, 1400]],
        ),
    )
    px, py = a.princess_centers[RED][TowerSlot.LEFT - 1]
    r = eng.step([DeployCommand(BLUE, 0, px, py)], 2)
    assert r[0].status == DeployStatus.OK
    s = eng.state()
    assert s.game_over
    assert s.winner == Winner.BLUE
    assert s.players[BLUE].crowns == 1


def test_king_kill_is_three_crowns_and_ends_the_game():
    eng = MockEngine()
    a = eng.arena()
    eng.reset(
        0,
        MatchSetup(
            decks=[[11, 0, 1, 2, 3, 4, 5, 6], DECK],
            shuffle=ShuffleMode.NONE,
            elixir_milli=[10000, 0],
            tower_hp=[[2400, 1400, 1400], [1, 1400, 1400]],
        ),
    )
    kx, ky = a.king_centers[RED]
    eng.step([DeployCommand(BLUE, 0, kx + a.subtile // 2, ky - a.subtile // 2)], 1)
    s = eng.state()
    assert s.game_over
    assert s.winner == Winner.BLUE
    assert s.players[BLUE].crowns == 3


def _ground_units_off_bridge(eng: MockEngine) -> list[tuple[int, int, int]]:
    a = eng.arena()
    lo = a.water_half_rows[0] * a.half_size
    hi = (a.water_half_rows[1] + 1) * a.half_size
    bad = []
    for e in eng.state().entities:
        if e.kind != EntityKind.TROOP or e.flying or not (lo <= e.y <= hi):
            continue
        hx = e.x // a.half_size
        if not any(b0 <= hx <= b1 for b0, b1 in a.bridges_half_cols):
            bad.append((e.uid, e.x, e.y))
    return bad


def _bridge_violations(seed: int, ticks: int) -> int:
    eng = MockEngine()
    parser = TileActionParser()
    parser.bind(eng)
    eng.reset(seed, MatchSetup(decks=[DECK, DECK]))
    rng = np.random.default_rng(seed)
    crossings = 0
    violations = 0
    for _ in range(ticks):
        eng.step(random_commands(eng, parser, rng, 0.05), 1)
        violations += len(_ground_units_off_bridge(eng))
        crossings += sum(
            1
            for e in eng.state().entities
            if e.kind == EntityKind.TROOP and not e.flying and 270000 <= e.y <= 306000
        )
        if eng.state().game_over:
            break
    assert crossings > 0, "no ground unit ever reached the river; the test proves nothing"
    return violations


def test_ground_units_cross_the_river_only_on_bridges():
    assert _bridge_violations(21, 1500) == 0


def test_plant_bridge_check_sees_units_walking_on_water(monkeypatch):
    monkeypatch.setattr(MockEngine, "_waypoint", lambda self, e, tgt: (tgt.x, tgt.y))
    assert _bridge_violations(21, 1500) > 0


def test_mirrored_knights_trade_symmetrically():
    """Invariant 2. The predecessor let Blue's Knight win this trade every time."""
    eng = MockEngine()
    a = eng.arena()
    x = a.subtile * 7 // 2
    yb = a.subtile * 12
    eng.reset(
        0,
        MatchSetup(
            decks=[DECK, DECK],
            spawns=[
                SpawnSpec(BLUE, 0, x, yb),
                SpawnSpec(RED, 0, a.width - x, a.height - yb),
            ],
        ),
    )
    for _ in range(600):
        eng.step([], 1)
        knights = {e.team: e for e in eng.state().entities if e.card_id == 0}
        assert set(knights) in ({BLUE, RED}, set()), f"one knight outlived the other: {knights}"
        if knights:
            assert knights[BLUE].hp == knights[RED].hp
            assert (knights[BLUE].x, knights[BLUE].y) == (
                a.width - knights[RED].x,
                a.height - knights[RED].y,
            )


# --- the dynamic mirror test: whole games, both seats playing mirrored moves -------


def _canonical(state, arena):
    return sorted(
        (e.team, e.kind, e.card_id, e.tower_slot, e.x, e.y, e.hp, e.max_hp, e.deploy_ticks)
        for e in state.entities
    )


def first_mirror_divergence(seed: int, max_steps: int = 400, noop_prob: float = 0.6) -> str | None:
    """Both seats run the same seeded policy on their own observation.

    If obs, masks and the engine are all exactly seat-symmetric, the two seats
    take the same action index every step and the battle stays a perfect mirror
    to the end. Returns a description of the first asymmetry, or None.
    """
    env = ClashParallelEnv(
        state_setter=DefaultStateSetter(decks=[ALL_TYPES_DECK, ALL_TYPES_DECK], mirror=True),
        terminal_conditions=[GameOverCondition(), StepLimitCondition(max_steps)],
    )
    obs, _ = env.reset(seed=seed)
    pol = {a: RandomLegalOpponent(noop_prob) for a in env.agents}
    rngs = {a: np.random.default_rng(seed) for a in env.agents}
    arena = env.engine.arena()
    step = 0
    deploys = 0
    while env.agents:
        for k in obs["blue"]:
            if not np.array_equal(obs["blue"][k], obs["red"][k]):
                return f"step {step}: observation {k!r} differs between seats"
        s = env.battle_state
        if _canonical(s, arena) != _canonical(mirror_state(arena, s), arena):
            return f"step {step}: engine state is not a mirror"
        acts = {a: pol[a].act(obs[a], obs[a]["action_mask"], rngs[a]) for a in env.agents}
        obs, rew, _, _, info = env.step(acts)
        deploys += int(info["blue"]["deploy_status"] == 0)
        if rew["blue"] != rew["red"]:
            return f"step {step}: rewards differ {rew}"
        step += 1
    assert deploys > 20, "mirror game had too few deploys to be evidence"
    return None


@pytest.mark.parametrize("seed", [1, 2])
def test_mirrored_games_stay_mirrored(seed):
    assert first_mirror_divergence(seed) is None


def test_plant_floor_division_breaks_the_mirror(monkeypatch):
    monkeypatch.setattr(mock_engine, "_tdiv", lambda a, b: a // b)
    assert first_mirror_divergence(1) is not None


def test_plant_blue_frame_tiebreaks_break_the_mirror(monkeypatch):
    monkeypatch.setattr(MockEngine, "_own_xy", lambda self, team, x, y: (x, y))
    assert first_mirror_divergence(1) is not None


# --- snapshots and catalogues ------------------------------------------------------
#
# Decoding any blob and reading its card ids through whatever catalogue loaded it
# plays a Knight snapshot as a Valkyrie, and crashes on a one-card catalogue. The rule
# is the Rust ``catalogue_violation``: refuse a snapshot naming a card outside the
# catalogue; match cards by NAME otherwise.

SHARED8 = ("Knight", "Minions", "Cannon", "Giant", "Archer", "Goblins", "MiniPekka", "HogRider")
SUPERSET = ("Valkyrie", *reversed(SHARED8[4:]), "Musketeer", *reversed(SHARED8[:4]))


def _named(eng: MockEngine) -> tuple:
    """The battle with every card id replaced by its name (uids kept: same battle)."""
    names = [c.name for c in eng.cards()]

    def nm(cid):
        return None if cid < 0 else names[cid]

    s = eng.state()
    players = [
        (p.elixir_milli, [nm(c) for c in p.hand], nm(p.next_card), p.crowns, p.tower_hp)
        for p in s.players
    ]
    ents = sorted(
        (
            e.uid,
            e.team,
            e.kind,
            nm(e.card_id) or "",
            e.tower_slot,
            e.x,
            e.y,
            e.hp,
            e.max_hp,
            e.radius,
            e.flying,
            e.deploy_ticks,
        )
        for e in s.entities
    )
    return s.tick, s.game_over, s.winner, players, ents


def _played_snapshot(steps: int = 80) -> tuple[MockEngine, bytes, list]:
    """A SHARED8 battle with units and pending deploys in the snapshot, and the
    command script that continues it (already run on the returned engine)."""
    eng = MockEngine(card_names=SHARED8)
    parser = TileActionParser()
    parser.bind(eng)
    eng.reset(4, MatchSetup(decks=[DECK, DECK[::-1]], elixir_milli=[10000, 10000]))
    rng = np.random.default_rng(4)
    for _ in range(steps):
        eng.step(random_commands(eng, parser, rng, 0.5), 5)
    # Pay without ticking, so pending deploys are in the blob (wait for elixir first).
    for _ in range(400):
        cmds = random_commands(eng, parser, np.random.default_rng(0), 1.0)
        if cmds:
            break
        eng.step([], 1)
    eng.step(cmds, 0)
    blob = eng.save_state()
    script = []
    for _ in range(60):
        cmds = random_commands(eng, parser, rng, 0.5)
        script.append(cmds)
        eng.step(cmds, 5)
    return eng, blob, script


def superset_load_mismatches() -> list[str]:
    """Load a SHARED8 snapshot into a permuted superset catalogue and replay the same
    commands on both: every step must be the same battle, card by NAME."""
    src_final, blob, script = _played_snapshot()
    src = MockEngine(card_names=SHARED8)
    src.load_state(blob)
    dst = MockEngine(card_names=SUPERSET)
    dst.load_state(blob)
    s = src.state()
    # Vacuity: the snapshot must hold pending deploys and board units, and the two
    # catalogues must give those cards DIFFERENT ids, or name-matching is untested.
    assert src._sim().pending, "snapshot has no pending deploys"
    board = {e.card_id for e in s.entities if e.card_id >= 0}
    assert board, "snapshot has no units on the board"
    assert any(SUPERSET.index(SHARED8[c]) != c for c in board), "no board card changes id"
    bad = []
    if _named(src) != _named(dst):
        bad.append("after load")
    for i, cmds in enumerate(script):
        r1 = src.step(cmds, 5)
        if bad:
            continue  # already diverged: finish the source replay only, for the check below
        r2 = dst.step(cmds, 5)
        if [r.status for r in r1] != [r.status for r in r2]:
            bad.append(f"step {i}: deploy statuses")
        elif _named(src) != _named(dst):
            bad.append(f"step {i}: state")
    assert src.state_hash() == src_final.state_hash(), "source replay itself diverged"
    return bad


def test_snapshot_loads_by_card_name_into_a_permuted_superset_catalogue():
    assert superset_load_mismatches() == []


def test_same_catalogue_round_trip_is_byte_exact():
    _, blob, _ = _played_snapshot(20)
    other = MockEngine(card_names=SHARED8)
    other.load_state(blob)
    assert other.save_state() == blob


def test_plant_ids_read_through_the_wrong_catalogue_are_caught(monkeypatch):
    """Regression plant: keep the refusal, drop the re-indexing -- ids are read through
    the loading catalogue again (the Knight-plays-as-Valkyrie defect)."""
    assert superset_load_mismatches() == [], "baseline must be green"
    monkeypatch.setattr(mock_engine, "_reindex_snapshot", lambda s, catalogue, cards: s)
    bad = superset_load_mismatches()
    assert bad[:1] == ["after load"], f"PLANT DID NOT LAND: {bad}"


def _snapshot_with_foreign_card(where: str) -> bytes:
    """A SHARED8 snapshot whose catalogue gains "Valkyrie", referenced in ONE place."""
    eng = MockEngine(card_names=SHARED8)
    eng.reset(2, MatchSetup(decks=[DECK, DECK], shuffle=ShuffleMode.NONE))
    a = eng.arena()
    sim = msgspec.msgpack.decode(eng.save_state(), type=mock_engine._Sim)
    valk = len(sim.catalogue)
    sim.catalogue.append("Valkyrie")
    if where == "hand":
        sim.hands[RED][2] = valk
    elif where == "queue":
        sim.queues[BLUE][-1] = valk
    elif where == "pending":
        sim.pending.append([BLUE, valk, a.width // 2, a.height // 4])
    else:
        sim.ents.append(msgspec.structs.replace(sim.ents[0], uid=99, tower_slot=-1, card_id=valk))
    return msgspec.msgpack.encode(sim)


LOAD_REFUSALS = {
    "hand": "snapshot hand card Valkyrie not in catalogue",
    "queue": "snapshot queue card Valkyrie not in catalogue",
    "pending": "snapshot pending spawn Valkyrie not in catalogue",
    "board": "snapshot board entity Valkyrie not in catalogue",
}


def load_refusal_failures() -> list[str]:
    """Every foreign-card snapshot must be refused with the Rust message, untouched."""
    bad = []
    for where, message in LOAD_REFUSALS.items():
        eng = MockEngine(card_names=SHARED8)
        eng.reset(7, MatchSetup(decks=[DECK, DECK]))
        eng.step([], 30)
        before = eng.save_state()
        try:
            eng.load_state(_snapshot_with_foreign_card(where))
        except ValueError as exc:
            if str(exc) != message:
                bad.append(f"{where}: refused with {exc!r}")
        else:
            bad.append(f"{where}: ACCEPTED")
        if eng.save_state() != before:
            bad.append(f"{where}: running battle changed")
    return bad


def test_snapshot_naming_a_card_outside_the_catalogue_is_refused():
    assert load_refusal_failures() == []


def test_a_smaller_catalogue_refuses_instead_of_crashing():
    _, blob, _ = _played_snapshot(20)
    with pytest.raises(ValueError, match="not in catalogue"):
        MockEngine(card_names=["Knight"]).load_state(blob)


def test_plant_load_without_catalogue_check_is_caught(monkeypatch):
    """Regression plant: the shipped defect, ``self._s = decode(blob)``."""
    assert load_refusal_failures() == [], "baseline must be green"
    monkeypatch.setattr(MockEngine, "_adopt_snapshot", lambda self, s: s)
    bad = load_refusal_failures()
    want = [m for w in LOAD_REFUSALS for m in (f"{w}: ACCEPTED", f"{w}: running battle changed")]
    assert bad == want, f"PLANT DID NOT LAND as aimed: {bad}"


# --- MatchSetup refusals and clamps (protocol.MatchSetup table) --------------------

REFUSED_SETUPS = {
    "destroyed king": dict(tower_hp=[[0, 10, 10], [1, 1, 1]], elixir_milli=[1234, 1234]),
    "start_tick -1": dict(start_tick=-1),
    "start_tick 2**32": dict(start_tick=2**32),
    "shuffle 7": dict(shuffle=7),
    "shuffle -1": dict(shuffle=-1),
    "elixir []": dict(elixir_milli=[]),
    "elixir three entries": dict(elixir_milli=[1, 2, 3]),
    "tower_hp short row": dict(tower_hp=[[5, 3], [1, 1, 1]]),
    "tower_hp one team": dict(tower_hp=[[5, 3, 3]]),
    "tower hp beyond i32": dict(tower_hp=[[2**31, 3, 3], [1, 1, 1]]),
    "deck of 7": dict(decks=[DECK[:7], DECK]),
    "unknown card": dict(decks=[DECK, [99] * 8]),
    "spawn on water": dict(spawns=[SpawnSpec(BLUE, 0, 9 * 18000, 31 * 9000)]),
}


def reset_refusal_failures() -> list[str]:
    bad = []
    for name, kw in REFUSED_SETUPS.items():
        kw = dict(kw)
        eng = MockEngine()
        eng.reset(1, MatchSetup(decks=[DECK, DECK], shuffle=ShuffleMode.NONE))
        eng.step([], 100)
        before = eng.save_state()
        decks = kw.pop("decks", [DECK, DECK[::-1]])
        try:
            eng.reset(2, MatchSetup(decks=decks, **kw))
        except ValueError:
            pass
        except Exception as exc:  # the contract is ValueError; anything else is a finding
            bad.append(f"{name}: raised {type(exc).__name__}")
        else:
            bad.append(f"{name}: ACCEPTED")
        if eng.save_state() != before:
            bad.append(f"{name}: running battle changed")
    return bad


def test_refused_reset_leaves_the_running_battle_untouched():
    assert reset_refusal_failures() == []


def test_plant_mutate_then_validate_reset_is_caught(monkeypatch):
    """Regression plant: build and commit the battle, THEN validate (the shipped order
    committed tick, elixir, hands and towers before refusing a destroyed king)."""
    assert reset_refusal_failures() == [], "baseline must be green"

    def mutate_first(self, seed, setup):
        # A setup the builder cannot even read never reaches the commit.
        with contextlib.suppress(IndexError, KeyError):
            self._s = self._new_battle(seed, setup)
        mock_engine.validate_setup(self._arena, self.cards(), setup)

    monkeypatch.setattr(MockEngine, "reset", mutate_first)
    bad = reset_refusal_failures()
    assert "destroyed king: running battle changed" in bad, f"PLANT DID NOT LAND: {bad}"
    assert not any(m.endswith("ACCEPTED") for m in bad), bad


def test_setup_clamps_elixir_and_accepts_the_u32_tick_edge():
    eng = MockEngine()
    eng.reset(0, MatchSetup(decks=[DECK, DECK], elixir_milli=[-1, 99999]))
    assert [p.elixir_milli for p in eng.state().players] == [0, 10000]
    eng.reset(0, MatchSetup(decks=[DECK, DECK], elixir_milli=[1234, 1]))
    assert [p.elixir_milli for p in eng.state().players] == [1234, 1]
    eng.reset(0, MatchSetup(decks=[DECK, DECK], start_tick=2**32 - 1))
    assert eng.state().tick == 2**32 - 1
    eng.reset(0, MatchSetup(decks=[DECK, DECK], tower_hp=[[5, -3, 10], [1, 1, 1]]))
    s = eng.state()
    assert s.players[BLUE].tower_hp == [5, 0, 10]
    assert s.players[RED].crowns == 1


def test_plant_setup_validation_removed_is_caught(monkeypatch):
    """Regression plant: no setup validation -- start_tick -1, shuffle 7 and elixir []
    are accepted again."""
    assert reset_refusal_failures() == [], "baseline must be green"
    monkeypatch.setattr(mock_engine, "validate_setup", lambda arena, cards, setup: None)
    bad = reset_refusal_failures()
    for name in ("start_tick -1", "shuffle 7", "elixir []"):
        assert f"{name}: ACCEPTED" in bad, f"PLANT DID NOT LAND for {name}: {bad}"

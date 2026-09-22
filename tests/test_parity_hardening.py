"""MockEngine / RustEngine parity on the contract gaps a shared rule has to close.

WHY IT EXISTS. Each item below is a way the two engines can disagree while each one
looks correct on its own:
  * Setup-spawn LIST ORDER decided the per-team spawn ordinal (Rust ``team_seq``, Mock
    ``_Ent.seq``), a hidden tie-break input: a rotation-mirrored ``MatchSetup`` with
    Red's specs listed in reverse desynced Rust at tick 1 and Mock at tick 155.
    ``protocol.SpawnSpec`` now says list order does not matter; both engines spawn in
    ``spawn_order_key`` order.
  * ``MockEngine.reset`` accepted start_tick -1, shuffle 7 and elixir -1 where Rust
    refuses / refuses / clamps; ``MockEngine.load_state`` had no catalogue check.
  * ``RustEngine.reset`` let out-of-range integers reach PyO3 (OverflowError) and
    indexed a malformed tower_hp (IndexError) instead of raising the Protocol's
    ValueError; and Mock's
    canonical spawn key compared cards by catalogue ID where Rust compares by NAME,
    so same-team specs stacked on one point with different cards got different spawn
    ordinals in the two engines.

WHAT IT CANNOT CATCH:
  * A Rust-side regression of canonical spawn order has no Python plant: the order is
    decided inside the compiled core, and its plant lives there with it (cargo cfg
    plants). Here the Rust half is a gate whose red state on a build without the rule
    is the evidence it can fail (see the doc of
    ``test_mirrored_setup_with_reversed_red_spawn_list_stays_rotated``).
  * A refusal the Protocol rule does not know about and both engines skip in the same
    way: parity cannot see a rule neither side implements.

The Rust tests need the compiled extension (``maturin develop --release`` in the sibling
RoyaleSim checkout, README.md "Install") and are skipped with the import error as the reason when it
is absent -- the same convention as tests/test_rust_engine.py. A skip is not a pass.
"""

from __future__ import annotations

import pytest

from royalegym import mock_engine, rust_engine
from royalegym.mock_engine import MockEngine
from royalegym.protocol import (
    BLUE,
    RED,
    MatchSetup,
    ShuffleMode,
    SpawnSpec,
    to_own,
)
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, SymmetricRustEngine, core_available

# Both engines have every one of these (Rust: data/derived/cards.json names).
CAT = ("Knight", "Minions", "Valkyrie", "Giant", "Archer", "Goblins", "MiniPekka", "HogRider")
KNIGHT, GIANT, GOBLINS = 0, 3, 5
DECK = list(range(8))
ROTATION_TICKS = 300

needs_rust = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
ENGINES = ["mock", pytest.param("rust", marks=needs_rust)]


def make(engine_name: str, card_names=CAT):
    # the rotation gates in this file run the Rust engine under its frame-planned
    # search (see SymmetricRustEngine): the shipped search is the game's own and is
    # not seat-symmetric by design
    return (
        MockEngine(card_names=card_names)
        if engine_name == "mock"
        else SymmetricRustEngine(card_names=card_names)
    )


# --------------------------------------------------------------------------
# 1. Setup-spawn list order
# --------------------------------------------------------------------------


def rotation_asymmetry(eng) -> str | None:
    """First field in which the battle differs from its own 180-degree rotation.

    Written out here (not ``protocol.mirror_state``) so a plant on a shared frame
    helper cannot also bend the instrument. Every EntityState field but uid; every
    PlayerState field but team.
    """
    a = eng.arena()
    s = eng.state()

    def row(e, rotate):
        x, y, team = (a.width - e.x, a.height - e.y, 1 - e.team) if rotate else (e.x, e.y, e.team)
        return (
            team,
            e.kind,
            e.card_id,
            e.tower_slot,
            x,
            y,
            e.hp,
            e.max_hp,
            e.radius,
            e.flying,
            e.deploy_ticks,
        )

    plain = sorted(row(e, False) for e in s.entities)
    rotated = sorted(row(e, True) for e in s.entities)
    if plain != rotated:
        only = sorted(set(plain) - set(rotated))[:2]
        return f"tick {s.tick}: entities {only}"
    pb, pr = s.players[BLUE], s.players[RED]
    for f in (
        "elixir_milli",
        "hand",
        "next_card",
        "crowns",
        "tower_hp",
        "tower_max_hp",
        "king_active",
    ):
        if getattr(pb, f) != getattr(pr, f):
            return f"tick {s.tick}: player.{f} {getattr(pb, f)} vs {getattr(pr, f)}"
    if s.winner not in (-1, 2):
        return f"tick {s.tick}: winner {s.winner}"
    return None


def blue_specs(eng) -> dict[str, list[SpawnSpec]]:
    """Blue's half of each scenario. Every scenario stacks non-identical units of one
    team on ONE point, so the spawn ordinal is the only thing that separates them."""
    t = eng.arena().subtile
    hp = {c.card_id: c.hitpoints for c in eng.cards()}
    return {
        # The case the rule exists for: Rust desynced at tick 1, Mock at tick 155.
        "stacked Knights full + partial": [
            SpawnSpec(BLUE, KNIGHT, 9 * t // 2, 10 * t),
            SpawnSpec(BLUE, KNIGHT, 9 * t // 2, 10 * t, hp[KNIGHT] * 3 // 7),
        ],
        "stacked Knight + Giant": [
            SpawnSpec(BLUE, KNIGHT, 5 * t, 11 * t),
            SpawnSpec(BLUE, GIANT, 5 * t, 11 * t),
        ],
        "four Goblins stacked, varied hp": [
            SpawnSpec(BLUE, GOBLINS, 14 * t, 9 * t, h)
            for h in (-1, hp[GOBLINS] * 3 // 4, hp[GOBLINS] // 2, hp[GOBLINS] // 4)
        ],
        # A lone enemy Knight two tiles from a stack must pick one of the stacked pair:
        # a target choice decided by the stacked team's ordinals, within a few ticks.
        "lone Knight facing a stack across the centre": [
            SpawnSpec(BLUE, KNIGHT, 9 * t, 13 * t),
            SpawnSpec(BLUE, KNIGHT, 9 * t, 13 * t, hp[KNIGHT] // 3),
            SpawnSpec(BLUE, KNIGHT, 9 * t, 21 * t),
        ],
    }


def spawn_order_divergences(engine_name: str, ticks: int = ROTATION_TICKS) -> dict[str, str]:
    """Scenario -> first rotation asymmetry, with Red's specs rotated and listed in
    REVERSE. Also run with Red's list in Blue's order, as the control: a scenario
    that breaks under the control is not evidence about list order."""
    eng = make(engine_name)
    a = eng.arena()
    out = {}
    for label, blue in blue_specs(eng).items():
        red = [SpawnSpec(RED, s.card_id, a.width - s.x, a.height - s.y, s.hp) for s in blue]
        for order, spawns in (("control", blue + red), ("red reversed", blue + red[::-1])):
            eng.reset(
                1, MatchSetup(decks=[DECK, DECK], shuffle=ShuffleMode.MIRRORED, spawns=spawns)
            )
            why = rotation_asymmetry(eng)
            for _ in range(ticks):
                if why or eng.state().game_over:
                    break
                eng.step([], 1)
                why = rotation_asymmetry(eng)
            if why:
                out[f"{label} / {order}"] = why
    return out


@pytest.mark.parametrize("engine_name", ENGINES)
def test_mirrored_setup_with_reversed_red_spawn_list_stays_rotated(engine_name):
    """Rust half: red on a core build without the canonical ``team_seq``, and that
    red state is this gate's evidence that it can
    fail on Rust; green after the engine rebuild."""
    assert spawn_order_divergences(engine_name) == {}


def test_plant_mock_spawns_in_list_order_is_caught(monkeypatch):
    """Regression plant: every spec gets the same key, so the stable sort keeps LIST
    order -- the shipped defect. The reversed-list runs must break; the controls not."""
    assert spawn_order_divergences("mock") == {}, "baseline must be green"
    monkeypatch.setattr(mock_engine, "spawn_order_key", lambda arena, cards, spec: 0)
    assert mock_engine.spawn_order_key(None, None, None) == 0, "plant did not land"
    got = spawn_order_divergences("mock")
    assert got, "PLANT DID NOT LAND: list-order spawn ordinals went unseen"
    assert all(k.endswith("/ red reversed") for k in got), f"a control broke: {got}"


def permuted_spawn_hash_splits(engine_name: str) -> list[str]:
    """Any permutation of one spawn list must give the same state_hash, at reset and
    after 60 ticks (uids and spawn ordinals included, not just the rotation)."""
    eng = make(engine_name)
    a = eng.arena()
    specs = [s for blue in blue_specs(eng).values() for s in blue]
    specs += [SpawnSpec(RED, s.card_id, a.width - s.x, a.height - s.y, s.hp) for s in specs]
    orders = {
        "given": specs,
        "reversed": specs[::-1],
        "interleaved": [
            s for pair in zip(specs[::2], specs[1::2], strict=False) for s in pair[::-1]
        ],
        "red first": [s for s in specs if s.team == RED] + [s for s in specs if s.team == BLUE],
    }
    assert sorted(map(repr, orders["interleaved"])) == sorted(map(repr, specs)), "not a permutation"
    hashes = {}
    for name, spawns in orders.items():
        eng.reset(3, MatchSetup(decks=[DECK, DECK], spawns=spawns))
        h0 = eng.state_hash()
        eng.step([], 60)
        hashes[name] = (h0, eng.state_hash())
    return [
        f"{n}: {h} != given {hashes['given']}" for n, h in hashes.items() if h != hashes["given"]
    ]


@pytest.mark.parametrize("engine_name", ENGINES)
def test_state_hash_is_invariant_under_spawn_list_permutation(engine_name):
    """Rust half: red on a core build without the canonical spawn batch -- reversed,
    interleaved and red-first orders all split the hash -- and green with it. Its Rust
    plant is the engine's cfg ``setup_spawn_team_list_order``."""
    assert permuted_spawn_hash_splits(engine_name) == []


def test_plant_mock_list_order_changes_state_hash(monkeypatch):
    assert permuted_spawn_hash_splits("mock") == [], "baseline must be green"
    monkeypatch.setattr(mock_engine, "spawn_order_key", lambda arena, cards, spec: 0)
    assert permuted_spawn_hash_splits("mock"), "PLANT DID NOT LAND"


# --------------------------------------------------------------------------
# 2. MatchSetup refusals and clamps: Mock and Rust decide the same way
# --------------------------------------------------------------------------

SETUP_CASES = {
    "start_tick -1": dict(start_tick=-1),
    "start_tick 2**32": dict(start_tick=2**32),
    "start_tick 2**32-1": dict(start_tick=2**32 - 1),
    "shuffle 7": dict(shuffle=7),
    "shuffle -1": dict(shuffle=-1),
    "elixir -1 / 99999": dict(elixir_milli=[-1, 99999]),
    "elixir 1234 / 1": dict(elixir_milli=[1234, 1]),
    "elixir []": dict(elixir_milli=[]),
    "elixir three entries": dict(elixir_milli=[1, 2, 3]),
    "destroyed king": dict(tower_hp=[[0, 10, 10], [1, 1, 1]]),
    "negative princess hp": dict(tower_hp=[[5, -3, 10], [1, 1, 1]]),
    "tower_hp short row": dict(tower_hp=[[5, 3], [1, 1, 1]]),
    "tower_hp one team": dict(tower_hp=[[5, 3, 3]]),
    "tower hp beyond i32": dict(tower_hp=[[2**31, 3, 3], [1, 1, 1]]),
    "tower hp below i32": dict(tower_hp=[[5, 3, 3], [1, -(2**31) - 1, 1]]),
    "deck of 7": dict(decks=[DECK[:7], DECK]),
    "unknown card": dict(decks=[DECK, [99] * 8]),
}
# RustEngine.reset calls protocol.validate_setup first, so ValueError is the only
# refusal either engine may raise -- not the PyO3 OverflowError or the IndexError a
# bare adapter produces. Anything else is reported as ("raised", type).


def setup_outcome(eng, kw: dict) -> tuple:
    """('refused', reason, untouched?) or ('accepted', tick, elixir, alive towers, crowns,
    kings) or ('raised', exception type) for anything but ValueError."""
    kw = dict(kw)
    eng.reset(1, MatchSetup(decks=[DECK, DECK], shuffle=ShuffleMode.NONE))
    eng.step([], 50)
    before = eng.state_hash()
    try:
        eng.reset(2, MatchSetup(decks=kw.pop("decks", [DECK, DECK[::-1]]), **kw))
    except ValueError as exc:
        return ("refused", str(exc), eng.state_hash() == before)
    except Exception as exc:  # outside the engine's refusal contract: a finding, not a crash
        return ("raised", type(exc).__name__)
    try:
        s = eng.state()
    except IndexError as exc:  # a wrongly accepted setup may not even be readable
        return ("accepted", f"unreadable: {exc!r}")
    return (
        "accepted",
        s.tick,
        [p.elixir_milli for p in s.players],
        [[hp > 0 for hp in p.tower_hp] for p in s.players],
        [p.crowns for p in s.players],
        [p.king_active for p in s.players],
    )


def setup_parity_splits() -> list[str]:
    mock, rust = MockEngine(card_names=CAT), RustEngine(card_names=CAT)
    bad = []
    outcomes = {}
    for name, kw in SETUP_CASES.items():
        m, r = setup_outcome(mock, kw), setup_outcome(rust, kw)
        outcomes[name] = r[0]
        if m != r:
            bad.append(f"{name}: mock {m} rust {r}")
        elif m[0] == "refused" and not m[2]:
            bad.append(f"{name}: refused but the running battle changed on both")
    # Vacuity: the integer cases are all refusals, and a clamp case is accepted.
    for name in ("start_tick -1", "start_tick 2**32", "shuffle -1", "tower hp beyond i32"):
        if outcomes[name] != "refused":
            bad.append(f"vacuous: {name} was {outcomes[name]} on rust")
    if outcomes["elixir -1 / 99999"] != "accepted":
        bad.append("vacuous: no accepted (clamped) case")
    return bad


@needs_rust
def test_mock_and_rust_refuse_and_clamp_the_same_setups():
    assert setup_parity_splits() == []


@needs_rust
def test_plant_rust_adapter_without_validate_setup_raises_pyo3_errors(monkeypatch):
    """Regression plant: RustEngine.reset without its protocol.validate_setup call.
    The integer cases must then split from Mock as non-ValueError raises: OverflowError
    from PyO3 for the integers, IndexError for the malformed tower_hp."""
    assert setup_parity_splits() == [], "baseline must be green"
    monkeypatch.setattr(rust_engine, "validate_setup", lambda arena, cards, setup: None)
    assert rust_engine.validate_setup(None, None, None) is None, "plant did not land"
    bad = setup_parity_splits()
    for name, exc in (
        ("start_tick -1", "OverflowError"),
        ("start_tick 2**32", "OverflowError"),
        ("shuffle -1", "OverflowError"),
        ("tower hp beyond i32", "OverflowError"),
        ("tower_hp short row", "IndexError"),
    ):
        hit = [b for b in bad if b.startswith(f"{name}:")]
        assert hit, f"PLANT DID NOT LAND for {name}: {bad}"
        assert f"('raised', '{exc}')" in hit[0], f"PLANT DID NOT LAND for {name}: {hit[0]}"


@needs_rust
def test_plant_mock_setup_unvalidated_splits_from_rust(monkeypatch):
    """Regression plant: MockEngine without setup validation (the shipped behaviour)."""
    assert setup_parity_splits() == [], "baseline must be green"
    monkeypatch.setattr(mock_engine, "validate_setup", lambda arena, cards, setup: None)
    bad = setup_parity_splits()
    for name in ("start_tick -1", "shuffle 7", "elixir []"):
        assert any(b.startswith(f"{name}:") for b in bad), f"PLANT DID NOT LAND for {name}: {bad}"


# --------------------------------------------------------------------------
# 3. Snapshots across catalogues: Mock and Rust decide the same way
# --------------------------------------------------------------------------

SHARED8 = ("Knight", "Minions", "Cannon", "Giant", "Archer", "Goblins", "MiniPekka", "HogRider")
CATALOGUES = {
    "permuted superset": ("Valkyrie", *reversed(SHARED8), "Musketeer"),
    "one card": ("Knight",),
}


def catalogue_outcomes(engine_name: str) -> dict[str, tuple]:
    src = make(engine_name, SHARED8)
    src.reset(3, MatchSetup(decks=[DECK, DECK[::-1]], shuffle=ShuffleMode.NONE))
    src.step([], 40)
    blob = src.save_state()
    names = [c.name for c in src.cards()]
    want = [[names[c] for c in p.hand] for p in src.state().players]
    out = {}
    for label, cat in CATALOGUES.items():
        dst = make(engine_name, cat)
        dst.reset(9, MatchSetup(decks=[[0] * 8, [0] * 8]))
        before = dst.state_hash()
        try:
            dst.load_state(blob)
        except ValueError as exc:
            out[label] = ("refused", str(exc), dst.state_hash() == before)
            continue
        dn = [c.name for c in dst.cards()]
        try:
            s = dst.state()
            out[label] = ("loaded", [[dn[c] for c in p.hand] for p in s.players] == want, s.tick)
        except IndexError as exc:  # a wrongly accepted snapshot may not even be readable
            out[label] = ("loaded", f"unreadable: {exc!r}")
    return out


def catalogue_parity_splits() -> list[str]:
    m, r = catalogue_outcomes("mock"), catalogue_outcomes("rust")
    bad = [f"{k}: mock {m[k]} rust {r[k]}" for k in CATALOGUES if m[k] != r[k]]
    if r["permuted superset"][:2] != ("loaded", True):
        bad.append(f"rust did not load the superset by name: {r['permuted superset']}")
    if r["one card"][0] != "refused":
        bad.append(f"rust accepted a one-card catalogue: {r['one card']}")
    return bad


@needs_rust
def test_mock_and_rust_load_snapshots_across_catalogues_the_same_way():
    assert catalogue_parity_splits() == []


@needs_rust
def test_plant_mock_load_without_catalogue_rule_splits_from_rust(monkeypatch):
    assert catalogue_parity_splits() == [], "baseline must be green"
    monkeypatch.setattr(MockEngine, "_adopt_snapshot", lambda self, s: s)
    bad = catalogue_parity_splits()
    assert any(b.startswith("one card:") for b in bad), f"PLANT DID NOT LAND: {bad}"


# --------------------------------------------------------------------------
# 4. Canonical spawn order compares cards by NAME in both engines
# --------------------------------------------------------------------------

# Catalogue ids ascending = card names DESCENDING, so an id-ordered key and a
# name-ordered key rank every pair of these cards oppositely.
NAME_REVERSED = (
    "Valkyrie",
    "Musketeer",
    "MiniPekka",
    "Knight",
    "HogRider",
    "Goblins",
    "Giant",
    "Archer",
)


def spawn_ordinals(engine_name: str) -> list[list[tuple]]:
    """Per team, the setup units in SPAWN-ORDINAL order (uid order within a team; Rust
    uid = 2 * team_seq + team, Mock uid a global counter), as (name, own x, own y) -- hp is
    left out because the engines run cards at different levels (rust_engine.py).

    Setup: each team stacks four different cards on one own-frame point, and two more
    on a second point differing only in card, listed in a scrambled order.
    """
    eng = make(engine_name, NAME_REVERSED)
    a = eng.arena()
    t_ = a.subtile
    ids = {c.name: c.card_id for c in eng.cards()}
    assert sorted(ids, key=ids.get) == sorted(ids, reverse=True), "fixture: ids must reverse names"
    blue = [
        SpawnSpec(BLUE, ids[n], 5 * t_, 11 * t_) for n in ("Knight", "Archer", "Valkyrie", "Giant")
    ]
    blue += [SpawnSpec(BLUE, ids[n], 13 * t_, 9 * t_) for n in ("MiniPekka", "Musketeer")]
    red = [SpawnSpec(RED, s.card_id, a.width - s.x, a.height - s.y, s.hp) for s in blue[::-1]]
    eng.reset(4, MatchSetup(decks=[DECK, DECK], shuffle=ShuffleMode.NONE, spawns=blue + red))
    names = [c.name for c in eng.cards()]
    out = []
    for team in (BLUE, RED):
        units = sorted(
            (e for e in eng.state().entities if e.team == team and e.card_id >= 0),
            key=lambda e: e.uid,
        )
        out.append([(names[e.card_id], *to_own(a, team, e.x, e.y)) for e in units])
    return out


def spawn_ordinal_splits() -> list[str]:
    m, r = spawn_ordinals("mock"), spawn_ordinals("rust")
    bad = [
        f"team {team}: mock {m[team]} != rust {r[team]}"
        for team in (BLUE, RED)
        if m[team] != r[team]
    ]
    for team in (BLUE, RED):
        if len(r[team]) != 6:
            bad.append(f"vacuous: team {team} spawned {len(r[team])} units, want 6")
        if r[team] != r[BLUE]:
            bad.append(f"rust seats differ: {r[team]} vs {r[BLUE]}")
    return bad


@needs_rust
def test_same_point_specs_with_different_cards_get_the_same_spawn_ordinals_on_both_engines():
    """Both engines number a stack of different cards on one own-frame point in the same
    order (card NAME), both seats. Ordering by card id instead makes Mock spawn the
    stack Valkyrie, Knight, Giant, Archer where Rust spawns Archer, Giant, Knight,
    Valkyrie (see the plant)."""
    assert spawn_ordinal_splits() == []


@needs_rust
def test_plant_mock_spawn_key_by_card_id_splits_from_rust(monkeypatch):
    """Regression plant: a Mock key of (team, own y, own x, card_id, hp)."""
    assert spawn_ordinal_splits() == [], "baseline must be green"

    def by_id(arena, cards, spec):
        ox, oy = to_own(arena, spec.team, spec.x, spec.y)
        hp = cards[spec.card_id].hitpoints if spec.hp == -1 else spec.hp
        return spec.team, oy, ox, spec.card_id, hp

    monkeypatch.setattr(mock_engine, "spawn_order_key", by_id)
    assert mock_engine.spawn_order_key is by_id, "plant did not land"
    bad = spawn_ordinal_splits()
    assert any(b.startswith("team 0: mock") for b in bad), f"PLANT DID NOT LAND: {bad}"
    assert any(b.startswith("team 1: mock") for b in bad), f"PLANT DID NOT LAND for Red: {bad}"

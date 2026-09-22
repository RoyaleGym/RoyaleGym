"""The engine contract, and the invariants the mock must share with the Rust engine."""

from __future__ import annotations

import ast
import inspect
import json
from fractions import Fraction
from pathlib import Path

import msgspec
import pytest

from royalegym import mock_engine, protocol
from royalegym.mock_engine import MockEngine, Pcg32
from royalegym.protocol import (
    BIT_LANE_LEFT,
    BIT_LANE_RIGHT,
    BLUE,
    RED,
    Arena,
    Calibration,
    CardInfo,
    DeployCommand,
    Engine,
    MatchSetup,
    ShuffleMode,
    data_dir,
)

PROTOCOL_METHODS = [
    "cards",
    "arena",
    "rules",
    "reset",
    "check_deploy",
    "step",
    "state",
    "save_state",
    "load_state",
    "state_hash",
]


def signature_mismatches(cls: type) -> list[str]:
    """Compare a class's method signatures with protocol.Engine, by parameter name."""
    out = []
    for name in PROTOCOL_METHODS:
        if not hasattr(cls, name):
            out.append(f"missing {name}")
            continue
        want = list(inspect.signature(getattr(Engine, name)).parameters)
        got = list(inspect.signature(getattr(cls, name)).parameters)
        if want != got:
            out.append(f"{name}: {got} != {want}")
    return out


def test_protocol_method_list_is_complete():
    declared = {n for n, v in vars(Engine).items() if callable(v) and not n.startswith("_")}
    assert declared == set(PROTOCOL_METHODS)


def test_mock_engine_satisfies_protocol():
    eng = MockEngine()
    assert isinstance(eng, Engine)
    assert signature_mismatches(MockEngine) == []


def test_plant_protocol_checker_sees_missing_and_misnamed_methods():
    class Broken(MockEngine):
        def step(self, cmds, n):  # type: ignore[override]
            return []

    assert any(m.startswith("step") for m in signature_mismatches(Broken))

    class Missing:
        pass

    assert len(signature_mismatches(Missing)) == len(PROTOCOL_METHODS)
    assert not isinstance(Missing(), Engine)


# --- invariant 1: no floats in the mock simulation --------------------------------


def float_violations(source: str) -> list[str]:
    """Float literals, true division, float() calls and math float functions."""
    bad = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            bad.append(f"line {node.lineno}: float literal {node.value}")
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            # `path / "file.csv"` is a pathlib join, not arithmetic. The first run of
            # this checker flagged exactly those, so a string right operand is exempt.
            if not (isinstance(node.right, ast.Constant) and isinstance(node.right.value, str)):
                bad.append(f"line {node.lineno}: true division")
        elif isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Div):
            bad.append(f"line {node.lineno}: true division")
        elif isinstance(node, ast.Name) and node.id == "float":
            bad.append(f"line {node.lineno}: float()")
        elif isinstance(node, ast.Attribute) and node.attr in {
            "sqrt",
            "hypot",
            "atan2",
            "cos",
            "sin",
        }:
            bad.append(f"line {node.lineno}: math.{node.attr}")
    return bad


def test_mock_engine_source_has_no_floats():
    src = Path(mock_engine.__file__).read_text(encoding="utf-8")
    assert float_violations(src) == []


def test_plant_float_checker_fires():
    planted = (
        "def f(a, b):\n    x = a / b\n    y = 1.5\n    z = float(a)\n    return math.sqrt(x)\n"
    )
    assert len(float_violations(planted)) == 4


# --- invariant 3: constants are read from calibration.json -----------------------


def _fill_ticks(cal: Calibration) -> int:
    eng = MockEngine(calibration=cal)
    eng.reset(
        0, MatchSetup(decks=[list(range(8))] * 2, shuffle=ShuffleMode.NONE, elixir_milli=[0, 0])
    )
    ticks = 0
    while eng.state().players[BLUE].elixir_milli < 10_000:
        eng.step([], 1)
        ticks += 1
    return ticks


def test_elixir_regen_reads_tick_ms_and_regen_from_calibration():
    cal = Calibration.load()
    base = _fill_ticks(cal)
    assert base == cal.int("match.MANA_REGEN_MS_1X") // cal.int("time.TICK_MS")
    # Planted calibration changes must change behaviour -- a baked constant would not.
    assert _fill_ticks(cal.with_override("time.TICK_MS", 25)) == 2 * base
    assert _fill_ticks(cal.with_override("match.MANA_REGEN_MS_1X", 14000)) == base // 2


def test_speed_multiplier_reads_calibration():
    cal = Calibration.load()
    a = MockEngine(calibration=cal)
    b = MockEngine(calibration=cal.with_override("time.SPEED_TO_SUBTILES_PER_TICK", 18))
    knight_a = next(u for u in a._units if u.name == "Knight")
    knight_b = next(u for u in b._units if u.name == "Knight")
    assert knight_a.speed * 18 == knight_b.speed * cal.int("time.SPEED_TO_SUBTILES_PER_TICK")


def test_calibration_missing_key_raises_instead_of_defaulting():
    cal = Calibration.load()
    with pytest.raises(KeyError):
        cal.value("time.NOT_A_REAL_CONSTANT")


def test_unimplemented_footprint_model_is_refused():
    cal = Calibration.load().with_override("collision.BUILDING_FOOTPRINT_MODEL", "static_bitmap")
    with pytest.raises(NotImplementedError):
        MockEngine(calibration=cal)


# --- arena --------------------------------------------------------------------------


def test_arena_is_invariant_under_the_seat_rotation():
    """The flip-exactness guarantees rest on this property of the shipped tilemap."""
    a = Arena.load(Calibration.load())

    def swap(v: int) -> int:
        lanes = v & (BIT_LANE_LEFT | BIT_LANE_RIGHT)
        rest = v & ~(BIT_LANE_LEFT | BIT_LANE_RIGHT)
        swapped = (BIT_LANE_RIGHT if lanes & BIT_LANE_LEFT else 0) | (
            BIT_LANE_LEFT if lanes & BIT_LANE_RIGHT else 0
        )
        return rest | swapped

    for hy in range(a.hy):
        for hx in range(a.hx):
            assert a.grid[hy][hx] == swap(a.grid[a.hy - 1 - hy][a.hx - 1 - hx])
    bk, rk = a.king_centers
    assert (a.width - bk[0], a.height - bk[1]) == rk
    for slot in range(2):
        bx, by = a.princess_centers[BLUE][slot]
        assert (a.width - bx, a.height - by) == a.princess_centers[RED][slot]


def test_arena_positions_are_integers_matching_the_json_convenience_fields():
    a = Arena.load(Calibration.load())
    text = (data_dir() / "derived" / "arena.json").read_text(encoding="utf-8")
    # parse_float=Fraction keeps this test float-free while still cross-checking the
    # integer derivation against the independent float fields extract_arena wrote.
    raw = json.loads(text, parse_float=Fraction)
    kings = sorted(raw["king_blocks"], key=lambda k: k["half_rows"][0])
    for (cx, cy), k in zip(a.king_centers, kings, strict=True):
        assert isinstance(cx, int)
        assert isinstance(cy, int)
        assert Fraction(cx, a.subtile) == k["center"][0]
        assert Fraction(cy, a.subtile) == k["center"][1]
    assert [Fraction(x, a.subtile) for x in a.bridge_centers_x()] == [
        b["center_x"] for b in raw["bridges"]
    ]


# --- rng ------------------------------------------------------------------------------


def test_pcg32_is_seed_reproducible_and_below_is_bounded():
    a, b = Pcg32.new(12345), Pcg32.new(12345)
    assert [a.next_u32() for _ in range(64)] == [b.next_u32() for _ in range(64)]
    c, d = Pcg32.new(1), Pcg32.new(2)
    assert [c.next_u32() for _ in range(4)] != [d.next_u32() for _ in range(4)]
    r = Pcg32.new(7)
    seen = [0] * 6
    for _ in range(6000):
        seen[r.below(6)] += 1
    assert all(900 < s < 1100 for s in seen), seen


def test_check_deploy_is_pure():
    eng = MockEngine()
    eng.reset(3, MatchSetup(decks=[list(range(8))] * 2))
    before = eng.state_hash()
    a = eng.arena()
    for x in range(a.subtile // 2, a.width, a.subtile):
        eng.check_deploy(DeployCommand(BLUE, 0, x, a.subtile * 5 + a.subtile // 2))
    assert eng.state_hash() == before


# --- MatchSetup rules shared by every engine -----------------------------------------


def spawn_key_rotation_breaks(key=protocol.spawn_order_key) -> list[str]:
    """The canonical spawn order must rank a rotated Red spec list exactly as it ranks
    Blue's: own frame, so both seats' spawn ordinals name the same own-frame unit."""
    eng = MockEngine()
    a, cards = eng.arena(), eng.cards()
    t = a.subtile
    blue = [
        protocol.SpawnSpec(BLUE, cid, x * t // 2, y * t // 2, hp)
        for cid, x, y, hp in [
            (0, 9, 20, -1),
            (0, 9, 20, 300),
            (0, 11, 20, -1),
            (3, 9, 20, -1),
            (2, 30, 12, 40),
            (2, 4, 12, 40),
            (2, 4, 25, 40),
            (0, 9, 20, 660),  # last: equal key to the first
        ]
    ]
    red = [protocol.SpawnSpec(RED, s.card_id, a.width - s.x, a.height - s.y, s.hp) for s in blue]
    rank_b = sorted(range(len(blue)), key=lambda i: key(a, cards, blue[i]))
    rank_r = sorted(range(len(red)), key=lambda i: key(a, cards, red[i]))
    keys_b = [key(a, cards, blue[i])[1:] for i in rank_b]
    keys_r = [key(a, cards, red[i])[1:] for i in rank_r]
    bad = []
    # Vacuity: exactly one designed tie (Knight hp -1 resolves to its 660 hitpoints).
    if len(set(keys_b)) != len(blue) - 1:
        bad.append(f"fixture has {len(blue) - len(set(keys_b))} equal keys, designed 1")
    if keys_b != keys_r:
        bad.append(f"blue order {rank_b} != red order {rank_r}")
    return bad


def test_spawn_order_key_ranks_the_rotated_list_identically():
    assert spawn_key_rotation_breaks() == []


def test_plant_engine_frame_spawn_key_is_caught():
    """Plant: rank by ENGINE-frame (y, x) -- Red's list comes out reversed."""

    def engine_frame(arena, cards, spec):
        hp = cards[spec.card_id].hitpoints if spec.hp == -1 else spec.hp
        return spec.team, spec.y, spec.x, spec.card_id, hp

    assert spawn_key_rotation_breaks(engine_frame), "PLANT DID NOT LAND"


def test_setup_violation_is_a_pure_function_of_its_arguments():
    """Engines call it BEFORE mutating; it must need no engine state and change nothing."""
    eng = MockEngine()
    setup = MatchSetup(decks=[list(range(8))] * 2, start_tick=-1, elixir_milli=[5, 5])
    snapshot = msgspec.json.encode(setup)
    why = protocol.setup_violation(eng.arena(), eng.cards(), setup)
    assert why == "start_tick -1 outside [0, 4294967295]"
    assert msgspec.json.encode(setup) == snapshot
    ok = MatchSetup(decks=[list(range(8))] * 2, elixir_milli=[-1, 10**9])  # clamped, not refused
    assert protocol.setup_violation(eng.arena(), eng.cards(), ok) is None


# --- which card table is in front of you -------------------------------------
#
# The derived card table is GENERATED from a raw client pack, and only the oldest
# pack is tracked, so which one a checkout has is a property of the MACHINE. Two
# engines reading different packs disagree about cards while both being correct,
# and a cross-engine comparison then measures the data rather than the engines.
# These are pure functions, so they run without the compiled engine.


def _card(name: str, **over) -> CardInfo:
    base = {
        "card_id": 0,
        "name": name,
        "elixir": 3,
        "placement": 0,
        "count": 1,
        "radius": 500,
        "flying": False,
        "hitpoints": 700,
    }
    return CardInfo(**{**base, **over})


def test_catalogue_vintage_split_is_silent_when_the_tables_agree():
    from royalegym.rust_engine import catalogue_vintage_split

    same = [_card("Knight"), _card("Goblins", count=3)]
    assert catalogue_vintage_split(same, list(same)) is None
    # hitpoints is the engine's card LEVEL, a known mechanics difference, not data
    levelled = [_card("Knight", hitpoints=2000), _card("Goblins", count=3, hitpoints=400)]
    assert catalogue_vintage_split(levelled, same) is None


def test_catalogue_vintage_split_names_the_field_the_card_and_both_vintages():
    from royalegym.rust_engine import catalogue_vintage_split

    mock = [_card("Knight"), _card("Goblins", count=3)]
    rust = [_card("Knight"), _card("Goblins", count=4)]
    why = catalogue_vintage_split(rust, mock)
    assert why is not None
    assert "Goblins 4/3" in why
    assert "count" in why
    assert "A SKIP IS NOT A PASS" in why
    assert mock_engine.RAW_CARD_PACK in why
    assert protocol.derived_cards_vintage() in why
    # a catalogue of a different SIZE is the same split, said plainly
    short = catalogue_vintage_split(rust, mock[:1])
    assert short is not None
    assert "2/1 cards" in short


def test_derived_cards_vintage_reads_the_provenance_the_extractor_wrote(tmp_path):
    assert protocol.derived_cards_vintage() != ""
    written = tmp_path / "cards.json"
    written.write_text(json.dumps({"provenance": {"vintage": "made up 1999"}}), encoding="utf-8")
    assert protocol.derived_cards_vintage(written) == "made up 1999"
    written.write_text(json.dumps({"cards": []}), encoding="utf-8")
    assert protocol.derived_cards_vintage(written) == "unknown"
    assert protocol.derived_cards_vintage(tmp_path / "absent.json") == "unknown"

"""Observation builders: BattleState -> what the policy sees.

PERSPECTIVE
    Every observation is in the ACTING player's own frame (see protocol.py):
    Red's board is rotated 180 degrees and its towers/crowns/elixir are "own",
    Blue's are "enemy". A mirrored battle therefore yields a bit-identical
    observation for the other seat, which tests assert, so one policy plays both.

FAIR INFORMATION, AND THE ``Reveal``
    Everything a builder writes by default is something a player watching the
    match could write down: the board, their own hand and cycle, the clock, and a
    COUNT of the opponent's elixir kept from the plays they saw and the
    regeneration rate everyone knows (``MatchMemory``). Nothing default-built is
    read out of the half of the state a player cannot see.

    ``Reveal`` opens that half, one field at a time, for curriculum, distillation
    and debugging. An enabled field ADDS its channels and vector slots; it is
    never present-but-zero, so a fair observation and a cheating one do not even
    have the same width, and a checkpoint cannot quietly be trained on one and
    evaluated on the other. ``ClashParallelEnv.config()`` records the Reveal for
    exactly that reason. The one field that changes a slot instead of adding one
    is ``enemy_elixir``: the fair slot already holds the counted value, and the
    reveal swaps in the value read from the state. The two agree on a played-out
    battle (tests/test_env_obs.py); a disagreement is a bug in the counter.

Floats appear here and only here-onwards (policy input). They are computed from
integer state by the same operations for both seats, so the flip is exact.

NO FLOAT MAY DEPEND ON ENTITY LIST ORDER
    ``state.entities`` order is engine-private and is NOT seat-canonical: the Rust
    engine lists entities by storage slot, and slots are reused, so an exactly
    rotation-mirrored battle can list Blue's twins in a different order from Red's.
    Float addition is not associative, so any per-entity float accumulation (or any
    sort whose key omits a feature it then writes) makes obs[blue] != obs[red]. The
    rule: accumulate INTEGERS, convert once; sort rows by every input they are
    built from. Accumulating ``sp[hp channel] += e.hp / HP_SCALE`` in list order
    instead costs a 1 ulp seat mismatch on mirrored states (channels own_hp and
    enemy_hp), measured on both hand-built boards and boards the Rust engine
    played out. ``MatchMemory`` obeys the same rule: integer fine elixir units and
    integer counts, converted once, per seat.

MOST OF AN OBSERVATION IS STRUCTURALLY CONSTANT, AND THAT DECIDES HOW TO READ IT
    Across eight boards differing in a unit's position, a destroyed tower and the
    elixir, 51 of the spatial observation's 11 749 numbers differ. 0.4%. The rest is
    the arena's static planes, the standing towers and an unchanged hand. So the
    greatest pairwise cosine between two observations is about 0.9998 with nothing
    whatever wrong, and a collapse detector that puts a threshold under a cosine is
    measuring how much of the tensor is static rather than whether it discriminates.
    ``measure_variability`` reports both numbers for a given builder and set of
    states, so a consumer can calibrate against its own configuration instead of
    against this paragraph. On the cells that CAN move, those same eight boards sit
    at 0.984.

PLACEMENT LEGALITY IS THE MASK'S JOB, NOT A CHANNEL'S
    The action space is ``Discrete(2305)`` = no-op + 4 hand slots x 18 x 32 tiles,
    so the mask ALREADY states per-slot, per-tile legality exactly. ``own_troop_zone``
    and ``own_building_zone`` restated a coarser version of it and cost two of the
    three ``PlacementOracle.point_grid`` calls the observation made per seat per
    step -- 6.66 calls per ``env.step`` before and 2.66 after, both seats, and
    ``env.step`` itself 766 -> 953 per second on the Rust engine; the mask itself is
    handed to the policy twice instead -- flat as ``action_mask`` for the head, and
    as ``mask_planes`` [4, 32, 18] (the mask minus the no-op, reshaped) for a
    convolutional trunk. ``enemy_troop_zone`` stays: it is about the OPPONENT's
    options and is in no mask.

SPELLS AND STATUS EFFECTS
    What the engine exposes, and nothing it does not: live spell objects
    (``BattleState.spells``: team, card, motion, centre, aim point, flight delay, roll
    progress, hits) and per-entity ``stun_ticks`` / ``knockback_ticks``. Spell
    objects are visible information in the live game (a Fireball in the air, a Log
    rolling), so both seats see both teams' spells. The one exception is where an
    ENEMY spell is aimed: the player chose where to throw their own, but reading
    the opponent's landing point out of a projectile still in flight is a reveal
    (``Reveal.enemy_spell_aim``); ``own_spell_aim`` is unconditional. The spatial
    builder rasterises spells in the ``own_spells``..``enemy_stunned`` channels; the
    entity-list builder adds status features to each entity row and a separate
    ``spells`` array. Not exposed, because the engine does not export it: a spell's
    hit radius or damage (use the card one-hot), buffs, a unit's current target.
    MockEngine resolves spells within a tick and has no status effects, so on it
    those channels and the ``spells`` array are always zero (mock_engine.py WHAT IT
    IS NOT); tests/test_rust_engine.py checks they carry information on RustEngine
    battles.
    KNOCKBACK: ``knockback_ticks`` is a per-entity feature only, never a spatial
    channel. Under the shipped calibration knockback.DURATION_MS = 0 the push is
    instant and the timer is 0 between ticks, so a channel for it would be constant
    zero and unverifiable by any coverage guard.

    STUN IS ALMOST INVISIBLE AT THE DEFAULT DECISION RATE, and that is a measured
    fact rather than a guess. A Zap's stun is 10 ticks; a decision is 10 ticks
    (500 ms at TICK_MS 50). Measured on the Rust engine, a Zap cast at tick k of a
    decision leaves stun_ticks = k (1, 4, 6, 10 for k = 0, 3, 5, 9) at the ONE
    observation that follows, and 0 at every observation after it. So
    ``own_stunned`` / ``enemy_stunned`` fire for at most one step per Zap, and the
    per-entity ``stun_ticks / 100`` feature reads between 0.01 and 0.10 for that one
    step. A policy at 500 ms decisions can barely perceive a stun.
    The features are left as "is stunned NOW", which is what the engine reports and
    what the seat-flip and cell-by-cell tests can check exactly. "Was stunned since
    the last observation" would be the feature a policy could actually use, but it
    is a different thing -- it depends on the decision rate, not only on the state --
    so it is recorded here as an open question rather than taken quietly.
    (Measured with Simulator 1 on the 2026-09-21 build.)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, NamedTuple

import numpy as np
from gymnasium import spaces

from .action import ActionParser, PlacementOracle
from .protocol import (
    DECK_SIZE,
    EMPTY_CARD,
    HAND_SIZE,
    RED,
    TEAMS,
    Arena,
    BattleState,
    Calibration,
    CardInfo,
    ElixirLaw,
    Engine,
    EntityKind,
    EntityState,
    Placement,
    SpellMotion,
    TowerSlot,
    default_calibration,
    to_own,
)

# Observation scaling. Presentation constants for the network, not physics: they
# only decide where a feature saturates, so they are named here rather than read
# from calibration.json, which holds the game's own numbers.
HP_SCALE = 1000.0
SPATIAL_CLIP = 64.0
# Observation keys that are the action mask in one shape or another. They are
# legality rather than representation, so ``measure_variability`` leaves them out.
MASK_OBS_KEYS = ("action_mask", "mask_planes")
LEAK_SCALE = 20.0  # elixir leaked at which ``own_elixir_leaked`` saturates
PLAY_GAP_TICKS = 600.0  # ticks since own last play at which that feature saturates
PLAYS_SCALE = 40.0  # enemy plays at which ``enemy_plays`` saturates


@dataclass(frozen=True)
class Reveal:
    """Which halves of the hidden state a builder is allowed to read.

    All-False (the default) is the fair observation. Every True field ADDS
    channels or vector slots (see the module doc), except ``enemy_elixir``, which
    swaps the source of the slot that already holds the counted value.
    """

    enemy_elixir: bool = False
    enemy_hand: bool = False
    enemy_next_card: bool = False
    enemy_deck: bool = False
    enemy_spell_aim: bool = False

    @property
    def any_enabled(self) -> bool:
        return any(asdict(self).values())

    def as_dict(self) -> dict[str, bool]:
        """JSON-able, for ``ClashParallelEnv.config()`` and checkpoint metadata."""
        return {k: bool(v) for k, v in asdict(self).items()}


# ---------------------------------------------------------------------------
# The flat vector, shared by both builders
# ---------------------------------------------------------------------------


class VectorField(NamedTuple):
    """One run of slots in the flat vector. ``fair`` is False for a reveal's slots."""

    key: str
    size: int
    doc: str
    fair: bool = True


def vector_layout(num_cards: int, reveal: Reveal | None = None) -> list[VectorField]:
    """The flat vector, field by field, in order. THE definition of the layout.

    Fair fields come first and in a fixed order, so enabling a reveal never moves
    a fair feature: the slice holding "own elixir" is the same in a fair run and in
    a cheating one. tests/test_env_obs.py asserts the resulting width rather than
    trusting arithmetic done here.
    """
    rev = reveal or Reveal()
    n = num_cards
    onehot = n + 1  # last index = empty slot / none
    enemy_elixir_doc = (
        "read from the state (Reveal.enemy_elixir)"
        if rev.enemy_elixir
        else "counted from the plays seen and the regeneration rate"
    )
    fields = [
        VectorField("own_elixir", 1, "own elixir / MAX_MANA"),
        VectorField("enemy_elixir", 1, f"enemy elixir / MAX_MANA, {enemy_elixir_doc}"),
        VectorField("own_hand_cards", HAND_SIZE * onehot, "hand slot card one-hot [4 x (n+1)]"),
        VectorField("own_hand_cost", HAND_SIZE, "hand slot elixir cost / MAX_MANA [4]"),
        VectorField("own_hand_affordable", HAND_SIZE, "hand slot affordable now [4]"),
        VectorField("own_next_card", onehot, "cycle position 5 (next card) one-hot [n+1]"),
        VectorField(
            "own_cycle_6_8",
            3 * onehot,
            "cycle positions 6, 7, 8 one-hot [3 x (n+1)]; index n = not deduced yet",
        ),
        VectorField("own_deck", n, "own deck multi-hot [n], as deduced from the cycle so far"),
        VectorField("own_last_card", onehot, "last card own played, one-hot [n+1]; n = none yet"),
        VectorField(
            "own_ticks_since_play",
            1,
            f"ticks since own last play / {PLAY_GAP_TICKS:.0f}, clipped",
        ),
        VectorField(
            "own_elixir_leaked", 1, f"own elixir leaked so far / {LEAK_SCALE:.0f}, clipped"
        ),
        VectorField("enemy_cards_seen", n, "enemy cards played at least once this match [n]"),
        VectorField(
            "enemy_possible_hand",
            n,
            "enemy cards that could be in hand now [n], by the 8-card cycle rule",
        ),
        VectorField(
            "enemy_plays", 1, f"enemy cards played this match / {PLAYS_SCALE:.0f}, clipped"
        ),
        VectorField("own_tower_hp", 3, "own tower hp / max [king, left, right]"),
        VectorField("enemy_tower_hp", 3, "enemy tower hp / max [king, left, right], own-frame"),
        VectorField("crowns", 2, "own crowns / 3, enemy crowns / 3"),
        VectorField("king_active", 2, "own king active, enemy king active"),
        VectorField(
            "clock",
            3,
            "regulation left / regulation, in overtime, overtime left / overtime",
        ),
        VectorField("elixir_rate", 2, "elixir rate one-hot [1x, 2x]"),
    ]
    if rev.enemy_hand:
        fields.append(
            VectorField(
                "enemy_hand_cards",
                HAND_SIZE * onehot,
                "enemy hand one-hot [4 x (n+1)] (Reveal.enemy_hand)",
                fair=False,
            )
        )
    if rev.enemy_next_card:
        fields.append(
            VectorField(
                "enemy_next_card",
                onehot,
                "enemy next card one-hot [n+1] (Reveal.enemy_next_card)",
                fair=False,
            )
        )
    if rev.enemy_deck:
        fields.append(
            VectorField(
                "enemy_deck", n, "enemy deck multi-hot [n] (Reveal.enemy_deck)", fair=False
            )
        )
    return fields


def vector_fields(num_cards: int, reveal: Reveal | None = None) -> list[tuple[str, int]]:
    """``(description, size)`` per field, in order. The suite checks the sizes add up."""
    return [(f"{f.key}: {f.doc}", f.size) for f in vector_layout(num_cards, reveal)]


def vector_offsets(num_cards: int, reveal: Reveal | None = None) -> dict[str, slice]:
    """``key -> slice`` into the flat vector, so nothing has to count slots by hand."""
    out: dict[str, slice] = {}
    at = 0
    for f in vector_layout(num_cards, reveal):
        out[f.key] = slice(at, at + f.size)
        at += f.size
    return out


# ---------------------------------------------------------------------------
# What a player watching the match remembers
# ---------------------------------------------------------------------------


def cards_that_left(before: Sequence[int], after: Sequence[int]) -> list[int]:
    """Cards played, read from hand slots changing. Slot order: see ``MatchMemory``."""
    if len(before) != len(after):
        return []
    return [b for b, a in zip(before, after, strict=True) if b != a and b != EMPTY_CARD]


class MatchMemory:
    """One seat's memory of the match so far, kept from the states it is shown.

    Everything here is derivable by a human watching: which cards the opponent has
    played and when, where their cycle therefore is, how much elixir they must
    have, and the player's own cycle, last play and wasted regeneration.

    THE ENEMY ELIXIR COUNT is the reason this class exists. It is seeded once, at
    the start of the match, from the opponent's bar -- the starting amount is
    public, and a curriculum start that hands one side extra elixir is equally
    public -- and never read again. From there it is the engine's own arithmetic
    (``protocol.ElixirLaw``): pay for each play seen, then regenerate over the
    ticks that passed, clamped at the cap. Integer fine units throughout, so the
    count is bit-exact against the bar the engine keeps rather than approximately
    right (tests/test_env_obs.py plays a battle out and compares the two every
    step, in both directions).

    HOW A PLAY IS SEEN. A play is a public event -- a unit appears, a spell is
    cast -- and it is read here from the player's hand changing between two
    observed states, which names the same event and names the CARD exactly. Two
    consequences, both deliberate: a deck holding the same card twice can hide a
    play (a real deck is eight distinct cards, and ``random_deck`` draws without
    replacement), and when states are sampled with GAPS rather than every decision,
    several plays inside one gap are attributed in hand-slot order -- own-frame, so
    still identical for the two seats of a mirrored battle, but not necessarily the
    order they happened in.

    AND IT SAYS WHEN IT CANNOT BE EXACT. ``exact`` goes False, and stays False for
    the match, the moment the count disagrees with the bar it is modelling -- on
    EITHER side. Two things make that happen: a play was missed (a deck that repeats
    a card can hide one, because a play that swaps a card for itself changes no hand
    slot), or the engine's elixir law is not the one in calibration.json.

    The enemy half of that check is the ONE place this class looks at the
    opponent's bar, and it does exactly one thing with it: set a boolean. The value
    is never read into ``foe_fine`` and never reaches an observation -- a wrong
    count is left wrong rather than quietly repaired, because repairing it is the
    cheat. Checking only the own bar was not enough and the gap was not theoretical:
    with a normal deck on one side and a repeating deck on the other, the own bar
    stays perfect while the enemy count drifts, and ``exact`` stayed True while the
    number it certified was wrong. The own bar IS resynced, since it is visible
    anyway, so it stops drifting further.

    IDEMPOTENT BY TICK. ``observe`` advances nothing when the state's tick has not
    moved, so building the same state twice -- which the seat-flip and list-order
    gates do dozens of times -- cannot drift. A tick that moves BACKWARDS re-seeds,
    because in a running battle the clock does not go back.

    THAT BACKWARDS-TICK RULE IS A BACKSTOP, NOT THE GUARANTEE. ``BattleState``
    carries no episode identity, so nothing in it distinguishes a new battle that
    starts at a HIGHER tick -- a Snapshot resume, or a MatchSetup with
    ``start_tick`` past the last episode's end -- from the same battle continuing.
    The guarantee is ``ObsBuilder.reset``, which the env calls on every episode and
    which forgets everything. A caller that drives builders itself must call it.
    """

    def __init__(self, num_cards: int, law: ElixirLaw) -> None:
        self.num_cards = num_cards
        self.law = law
        self.cost: list[int] = [0] * num_cards
        self.tick = -1
        self.exact = True
        self.own_fine = 0
        self.foe_fine = 0
        self.leak_fine = 0
        self.own_hand: list[int] = []
        self.foe_hand: list[int] = []
        # Own cycle positions 5..8. Position 5 is ``next_card`` and always known;
        # 6..8 start unknown and are learned one per play as the queue rotates.
        self.own_cycle: list[int] = [EMPTY_CARD] * (DECK_SIZE - HAND_SIZE)
        self.own_deck = np.zeros(num_cards, dtype=bool)
        self.own_last_card = EMPTY_CARD
        self.own_last_play_tick = -1
        self.foe_seen = np.zeros(num_cards, dtype=bool)
        self.foe_plays = 0
        self.foe_recent: list[int] = []  # the cards behind the enemy hand, oldest first

    def bind(self, cards: Sequence[CardInfo]) -> None:
        self.cost = [c.elixir for c in cards]

    # -- lifecycle ----------------------------------------------------------

    def seed(self, state: BattleState, team: int) -> None:
        """Start of a match: everything forgotten, the two bars read once."""
        me, foe = state.players[team], state.players[1 - team]
        self.tick = state.tick
        self.exact = True
        self.own_fine = self.law.seed_fine(me.elixir_milli)
        self.foe_fine = self.law.seed_fine(foe.elixir_milli)
        self.leak_fine = 0
        self.own_hand = list(me.hand)
        self.foe_hand = list(foe.hand)
        self.own_cycle = [me.next_card] + [EMPTY_CARD] * (DECK_SIZE - HAND_SIZE - 1)
        self.own_deck = np.zeros(self.num_cards, dtype=bool)
        self.own_last_card = EMPTY_CARD
        self.own_last_play_tick = -1
        self.foe_seen = np.zeros(self.num_cards, dtype=bool)
        self.foe_plays = 0
        self.foe_recent = []
        self._note_own_cards()

    def observe(self, state: BattleState, team: int) -> None:
        """Advance to ``state``. A no-op unless the clock moved forward."""
        if self.tick < 0 or state.tick < self.tick:
            self.seed(state, team)
            return
        if state.tick == self.tick:
            return
        me, foe = state.players[team], state.players[1 - team]
        own_played = cards_that_left(self.own_hand, me.hand)
        foe_played = cards_that_left(self.foe_hand, foe.hand)
        spent_own = sum(self.cost[c] for c in own_played)
        spent_foe = sum(self.cost[c] for c in foe_played)
        self.own_fine, leaked = self.law.advance(
            self.own_fine, spent_own, self.tick, state.tick, state.regular_ticks, state.overtime
        )
        self.foe_fine, _ = self.law.advance(
            self.foe_fine, spent_foe, self.tick, state.tick, state.regular_ticks, state.overtime
        )
        self.leak_fine += leaked
        if self.law.to_milli(self.own_fine) != me.elixir_milli:
            self.exact = False
            self.own_fine = self.law.seed_fine(me.elixir_milli)
        if self.law.to_milli(self.foe_fine) != foe.elixir_milli:
            self.exact = False
        for card in own_played:
            self.own_cycle = [*self.own_cycle[1:], card]
            self.own_last_card = card
            self.own_last_play_tick = state.tick
        for card in foe_played:
            self.foe_seen[card] = True
            self.foe_plays += 1
            self.foe_recent.append(card)
            del self.foe_recent[: -(DECK_SIZE - HAND_SIZE)]
        self.own_hand = list(me.hand)
        self.foe_hand = list(foe.hand)
        self.own_cycle[0] = me.next_card
        self.tick = state.tick
        self._note_own_cards()

    def _note_own_cards(self) -> None:
        for c in (*self.own_hand, *self.own_cycle):
            if c != EMPTY_CARD:
                self.own_deck[c] = True

    # -- what the vector reads ----------------------------------------------

    def enemy_elixir_milli(self) -> int:
        """The opponent's bar, counted. Exact while ``exact``; an estimate after."""
        return self.law.to_milli(self.foe_fine)

    def own_elixir_milli(self) -> int:
        """The same count for the player's own bar; only the law's tests read it."""
        return self.law.to_milli(self.own_fine)

    def leaked_elixir(self) -> float:
        return self.leak_fine / self.law.scale

    def ticks_since_own_play(self, tick: int) -> int:
        if self.own_last_play_tick < 0:
            return tick
        return tick - self.own_last_play_tick

    def enemy_possible_hand(self) -> np.ndarray:
        """Cards that could be in the enemy hand right now, by the cycle rule.

        A card the opponent played goes to the back of their 8-card cycle, so it
        cannot be in hand again until four more plays have happened: the last four
        cards they played are exactly the four behind their hand, and everything
        else is possible. "Everything else" is the whole catalogue until eight
        distinct cards have been seen, at which point the deck is known and the
        answer narrows to it.
        """
        known = int(self.foe_seen.sum())
        out = self.foe_seen.copy() if known >= DECK_SIZE else np.ones(self.num_cards, dtype=bool)
        for card in self.foe_recent:
            out[card] = False
        return out


def build_vector(
    state: BattleState,
    team: int,
    cards: list[CardInfo],
    max_mana: int,
    reveal: Reveal,
    memory: MatchMemory,
) -> np.ndarray:
    """The flat vector of ``vector_layout``. ``memory`` must already have seen ``state``."""
    me, foe = state.players[team], state.players[1 - team]
    num_cards = len(cards)
    onehot = num_cards + 1
    full = 1000 * max_mana
    out: list[np.ndarray] = []
    out.append(np.array([me.elixir_milli / full], dtype=np.float32))
    foe_milli = foe.elixir_milli if reveal.enemy_elixir else memory.enemy_elixir_milli()
    out.append(np.array([foe_milli / full], dtype=np.float32))
    out += _hand_block(me.hand, cards, me.elixir_milli, num_cards, max_mana)
    out.append(_one_hot(me.next_card, onehot, num_cards))
    cycle = np.zeros((DECK_SIZE - HAND_SIZE - 1, onehot), dtype=np.float32)
    for i, card in enumerate(memory.own_cycle[1:]):
        cycle[i, num_cards if card == EMPTY_CARD else card] = 1
    out.append(cycle.reshape(-1))
    out.append(memory.own_deck.astype(np.float32))
    out.append(_one_hot(memory.own_last_card, onehot, num_cards))
    out.append(
        np.array(
            [
                min(1.0, memory.ticks_since_own_play(state.tick) / PLAY_GAP_TICKS),
                min(1.0, memory.leaked_elixir() / LEAK_SCALE),
            ],
            dtype=np.float32,
        )
    )
    out.append(memory.foe_seen.astype(np.float32))
    out.append(memory.enemy_possible_hand().astype(np.float32))
    out.append(np.array([min(1.0, memory.foe_plays / PLAYS_SCALE)], dtype=np.float32))
    for p in (me, foe):
        out.append(
            np.array(
                [p.tower_hp[s] / max(1, p.tower_max_hp[s]) for s in TowerSlot], dtype=np.float32
            )
        )
    out.append(np.array([me.crowns / 3.0, foe.crowns / 3.0], dtype=np.float32))
    out.append(np.array([float(me.king_active), float(foe.king_active)], dtype=np.float32))
    reg_left = max(0, state.regular_ticks - state.tick) / max(1, state.regular_ticks)
    ot_left = 0.0
    if state.overtime:
        ot_end = state.regular_ticks + state.overtime_ticks
        ot_left = max(0, ot_end - state.tick) / max(1, state.overtime_ticks)
    out.append(np.array([reg_left, float(state.overtime), ot_left], dtype=np.float32))
    out.append(
        np.array([float(state.elixir_rate == 1), float(state.elixir_rate == 2)], dtype=np.float32)
    )
    if reveal.enemy_hand:
        one, _cost, _afford = _hand_block(foe.hand, cards, foe.elixir_milli, num_cards, max_mana)
        out.append(one)
    if reveal.enemy_next_card:
        out.append(_one_hot(foe.next_card, onehot, num_cards))
    if reveal.enemy_deck:
        deck = np.zeros(num_cards, dtype=np.float32)
        for c in (*foe.hand, foe.next_card):
            if c != EMPTY_CARD:
                deck[c] = 1
        deck[memory.foe_seen] = 1
        out.append(deck)
    vec: np.ndarray = np.clip(np.concatenate(out), 0.0, 1.0).astype(np.float32)
    return vec


class Variability(NamedTuple):
    """How much of an observation can actually move. See ``measure_variability``."""

    cells: int  #: numbers in one observation, counting every key except the masks
    varying: int  #: how many of them differ across the states measured
    fraction: float  #: ``varying / cells``
    cosine_raw: float  #: greatest pairwise cosine over the whole observation
    cosine_varying: float  #: the same over the varying cells alone
    states: int

    def __str__(self) -> str:
        return (
            f"{self.varying}/{self.cells} cells move ({100 * self.fraction:.1f}%) over "
            f"{self.states} states; max pairwise cosine {self.cosine_raw:.4f} raw, "
            f"{self.cosine_varying:.4f} on the cells that move"
        )


def measure_variability(
    builder: ObsBuilder,
    states: Sequence[BattleState],
    action_masks: Sequence[np.ndarray],
    team: int = 0,
) -> Variability:
    """What fraction of this builder's observation responds to the battle at all.

    WHY A CONSUMER WANTS THIS. A representation that has collapsed -- every board
    encoding to nearly the same vector -- looks exactly like slow learning, and the
    usual detector is a cosine between encoded states with a threshold under it.
    That threshold is meaningless without this number. Measured on the shipped
    spatial builder over eight boards differing in a unit's position, a destroyed
    tower and the elixir: 51 of 11 749 cells move, 0.4%, and the greatest pairwise
    cosine is 0.9998 with nothing whatever wrong. An encoder reporting 0.99 on that
    input is INCREASING discrimination, not losing it.

    So measure the input the same way you measure the encoding, on the same states,
    and compare the two. A cosine threshold chosen without the input's own cosine is
    worse than no threshold, because it will fire on a healthy encoder or stay quiet
    on a dead one depending only on how much of the observation happens to be
    static.

    The masks are excluded: they are legality, they are handed to the policy
    separately, and their variability says nothing about the representation.
    """
    if len(states) != len(action_masks):
        raise ValueError("one action mask per state")
    if len(states) < 2:
        raise ValueError("variability needs at least two states")
    rows = []
    for state, mask in zip(states, action_masks, strict=True):
        obs = builder.build(state, team, mask)
        rows.append(
            np.concatenate(
                [np.asarray(v, dtype=np.float64).ravel() for k, v in sorted(obs.items())
                 if k not in MASK_OBS_KEYS]
            )
        )
    block = np.asarray(rows)
    varying = block.max(axis=0) != block.min(axis=0)

    def worst_cosine(m: np.ndarray) -> float:
        if m.shape[1] == 0:
            return 1.0
        unit = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
        cos = unit @ unit.T
        return float(cos[~np.eye(len(m), dtype=bool)].max())

    return Variability(
        cells=int(block.shape[1]),
        varying=int(varying.sum()),
        fraction=float(varying.mean()),
        cosine_raw=worst_cosine(block),
        cosine_varying=worst_cosine(block[:, varying]),
        states=len(states),
    )


def _one_hot(card: int, onehot: int, empty_index: int) -> np.ndarray:
    v = np.zeros(onehot, dtype=np.float32)
    v[empty_index if card == EMPTY_CARD else card] = 1
    return v


def _hand_block(
    hand: Sequence[int], cards: list[CardInfo], elixir_milli: int, num_cards: int, max_mana: int
) -> list[np.ndarray]:
    """[card one-hots, costs, affordable] for one hand."""
    onehot = num_cards + 1
    one = np.zeros((HAND_SIZE, onehot), dtype=np.float32)
    cost = np.zeros(HAND_SIZE, dtype=np.float32)
    afford = np.zeros(HAND_SIZE, dtype=np.float32)
    for i, c in enumerate(hand):
        if c == EMPTY_CARD:
            one[i, num_cards] = 1
            continue
        one[i, c] = 1
        cost[i] = cards[c].elixir / max_mana
        afford[i] = 1.0 if elixir_milli >= cards[c].elixir * 1000 else 0.0
    return [one.reshape(-1), cost, afford]


# ---------------------------------------------------------------------------
# The base builder
# ---------------------------------------------------------------------------


class ObsBuilder(ABC):
    """state -> observation dict. Always includes ``action_mask``."""

    calibration: Calibration
    reveal: Reveal

    def __init__(
        self, reveal: Reveal | None = None, calibration: Calibration | None = None
    ) -> None:
        if reveal is not None and not isinstance(reveal, Reveal):
            # The old constructors took ``reveal_enemy_elixir: bool`` in this
            # position. A bare True would otherwise be carried all the way to the
            # first ``reveal.enemy_elixir`` lookup and fail there, in a builder
            # that has already declared its observation space.
            raise TypeError(f"reveal must be a Reveal, not {reveal!r} (see obs.Reveal)")
        self.reveal = reveal or Reveal()
        if calibration is not None:
            self.calibration = calibration

    def bind(self, engine: Engine, action_parser: ActionParser) -> None:
        if not hasattr(self, "calibration"):
            self.calibration = default_calibration()
        self.arena = engine.arena()
        self.cards = list(engine.cards())
        self.num_cards = len(self.cards)
        self.oracle = PlacementOracle(self.arena, engine.rules(), self.cards)
        self.mask_space = spaces.Box(0, 1, shape=(int(action_parser.space.n),), dtype=np.int8)
        self.mask_plane_shape = action_parser.mask_plane_shape()
        self._troop_probe = next((c for c in self.cards if c.placement == Placement.TROOP), None)
        self.max_mana = self.calibration.int("match.MAX_MANA")
        self.law = ElixirLaw.load(self.calibration)
        self.memory = {t: MatchMemory(self.num_cards, self.law) for t in TEAMS}
        for m in self.memory.values():
            m.bind(self.cards)
        self.vec_size = sum(f.size for f in self.vector_layout())

    def reset(self, state: BattleState) -> None:
        """Called at the start of every episode: both seats forget the last one."""
        for team, memory in self.memory.items():
            memory.seed(state, team)

    def counts_are_exact(self, team: int) -> bool:
        """Whether this seat's counted features are still provably right.

        False once the opponent's elixir count has been caught disagreeing with the
        bar it models (``MatchMemory``). A training run should log it: a policy
        trained on a match where it went False was reading an estimate in a slot
        documented as exact, and the whole argument for putting that slot in the
        fair set is that it is not an estimate.
        """
        return self.memory[team].exact

    def vector_layout(self) -> list[VectorField]:
        """The flat vector's fields, in order, for this builder's ``Reveal``."""
        return vector_layout(self.num_cards, self.reveal)

    def vector_offsets(self) -> dict[str, slice]:
        """``key -> slice`` into the flat vector this builder writes."""
        return vector_offsets(self.num_cards, self.reveal)

    @abstractmethod
    def channel_names(self) -> list[str]:
        """The names of the feature planes / columns this builder writes, in order."""

    def spatial_layout(self) -> tuple[tuple[str, bool], ...]:
        """``(channel name, is static)`` per spatial plane, or empty if there are none.

        STATIC means a function of the arena alone: the same numbers every tick, in
        every battle, for a given seat. A consumer that stores observations can hold
        those planes once instead of once per transition, and this says which they
        are rather than leaving it to be inferred by sampling states and hoping none
        of the others happened to be constant.
        """
        return ()

    def config(self) -> dict[str, Any]:
        """Constructor state, JSON-able, for ``ClashParallelEnv.config()``."""
        return {"reveal": self.reveal.as_dict()}

    def _mask_space_entries(self) -> dict[str, spaces.Space[Any]]:
        out: dict[str, spaces.Space[Any]] = {"action_mask": self.mask_space}
        if self.mask_plane_shape is not None:
            out["mask_planes"] = spaces.Box(0, 1, shape=self.mask_plane_shape, dtype=np.int8)
        return out

    def _mask_entries(self, action_mask: np.ndarray) -> dict[str, np.ndarray]:
        """``action_mask`` flat, and ``mask_planes`` as a VIEW of the same buffer.

        A view because the action space is laid out as ``1 + slot * ny * nx + y * nx
        + x`` (action.py), so the planes need no arithmetic at all -- which is the
        whole reason they are cheaper than the zone channels they replace. The two
        keys therefore alias each other and the env's own mask, as ``action_mask``
        already did: read them, do not write to them. Every vector env copies on
        the way into its batch.
        """
        flat = action_mask.astype(np.int8, copy=False)
        out = {"action_mask": flat}
        if self.mask_plane_shape is not None:
            out["mask_planes"] = flat[1:].reshape(self.mask_plane_shape)
        return out

    def _vector(self, state: BattleState, team: int) -> np.ndarray:
        memory = self.memory[team]
        memory.observe(state, team)
        return build_vector(state, team, self.cards, self.max_mana, self.reveal, memory)

    @abstractmethod
    def observation_space(self) -> spaces.Dict: ...

    @abstractmethod
    def build(self, state: BattleState, team: int, action_mask: np.ndarray) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Spatial builder (the default)
# ---------------------------------------------------------------------------

FAIR_SPATIAL_CHANNELS: list[tuple[str, str]] = [
    ("own_ground_troops", "count of own ground troops whose centre is in the tile"),
    ("own_air_troops", "count of own flying troops"),
    ("own_buildings", "count of own buildings (crown towers have their own channel)"),
    ("own_towers", "count of own crown towers by centre"),
    ("own_hp", "sum of own entity hp / 1000 in the tile"),
    ("enemy_ground_troops", "count of enemy ground troops"),
    ("enemy_air_troops", "count of enemy flying troops"),
    ("enemy_buildings", "count of enemy buildings"),
    ("enemy_towers", "count of enemy crown towers"),
    ("enemy_hp", "sum of enemy entity hp / 1000"),
    ("own_deploying", "count of own entities still in their deploy timer"),
    ("enemy_deploying", "count of enemy entities still in their deploy timer"),
    ("water", "fraction of the tile's 4 half-cells that are water (static)"),
    ("no_deploy", "fraction of the tile's 4 half-cells flagged no-deploy (static)"),
    ("enemy_troop_zone", "1 where the enemy could place a TROOP now (in no mask)"),
    ("own_spells", "count of my live spell objects whose current centre is in the tile"),
    ("enemy_spells", "count of enemy live spell objects whose current centre is in the tile"),
    (
        "own_spell_aim",
        "count of my live spells whose aim point is in the tile (landing point; roll end)",
    ),
    ("own_stunned", "count of my entities with stun_ticks > 0"),
    ("enemy_stunned", "count of enemy entities with stun_ticks > 0"),
]

REVEAL_SPATIAL_CHANNELS: dict[str, tuple[str, str]] = {
    "enemy_spell_aim": (
        "enemy_spell_aim",
        "count of enemy live spells whose aim point is in the tile (Reveal.enemy_spell_aim)",
    ),
}


def spatial_channels(reveal: Reveal | None = None) -> list[tuple[str, str]]:
    """The spatial channels for a ``Reveal``: the fair ones, then the revealed ones."""
    rev = reveal or Reveal()
    out = list(FAIR_SPATIAL_CHANNELS)
    for field, entry in REVEAL_SPATIAL_CHANNELS.items():
        if getattr(rev, field):
            out.append(entry)
    return out


# The fair channel list, for callers that want the default set without a Reveal.
SPATIAL_CHANNELS: list[tuple[str, str]] = spatial_channels()

# Channels that are a function of the arena alone -- see ObsBuilder.spatial_layout.
STATIC_CHANNELS = frozenset({"water", "no_deploy"})

ENTITY_CHANNELS = 12  # the first 12 channels are rasterised from state.entities
TEAM_STRIDE = 5  # own_ground .. own_hp, then the same five for the enemy
HP_CHANNELS = (4, 9)  # own_hp, enemy_hp
# ``spell_channels`` rows, in its own fixed order (independent of the Reveal).
SPELL_ROWS = (
    "own_spells",
    "enemy_spells",
    "own_spell_aim",
    "enemy_spell_aim",
    "own_stunned",
    "enemy_stunned",
)


def entity_channels(entities: Sequence[EntityState], team: int, arena: Arena) -> np.ndarray:
    """The first ``ENTITY_CHANNELS`` channels, float32 [12, tiles_y, tiles_x], seen by ``team``.

    Every cell is an INTEGER sum (counts, raw hp) converted to float32 exactly once,
    so the result is a function of the entity SET and cannot depend on list order
    (module doc). hp: int64 / HP_SCALE in float64 (exact for any sum below 2**53),
    then one rounding to float32. Crown towers are counted in their OWN channel and
    not in ``own_buildings`` / ``enemy_buildings``: a Cannon and a princess tower
    pose different problems, and one channel holding both could not say which it
    was. Module-level so a test can plant the old order-dependent float
    accumulation back in.
    """
    acc = np.zeros((ENTITY_CHANNELS, arena.tiles_y, arena.tiles_x), dtype=np.int64)
    for e in entities:
        ox, oy = to_own(arena, team, e.x, e.y)
        tx = min(max(ox // arena.subtile, 0), arena.tiles_x - 1)
        ty = min(max(oy // arena.subtile, 0), arena.tiles_y - 1)
        base = 0 if e.team == team else TEAM_STRIDE
        if e.kind == EntityKind.TROOP:
            acc[base + (1 if e.flying else 0), ty, tx] += 1
        elif e.kind == EntityKind.BUILDING:
            acc[base + 2, ty, tx] += 1
        else:
            acc[base + 3, ty, tx] += 1
        acc[base + 4, ty, tx] += e.hp
        if e.deploy_ticks > 0:
            acc[10 if e.team == team else 11, ty, tx] += 1
    out = acc.astype(np.float32)
    for c in HP_CHANNELS:
        out[c] = (acc[c] / HP_SCALE).astype(np.float32)
    return out


def _tile(arena: Arena, team: int, x: int, y: int) -> tuple[int, int]:
    """Own-frame tile of an engine-frame point, clamped onto the board (a Log's roll
    end point can lie past the arena edge)."""
    ox, oy = to_own(arena, team, x, y)
    return (
        min(max(oy // arena.subtile, 0), arena.tiles_y - 1),
        min(max(ox // arena.subtile, 0), arena.tiles_x - 1),
    )


def spell_channels(state: BattleState, team: int, arena: Arena) -> np.ndarray:
    """The ``SPELL_ROWS``, float32 [6, tiles_y, tiles_x], seen by ``team``.

    Always all six rows, whatever the Reveal: the builder decides which of them
    reach the observation, so this stays one function with one layout that a test
    can plant a defect into. Integer counts converted once, like
    ``entity_channels``, so the result is a function of the spell and entity SETS.
    """
    acc = np.zeros((len(SPELL_ROWS), arena.tiles_y, arena.tiles_x), dtype=np.int64)
    for sp in state.spells:
        side = 0 if sp.team == team else 1
        ty, tx = _tile(arena, team, sp.x, sp.y)
        acc[side, ty, tx] += 1
        ty, tx = _tile(arena, team, sp.aim_x, sp.aim_y)
        acc[2 + side, ty, tx] += 1
    for e in state.entities:
        if e.stun_ticks > 0:
            ty, tx = _tile(arena, team, e.x, e.y)
            acc[4 if e.team == team else 5, ty, tx] += 1
    out: np.ndarray = acc.astype(np.float32)
    return out


class SpatialObsBuilder(ObsBuilder):
    """Dict(spatial [C, 32, 18], mask_planes [4, 32, 18], vector [V], action_mask [A]).

    Entities are rasterised by the tile containing their centre in the own frame
    (``x_own // SUBTILE``, clamped). ``channel_names()`` lists the channels;
    ``spatial_channels(reveal)`` is the same list with each channel's meaning.
    """

    def bind(self, engine: Engine, action_parser: ActionParser) -> None:
        super().bind(engine, action_parser)
        a = self.arena
        self.channels = spatial_channels(self.reveal)
        self._channel_index = {name: i for i, (name, _) in enumerate(self.channels)}
        self.shape = (len(self.channels), a.tiles_y, a.tiles_x)
        h = a.half
        water = self.oracle.water.reshape(a.tiles_y, h, a.tiles_x, h).mean(axis=(1, 3))
        nodep = self.oracle.nodeploy.reshape(a.tiles_y, h, a.tiles_x, h).mean(axis=(1, 3))
        # Static channels in each team's own frame.
        self._static = [
            np.stack([water, nodep]).astype(np.float32),
            np.stack([water[::-1, ::-1], nodep[::-1, ::-1]]).astype(np.float32),
        ]
        # Which ``spell_channels`` row goes to which channel, resolved once at bind.
        self._spell_map = [
            (row, self._channel_index[name])
            for row, name in enumerate(SPELL_ROWS)
            if name in self._channel_index
        ]
        self._space = spaces.Dict(
            {
                "spatial": spaces.Box(0.0, SPATIAL_CLIP, shape=self.shape, dtype=np.float32),
                "vector": spaces.Box(0.0, 1.0, shape=(self.vec_size,), dtype=np.float32),
                **self._mask_space_entries(),
            }
        )

    def channel_names(self) -> list[str]:
        return [name for name, _ in self.channels]

    def spatial_layout(self) -> tuple[tuple[str, bool], ...]:
        return tuple((name, name in STATIC_CHANNELS) for name, _ in self.channels)

    def observation_space(self) -> spaces.Dict:
        return self._space

    def _enemy_troop_zone(self, state: BattleState, viewer: int) -> np.ndarray:
        """Where the OPPONENT could put a troop now, in ``viewer``'s frame."""
        card = self._troop_probe
        a = self.arena
        if card is None:
            return np.zeros((a.tiles_y, a.tiles_x), dtype=np.float32)
        g = self.oracle.point_grid(state, 1 - viewer, card, 1)
        if viewer == RED:
            g = g[::-1, ::-1]
        return g.astype(np.float32)

    def build(self, state: BattleState, team: int, action_mask: np.ndarray) -> dict[str, Any]:
        sp = np.zeros(self.shape, dtype=np.float32)
        idx = self._channel_index
        sp[:ENTITY_CHANNELS] = entity_channels(state.entities, team, self.arena)
        sp[idx["water"] : idx["no_deploy"] + 1] = self._static[team]
        sp[idx["enemy_troop_zone"]] = self._enemy_troop_zone(state, team)
        rows = spell_channels(state, team, self.arena)
        for row, channel in self._spell_map:
            sp[channel] = rows[row]
        np.clip(sp, 0.0, SPATIAL_CLIP, out=sp)
        return {
            "spatial": sp,
            "vector": self._vector(state, team),
            **self._mask_entries(action_mask),
        }


# ---------------------------------------------------------------------------
# Entity-list builder (for attention / transformer policies)
# ---------------------------------------------------------------------------


def entity_row_key(row: tuple[Any, ...]) -> tuple[Any, ...]:
    """Sort key of an EntityListObsBuilder row: every field but the entity itself.
    Module-level so a test can plant a shorter six-field key back in."""
    return row[:-1]


def spell_row_key(row: tuple[Any, ...]) -> tuple[Any, ...]:
    """Sort key of a ``spells`` row: every field but the spell itself (as entity rows).

    ``state.spells`` is in engine cast order, which is not seat-canonical. The row is
    built with the AIM POINT LAST, for the reason in ``EntityListObsBuilder``: an
    enemy spell's aim is hidden, and a key that ranks it before a visible field
    orders visible content by hidden data.
    """
    return row[:-1]


ENTITY_FEATURE_NAMES = (
    "present",
    "own",
    "enemy",
    "kind_troop",
    "kind_building",
    "kind_king",
    "kind_princess",
    "x_own",
    "y_own",
    "hp_frac",
    "hp_scaled",
    "radius",
    "flying",
    "deploying",
    "deploy_ticks",
    "stunned",
    "stun_ticks",
    "knockback",
)

# Columns 9 and 10 hold the aim point of the VIEWER's own spells and are 0 on an
# enemy row; ``Reveal.enemy_spell_aim`` appends two more for the opponent's.
SPELL_FEATURE_NAMES = (
    "present",
    "own",
    "enemy",
    "motion_flight",
    "motion_airborne",
    "motion_rolling",
    "motion_area",
    "x_own",
    "y_own",
    "own_aim_x_own",
    "own_aim_y_own",
    "delay_ticks",
    "roll_progress",
    "hits",
)


class EntityListObsBuilder(ObsBuilder):
    """Dict(entities [N, F], spells [M, S], mask_planes, vector [V], action_mask [A]).

    Per-entity features (F = 18 + num_cards + 1), named by ``ENTITY_FEATURE_NAMES``:
        0 present, 1 own, 2 enemy, 3..6 kind one-hot (troop, building, king, princess),
        7 x_own / width, 8 y_own / height, 9 hp / max_hp, 10 hp / 1000 (clipped to 1),
        11 radius / tile, 12 flying, 13 deploying, 14 deploy_ticks / 100 (clipped),
        15 stunned, 16 stun_ticks / 100 (clipped), 17 knockback slide in progress,
        18.. card one-hot (last index = crown tower / no card). A unit a spell
        released (Goblin Barrel's Goblins) carries that spell's card.
    Rows are sorted canonically by ``entity_row_key``: (enemy, y_own, x_own, kind,
    card, hp, max_hp, radius, flying, deploy_ticks, stun_ticks, knockback_ticks) --
    EVERY entity field a row is built from, so two rows that tie are identical and
    the order is seat-invariant whatever order the engine listed them in. A shorter
    key -- (enemy, y_own, x_own, kind, card, hp) -- leaves stacked twins that differ
    only in deploy_ticks in engine list order. Entities beyond
    ``max_entities`` are dropped in that order; the drop is silent, so size
    ``max_entities`` generously.

    Live spell objects, ``spells`` [max_spells, S] (S = 14 + num_cards), named by
    ``SPELL_FEATURE_NAMES``:
        0 present, 1 own, 2 enemy, 3..6 motion one-hot (flight, airborne, rolling,
        area; a pulsing area effect sets the area bit), 7 x_own / width, 8 y_own /
        height, 9 and 10 the aim point of YOUR spells (0 on an enemy row),
        11 delay_ticks / 100 (clipped; a pulsing area's life left), 12 travelled /
        length (0 when length is 0), 13 hits / 16 (clipped), 14.. card one-hot, and
        then two APPENDED columns under ``Reveal.enemy_spell_aim`` holding the
        opponent's aim point. Rows past ``max_spells`` are dropped in sort order.
        Positions are clipped to [0, 1] (a roll end point can lie past the arena
        edge).

        THE SORT KEY PUTS THE AIM POINT LAST, AND THAT IS NOT COSMETIC. Row ORDER is
        observable: it decides whose delay and hit count appear first. The key must
        still name every field a row is built from, so the aim cannot leave it -- but
        with the aim ranked early, two enemy spells alike in everything visible and
        different in where they were going came out in an order set by where they
        were going, and the fair observation changed when only the hidden aim
        changed. Measured: two states differing ONLY in two enemy aim points gave
        delay columns [0.03, 0.07] and [0.07, 0.03]. With the aim last, hidden data
        can only order rows whose every visible field is equal, and those rows write
        the same numbers, so their order cannot be seen.
    """

    BASE_FEATURES = 18
    SPELL_BASE_FEATURES = 14

    def __init__(
        self,
        max_entities: int = 96,
        reveal: Reveal | None = None,
        calibration: Calibration | None = None,
        max_spells: int = 16,
    ) -> None:
        super().__init__(reveal, calibration)
        self.max_entities = max_entities
        self.max_spells = max_spells

    def bind(self, engine: Engine, action_parser: ActionParser) -> None:
        super().bind(engine, action_parser)
        self.features = self.BASE_FEATURES + self.num_cards + 1
        self.spell_features = self.SPELL_BASE_FEATURES + self.num_cards
        # Reveal.enemy_spell_aim APPENDS two columns rather than filling the two the
        # fair rows already have: a reveal must change the width (module doc), and
        # the fair aim columns mean "where MY spell is going", which is a different
        # feature from where the opponent's is.
        self._enemy_aim_at = self.spell_features if self.reveal.enemy_spell_aim else None
        if self._enemy_aim_at is not None:
            self.spell_features += 2
        self._space = spaces.Dict(
            {
                "entities": spaces.Box(
                    0.0, 1.0, shape=(self.max_entities, self.features), dtype=np.float32
                ),
                "spells": spaces.Box(
                    0.0, 1.0, shape=(self.max_spells, self.spell_features), dtype=np.float32
                ),
                "vector": spaces.Box(0.0, 1.0, shape=(self.vec_size,), dtype=np.float32),
                **self._mask_space_entries(),
            }
        )

    def channel_names(self) -> list[str]:
        """Entity columns, then spell columns; the card one-hots are the trailing run."""
        names = [
            *(f"entity.{n}" for n in ENTITY_FEATURE_NAMES),
            "entity.card_one_hot",
            *(f"spell.{n}" for n in SPELL_FEATURE_NAMES),
            "spell.card_one_hot",
        ]
        if self._enemy_aim_at is not None:
            names += ["spell.enemy_aim_x_own", "spell.enemy_aim_y_own"]
        return names

    def config(self) -> dict[str, Any]:
        return {
            **super().config(),
            "max_entities": self.max_entities,
            "max_spells": self.max_spells,
        }

    def observation_space(self) -> spaces.Dict:
        return self._space

    def build(self, state: BattleState, team: int, action_mask: np.ndarray) -> dict[str, Any]:
        a = self.arena
        rows = []
        for e in state.entities:
            ox, oy = to_own(a, team, e.x, e.y)
            enemy = int(e.team != team)
            card = self.num_cards if e.card_id == EMPTY_CARD else e.card_id
            rows.append(
                (
                    enemy,
                    oy,
                    ox,
                    int(e.kind),
                    card,
                    e.hp,
                    e.max_hp,
                    e.radius,
                    int(e.flying),
                    e.deploy_ticks,
                    e.stun_ticks,
                    e.knockback_ticks,
                    e,
                )
            )
        rows.sort(key=entity_row_key)
        out = np.zeros((self.max_entities, self.features), dtype=np.float32)
        for i, (enemy, oy, ox, kind, card, hp, *_, e) in enumerate(rows[: self.max_entities]):
            f = out[i]
            f[0] = 1
            f[1 + enemy] = 1
            f[3 + kind] = 1
            f[7] = ox / a.width
            f[8] = oy / a.height
            f[9] = hp / max(1, e.max_hp)
            f[10] = min(1.0, hp / HP_SCALE)
            f[11] = e.radius / a.subtile
            f[12] = float(e.flying)
            f[13] = float(e.deploy_ticks > 0)
            f[14] = min(1.0, e.deploy_ticks / 100.0)
            f[15] = float(e.stun_ticks > 0)
            f[16] = min(1.0, e.stun_ticks / 100.0)
            f[17] = float(e.knockback_ticks > 0)
            f[self.BASE_FEATURES + card] = 1
        np.clip(out, 0.0, 1.0, out=out)
        return {
            "entities": out,
            "spells": self._spell_rows(state, team),
            "vector": self._vector(state, team),
            **self._mask_entries(action_mask),
        }

    def _spell_rows(self, state: BattleState, team: int) -> np.ndarray:
        a = self.arena
        rows = []
        for sp in state.spells:
            ox, oy = to_own(a, team, sp.x, sp.y)
            ax, ay = to_own(a, team, sp.aim_x, sp.aim_y)
            rows.append(
                (
                    int(sp.team != team),
                    oy,
                    ox,
                    sp.motion,
                    sp.card_id,
                    sp.delay_ticks,
                    sp.travelled,
                    sp.length,
                    sp.hits,
                    ay,
                    ax,
                    sp,
                )
            )
        rows.sort(key=spell_row_key)
        aim_at = self._enemy_aim_at
        out = np.zeros((self.max_spells, self.spell_features), dtype=np.float32)
        for i, (enemy, oy, ox, motion, card, delay, trav, length, hits, ay, ax, _) in enumerate(
            rows[: self.max_spells]
        ):
            f = out[i]
            f[0] = 1
            f[1 + enemy] = 1
            if 0 <= motion < 4:
                f[3 + motion] = 1
            elif motion == SpellMotion.PULSING:
                f[3 + SpellMotion.AREA] = 1
            f[7] = ox / a.width
            f[8] = oy / a.height
            if enemy:
                if aim_at is not None:
                    f[aim_at] = ax / a.width
                    f[aim_at + 1] = ay / a.height
            else:
                f[9] = ax / a.width
                f[10] = ay / a.height
            f[11] = min(1.0, delay / 100.0)
            f[12] = trav / length if length > 0 else 0.0
            f[13] = min(1.0, hits / 16.0)
            if 0 <= card < self.num_cards:
                f[self.SPELL_BASE_FEATURES + card] = 1
        np.clip(out, 0.0, 1.0, out=out)
        return out

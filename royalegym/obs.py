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
    read out of the half of the state a player cannot see, with one known exception:
    the builders read no status flags, so a unit invisible to its enemy (a Ghost,
    ``STATUS_INVISIBLE``) is still shown to the enemy seat where it stands. Hiding it
    or marking it waits on a measurement of what the live client's opponent sees.

    ``Reveal`` opens that half, one field at a time, for curriculum, distillation
    and debugging. An enabled field ADDS its channels and vector slots; it is
    never present-but-zero, so a fair observation and a cheating one do not even
    have the same width, and a checkpoint cannot quietly be trained on one and
    evaluated on the other. ``ClashParallelEnv.config()`` records the Reveal for
    exactly that reason. The one field that changes a slot instead of adding one
    is ``enemy_elixir``: the fair slot already holds the counted value, and the
    reveal swaps in the value read from the state. The two agree on a played-out
    battle, presses included (tests/test_env_obs.py, tests/test_press_memory.py); a
    disagreement is a bug in the counter, or one of the misses ``MatchMemory`` names and
    flags with ``exact``.

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
    ``spells`` array. Not exposed: a spell's hit radius or damage, which the engine
    does not export (use the card one-hot); and a unit's buffs and current target,
    which it does (``EntityState.buffs``, ``target_uid``) and no builder reads yet.
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
from functools import lru_cache
from typing import Any, NamedTuple

import numpy as np
from gymnasium import spaces

from .action import ActionParser, PlacementOracle
from .protocol import (
    ABILITY_BUTTONS,
    DECK_SIZE,
    EMPTY_CARD,
    HAND_SIZE,
    RED,
    STATUS_EVOLVED,
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
    ability_row,
    card_is_spell,
    default_calibration,
    default_elixir_law,
    status_of,
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

    All-False (the default) is the fair observation, apart from the invisible units
    the module doc names. Every True field ADDS
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


def vector_layout(
    num_cards: int,
    reveal: Reveal | None = None,
    enemy_last_card: bool = False,
    evolutions: bool = False,
    evolution_progress: bool = False,
) -> list[VectorField]:
    """The flat vector, field by field, in order. THE definition of the layout.

    Fair fields come first and in a fixed order, so enabling a reveal never moves
    a fair feature: the slice holding "own elixir" is the same in a fair run and in
    a cheating one. tests/test_env_obs.py asserts the resulting width rather than
    trusting arithmetic done here.

    ``enemy_last_card`` (D2, 2026-09-22; off by default) appends one FAIR field, the
    enemy's last play, at the END of the fair block -- after every fair field that
    existed before it, so turning it on moves no existing fair offset, and before the
    reveal fields, so the fair block stays contiguous. It is fair because anyone
    watching sees what the opponent just played.

    ``evolutions`` (off by default) appends the OWN evolution fields after that, for the
    same reason: ``own_hand_evolved`` and ``own_next_evolved``, and with
    ``evolution_progress`` also ``own_hand_evo_progress``. They are fair because the
    client shows the charge on a player's own cards. The enemy's charge is not shown, so
    nothing here reads it.
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
        VectorField(
            "own_hand_pending", HAND_SIZE, "hand slot card has a play waiting to run [4]"
        ),
        VectorField("own_pending_cost", 1, "elixir the own waiting commands hold / MAX_MANA"),
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
        VectorField("elixir_rate", 3, "elixir rate one-hot [1x, 2x, 3x]"),
    ]
    if enemy_last_card:
        fields.append(
            VectorField(
                "enemy_last_card", onehot, "last card the enemy played, one-hot [n+1]; n = none yet"
            )
        )
    if evolution_progress and not evolutions:
        raise ValueError("evolution_progress needs evolutions=True: it extends those fields")
    if evolutions:
        fields.append(
            VectorField(
                "own_hand_evolved",
                HAND_SIZE,
                "hand slot puts down its evolved form if played now [4]",
            )
        )
        fields.append(
            VectorField("own_next_evolved", 1, "the next card's next play is its evolved form")
        )
    if evolution_progress:
        fields.append(
            VectorField(
                "own_hand_evo_progress",
                HAND_SIZE,
                "hand slot's evolution counter: plays / the cycle length, 0 without one [4]",
            )
        )
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


def vector_fields(
    num_cards: int,
    reveal: Reveal | None = None,
    enemy_last_card: bool = False,
    evolutions: bool = False,
    evolution_progress: bool = False,
) -> list[tuple[str, int]]:
    """``(description, size)`` per field, in order. The suite checks the sizes add up."""
    return [
        (f"{f.key}: {f.doc}", f.size)
        for f in vector_layout(
            num_cards, reveal, enemy_last_card, evolutions, evolution_progress
        )
    ]


def vector_offsets(
    num_cards: int,
    reveal: Reveal | None = None,
    enemy_last_card: bool = False,
    evolutions: bool = False,
    evolution_progress: bool = False,
) -> dict[str, slice]:
    """``key -> slice`` into the flat vector, so nothing has to count slots by hand."""
    out: dict[str, slice] = {}
    at = 0
    for f in vector_layout(num_cards, reveal, enemy_last_card, evolutions, evolution_progress):
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


def presses_seen(
    before: Sequence[Sequence[int]], after: Sequence[Sequence[int]], standing: set[int]
) -> list[int]:
    """The costs of the ability presses one side made between two observations of its
    ``abilities`` rows, button by button: a hero's charge turning spent, or a champion's
    button going dark while a unit of its card still stands (``standing``: the card ids of
    that side's units on the board). A button gone dark with its unit gone is a death, not a
    press. See ``MatchMemory``."""
    out = []
    for b, a in zip(map(ability_row, before), map(ability_row, after), strict=False):
        spent = not b.spent and a.spent
        dark = b.available and not a.available and not a.spent and a.card_id in standing
        if spent or dark:
            out.append(a.cost)
    return out


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

    AND A PRESS. An ability press (a hero's, a champion's) is paid from the bar and changes
    no hand slot, so it is read from the side's ``abilities`` rows instead, which name the
    same public event: a hero's charge turns spent, or a champion's button goes dark while
    he still stands (``presses_seen``). It costs the row's price, dated like a play, and
    is no play: the cycle, the last play a Mirror copies and the play counts stay as they
    were. Uncharged, one enemy Golden Knight press left the count 1000 high for the rest of
    the match (found by the docs session, 2026-09-29).

    AND IT SAYS WHEN IT CANNOT BE EXACT. ``exact`` goes False, and stays False for
    the match, the moment the count disagrees with the bar it is modelling -- on
    EITHER side. Three things make that happen: a play was missed (a deck that repeats
    a card can hide one, because a play that swaps a card for itself changes no hand
    slot), a press was missed (read off the rows, a hero or champion pressed and killed
    between two observations shows only its death; the env hands ``observe`` the presses
    it saw accepted, so its memories cannot miss one), or the engine's elixir law is not
    the one in calibration.json.

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
        # A MIRROR costs the card it copies plus its own one (match.MIRROR_COST_RULE), and it
        # copies its side's last ACCEPTED play that was not a Mirror (match.MIRROR_RECORD, the
        # engine's ``mirror_target``). Both are public: every play is. [own, enemy].
        self.is_mirror: list[bool] = [False] * num_cards
        self.last_play: list[int] = [EMPTY_CARD, EMPTY_CARD]
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
        self.unaffordable = [0, 0]  # plays the counted bar could not pay: own, enemy
        # Each side's ability rows as last seen, [own, enemy]; None until a state shows them.
        self.rows: list[list[list[int]] | None] = [None, None]
        # THE COMMAND DELAY, [own, enemy] ticks (``ObsBuilder.command_delay``): a play or a
        # press accepted on tick T runs, and is paid, on T + delay. The waiting commands the
        # last state showed, and the presses accepted and not run yet as (tick, elixir).
        self.delay = [0, 0]
        self.waiting: list[list[list[int]]] = [[], []]
        self.press_queue: list[list[tuple[int, int]]] = [[], []]

    def bind(self, cards: Sequence[CardInfo]) -> None:
        self.cost = [c.elixir for c in cards]
        self.is_mirror = [c.placement == Placement.MIRROR for c in cards]

    # -- lifecycle ----------------------------------------------------------

    def seed(self, state: BattleState, team: int) -> None:
        """Start of a match: everything forgotten, the two bars read once."""
        me, foe = state.players[team], state.players[1 - team]
        self.start(
            state.tick, me.elixir_milli, foe.elixir_milli, me.hand, me.next_card, foe.hand
        )
        self.rows = [list(me.abilities), list(foe.abilities)]
        self.waiting = [list(me.pending), list(foe.pending)]
        # A press already waiting is paid when it runs: its tick is the state's plus its
        # ticks left, and its cost is the row's.
        self.press_queue = [
            [(state.tick + row[4], row[5]) for row in p.pending if row[0] == 1] for p in (me, foe)
        ]

    def start(
        self,
        tick: int,
        own_elixir_milli: int,
        enemy_elixir_milli: int,
        own_hand: Sequence[int],
        next_card: int,
        enemy_hand: Sequence[int] | None = None,
    ) -> None:
        """``seed`` without a state: the start of a match, from what a player sees at it.

        Both bars are public at the start, and so is the player's own hand. The enemy
        hand is only used by ``observe`` to notice plays, so a caller that feeds dated
        plays to ``advance`` instead leaves it out.
        """
        self.tick = tick
        self.exact = True
        self.own_fine = self.law.seed_fine(own_elixir_milli)
        self.foe_fine = self.law.seed_fine(enemy_elixir_milli)
        self.leak_fine = 0
        self.own_hand = list(own_hand)
        self.foe_hand = list(enemy_hand) if enemy_hand is not None else [EMPTY_CARD] * HAND_SIZE
        self.own_cycle = [next_card] + [EMPTY_CARD] * (DECK_SIZE - HAND_SIZE - 1)
        self.own_deck = np.zeros(self.num_cards, dtype=bool)
        self.own_last_card = EMPTY_CARD
        self.own_last_play_tick = -1
        self.foe_seen = np.zeros(self.num_cards, dtype=bool)
        self.foe_plays = 0
        self.foe_recent = []
        self.unaffordable = [0, 0]
        self.last_play = [EMPTY_CARD, EMPTY_CARD]
        self.rows = [None, None]
        self.waiting = [[], []]
        self.press_queue = [[], []]
        self._note_own_cards()

    def observe(
        self,
        state: BattleState,
        team: int,
        presses: Sequence[tuple[int, int]] | None = None,
    ) -> None:
        """Advance to ``state``. A no-op unless the clock moved forward.

        ``presses``, where the caller has them, are the ability presses accepted since the
        last observation, each ``(team, button)``: the env's own results, the same public
        event a player sees. They are charged at the price the button's row showed, and are
        exact at any gap. Without them the presses are read off the rows (``presses_seen``),
        which misses a unit pressed and killed between two observations.
        """
        if self.tick < 0 or state.tick < self.tick:
            self.seed(state, team)
            return
        if state.tick == self.tick:
            return
        me, foe = state.players[team], state.players[1 - team]
        last = state.tick - 1
        plays_by_side: list[list[tuple[int, int]]] = []
        presses_by_side: list[list[tuple[int, int]]] = []
        for side, (p, hand) in enumerate(((me, self.own_hand), (foe, self.foe_hand))):
            before, self.rows[side] = self.rows[side], list(p.abilities)
            if presses is not None:
                shown = before if before is not None else p.abilities
                seen = [
                    ability_row(shown[k]).cost
                    for t, k in presses
                    if t == p.team and 0 <= k < len(shown)
                ]
            elif before is not None:
                standing = {e.card_id for e in state.entities if e.team == p.team}
                seen = presses_seen(before, p.abilities, standing)
            else:
                seen = []
            # The engine pays every accepted command before the first tick of a step, so a
            # play or a press accepted at the earlier observation is paid there -- or, under
            # a command delay, the delay later, when it runs. A play is seen when its card
            # leaves the hand, which is when it runs: one the last state showed waiting runs
            # its ticks left after that state, and one accepted since runs the delay after.
            waiting = list(self.waiting[side])
            plays = []
            for card in cards_that_left(hand, p.hand):
                due = self.tick + self.delay[side]
                for i, row in enumerate(waiting):
                    if row[0] == 0 and row[1] == card:
                        due = self.tick + row[4]
                        del waiting[i]
                        break
                plays.append((min(max(due, self.tick), last), card))
            plays_by_side.append(plays)
            self.waiting[side] = list(p.pending)
            queue = self.press_queue[side] + [(self.tick + self.delay[side], c) for c in seen]
            presses_by_side.append([(max(due, self.tick), c) for due, c in queue if due <= last])
            self.press_queue[side] = [(due, c) for due, c in queue if due > last]
        self.advance(
            state.tick,
            state.regular_ticks,
            state.overtime,
            plays_by_side[0],
            plays_by_side[1],
            own_presses=presses_by_side[0],
            foe_presses=presses_by_side[1],
        )
        if self.law.to_milli(self.own_fine) != me.elixir_milli:
            self.exact = False
            self.own_fine = self.law.seed_fine(me.elixir_milli)
        if self.law.to_milli(self.foe_fine) != foe.elixir_milli:
            self.exact = False
        self.foe_hand = list(foe.hand)
        self.show_own_hand(me.hand, me.next_card)

    def show_own_hand(self, hand: Sequence[int], next_card: int) -> None:
        """The player's own hand and next card, which the player always sees."""
        self.own_hand = list(hand)
        self.own_cycle[0] = next_card
        self._note_own_cards()

    def advance(
        self,
        tick: int,
        regular_ticks: int,
        overtime: bool,
        own_plays: Sequence[tuple[int, int]],
        foe_plays: Sequence[tuple[int, int]],
        own_presses: Sequence[tuple[int, int]] = (),
        foe_presses: Sequence[tuple[int, int]] = (),
    ) -> None:
        """Move to ``tick`` through the plays made since the last tick, each ``(tick, card)``,
        and the ability presses, each ``(tick, elixir)``: a press is paid from the bar like a
        play and is no play (see ``MatchMemory``).

        THE ONE PLACE A PLAY CHANGES THIS MEMORY. ``observe`` reads plays off hand slots
        and dates them all at the previous observation; a caller holding a timed log of
        plays dates each one at its own tick. Both come here, so the counts, the cycle
        and the leak have one set of formulas.

        A play dated inside the interval splits the regeneration at that tick: the bar
        fills up to it, pays, and fills on. Splitting is exact, because regeneration is
        never negative, so clamping at a split point and again at the end gives the same
        bar and the same leak as clamping once. Plays dated at the start of the interval
        therefore cost the single ``ElixirLaw.advance`` call ``observe`` always made.

        A play the counted bar cannot pay is counted in ``unaffordable`` (own, enemy) and
        the bar floors at zero. The engine refuses such a play, so on an engine's own log
        this stays 0; anywhere else it means a missed play or a different elixir law.

        A MIRROR play is charged the card it copies plus its own elixir: its side's last
        play that was not a Mirror, which this memory has seen, since plays are public. A
        Mirror with no earlier play to copy is refused by the engine, so seeing one means a
        play was missed: it is charged its own elixir and ``exact`` goes False.

        ``own_last_play_tick`` becomes ``tick``, the moment the play is SEEN, not the
        moment it was made. That is what ``observe`` has always recorded, and a policy
        trained on it reads ``own_ticks_since_play`` that way.
        """
        clock = (tick, regular_ticks, overtime)
        self.own_fine, leaked = self._bar(self.own_fine, own_plays, own_presses, 0, *clock)
        self.foe_fine, _ = self._bar(self.foe_fine, foe_plays, foe_presses, 1, *clock)
        self.leak_fine += leaked
        for _, card in sorted(own_plays, key=lambda p: p[0]):
            self.own_cycle = [*self.own_cycle[1:], card]
            self.own_last_card = card
            self.own_last_play_tick = tick
        for _, card in sorted(foe_plays, key=lambda p: p[0]):
            self.foe_seen[card] = True
            self.foe_plays += 1
            self.foe_recent.append(card)
            del self.foe_recent[: -(DECK_SIZE - HAND_SIZE)]
        self.tick = tick

    def _bar(
        self,
        fine: int,
        plays: Sequence[tuple[int, int]],
        presses: Sequence[tuple[int, int]],
        side: int,
        tick: int,
        regular_ticks: int,
        overtime: bool,
    ) -> tuple[int, int]:
        """One bar from ``self.tick`` to ``tick``: (fine units, fine units lost to the cap)."""
        at, due, lost = self.tick, 0, 0
        paid = [(when, card, None) for when, card in plays]
        paid += [(when, None, cost) for when, cost in presses]
        for when, card, cost in sorted(paid, key=lambda p: p[0]):
            if not self.tick <= when < tick:
                raise ValueError(f"a play at tick {when} is outside [{self.tick}, {tick})")
            if when != at:
                fine, spilled = self.law.advance(fine, due, at, when, regular_ticks, overtime)
                at, due, lost = when, 0, lost + spilled
            due += self._price(card, side) if card is not None else cost
            if due * self.law.scale > fine:
                self.unaffordable[side] += 1
        fine, spilled = self.law.advance(fine, due, at, tick, regular_ticks, overtime)
        return fine, lost + spilled

    def _price(self, card: int, side: int) -> int:
        """What a play of ``card`` by ``side`` (0 own, 1 enemy) cost, and the copy target it
        leaves: a Mirror costs its side's last non-Mirror play plus its own and changes
        nothing; any other card costs its elixir and becomes that side's last play."""
        if not self.is_mirror[card]:
            self.last_play[side] = card
            return self.cost[card]
        copied = self.last_play[side]
        if copied == EMPTY_CARD:
            self.exact = False
            return self.cost[card]
        return self.cost[copied] + self.cost[card]

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


class MatchClock(NamedTuple):
    """Where a match is in time: everything the ``clock`` and ``elixir_rate`` fields read."""

    tick: int
    regular_ticks: int
    overtime_ticks: int
    overtime: bool
    elixir_rate: int  # 1, 2 or 3

    @classmethod
    def of(cls, state: BattleState) -> MatchClock:
        """The clock an engine reports."""
        return cls(
            state.tick, state.regular_ticks, state.overtime_ticks, state.overtime, state.elixir_rate
        )

    @classmethod
    def at(cls, tick: int, calibration: Calibration | None = None) -> MatchClock:
        """The clock of a match still running at ``tick``, from the rules alone.

        A match decided at the end of regulation has no later ticks, so one still running
        there is in overtime. The rate is ``ElixirLaw.rate_at``. Both rules are checked
        against MockEngine and RustEngine, tick by tick, in tests/test_fair_fields.py.
        """
        cal = calibration if calibration is not None else default_calibration()
        law = default_elixir_law() if calibration is None else ElixirLaw.load(cal)
        tick_ms = cal.int("time.TICK_MS")
        regular = -(-cal.int("match.REGULAR_TIME_S") * 1000 // tick_ms)
        overtime_ticks = -(-cal.int("match.OVERTIME_S") * 1000 // tick_ms)
        overtime = tick >= regular
        return cls(tick, regular, overtime_ticks, overtime, law.rate_at(tick, regular, overtime))


#: The fair vector fields that need no board, in vector order. ``fair_fields`` writes them.
FAIR_FIELDS = (
    "own_elixir",
    "enemy_elixir",
    "own_hand_cards",
    "own_hand_cost",
    "own_hand_affordable",
    "own_hand_pending",
    "own_pending_cost",
    "own_next_card",
    "own_cycle_6_8",
    "own_deck",
    "own_last_card",
    "own_ticks_since_play",
    "own_elixir_leaked",
    "enemy_cards_seen",
    "enemy_possible_hand",
    "enemy_plays",
    "clock",
    "elixir_rate",
)
#: The four fair fields the board decides. Only ``build_vector`` writes them.
BOARD_FIELDS = ("own_tower_hp", "enemy_tower_hp", "crowns", "king_active")


def fair_fields(
    memory: MatchMemory,
    clock: MatchClock,
    hand: Sequence[int],
    next_card: int,
    own_elixir_milli: int,
    cards: Sequence[CardInfo],
    max_mana: int,
    *,
    enemy_elixir_milli: int | None = None,
    enemy_last_card: bool = False,
    hand_costs: Sequence[int] | None = None,
    own_pending: Sequence[Sequence[int]] = (),
    evolutions: bool = False,
    evolution_progress: bool = False,
    own_evo: Sequence[Sequence[int]] = (),
) -> dict[str, np.ndarray]:
    """Every fair vector field but the board's four, by name, from what a player sees.

    THE CONTRACT. These are the exact numbers ``build_vector`` puts in the env's vector:
    it calls this function for them. So anything that can keep a ``MatchMemory`` without
    an engine -- dated plays through ``MatchMemory.advance`` -- gets the env's fields
    without building a state. The player supplies what a player sees anyway: the own
    hand, the next card, the own bar, and the clock (``MatchClock.of`` a state, or
    ``MatchClock.at`` a tick). ``memory`` must already have been moved to ``clock.tick``.

    ``enemy_elixir_milli`` is the true enemy bar for ``Reveal.enemy_elixir`` only; left
    at None, the field is the memory's count, which is the fair one.

    ``hand_costs`` is what each own hand slot costs right now (``PlayerState.hand_costs``:
    a Mirror costs the card it copies plus its own one, -1 when there is nothing to copy).
    Left at None, each slot is priced at its card's listed elixir, exactly as before it
    existed. A price p >= 0 is written p / MAX_MANA and is affordable when the own bar
    holds p; p < 0 keeps the listed elixir and is never affordable. The env passes the
    engine's prices; RoyaleImitate rebuilds the same ones from its log (from its cc78f2f).

    ``own_pending`` is the player's OWN commands accepted and not run yet, under a command
    delay (``PlayerState.pending``: [kind, what, x, y, ticks_left, cost] rows). A player
    knows its own taps, so it is fair: a hand slot whose card has a play waiting is flagged
    and not affordable, and the bar a new play is paid from is the own bar less the waiting
    cost, as the engine accepts. The own bar itself stays unspent until a command runs, as
    the client's does. The enemy's waiting commands are never an input: the client shows
    an opponent's play only when it runs, and so does the enemy count.

    ``own_evo`` is the player's OWN evolution counters (``PlayerState.evo``: [card_id,
    plays, next play evolved, cycle length] rows, one per evolved deck card), read only
    with ``evolutions``. The client shows the charge on a player's own cards, so it is
    fair; the enemy's is not shown, and nothing takes it. ``evolution_progress`` needs
    the cycle length, the fourth column, and refuses rows without it.

    Keys are ``FAIR_FIELDS`` in order, then ``enemy_last_card`` and the evolution fields
    when asked for. Each array is float32 and already clipped to [0, 1], as in the
    vector. They are views of one buffer, laid out in that order.
    """
    off, width = _fair_slots(len(cards), enemy_last_card, evolutions, evolution_progress)
    buf = np.zeros(width, dtype=np.float32)
    _write_fair(
        buf, off, memory, clock, hand, next_card, own_elixir_milli, cards, max_mana,
        enemy_elixir_milli, enemy_last_card, hand_costs, own_pending,
        own_evo if evolutions else None, evolution_progress,
    )
    np.clip(buf, 0.0, 1.0, out=buf)
    return {k: buf[s] for k, s in off.items()}


def _write_fair(
    out: np.ndarray,
    off: dict[str, slice],
    memory: MatchMemory,
    clock: MatchClock,
    hand: Sequence[int],
    next_card: int,
    own_elixir_milli: int,
    cards: Sequence[CardInfo],
    max_mana: int,
    enemy_elixir_milli: int | None,
    enemy_last_card: bool,
    hand_costs: Sequence[int] | None = None,
    own_pending: Sequence[Sequence[int]] = (),
    own_evo: Sequence[Sequence[int]] | None = None,
    evolution_progress: bool = False,
) -> None:
    """Write every fair field but the board's four into ``out`` at ``off``, NOT clipped.

    The one set of formulas behind ``fair_fields`` and ``build_vector``. Every field is
    computed as it was when each was an array of its own: the same Python arithmetic,
    rounded to float32 as it is stored. The caller clips once, after. The old path also
    clipped after rounding to float32, and rounding is monotone with 0 and 1 exact, so the
    bytes are the same. tests/test_vector_writer.py holds that against a frozen copy of
    the old path.

    WHY IT WRITES INTO ONE BUFFER. Building the vector used to make about twenty small
    arrays per seat per step, clip each, concatenate them and clip again, and that was
    the largest single cost of building an observation.
    """
    num_cards = len(cards)
    onehot = num_cards + 1
    full = 1000 * max_mana
    foe_milli = memory.enemy_elixir_milli() if enemy_elixir_milli is None else enemy_elixir_milli
    held = sum(int(row[5]) for row in own_pending)
    waiting = {int(row[1]) for row in own_pending if int(row[0]) == 0}
    flags = out[off["own_hand_pending"]]
    for i, c in enumerate(hand):
        flags[i] = 1.0 if c != EMPTY_CARD and c in waiting else 0.0
    out[off["own_pending_cost"]] = held / max_mana
    _write_hand(
        out[off["own_hand_cards"]], out[off["own_hand_cost"]], out[off["own_hand_affordable"]],
        hand, cards, own_elixir_milli - 1000 * held, num_cards, max_mana, hand_costs,
    )
    if waiting:
        out[off["own_hand_affordable"]] *= 1.0 - flags
    cycle = out[off["own_cycle_6_8"]].reshape(DECK_SIZE - HAND_SIZE - 1, onehot)
    for i, card in enumerate(memory.own_cycle[1:]):
        cycle[i, num_cards if card == EMPTY_CARD else card] = 1
    reg_left = max(0, clock.regular_ticks - clock.tick) / max(1, clock.regular_ticks)
    ot_left = 0.0
    if clock.overtime:
        ot_end = clock.regular_ticks + clock.overtime_ticks
        ot_left = max(0, ot_end - clock.tick) / max(1, clock.overtime_ticks)
    out[off["own_elixir"]] = own_elixir_milli / full
    out[off["enemy_elixir"]] = foe_milli / full
    _put_one(out[off["own_next_card"]], next_card, num_cards)
    out[off["own_deck"]] = memory.own_deck
    _put_one(out[off["own_last_card"]], memory.own_last_card, num_cards)
    out[off["own_ticks_since_play"]] = min(
        1.0, memory.ticks_since_own_play(clock.tick) / PLAY_GAP_TICKS
    )
    out[off["own_elixir_leaked"]] = min(1.0, memory.leaked_elixir() / LEAK_SCALE)
    out[off["enemy_cards_seen"]] = memory.foe_seen
    out[off["enemy_possible_hand"]] = memory.enemy_possible_hand()
    out[off["enemy_plays"]] = min(1.0, memory.foe_plays / PLAYS_SCALE)
    out[off["clock"]] = (reg_left, float(clock.overtime), ot_left)
    out[off["elixir_rate"]] = tuple(float(clock.elixir_rate == r) for r in (1, 2, 3))
    if enemy_last_card:
        # The newest entry of the cycle memory the vector already uses for
        # enemy_possible_hand, so it is derived from tested state, not kept twice.
        last = memory.foe_recent[-1] if memory.foe_recent else EMPTY_CARD
        _put_one(out[off["enemy_last_card"]], last, num_cards)
    if own_evo is not None:
        _write_evolutions(out, off, hand, next_card, own_evo, evolution_progress)


def _write_evolutions(
    out: np.ndarray,
    off: dict[str, slice],
    hand: Sequence[int],
    next_card: int,
    own_evo: Sequence[Sequence[int]],
    progress: bool,
) -> None:
    """The own evolution fields, from the own counters only. Module-level for plants."""
    rows = {int(r[0]): r for r in own_evo}
    evolved = out[off["own_hand_evolved"]]
    for i, c in enumerate(hand):
        row = rows.get(int(c)) if c != EMPTY_CARD else None
        evolved[i] = 1.0 if row is not None and int(row[2]) else 0.0
    row = rows.get(int(next_card)) if next_card != EMPTY_CARD else None
    out[off["own_next_evolved"]] = 1.0 if row is not None and int(row[2]) else 0.0
    if not progress:
        return
    bar = out[off["own_hand_evo_progress"]]
    for i, c in enumerate(hand):
        row = rows.get(int(c)) if c != EMPTY_CARD else None
        if row is None:
            bar[i] = 0.0
            continue
        if len(row) < 4 or int(row[3]) < 1:
            raise ValueError(
                f"evolution_progress needs each evolution counter's cycle length, the fourth "
                f"column of PlayerState.evo, and this engine's row for card {int(row[0])} is "
                f"{list(row)}. An engine from before that column cannot drive it; leave "
                "evolution_progress off, or update the engine."
            )
        bar[i] = min(1.0, int(row[1]) / int(row[3]))


@lru_cache(maxsize=64)
def _fair_slots(
    num_cards: int,
    enemy_last_card: bool,
    evolutions: bool = False,
    evolution_progress: bool = False,
) -> tuple[dict[str, slice], int]:
    """Where ``fair_fields`` puts each field in its own buffer, and the buffer's width.

    ``FAIR_FIELDS`` in order, then ``enemy_last_card`` and the evolution fields when
    asked for, packed, at the sizes ``vector_layout`` gives them. Cached and read-only.
    """
    layout = vector_layout(num_cards, None, enemy_last_card, evolutions, evolution_progress)
    size = {f.key: f.size for f in layout}
    extra = [
        *(("enemy_last_card",) if enemy_last_card else ()),
        *(("own_hand_evolved", "own_next_evolved") if evolutions else ()),
        *(("own_hand_evo_progress",) if evolution_progress else ()),
    ]
    off: dict[str, slice] = {}
    at = 0
    for key in (*FAIR_FIELDS, *extra):
        off[key] = slice(at, at + size[key])
        at += size[key]
    return off, at


@lru_cache(maxsize=64)
def _vector_slots(
    num_cards: int,
    reveal: Reveal,
    enemy_last_card: bool,
    evolutions: bool = False,
    evolution_progress: bool = False,
) -> tuple[dict[str, slice], int]:
    """``vector_offsets`` and the vector's width, cached and read-only.

    The layout is a function of these alone, and ``vector_layout`` is the one
    definition of it.
    """
    flags = (enemy_last_card, evolutions, evolution_progress)
    off = vector_offsets(num_cards, reveal, *flags)
    return off, sum(f.size for f in vector_layout(num_cards, reveal, *flags))


def build_vector(
    state: BattleState,
    team: int,
    cards: list[CardInfo],
    max_mana: int,
    reveal: Reveal,
    memory: MatchMemory,
    enemy_last_card: bool = False,
    evolutions: bool = False,
    evolution_progress: bool = False,
) -> np.ndarray:
    """The flat vector of ``vector_layout``. ``memory`` must already have seen ``state``.

    The fields a player's own view decides are ``fair_fields``' (both write them through
    ``_write_fair``); this adds the four the board decides and any reveal, each at its
    ``vector_layout`` slot in one float32 buffer, and clips once.
    """
    me, foe = state.players[team], state.players[1 - team]
    num_cards = len(cards)
    off, width = _vector_slots(
        num_cards, reveal, enemy_last_card, evolutions, evolution_progress
    )
    vec = np.zeros(width, dtype=np.float32)
    # Each own slot priced as the engine prices it (a Mirror: its copy plus one), where the
    # engine states the prices. RoyaleImitate passes the same prices rebuilt from its log
    # (RoyaleImitate cc78f2f, checked against these at every step), so the two agree.
    _write_fair(
        vec, off, memory, MatchClock.of(state), me.hand, me.next_card, me.elixir_milli,
        cards, max_mana, foe.elixir_milli if reveal.enemy_elixir else None, enemy_last_card,
        me.hand_costs if len(me.hand_costs) == HAND_SIZE else None, me.pending,
        me.evo if evolutions else None, evolution_progress,
    )
    vec[off["own_tower_hp"]] = [me.tower_hp[s] / max(1, me.tower_max_hp[s]) for s in TowerSlot]
    vec[off["enemy_tower_hp"]] = [
        foe.tower_hp[s] / max(1, foe.tower_max_hp[s]) for s in TowerSlot
    ]
    vec[off["crowns"]] = (me.crowns / 3.0, foe.crowns / 3.0)
    vec[off["king_active"]] = (float(me.king_active), float(foe.king_active))
    if reveal.enemy_hand:
        # Cost and affordability are worked out and dropped, as they always were: the
        # reveal carries the card one-hots only.
        _write_hand(
            vec[off["enemy_hand_cards"]], np.zeros(HAND_SIZE, dtype=np.float32),
            np.zeros(HAND_SIZE, dtype=np.float32), foe.hand, cards, foe.elixir_milli,
            num_cards, max_mana,
        )
    if reveal.enemy_next_card:
        _put_one(vec[off["enemy_next_card"]], foe.next_card, num_cards)
    if reveal.enemy_deck:
        deck = vec[off["enemy_deck"]]
        for c in (*foe.hand, foe.next_card):
            if c != EMPTY_CARD:
                deck[c] = 1
        deck[memory.foe_seen] = 1
    np.clip(vec, 0.0, 1.0, out=vec)
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

    THE SAME STATES IS A CONSTRAINT, NOT A CONVENIENCE. With 0.4% of cells moving,
    the baseline is dominated by which states were sampled: boards that differ only
    in elixir barely move the number, a board with a tower down moves it much more.
    Two people measuring the SAME encoder against baselines taken on different state
    sets will disagree about that encoder, and both will be right about what they
    measured. Take the baseline on the states the encoding was measured on, or the
    two numbers are not a pair.

    WHAT THIS IS NOT FOR: comparing two DIFFERENT builders with each other. The
    number is built to compare one representation against ITSELF -- an encoding
    against the input it came from, on the same states, as a ratio. Across builders
    it is not measuring the same property twice. ``EntityListObsBuilder`` is mostly
    empty canonically-sorted rows where one unit moving can permute a whole row;
    ``SpatialObsBuilder`` is a dense grid of counts where the same unit touches two
    tiles. Different sparsity, different magnitudes, different response to a small
    change in the state, so a lower cosine on one may mean it discriminates less or
    may mean cosine reads a sorted sparse row-set differently from a dense grid, and
    nothing in the scalar separates those. Measured, for the record: on eight boards
    the entity-list builder scores 0.9997 on its moving cells against the spatial
    builder's 0.9841, and that difference is NOT evidence that one is worse.

    What would settle it is whether a policy trained on each can tell the boards
    apart, which is a training question; a cheaper proxy is whether a small probe
    can recover a known state variable from each, which measures usable information
    rather than geometric spread. Neither is this function.

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


def _put_one(v: np.ndarray, card: int, empty_index: int) -> None:
    """One-hot ``card`` into the zeroed vector field ``v``; an empty slot sets ``empty_index``."""
    v[empty_index if card == EMPTY_CARD else card] = 1


def _write_hand(
    one_flat: np.ndarray,
    cost: np.ndarray,
    afford: np.ndarray,
    hand: Sequence[int],
    cards: Sequence[CardInfo],
    elixir_milli: int,
    num_cards: int,
    max_mana: int,
    hand_costs: Sequence[int] | None = None,
) -> None:
    """Card one-hots, costs and affordable flags for one hand, into zeroed views. A slot
    is priced at ``hand_costs`` where given (see ``fair_fields``), else its listed elixir."""
    one = one_flat.reshape(HAND_SIZE, num_cards + 1)
    for i, c in enumerate(hand):
        if c == EMPTY_CARD:
            one[i, num_cards] = 1
            continue
        one[i, c] = 1
        listed = cards[c].elixir
        price = listed if hand_costs is None else int(hand_costs[i])
        cost[i] = (price if price >= 0 else listed) / max_mana
        afford[i] = 1.0 if price >= 0 and elixir_milli >= price * 1000 else 0.0


# ---------------------------------------------------------------------------
# The base builder
# ---------------------------------------------------------------------------


class ObsBuilder(ABC):
    """state -> observation dict. Always includes ``action_mask``."""

    calibration: Calibration
    reveal: Reveal
    #: Whether the vector carries ``enemy_last_card``. Only SpatialObsBuilder's
    #: ``card_identity`` turns it on; every other builder keeps the shipped vector.
    enemy_last_card: bool = False
    #: Whether the vector carries the own evolution fields (and their progress), and the
    #: spatial planes the evolved units. Only SpatialObsBuilder's ``evolutions`` sets them.
    evolutions: bool = False
    evolution_progress: bool = False

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
        rules = engine.rules()
        # The parser's oracle when it was built from the same arena, rules and cards. Its
        # grids are memoised on everything they read (``PlacementOracle.grid_key``), so one
        # oracle serves both: Blue's ``enemy_troop_zone`` is the very grid Red's troop
        # mask needs, and with two oracles it was worked out twice per step.
        shared = getattr(action_parser, "oracle", None)
        if (
            isinstance(shared, PlacementOracle)
            and shared.arena == self.arena
            and shared.rules == rules
            and shared.cards == self.cards
        ):
            self.oracle = shared
        else:
            self.oracle = PlacementOracle(self.arena, rules, self.cards)
        self.mask_space = spaces.Box(0, 1, shape=(int(action_parser.space.n),), dtype=np.int8)
        self.mask_plane_shape = action_parser.mask_plane_shape()
        # A parser with ability buttons puts them last in the mask; they also go out on
        # their own as ``ability_ready``, the minimal observation of a hero's or a
        # champion's ability.
        with_buttons = getattr(action_parser, "ability_buttons", False)
        self.ability_buttons = (
            int(getattr(action_parser, "n_buttons", ABILITY_BUTTONS)) if with_buttons else 0
        )
        # A troop by KIND: a spell may carry a troop's placement (Heal) without its laws.
        self._troop_probe = next(
            (c for c in self.cards if c.placement == Placement.TROOP and not card_is_spell(c)),
            None,
        )
        self.max_mana = self.calibration.int("match.MAX_MANA")
        self.law = ElixirLaw.load(self.calibration)
        self.memory = {t: MatchMemory(self.num_cards, self.law) for t in TEAMS}
        for m in self.memory.values():
            m.bind(self.cards)
        self.vec_size = sum(f.size for f in self.vector_layout())

    def reset(self, state: BattleState) -> None:
        """Called at the start of every episode: both seats forget the last one."""
        self.presses = None
        delay = getattr(self, "command_delay", (0, 0))
        for team, memory in self.memory.items():
            memory.seed(state, team)
            memory.delay = [delay[team], delay[1 - team]]

    def see_presses(self, presses: Sequence[tuple[int, int]] | None) -> None:
        """The ability presses accepted since the last build, each ``(team, button)``, for
        the memories to charge (``MatchMemory.observe``); None reads them off the rows."""
        self.presses = None if presses is None else list(presses)

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
        return vector_layout(
            self.num_cards, self.reveal, self.enemy_last_card, self.evolutions,
            self.evolution_progress,
        )

    def vector_offsets(self) -> dict[str, slice]:
        """``key -> slice`` into the flat vector this builder writes."""
        return vector_offsets(
            self.num_cards, self.reveal, self.enemy_last_card, self.evolutions,
            self.evolution_progress,
        )

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
        if self.ability_buttons:
            out["ability_ready"] = spaces.Box(0, 1, shape=(self.ability_buttons,), dtype=np.int8)
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
            n = int(np.prod(self.mask_plane_shape))
            out["mask_planes"] = flat[1 : 1 + n].reshape(self.mask_plane_shape)
        if self.ability_buttons:
            out["ability_ready"] = flat[len(flat) - self.ability_buttons :]
        return out

    def _vector(self, state: BattleState, team: int) -> np.ndarray:
        memory = self.memory[team]
        # By keyword and only when there are any, as below: an observe that wraps or stands in
        # for this one with the old (state, team) keeps working, and no press accepted since
        # the last build is the same as the rows showing none.
        presses = getattr(self, "presses", None)
        if presses:
            memory.observe(state, team, presses=presses)
        else:
            memory.observe(state, team)
        # The flag goes only when it is ON, and by keyword. With it off this is the exact call
        # it was before D2, so anything that wraps or substitutes build_vector with the old
        # six arguments keeps working -- this suite's own plant tests do, and the first
        # version of this line broke two of them by always passing a seventh.
        extra: dict[str, bool] = {"enemy_last_card": True} if self.enemy_last_card else {}
        if self.evolutions:
            extra["evolutions"] = True
        if self.evolution_progress:
            extra["evolution_progress"] = True
        return build_vector(
            state, team, self.cards, self.max_mana, self.reveal, memory, **extra
        )

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


#: The two planes ``SpatialObsBuilder(evolutions=True)`` adds, after the fair ones and
#: before any revealed one. Fair: an evolved unit looks different on the board.
EVOLVED_SPATIAL_CHANNELS: list[tuple[str, str]] = [
    ("own_evolved", "count of own evolved units (troops and buildings) whose centre is in the tile"),
    ("enemy_evolved", "count of enemy evolved units whose centre is in the tile"),
]


def spatial_channels(
    reveal: Reveal | None = None, evolutions: bool = False
) -> list[tuple[str, str]]:
    """The spatial channels: the fair ones, the evolved-unit pair when asked for, then the
    revealed ones. The fair block keeps its offsets either way."""
    rev = reveal or Reveal()
    out = list(FAIR_SPATIAL_CHANNELS)
    if evolutions:
        out.extend(EVOLVED_SPATIAL_CHANNELS)
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


def evolved_channels(entities: Sequence[EntityState], team: int, arena: Arena) -> np.ndarray:
    """float32 [2, tiles_y, tiles_x], seen by ``team``: own, then enemy, evolved units.

    Counted on the centre tile, as ``entity_channels`` counts troops, from each unit's
    ``STATUS_EVOLVED`` bit. A hero's unit carries ``STATUS_HERO`` instead and is not
    counted. An engine that does not report the bits (``status_flags`` -1) is refused:
    reading "not reported" as "not evolved" would hand a network zeros that look like an
    answer. Module-level so a test can plant a defect in it.
    """
    acc = np.zeros((2, arena.tiles_y, arena.tiles_x), dtype=np.int64)
    for e in entities:
        status = status_of(e)  # towers too: an engine that reports the bits reports theirs
        if status is None:
            raise ValueError(
                f"evolutions=True, but this engine does not report which units are evolved "
                f"(entity uid {e.uid} has no status_flags). Use an engine that reports them, "
                "such as RustEngine, or leave evolutions off."
            )
        if status & STATUS_EVOLVED and e.kind not in TOWER_KINDS:
            ty, tx = _tile(arena, team, e.x, e.y)
            acc[0 if e.team == team else 1, ty, tx] += 1
    out: np.ndarray = acc.astype(np.float32)
    return out


#: ``card_ids`` value for a tile nothing stands on.
CARD_ID_EMPTY = 0
#: ``card_ids`` value for a crown tower, the only entity class that carries no card. Kept
#: apart from EMPTY so that 0 never means both "empty ground" and "a tower stands here" --
#: which also makes a destroyed tower's tile read as genuinely empty.
CARD_ID_TOWER = 1
#: A catalogue card id ``c`` is written as ``CARD_ID_OFFSET + c``.
CARD_ID_OFFSET = 2
TOWER_KINDS = (EntityKind.KING_TOWER, EntityKind.PRINCESS_TOWER)


def card_id_planes(
    entities: Sequence[EntityState], team: int, arena: Arena, num_cards: int
) -> np.ndarray:
    """uint8 [2, tiles_y, tiles_x], seen by ``team``: plane 0 own, plane 1 enemy.

    Which CARD occupies each tile, which the float ``spatial`` planes cannot say: they
    count troops and sum hp, so a Giant and a Knight on one tile look alike
    (docs/observation-spec.md 3b; decision D2). Tiles are the same own-frame centre tiles
    ``entity_channels`` uses, so the two line up cell for cell.

    TIES GO TO THE LOWEST UID, and that is not cosmetic. ``BattleState.entities`` is not in
    uid order -- a live battle gives [0, 2, 4, 1, 3, 5] -- so "whichever comes first" would
    make a plane a function of iteration order, and two runs of one seed could differ. A
    uid is unique for a whole battle and never reused.

    Spells in flight never appear: ``BattleState.spells`` is a separate list, so a live
    spell has no entity and no tile. Units a spell releases do appear, under the releasing
    spell's catalogue id, because that is what the engine reports for them.

    Module-level so a test can plant a defect in it.
    """
    out = np.zeros((2, arena.tiles_y, arena.tiles_x), dtype=np.uint8)
    for e in sorted(entities, key=lambda ent: ent.uid):
        ox, oy = to_own(arena, team, e.x, e.y)
        tx = min(max(ox // arena.subtile, 0), arena.tiles_x - 1)
        ty = min(max(oy // arena.subtile, 0), arena.tiles_y - 1)
        plane = 0 if e.team == team else 1
        if out[plane, ty, tx] != CARD_ID_EMPTY:
            continue  # a lower uid already holds this tile
        if e.kind in TOWER_KINDS:
            out[plane, ty, tx] = CARD_ID_TOWER
        elif 0 <= e.card_id < num_cards:
            out[plane, ty, tx] = CARD_ID_OFFSET + e.card_id
        else:
            # Writing it anyway would put an id outside the declared vocabulary into an
            # embedding lookup, or fold an unknown entity into "tower". Neither is a value.
            raise ValueError(
                f"entity uid {e.uid} (kind {e.kind}) has card_id {e.card_id}, outside the "
                f"{num_cards}-card catalogue, and is not a crown tower, so card_ids has no "
                "index for it"
            )
    return out


class SpatialObsBuilder(ObsBuilder):
    """Dict(spatial [C, 32, 18], mask_planes [4, 32, 18], vector [V], action_mask [A]).

    Entities are rasterised by the tile containing their centre in the own frame
    (``x_own // SUBTILE``, clamped). ``channel_names()`` lists the channels;
    ``spatial_channels(reveal)`` is the same list with each channel's meaning.

    ``card_identity=True`` (D2; OFF by default) adds two things, as one switch because
    they are one change to what a network sees: a ``card_ids`` key, uint8 [2, 32, 18],
    naming the card on each tile (``card_id_planes``), and ``enemy_last_card`` in the
    vector. It is its OWN key and not a channel of ``spatial``, because ``spatial`` is a
    float Box: a learner's codec stores a float box as a scaled half, a card id comes
    back as 6.997, and ``.long()`` reads card 6 with nothing failing.

    THE VOCABULARY SIZE is ``num_cards + 2`` from the LOADED table, and it is published
    as the ``card_ids`` Box's upper bound plus one, so a network sizes its embedding from
    ``observation_space["card_ids"].high.max() + 1`` at construction.

    CATALOGUE IDS ARE POSITIONAL: making one more card loadable renumbers every later id,
    and an embedding indexed by them would read a different game from the same checkpoint
    with nothing failing. So ``config()`` records the card NAMES in order, and a builder
    constructed with ``card_names`` REFUSES to bind to an engine whose catalogue differs.
    """

    def __init__(
        self,
        reveal: Reveal | None = None,
        calibration: Calibration | None = None,
        card_identity: bool = False,
        card_names: Sequence[str] | None = None,
        evolutions: bool = False,
        evolution_progress: bool = False,
    ) -> None:
        super().__init__(reveal, calibration)
        self.card_identity = bool(card_identity)
        self.enemy_last_card = self.card_identity
        if evolution_progress and not evolutions:
            raise ValueError("evolution_progress needs evolutions=True: it extends those fields")
        self.evolutions = bool(evolutions)
        self.evolution_progress = bool(evolution_progress)
        if card_names is not None and not self.card_identity:
            raise ValueError(
                "card_names pins the card_ids vocabulary, which only exists with "
                "card_identity=True"
            )
        self._pinned_card_names = list(card_names) if card_names is not None else None

    def _check_card_names(self) -> list[str]:
        """The catalogue's names, or a refusal if they are not the ones pinned."""
        names = [c.name for c in self.cards]
        pinned = self._pinned_card_names
        if pinned is not None and pinned != names:
            i = next(
                (k for k, (a, b) in enumerate(zip(pinned, names, strict=False)) if a != b),
                min(len(pinned), len(names)),
            )
            was = pinned[i] if i < len(pinned) else "<end>"
            now = names[i] if i < len(names) else "<end>"
            raise ValueError(
                f"card_ids was built for a catalogue that has {was!r} at id {i}; this "
                f"engine has {now!r} there ({len(pinned)} pinned names, {len(names)} "
                "loaded). Catalogue ids are positional, so every card_ids value from here "
                "on would name a different card than the checkpoint learned. Refusing "
                "rather than reinterpreting."
            )
        return names

    def bind(self, engine: Engine, action_parser: ActionParser) -> None:
        super().bind(engine, action_parser)
        a = self.arena
        self.channels = spatial_channels(self.reveal, self.evolutions)
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
        entries: dict[str, spaces.Space[Any]] = {
            "spatial": spaces.Box(0.0, SPATIAL_CLIP, shape=self.shape, dtype=np.float32),
            "vector": spaces.Box(0.0, 1.0, shape=(self.vec_size,), dtype=np.float32),
            **self._mask_space_entries(),
        }
        if self.card_identity:
            self.card_names = self._check_card_names()
            self.card_vocab = self.num_cards + CARD_ID_OFFSET
            top = int(np.iinfo(np.uint8).max)
            if self.card_vocab - 1 > top:
                raise ValueError(
                    f"card_ids needs {self.card_vocab} values for {self.num_cards} cards and "
                    f"uint8 holds {top + 1}. Widening the dtype is a storage decision for "
                    "every consumer of this key, so it is refused here rather than made "
                    "silently."
                )
            entries["card_ids"] = spaces.Box(
                0, self.card_vocab - 1, shape=(2, a.tiles_y, a.tiles_x), dtype=np.uint8
            )
        self._space = spaces.Dict(entries)

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
        if self.evolutions:
            at = idx["own_evolved"]
            sp[at : at + 2] = evolved_channels(state.entities, team, self.arena)
        np.clip(sp, 0.0, SPATIAL_CLIP, out=sp)
        out: dict[str, Any] = {
            "spatial": sp,
            "vector": self._vector(state, team),
            **self._mask_entries(action_mask),
        }
        if self.card_identity:
            out["card_ids"] = card_id_planes(state.entities, team, self.arena, self.num_cards)
        return out

    def config(self) -> dict[str, Any]:
        """Constructor state, including the card names the ``card_ids`` ids refer to."""
        out = super().config()
        if self.evolutions:
            out["evolutions"] = True
        if self.evolution_progress:
            out["evolution_progress"] = True
        if self.card_identity:
            out["card_identity"] = True
            names = getattr(self, "card_names", None) or self._pinned_card_names or []
            out["card_names"] = list(names)
        return out


# ---------------------------------------------------------------------------
# Entity-list builder (for attention / transformer policies)
# ---------------------------------------------------------------------------


def entity_row_key(row: tuple[Any, ...]) -> tuple[Any, ...]:
    """Sort key of an EntityListObsBuilder row: every field but the entity itself.
    Module-level so a test can plant a shorter six-field key back in."""
    return row[:-1]


# Which of the four motion bits each engine motion sets. A new motion maps onto an old
# bit, so a spells row keeps its width (a learner's network is sized by that width).
# FUSE reads as FLIGHT, not AREA: a Rage bottle's row (its centre, aim at the centre,
# delay = the fuse left) is otherwise the same row as the Rage area it releases in that
# area's last ticks, and "Rage starts in 10 ticks" would read as "Rage ends in 10 ticks".
# FLIGHT already means "acts at its aim after delay_ticks". STRIKES (Lightning) and
# SCHEDULED (the Graveyard) are each the only motion their card has, so the card tells
# them apart from any other area.
MOTION_BIT = {
    SpellMotion.FLIGHT: SpellMotion.FLIGHT,
    SpellMotion.AIRBORNE: SpellMotion.AIRBORNE,
    SpellMotion.ROLLING: SpellMotion.ROLLING,
    SpellMotion.AREA: SpellMotion.AREA,
    SpellMotion.PULSING: SpellMotion.AREA,
    SpellMotion.FUSE: SpellMotion.FLIGHT,
    SpellMotion.STRIKES: SpellMotion.AREA,
    SpellMotion.SCHEDULED: SpellMotion.AREA,
}


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
        area; ``MOTION_BIT`` maps every engine motion onto these four: a fuse sets
        flight, and pulsing, strikes and scheduled set area; a motion code it does not
        name is refused, on every row and not only the kept ones), 7 x_own / width,
        8 y_own / height, 9 and 10 the aim point of YOUR spells (0 on an enemy row),
        11 delay_ticks / 100 (clipped; a pulsing area's life left, a fuse's time left),
        12 travelled /
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
            if sp.motion not in MOTION_BIT:
                # Checked on EVERY row, before any is dropped past max_spells: a check on
                # the kept rows alone passes until the unknown one sorts into them.
                # Written as nothing, it would be a spell with no motion bit set, which a
                # network reads as a motion of its own that no test ever named.
                raise ValueError(
                    f"spell motion {sp.motion} (card {sp.card_id}) is not a protocol."
                    "SpellMotion. The engine added a motion; add it to SpellMotion and "
                    "to obs.MOTION_BIT, deciding which bit it sets."
                )
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
            f[3 + MOTION_BIT[motion]] = 1
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

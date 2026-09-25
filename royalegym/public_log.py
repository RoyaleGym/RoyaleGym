"""The fair observation fields a timed log of card plays determines, with no engine.

WHAT IT IS FOR. A log of who played which card when, plus the rules everyone knows,
fixes most of what a player remembers: both elixir bars, the own hand and cycle, the
cards the opponent has shown, how long since the last play. ``PublicLogMemory`` rebuilds
those fields from such a log, so a model can be trained on recorded matches without
replaying them through an engine, and read exactly the numbers the env would show.

ONE SET OF FORMULAS. It holds a ``MatchMemory`` and moves it with
``MatchMemory.advance``, the same call the env's ``observe`` makes, and it reads the
fields through ``build_vector``, the function that writes the env's vector. The only
thing this module adds is where the inputs come from: plays from the log instead of
from hand slots changing, and the hand from the deck order instead of from the engine.

WHAT IT CANNOT FILL. The board: tower hitpoints, crowns and which kings are awake
(``BOARD_FIELDS``). A log of plays does not say what the plays did.

THE ASSUMPTIONS, EACH CHECKED AGAINST AN ENGINE IN tests/test_public_log.py:
    * a played card's hand slot takes the next card, and the played card goes to the
      back of the queue;
    * a play at tick p is paid before tick p runs and shows in any observation at a
      tick after p;
    * a match still running at the end of regulation is in overtime;
    * the elixir rate at a tick is ``ElixirLaw.rate_at``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .obs import MatchMemory, Reveal, build_vector, vector_offsets
from .protocol import (
    DECK_SIZE,
    EMPTY_CARD,
    HAND_SIZE,
    BattleState,
    Calibration,
    CardInfo,
    ElixirLaw,
    PlayerState,
    Winner,
    default_calibration,
)

#: Vector fields a log of plays cannot fill. ``observe`` leaves them out.
BOARD_FIELDS = ("own_tower_hp", "enemy_tower_hp", "crowns", "king_active")

_OWN, _ENEMY = 0, 1  # seats of the stand-in state build_vector reads


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


class PublicLogMemory:
    """One seat's fair, non-board observation fields, kept from a timed log of plays.

    ``cards`` is the run's catalogue, in the run's order: ids are positions, and the
    one-hot fields are as wide as it is. ``deck_order`` is the seat's own eight cards
    as the match dealt them: the first four are the hand, slot by slot, and the rest
    the queue, next card first.

    Feed plays with ``own_play`` and ``enemy_play``, in any amount ahead of time, and
    read the fields with ``observe(tick)``. An observation at tick T sees exactly the
    plays made before T, as the env's does.

    ``unaffordable`` counts plays the counted bar could not pay, (own, enemy). The
    engine refuses those, so on a log an engine produced it stays (0, 0). Anywhere else
    a non-zero count means a play is missing from the log or the elixir law is not the
    one this memory counts with, and from then on the elixir fields are estimates.
    """

    def __init__(
        self,
        cards: Sequence[CardInfo],
        deck_order: Sequence[int],
        *,
        calibration: Calibration | None = None,
        start_tick: int = 0,
        own_elixir_milli: int | None = None,
        enemy_elixir_milli: int | None = None,
        enemy_last_card: bool = False,
    ) -> None:
        self.cards = list(cards)
        n = len(self.cards)
        deck = [int(c) for c in deck_order]
        if len(deck) != DECK_SIZE or len(set(deck)) != DECK_SIZE:
            raise ValueError(f"deck_order must be {DECK_SIZE} distinct card ids, got {deck}")
        if not all(0 <= c < n for c in deck):
            raise ValueError(f"deck_order {deck} has ids outside a {n}-card catalogue")
        cal = calibration if calibration is not None else default_calibration()
        self.law = ElixirLaw.load(cal)
        self.max_mana = cal.int("match.MAX_MANA")
        self.tick_ms = cal.int("time.TICK_MS")
        self.regular_ticks = _ceil_div(cal.int("match.REGULAR_TIME_S") * 1000, self.tick_ms)
        self.overtime_ticks = _ceil_div(cal.int("match.OVERTIME_S") * 1000, self.tick_ms)
        start = 1000 * cal.int("match.START_MANA")
        self.enemy_last_card = enemy_last_card
        self.hand = deck[:HAND_SIZE]
        self.queue = deck[HAND_SIZE:]
        self._pending: list[tuple[int, int, int, int]] = []  # (tick, order fed, side, card)
        self._fed = 0
        self.memory = MatchMemory(n, self.law)
        self.memory.bind(self.cards)
        self.memory.seed(
            self._state(
                start_tick,
                start if own_elixir_milli is None else own_elixir_milli,
                start if enemy_elixir_milli is None else enemy_elixir_milli,
            ),
            _OWN,
        )
        self._offsets = {
            k: s
            for k, s in vector_offsets(n, None, enemy_last_card).items()
            if k not in BOARD_FIELDS
        }

    # -- the log ---------------------------------------------------------------

    def own_play(self, tick: int, card: int) -> None:
        self._feed(tick, _OWN, card)

    def enemy_play(self, tick: int, card: int) -> None:
        self._feed(tick, _ENEMY, card)

    def _feed(self, tick: int, side: int, card: int) -> None:
        if tick < self.memory.tick:
            raise ValueError(
                f"a play at tick {tick} arrived after tick {self.memory.tick} was observed"
            )
        if not 0 <= card < len(self.cards):
            raise ValueError(f"card id {card} is outside a {len(self.cards)}-card catalogue")
        self._pending.append((tick, self._fed, side, card))
        self._fed += 1

    # -- reading ---------------------------------------------------------------

    @property
    def unaffordable(self) -> tuple[int, int]:
        own, enemy = self.memory.unaffordable
        return own, enemy

    def overtime_at(self, tick: int) -> bool:
        """Whether a match still running at ``tick`` is in overtime."""
        return tick >= self.regular_ticks

    def fields(self) -> list[str]:
        """The field names ``observe`` returns, in vector order."""
        return list(self._offsets)

    def observe(self, tick: int) -> dict[str, np.ndarray]:
        """The fields at ``tick``, from every play made before it. The clock only moves on."""
        memory = self.memory
        if tick < memory.tick:
            raise ValueError(f"tick {tick} is before the last observation, {memory.tick}")
        if tick > memory.tick:
            due = sorted(p for p in self._pending if p[0] < tick)
            self._pending = [p for p in self._pending if p[0] >= tick]
            own: list[tuple[int, int]] = []
            enemy: list[tuple[int, int]] = []
            for when, _, side, card in due:
                if side == _OWN:
                    self._cycle(when, card)
                    own.append((when, card))
                else:
                    enemy.append((when, card))
            memory.advance(tick, self.regular_ticks, self.overtime_at(tick), own, enemy)
            memory.show_own_hand(self.hand, self.queue[0])
        state = self._state(
            tick, self.law.to_milli(memory.own_fine), self.law.to_milli(memory.foe_fine)
        )
        extra = {"enemy_last_card": True} if self.enemy_last_card else {}
        vec = build_vector(state, _OWN, self.cards, self.max_mana, Reveal(), memory, **extra)
        return {k: vec[s].copy() for k, s in self._offsets.items()}

    # -- internals ---------------------------------------------------------------

    def _cycle(self, tick: int, card: int) -> None:
        """The played slot takes the next card; the played card joins the back of the queue."""
        if card not in self.hand:
            raise ValueError(
                f"own play of card {card} at tick {tick}, but the hand is {self.hand}: "
                "the log or the deck order is wrong"
            )
        self.hand[self.hand.index(card)] = self.queue.pop(0)
        self.queue.append(card)

    def _state(self, tick: int, own_milli: int, enemy_milli: int) -> BattleState:
        """A stand-in for the state build_vector reads. Its board is empty and unread."""
        overtime = self.overtime_at(tick)

        def player(team: int, milli: int, hand: list[int], next_card: int) -> PlayerState:
            return PlayerState(
                team=team,
                elixir_milli=milli,
                hand=hand,
                next_card=next_card,
                crowns=0,
                tower_hp=[0, 0, 0],
                tower_max_hp=[1, 1, 1],
                king_active=False,
            )

        return BattleState(
            tick=tick,
            tick_ms=self.tick_ms,
            regular_ticks=self.regular_ticks,
            overtime_ticks=self.overtime_ticks,
            elixir_rate=self.law.rate_at(tick, self.regular_ticks, overtime),
            overtime=overtime,
            players=[
                player(_OWN, own_milli, list(self.hand), self.queue[0]),
                player(_ENEMY, enemy_milli, [EMPTY_CARD] * HAND_SIZE, EMPTY_CARD),
            ],
            entities=[],
            game_over=False,
            winner=Winner.NONE,
        )

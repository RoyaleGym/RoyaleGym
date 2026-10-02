#!/usr/bin/env python3
"""Showcase tile: a to-scale map of what one seat's observation vector holds about the other.

Three runs are drawn, and every position and size is read out of
``royalegym.obs.vector_layout`` / ``vector_offsets`` for this checkout's card table:

* the run your OWN hand occupies;
* the run holding what this seat worked out for itself (the enemy cards it has seen, the
  cards that could be in the enemy hand now by the cycle rule, the plays counted);
* the run ``Reveal(enemy_hand=True)`` APPENDS, which is the enemy hand and comes out
  exactly as wide as your own -- and is not in the default vector at all. Hidden is absent
  here, not zeroed, so a checkpoint cannot quietly be trained on one and evaluated on the
  other. That the two are the same width is a fact about the layout, not a coincidence
  arranged here: both are ``HAND_SIZE`` one-hots over the card table.

The two totals come from a different source than the map: two environments are built and
``obs["vector"].shape`` is read off the arrays they hand out. The module asserts the built
widths match the declared layout, that enabling the reveal moved no fair field, and that
it changed nothing outside the vector -- which is what lets the tile draw the vector and
say "the observation's vector" rather than "the observation".

The counted enemy elixir is checked after EVERY step, on both seats, against the engine's
own bar -- never at reset, where the count was just seeded from that bar and agreement
would be free. Beside it on the tile runs a control: the same counter, blind to the first
play it should have seen. A check that could not fail would be no evidence, and this one
prints the number of times the blinded counter is caught.

What the tile does NOT claim: that the enemy hand is unreachable inside the library.
``MatchMemory`` spots a play by the opponent's hand array changing, which is the public
event "a unit appeared" but is read from the battle state. The claim drawn here is about
what reaches the policy, which is the vector.

The battle is one battle: a different 8-card deck each side (not a mirror, which is the
one arrangement where a defect hitting both seats the same way cannot show), moves drawn
at random from the legal ones, on the seed the READMEs quote. Nothing is drawn that was
not measured in this run.
"""

from __future__ import annotations

import pathlib
from itertools import pairwise

import make_media as M

SEED = 0
# A different deck each side, and by NAME: an id is a position in a catalogue that moves
# between tables. Two decks rather than one mirrored deck, because a mirror is the one
# configuration in which a defect that hits both seats the same way cannot show up.
DECKS = (("Knight", "Archer", "Giant", "Minions", "Fireball", "Cannon", "Zap", "Musketeer"),
         ("Goblins", "Skeletons", "Valkyrie", "Bomber", "HogRider", "Tesla", "Arrows",
          "MiniPekka"))
NOOP_PROB = 0.7


# --------------------------------------------------------------------------- measuring


def _decks(engine):
    by_name = {c.name: c.card_id for c in engine.cards()}
    return [[by_name[n] for n in names] for names in DECKS]


def _env(engine, reveal):
    from royalegym import ClashParallelEnv, DefaultStateMutator, SpatialObsBuilder

    return ClashParallelEnv(engine=engine, obs_builder=SpatialObsBuilder(reveal=reveal),
                            state_mutator=DefaultStateMutator(decks=_decks(engine)))


def _map() -> dict:
    """Widths from the built arrays, runs from the declared layout, and they must agree."""
    from royalegym import Reveal, RustEngine
    from royalegym.obs import vector_layout, vector_offsets

    engine = RustEngine()
    n = len(engine.cards())
    built, planes = {}, {}
    for key, reveal in (("fair", Reveal()), ("hand", Reveal(enemy_hand=True))):
        env = _env(engine, reveal)
        obs, _ = env.reset(seed=SEED)
        built[key] = int(obs[env.agents[0]]["vector"].shape[0])
        planes[key] = {k: v.shape for k, v in obs[env.agents[0]].items() if k != "vector"}
    # The tile draws the vector and says so, which is only fair if this reveal leaves the
    # rest of the observation alone. If a later one does not, this stops the build.
    assert planes["fair"] == planes["hand"], "Reveal(enemy_hand) moved more than the vector"

    fair_at = vector_offsets(n, Reveal())
    rev_at = vector_offsets(n, Reveal(enemy_hand=True))
    for layout, width in ((vector_layout(n, Reveal()), built["fair"]),
                          (vector_layout(n, Reveal(enemy_hand=True)), built["hand"])):
        assert sum(f.size for f in layout) == width, "declared layout and built vector differ"
    for key, sl in fair_at.items():  # a reveal must not move a fair field
        assert rev_at[key] == sl, f"Reveal moved the fair field {key}"

    # What this seat worked out for itself: the cards it has seen, the cards that could be
    # in the enemy hand now, and how many plays it has counted. One contiguous run.
    worked = [fair_at[k] for k in ("enemy_cards_seen", "enemy_possible_hand", "enemy_plays")]
    assert all(a.stop == b.start for a, b in pairwise(worked)), "the run is not contiguous"

    # Two sentences this module prints would go stale silently if the layout moved under
    # them: that the revealed block is as wide as your own hand, and that a seat's
    # opponent holds as many cards as it does. Neither is drawn as a check on the tile.
    own, foe = fair_at["own_hand_cards"], rev_at["enemy_hand_cards"]
    assert own.stop - own.start == foe.stop - foe.start, "the two hand blocks differ in width"
    sizes = {len(deck) for deck in _decks(engine)}
    assert len(sizes) == 1, "the two decks are not the same size"
    return {"fair": built["fair"], "revealed": built["hand"],
            "own_hand": fair_at["own_hand_cards"], "worked": slice(worked[0].start,
                                                                   worked[-1].stop),
            "enemy_hand": rev_at["enemy_hand_cards"], "deck": len(_decks(engine)[0])}


class _Deaf:
    """The counter with one play hidden from it: the control this figure is judged by.

    It is the shipped ``MatchMemory`` fed the same states, with the cost of the first
    enemy play it would have seen set to zero -- exactly the "a play was missed" failure
    the class documents. If the comparison on the tile cannot tell this apart from the
    honest count, the comparison is not evidence of anything.
    """

    def __init__(self, builder, state, team):
        from royalegym.obs import MatchMemory

        self.mem = MatchMemory(builder.num_cards, builder.law)
        self.mem.bind(builder.cards)
        self.mem.seed(state, team)
        self.team = team
        self.missed = False

    def observe(self, state):
        from royalegym.obs import cards_that_left

        m, foe = self.mem, state.players[1 - self.team]
        moved = state.tick > m.tick >= 0
        played = cards_that_left(m.foe_hand, foe.hand) if moved else []
        if played and not self.missed:
            self.missed = True
            kept = list(m.cost)
            for card in played:
                m.cost[card] = 0
            m.observe(state, self.team)
            m.cost = kept
        else:
            m.observe(state, self.team)

    def milli(self) -> int:
        return self.mem.enemy_elixir_milli()


def _battle() -> dict:
    """Play one battle out; compare both counts to the engine after every step."""
    import numpy as np

    from royalegym import RandomLegalOpponent, Reveal, RustEngine

    engine = RustEngine()
    env = _env(engine, Reveal())
    obs, _ = env.reset(seed=SEED)
    teams = {a: i for i, a in enumerate(env.agents)}
    rng, policy = np.random.default_rng(SEED), RandomLegalOpponent(noop_prob=NOOP_PROB)
    deaf = {t: _Deaf(env.obs_builder, env.battle_state, t) for t in teams.values()}

    checks = agree = worst = deaf_wrong = deaf_worst = steps = 0
    while env.agents:
        obs, *_ = env.step({a: policy.act(obs[a], obs[a]["action_mask"], rng)
                            for a in env.agents})
        steps += 1
        state = env.battle_state
        for team in teams.values():
            # Never at reset: there the count was just seeded from this same bar.
            actual = state.players[1 - team].elixir_milli
            counted = env.obs_builder.memory[team].enemy_elixir_milli()
            agree += int(counted == actual)
            worst = max(worst, abs(counted - actual))
            deaf[team].observe(state)
            deaf_wrong += int(deaf[team].milli() != actual)
            deaf_worst = max(deaf_worst, abs(deaf[team].milli() - actual))
            checks += 1
    seen = tuple(int(env.obs_builder.memory[t].foe_seen.sum()) for t in sorted(teams.values()))
    return {"checks": checks, "agree": agree, "worst": worst, "steps": steps,
            "deaf_wrong": deaf_wrong, "deaf_worst": deaf_worst, "seen": seen,
            "seats": len(teams)}


# --------------------------------------------------------------------------- drawing


def _text(d, xy, s, size, fill, mono=False):
    # These tiles are read at a third of this size, so a line that runs off the canvas on
    # some other card table has to stop the build rather than be cropped by the page.
    assert xy[0] + _w(d, s, size, mono) <= 984, f"too wide for the canvas: {s!r}"
    d.text(xy, s, font=M.theme_font(size, mono=mono), fill=fill)


def _w(d, s, size, mono=False):
    return d.textlength(s, font=M.theme_font(size, mono=mono))


def draw(out_path: pathlib.Path) -> str:
    m = _map()
    fair, revealed = m["fair"], m["revealed"]
    gap = revealed - fair
    el = _battle()
    seen = el["seen"]
    seen_txt = str(seen[0]) if len(set(seen)) == 1 else " and ".join(str(v) for v in seen)

    W, H = 1000, 640
    im, d = M.canvas(W, H)
    x0 = 44

    # ---- headline: the one number, and what makes it appear ----------------
    _text(d, (x0, 18), "Reveal the enemy hand:", 38, M.DIM)
    head = 66
    plus = f"+{gap}"
    _text(d, (x0, 62), plus, head, M.AMBER)
    _text(d, (x0 + _w(d, plus + " ", head), 62), "slots appear", head, M.TEXT)

    # ---- the vector, to scale, with the runs where they actually are -------
    bar_max, bar_h = 742, 64
    px = bar_max / revealed

    def run(top, sl, colour, label=""):
        lo, hi = x0 + round(sl.start * px), x0 + round(sl.stop * px)
        d.rectangle([lo, top, hi, top + bar_h], fill=colour)
        # The label goes in at the largest size that fits the block on this card table,
        # and is left out rather than shrunk under the size these tiles are read at.
        for size in (40, 38, 36, 34) if label else ():
            if _w(d, label, size) <= hi - lo - 14:
                _text(d, (lo + (hi - lo - _w(d, label, size)) / 2, top + 14), label, size,
                      M.BG)
                break

    rows = ((168, "the observation's vector, by default", fair, False),
            (296, "with Reveal(enemy_hand)", revealed, True))
    for y, label, value, revealed_row in rows:
        _text(d, (x0, y), label, 34, M.DIM)
        top = y + 46
        d.rectangle([x0, top, x0 + round(value * px), top + bar_h], fill=M.BORDER)
        run(top, m["own_hand"], M.BLUE, "" if revealed_row else "your hand")
        run(top, m["worked"], M.GREEN)
        if revealed_row:
            run(top, m["enemy_hand"], M.AMBER, "their hand")
        _text(d, (x0 + bar_max + 26, top + 12), str(value), 40, M.TEXT)

    # ---- what sits in the fair vector instead ------------------------------
    y = 428
    d.rectangle([x0, y + 12, x0 + 26, y + 38], fill=M.GREEN)
    worked = m["worked"].stop - m["worked"].start
    _text(d, (x0 + 40, y), f"{worked} slots you work out: {seen_txt} of their "
                           f"{m['deck']} cards, by the end", 34, M.TEXT)

    # ---- and the check that says the counting works, next to its control ---
    ok = el["agree"] == el["checks"]
    wrong = el["checks"] - el["agree"]
    lead = "Elixir counted, both seats, every step: "
    _text(d, (x0, 480), lead, 34, M.TEXT)
    _text(d, (x0 + _w(d, lead, 34), 480), f"wrong at {wrong} of {el['checks']}", 34,
          M.GREEN if ok else M.RED)
    lead2 = "the same counter blind to one play: "
    _text(d, (x0, 528), lead2, 34, M.DIM)
    _text(d, (x0 + _w(d, lead2, 34), 528), f"wrong at {el['deaf_wrong']}", 34, M.RED)

    _text(d, (x0, 580), f"one battle, seed {SEED}, random legal moves, different decks",
          34, M.DIM)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)
    tick = "OK" if ok and el["deaf_wrong"] else "NO"
    return (f"{tick}: fair vector {fair} slots, Reveal(enemy_hand) {revealed}, gap +{gap} "
            f"(the enemy hand, as wide as your own); {worked} fair slots of deduction, "
            f"{seen_txt} of {m['deck']} cards seen; counted enemy elixir matched the engine "
            f"{el['agree']}/{el['checks']} times (worst {el['worst']} milli) over "
            f"{el['steps']} steps on seed {SEED}, while the same counter blind to one play "
            f"was wrong {el['deaf_wrong']}/{el['checks']} times (worst {el['deaf_worst']})")

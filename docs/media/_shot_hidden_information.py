#!/usr/bin/env python3
"""Showcase tile: hidden information is ABSENT, not zeroed.

The evidence is the observation vector's WIDTH. A fair observation and one built with
``Reveal(enemy_hand=True)`` do not have the same number of slots, so a checkpoint cannot
quietly be trained on one and evaluated on the other. Both widths here come from building
the two environments and reading ``obs["vector"].shape``.

The second half of the claim is that hidden does not mean guessed: the fair slot holding
the opponent's elixir is COUNTED from the plays seen, and it is compared against the
engine's own bar at every step of the documented battle, for both seats.

Nothing is drawn that was not measured in this run.
"""

from __future__ import annotations

import pathlib

import make_media as M

SEED = 0
DECK_NAMES = ("Knight", "Archer", "Giant", "Minions", "Fireball", "Cannon", "Zap",
              "Musketeer")
NOOP_PROB = 0.7


# --------------------------------------------------------------------------- measuring


def _deck(engine):
    """Cards BY NAME: an id is a position in a catalogue that moves between tables."""
    by_name = {c.name: c.card_id for c in engine.cards()}
    return [by_name[n] for n in DECK_NAMES]


def _widths():
    """Vector width with no reveal and with the enemy hand revealed."""
    from royalegym import (ClashParallelEnv, DefaultStateMutator, Reveal, RustEngine,
                           SpatialObsBuilder)

    out = {}
    for key, reveal in (("fair", Reveal()), ("hand", Reveal(enemy_hand=True))):
        engine = RustEngine()
        env = ClashParallelEnv(engine=engine,
                               obs_builder=SpatialObsBuilder(reveal=reveal),
                               state_mutator=DefaultStateMutator(decks=[_deck(engine)] * 2))
        obs, _ = env.reset(seed=SEED)
        out[key] = int(obs[env.agents[0]]["vector"].shape[0])
    return out["fair"], out["hand"]


def _count_vs_engine():
    """Play the documented battle; compare the counted enemy bar to the engine's own.

    The count lives in the observation builder's per-seat memory, which is fed nothing but
    the states that seat is shown. The engine's value is read straight off the battle state.
    """
    import numpy as np
    from royalegym import (ClashParallelEnv, DefaultStateMutator, RandomLegalOpponent,
                           Reveal, RustEngine, SpatialObsBuilder)

    engine = RustEngine()
    env = ClashParallelEnv(engine=engine, obs_builder=SpatialObsBuilder(reveal=Reveal()),
                           state_mutator=DefaultStateMutator(decks=[_deck(engine)] * 2))
    obs, _ = env.reset(seed=SEED)
    teams = {a: i for i, a in enumerate(env.agents)}
    rng, policy = np.random.default_rng(SEED), RandomLegalOpponent(noop_prob=NOOP_PROB)

    checks = agree = steps = 0
    worst = 0
    while env.agents:
        state = env.battle_state
        for agent in env.agents:
            team = teams[agent]
            counted = env.obs_builder.memory[team].enemy_elixir_milli()
            actual = state.players[1 - team].elixir_milli
            worst = max(worst, abs(counted - actual))
            agree += int(counted == actual)
            checks += 1
        obs, *_ = env.step({a: policy.act(obs[a], obs[a]["action_mask"], rng)
                            for a in env.agents})
        steps += 1
    return {"checks": checks, "agree": agree, "worst": worst, "steps": steps,
            "seats": len(teams)}


# --------------------------------------------------------------------------- drawing


def _text(d, xy, s, size, fill, mono=False):
    d.text(xy, s, font=M.theme_font(size, mono=mono), fill=fill)


def _w(d, s, size, mono=False):
    return d.textlength(s, font=M.theme_font(size, mono=mono))


def draw(out_path: pathlib.Path) -> str:
    fair, revealed = _widths()
    gap = revealed - fair
    el = _count_vs_engine()

    W, H = 1000, 640
    im, d = M.canvas(W, H)
    x0 = 44

    # ---- headline: the one number, and what makes it appear ----------------
    head = 58
    _text(d, (x0, 24), "Reveal the enemy hand:", head, M.TEXT)
    plus = f"+{gap}"
    _text(d, (x0, 94), plus, head, M.AMBER)
    _text(d, (x0 + _w(d, plus + " ", head), 94), "slots appear", head, M.TEXT)

    # ---- the two widths, to scale ------------------------------------------
    bar_x, bar_max, bar_h = x0, 742, 76
    fair_w = round(bar_max * fair / revealed)
    gap_w = bar_max - fair_w
    num_x = bar_x + bar_max + 28

    rows = (("default observation", fair, False), ("Reveal(enemy_hand)", revealed, True))
    y = 212
    for label, value, with_gap in rows:
        _text(d, (bar_x, y), label, 36, M.DIM)
        top = y + 44
        d.rectangle([bar_x, top, bar_x + fair_w, top + bar_h], fill=M.BLUE)
        if with_gap:
            d.rectangle([bar_x + fair_w, top, bar_x + bar_max, top + bar_h], fill=M.AMBER)
            s = plus
            _text(d, (bar_x + fair_w + (gap_w - _w(d, s, 40)) / 2, top + 18), s, 40, M.BG)
        _text(d, (num_x, top + 16), str(value), 44, M.TEXT)
        y = top + bar_h + 40

    # ---- hidden is not guessed ---------------------------------------------
    tick = "OK" if el["agree"] == el["checks"] else "NO"
    colour = M.GREEN if el["agree"] == el["checks"] else M.RED
    _text(d, (x0, 528), "Counted enemy elixir:", 40, M.TEXT)
    s = f"{el['agree']} / {el['checks']} exact"
    _text(d, (x0 + _w(d, "Counted enemy elixir: ", 40), 528), s, 40, colour)

    _text(d, (x0, 588),
          f"seed {SEED}, {el['steps']} steps, {el['seats']} seats", 36, M.DIM)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)
    return (f"{tick}: fair vector {fair} slots, Reveal(enemy_hand) {revealed}, gap +{gap}; "
            f"counted enemy elixir matched the engine {el['agree']}/{el['checks']} times "
            f"(worst miss {el['worst']} milli) over {el['steps']} steps on seed {SEED}")

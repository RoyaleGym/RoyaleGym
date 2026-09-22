#!/usr/bin/env python3
"""One saved battle position, reloaded into several engines and played on differently.

Everything on the picture is measured in ``draw``: the battle is played from a fixed
seed with the deck the README's Try-it program uses, snapshotted mid-match, and the
snapshot is loaded into one fresh engine per branch. Every branch is checked to start on
the snapshot's state hash, then gets one different command on the SAME tile before the
same number of ticks runs, so the command is the only input that differs.

Endings are graded on the BOARD -- every entity's team, kind, card, position and hp, the
live spell objects and both players' tower hp -- not on the engine's state hash. Two
branches that spend different elixir but leave the same board are counted as one ending
and said so on the picture.
"""

from __future__ import annotations

import hashlib
import pathlib

import numpy as np

import make_media as M
from royalegym import RustEngine
from royalegym.protocol import BLUE, RED, DeployCommand, DeployStatus, MatchSetup

# The deck and seed the README's Try-it battle uses, so this figure is that battle.
DECK = ("Knight", "Archer", "Giant", "Minions", "Fireball", "Cannon", "Goblins",
        "Musketeer")
SEED = 0
LEAD_IN_STEPS = 40      # env-sized steps played before the snapshot is taken
STEP_TICKS = 10         # ticks per lead-in step (one half second of game time)
BRANCH_TICKS = 120      # ticks every branch runs after its one different command
PLAY_PROB = 0.35        # chance a seat plays on a lead-in step
MAX_CARD_BRANCHES = 3   # card branches drawn, plus the one that plays nothing


# --------------------------------------------------------------------------- engine


def _engine() -> tuple[RustEngine, list[int]]:
    e = RustEngine()
    ids = {c.name: c.card_id for c in e.cards()}
    missing = [n for n in DECK if n not in ids]
    if missing:
        raise RuntimeError(f"this build's catalogue has no {missing}")
    return e, [ids[n] for n in DECK]


def _tile_xy(arena, tx: int, ty: int) -> tuple[int, int]:
    return tx * arena.subtile + arena.subtile // 2, ty * arena.subtile + arena.subtile // 2


def _legal_tiles(engine: RustEngine, team: int) -> dict[int, list[tuple[int, int]]]:
    """{hand slot: [(tile x, tile y), ...]} the engine would accept right now."""
    arena = engine.arena()
    ok = int(DeployStatus.OK)
    out: dict[int, list[tuple[int, int]]] = {}
    for slot in range(4):
        tiles = []
        for tx in range(arena.tiles_x):
            for ty in range(arena.tiles_y):
                x, y = _tile_xy(arena, tx, ty)
                if engine.check_deploy(DeployCommand(team, slot, x, y)) == ok:
                    tiles.append((tx, ty))
        if tiles:
            out[slot] = tiles
    return out


def _play_in(engine: RustEngine) -> None:
    """Play the battle a little way in: both seats choose among their legal moves."""
    rng = np.random.default_rng(SEED)
    arena = engine.arena()
    for _ in range(LEAD_IN_STEPS):
        commands = []
        for team in (BLUE, RED):
            if rng.random() >= PLAY_PROB:
                continue
            legal = _legal_tiles(engine, team)
            if not legal:
                continue
            slot = sorted(legal)[int(rng.integers(len(legal)))]
            tiles = legal[slot]
            tx, ty = tiles[int(rng.integers(len(tiles)))]
            commands.append(DeployCommand(team, slot, *_tile_xy(arena, tx, ty)))
        engine.step(commands, STEP_TICKS)


def _shared_tile(engine: RustEngine, arena, legal: dict[int, list[tuple[int, int]]],
                 ) -> tuple[int, int]:
    """One tile every playable card may go on: the legal tile nearest the enemy push.

    Every branch plays on this same tile, so the card is the only thing that differs.
    The tile is chosen by where the opponent's units are, not by what it would do to the
    ending, and it is named on the picture.
    """
    slots = sorted(legal)[:MAX_CARD_BRANCHES]
    common = set(legal[slots[0]]).intersection(*(legal[s] for s in slots[1:]))
    if not common:
        raise RuntimeError("no tile is legal for every playable card in this position")
    enemy = [e for e in engine.state().entities if e.team == RED and e.tower_slot < 0]
    if enemy:
        ex = sum(e.x for e in enemy) / len(enemy)
        ey = sum(e.y for e in enemy) / len(enemy)
    else:  # no push to answer: the forward-most tile of the player's own half
        ex, ey = _tile_xy(arena, arena.tiles_x // 2, max(t for _, t in common))
    return min(sorted(common),
               key=lambda t: sum((a - b) ** 2 for a, b in zip(_tile_xy(arena, *t), (ex, ey))))


def _board(state) -> str:
    """A fingerprint of what is ON the board: entities, live spells, tower hp.

    Deliberately excludes elixir, hands and the deck cursor, so two branches that cost
    different elixir but leave the same units in the same places grade as one ending.
    """
    ents = sorted((e.team, e.kind, e.card_id, e.tower_slot, e.x, e.y, e.hp, e.max_hp,
                   e.deploy_ticks, e.stun_ticks, e.knockback_ticks) for e in state.entities)
    spells = sorted((s.team, s.card_id, s.motion, s.x, s.y, s.aim_x, s.aim_y,
                     s.delay_ticks, s.travelled, s.hits) for s in state.spells)
    towers = [list(p.tower_hp) for p in state.players]
    return hashlib.blake2b(repr((ents, spells, towers)).encode(), digest_size=8).hexdigest()


def _prefix_len(digests: list[str], start: int = 6) -> int:
    """Shortest prefix that still separates every pair of different digests."""
    uniq = set(digests)
    for n in range(start, len(digests[0]) + 1):
        if len({d[:n] for d in uniq}) == len(uniq):
            return n
    return len(digests[0])


# --------------------------------------------------------------------------- drawing


def _panel(d, box, *, fill=None, outline=None, width=2, radius=14) -> None:
    d.rounded_rectangle(box, radius=radius, fill=fill if fill is not None else M.PANEL,
                        outline=outline if outline is not None else M.BORDER, width=width)


def _right(d, x: int, y: int, text: str, font, fill) -> None:
    d.text((x - d.textlength(text, font=font), y), text, font=font, fill=fill)


def draw(out_path: pathlib.Path) -> str:
    """Compute the branches, draw the picture, save it, return the numbers on it."""
    # ---- the shared position -------------------------------------------------
    engine, deck = _engine()
    engine.reset(SEED, MatchSetup(decks=[deck, deck]))
    _play_in(engine)
    arena = engine.arena()
    root = engine.state()
    blob = engine.save_state()
    root_hash = engine.state_hash()
    names = [c.name for c in engine.cards()]

    # ---- the branches: same tile, one different card each, and one that passes
    legal = _legal_tiles(engine, BLUE)
    if len(legal) < 2:
        raise RuntimeError("this position has fewer than two playable cards")
    tile = _shared_tile(engine, arena, legal)
    slots = sorted(legal)[:MAX_CARD_BRANCHES]

    branches = []
    starts_matched = 0
    for slot in [*slots, None]:
        fresh = RustEngine()
        fresh.load_state(blob)
        starts_matched += int(fresh.state_hash() == root_hash)
        commands = ([] if slot is None
                    else [DeployCommand(BLUE, slot, *_tile_xy(arena, *tile))])
        fresh.step(commands, BRANCH_TICKS)
        end = fresh.state()
        branches.append({
            "card": "nothing" if slot is None else names[root.players[BLUE].hand[slot]],
            "slot": slot,
            "board": _board(end),
            "units": sum(1 for e in end.entities if e.tower_slot < 0),
        })

    # A branch replayed from the same bytes lands on the same board: the endings differ
    # because the commands did, not because anything drifted.
    repeat = RustEngine()
    repeat.load_state(blob)
    repeat.step([DeployCommand(BLUE, slots[0], *_tile_xy(arena, *tile))], BRANCH_TICKS)
    repeatable = _board(repeat.state()) == branches[0]["board"]

    n = len(branches)
    boards = [b["board"] for b in branches]
    distinct = len(set(boards))
    cut = _prefix_len(boards)                       # never show a prefix that lies
    seen: dict[str, str] = {}
    for b in branches:
        b["same_as"] = seen.get(b["board"])
        seen.setdefault(b["board"], b["card"])
    secs = BRANCH_TICKS * root.tick_ms / 1000
    thousands = lambda v: f"{v:,}".replace(",", " ")  # noqa: E731

    # ---- the picture ---------------------------------------------------------
    # 1000 x 640, and every type size here is set so the figure survives a 2.8x downscale
    # into a README table cell: nothing smaller than 34 px, headline 66 px.
    W, H = 1000, 640
    im, d = M.canvas(W, H)
    f_head = M.theme_font(66)
    f_sub = M.theme_font(37)
    f_hash = M.theme_font(46, mono=True)
    f_note = M.theme_font(34)
    f_col = M.theme_font(34)
    f_card = M.theme_font(44)
    f_cell = M.theme_font(40)
    f_board = M.theme_font(40, mono=True)

    L, R = 40, W - 40
    d.text((L, 22), f"One save, {distinct} endings", font=f_head, fill=M.TEXT)
    d.text((L, 102),
           f"seed {SEED} · tick {root.tick} · {thousands(len(blob))} byte save",
           font=f_sub, fill=M.DIM)

    band = (L, 158, R, 258)
    _panel(d, band, outline=M.AMBER, width=3)
    d.text((L + 26, 190), "SAVE", font=f_note, fill=M.DIM)
    d.text((L + 116, 180), f"{root_hash:016x}"[:cut], font=f_hash, fill=M.AMBER)
    _right(d, R - 26, 172, f"{starts_matched} of {n} reload the same", f_note, M.GREEN)
    _right(d, R - 26, 212,
           f"replay of {branches[0]['card']} repeats" if repeatable
           else f"replay of {branches[0]['card']} DID NOT repeat", f_note, M.GREEN)

    col_card, col_units, col_board, col_tag = L + 22, 452, 560, 748
    d.text((col_card, 284), "PLAYS", font=f_col, fill=M.DIM)
    _right(d, col_units, 284, "UNITS LEFT", font=f_col, fill=M.DIM)
    d.text((col_board, 284), f"BOARD AFTER {secs:.0f} s", font=f_col, fill=M.DIM)

    top, row_h = 330, 62
    d.line([(L, top - 8), (R, top - 8)], fill=M.BORDER, width=2)
    d.line([(14, top - 8), (14, top + row_h * n - 18)], fill=M.AMBER, width=5)
    for i, b in enumerate(branches):
        y = top + i * row_h
        d.line([(14, y + 24), (L - 6, y + 24)], fill=M.AMBER, width=5)
        plays = b["card"] if b["slot"] is not None else "nothing"
        d.text((col_card, y), plays, font=f_card,
               fill=M.BLUE if b["slot"] is not None else M.DIM)
        _right(d, col_units, y + 4, str(b["units"]), f_cell, M.TEXT)
        d.text((col_board, y + 4), b["board"][:cut], font=f_board,
               fill=M.DIM if b["same_as"] else M.GREEN)
        if b["same_as"]:
            d.text((col_tag, y + 6), f"= {b['same_as']}", font=f_note, fill=M.DIM)
        if i:
            d.line([(L, y - 10), (R, y - 10)], fill=M.PANEL, width=2)

    d.text((L, H - 52), f"every branch plays tile {tile[0]},{tile[1]}, nearest the push",
           font=f_sub, fill=M.DIM)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)

    plays = ", ".join(f"{b['card']}"
                      + ("" if b["slot"] is None else f"@{tile[0]},{tile[1]}")
                      + (f" (same board as {b['same_as']})" if b["same_as"] else "")
                      for b in branches)
    return (f"tick {root.tick}, snapshot {len(blob)} bytes, shared hash "
            f"{root_hash:016x}; {n} branches ({plays}) ran {BRANCH_TICKS} ticks "
            f"({secs:.1f} s) each, {starts_matched}/{n} started on the shared hash, "
            f"{distinct}/{n} endings are distinct BOARDS, replay of "
            f"{branches[0]['card']} repeatable={repeatable}")

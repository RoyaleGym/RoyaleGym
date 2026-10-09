"""``reset(options={"mask_only": [...]})``: a seat played by something that reads only legality
gets an observation of masks and zeros, and nothing else about the episode changes.

The skipped seat's observation has every key a build has, at the same shape and dtype, the mask
entries byte for byte, and zero everywhere else. The other seat's observations are byte-equal
to an episode that builds both. Every scripted opponent of the ladder picks the same action from
either observation, since they read only ``mask_planes``. A snapshot (the skipped seat kept no
memory) and Blue's global state are refused; the option lasts one episode.

SKIPS
    Everything here needs the engine. Not a pass without it.
"""

from __future__ import annotations

import numpy as np
import pytest

from royalegym.action import TileActionParser
from royalegym.env import ClashParallelEnv
from royalegym.obs import SAME_UNIT_ALIASES, EntityListObsBuilder, SpatialObsBuilder
from royalegym.opponents import ladder
from royalegym.protocol import ShuffleMode
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, core_available
from royalegym.state_mutator import DefaultStateMutator, deck_ids

pytestmark = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))

BLUE_DECK = (["Musketeer", "Wallbreakers", "Ghost", "Miner", "GoblinDrill", "Tesla", "Rage",
              "IceWizard"], [2, 1, 1, 0, 0, 0, 0, 0])
RED_DECK = (["GoldenKnight", "DarkPrince", "Prince", "ZapMachine", "Clone", "Vines", "Firecracker",
             "Graveyard"], [0, 2, 0, 0, 0, 0, 1, 0])
RG2 = dict(
    card_identity=True, spell_aim_after_ticks=0, spell_identity=True, effect_identity=True,
    evolutions=True, evolution_progress=True, heroes=True, unit_status=True, card_status=True,
    unit_actions=True, unit_identity=True, unit_aliases=dict(SAME_UNIT_ALIASES),
    button_index=True, enemy_queue=True,
)
BUILDERS = {"spatial_rg2": lambda: SpatialObsBuilder(**RG2), "entity_list": EntityListObsBuilder}
MASK_KEYS = ("action_mask", "mask_planes", "ability_ready")


def _env(builder):
    eng = RustEngine()
    mutator = DefaultStateMutator(
        decks=[deck_ids(BLUE_DECK[0], eng.cards()), deck_ids(RED_DECK[0], eng.cards())],
        forms=[BLUE_DECK[1], RED_DECK[1]], shuffle=ShuffleMode.INDEPENDENT,
    )
    return ClashParallelEnv(eng, action_parser=TileActionParser(ability_buttons=True),
                            obs_builder=builder, state_mutator=mutator)


def _episode(builder, seed, options=None, steps=160):
    """The observations of one episode under random legal play read off the masks alone, so
    two runs that agree on the masks play the same moves."""
    env = _env(builder)
    obs, _ = env.reset(seed=seed, options=options)
    rng = np.random.default_rng(seed)
    out = [obs]
    while env.agents and len(out) < steps:
        acts = {}
        for agent in env.agents:
            legal = np.flatnonzero(obs[agent]["action_mask"])
            legal = legal[legal != 0]
            acts[agent] = int(rng.choice(legal)) if len(legal) and rng.random() < 0.3 else 0
        obs, *_ = env.step(acts)
        out.append(obs)
    return out


@pytest.mark.parametrize("name", sorted(BUILDERS))
@pytest.mark.parametrize("skipped", ["red", "blue"])
def test_the_skipped_seat_holds_the_masks_and_zeros_and_the_other_is_unchanged(name, skipped):
    kept = "blue" if skipped == "red" else "red"
    full = _episode(BUILDERS[name](), 31)
    lean = _episode(BUILDERS[name](), 31, options={"mask_only": [skipped]})
    assert len(full) == len(lean) > 100
    moved = {k: 0 for k in full[0][skipped] if k not in MASK_KEYS}
    for i, (f, g) in enumerate(zip(full, lean, strict=True)):
        assert f.keys() == g.keys()
        assert f[kept].keys() == g[kept].keys()
        for k in f[kept]:
            assert f[kept][k].dtype == g[kept][k].dtype, (i, k)
            assert f[kept][k].tobytes() == g[kept][k].tobytes(), (i, k)
        assert set(f[skipped]) == set(g[skipped])
        for k, want in f[skipped].items():
            got = g[skipped][k]
            assert (got.dtype, got.shape) == (want.dtype, want.shape), (i, k)
            if k in MASK_KEYS:
                assert got.tobytes() == want.tobytes(), (i, k)
            else:
                assert not got.any(), (i, k)
                moved[k] += bool(want.any())
    # Vacuity: every zeroed key holds something in the built episode, so "all zero" says a build
    # was skipped and not that the key is always empty.
    assert all(n > 0 for n in moved.values()), moved
    assert set(MASK_KEYS[:2]) <= set(full[0][skipped])


def test_every_scripted_opponent_plays_the_same_from_either_observation():
    full = _episode(SpatialObsBuilder(**RG2), 32)
    lean = _episode(SpatialObsBuilder(**RG2), 32, options={"mask_only": ["red"]})
    played = 0
    for (name, a), (_, b) in zip(ladder(), ladder(), strict=True):
        ra, rb = np.random.default_rng(5), np.random.default_rng(5)
        for f, g in zip(full, lean, strict=True):
            if "red" not in f:
                continue
            got = b.act(g["red"], g["red"]["action_mask"], rb)
            want = a.act(f["red"], f["red"]["action_mask"], ra)
            assert got == want, name
            played += want != 0
    assert played > 100


def test_refusals_and_one_episode_only():
    env = _env(SpatialObsBuilder(**RG2))
    with pytest.raises(ValueError, match="mask_only names"):
        env.reset(seed=1, options={"mask_only": ["green"]})
    env.reset(seed=1, options={"mask_only": ["red"]})
    env.state()  # Blue is built
    with pytest.raises(ValueError, match="never kept"):
        env.snapshot()
    env.reset(seed=1, options={"mask_only": ["blue"]})
    with pytest.raises(ValueError, match="Blue's observation"):
        env.state()
    # The next episode builds both again.
    obs, _ = env.reset(seed=1)
    fresh, _ = _env(SpatialObsBuilder(**RG2)).reset(seed=1)
    for agent in ("blue", "red"):
        for k in fresh[agent]:
            assert obs[agent][k].tobytes() == fresh[agent][k].tobytes(), (agent, k)
    env.snapshot()
    stub = SpatialObsBuilder(**RG2)
    stub.mask_only = None  # type: ignore[assignment,method-assign]
    with pytest.raises(ValueError, match="has none"):
        _env(stub).reset(seed=1, options={"mask_only": ["red"]})

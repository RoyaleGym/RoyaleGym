"""What a user who only ran ``pip install "royalegym[all]"`` gets.

They have no RoyaleSim checkout beside this one and no environment variable set, so the
engine's data has to be found through the installed engine, and one install line has to
pull in every piece: the engine, the learner, the viewer and imitation.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from royalegym import protocol

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
PIECES = {"sim": "royalesim", "learn": "royalelearn", "viser": "royaleviser",
          "imitate": "royaleimitate"}


def _extras() -> dict[str, list[str]]:
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)["project"]["optional-dependencies"]


def _names(reqs: list[str]) -> set[str]:
    return {r.split("[")[0].split(">")[0].split("=")[0].split("<")[0].strip() for r in reqs}


def test_one_extra_per_piece_and_all_installs_every_piece():
    extras = _extras()
    for extra, package in PIECES.items():
        assert package in _names(extras.get(extra, [])), f"[{extra}] does not install {package}"
    assert set(PIECES.values()) <= _names(extras.get("all", [])), extras.get("all")


def test_every_piece_names_a_minimum_so_an_upgrade_moves_it():
    """`pip install --upgrade "royalegym[all]"` upgrades a dependency only when a requirement
    forces it (pip's default only-if-needed strategy). Measured 2026-10-03 from a v0.1.3
    install: [all] named royalesim with no minimum, so that upgrade kept royalesim 0.1.3
    beside the new royalegym. Every piece names a minimum, in [all] and in its own extra."""
    extras = _extras()
    for extra in (*PIECES, "all"):
        for req in extras[extra]:
            if _names([req]) & set(PIECES.values()):
                assert ">=" in req, f"[{extra}] names {req!r} with no minimum version"
    floors = sorted(r for r in extras["all"] if _names([r]) & {"royalesim", "royaleviser"})
    assert floors == ["royalesim>=0.1.10", "royaleviser[media]>=0.1.1"], floors


def test_the_learner_extras_require_a_royalelearn_that_has_the_learner():
    """The quickstart uses the Learner's named settings (0.4.0) and is stopped with Ctrl+C, which
    finishes the update, saves a checkpoint and returns (0.4.1); it starts on macOS from 0.4.2.
    Before 0.5.2 a run with an extension section (royaleimitate's warm_start) was refused for any
    package installed from a wheel, which is every user of the install line. From 0.5.3 its
    preflight battle may run past t6000, where RoyaleSim 0.1.4 ends a level overtime. 0.5.4
    stores and embeds ``spell_ids`` (``SpatialObsBuilder(spell_identity=True)``) and refused it
    before."""
    extras = _extras()
    for extra in ("learn", "all"):
        pins = [r for r in extras[extra] if r.startswith("royalelearn")]
        assert pins == ["royalelearn[torch]>=0.5.4"], f"[{extra}]: {pins}"


def test_the_engine_data_comes_from_the_installed_engine(monkeypatch, tmp_path):
    """No env var and no sibling checkout: the installed royalesim says where its data is."""
    monkeypatch.delenv(protocol.DATA_DIR_ENV, raising=False)
    monkeypatch.setattr(protocol, "DEFAULT_DATA_DIR", tmp_path / "no-sibling-checkout")
    shipped = tmp_path / "royalesim-data"
    shipped.mkdir()
    monkeypatch.setitem(sys.modules, "royalesim", SimpleNamespace(data_dir=lambda: shipped))
    assert protocol.data_dir() == shipped


def test_the_env_var_still_wins_and_the_sibling_is_the_fallback(monkeypatch, tmp_path):
    shipped, mine, sibling = (tmp_path / n for n in ("shipped", "mine", "sibling"))
    for p in (shipped, mine, sibling):
        p.mkdir()
    monkeypatch.setattr(protocol, "DEFAULT_DATA_DIR", sibling)
    monkeypatch.setitem(sys.modules, "royalesim", SimpleNamespace(data_dir=lambda: shipped))
    monkeypatch.setenv(protocol.DATA_DIR_ENV, str(mine))
    assert protocol.data_dir() == mine
    monkeypatch.delenv(protocol.DATA_DIR_ENV)
    monkeypatch.setitem(sys.modules, "royalesim", SimpleNamespace())  # an engine without it
    assert protocol.data_dir() == sibling


def test_with_nothing_found_the_error_says_how_to_install(monkeypatch, tmp_path):
    monkeypatch.delenv(protocol.DATA_DIR_ENV, raising=False)
    monkeypatch.setattr(protocol, "DEFAULT_DATA_DIR", tmp_path / "none")
    monkeypatch.setitem(sys.modules, "royalesim", SimpleNamespace())
    with pytest.raises(FileNotFoundError, match=re.escape(protocol.INSTALL_PAGE)):
        protocol.data_dir()


# -- make_env: the one-line environment the quickstart builds --------------------------


def test_make_env_is_a_two_seat_battle_with_the_starter_deck_and_tower_damage():
    from royalegym import STARTER_DECK, ClashParallelEnv, TowerHPReward, make_env

    env = make_env(engine="mock")
    assert isinstance(env, ClashParallelEnv)
    assert isinstance(env.reward_fn, TowerHPReward)
    obs, _ = env.reset(seed=1)
    assert set(obs) == {"blue", "red"}
    names = {c.card_id: c.name for c in env.engine.cards()}
    for p in env.battle_state.players:
        assert {names[c] for c in p.hand} <= set(STARTER_DECK), p.hand


def test_make_env_names_a_card_it_does_not_know():
    from royalegym import make_env

    with pytest.raises(ValueError, match="NotACard"):
        make_env(engine="mock", deck=["NotACard", *["Knight"] * 7])


def test_make_env_without_the_engine_says_how_to_install_it(monkeypatch):
    from royalegym import env as env_mod
    from royalegym import make_env

    monkeypatch.setattr(env_mod, "core_available", lambda: False)
    with pytest.raises(ImportError, match=re.escape(protocol.INSTALL_PAGE)):
        make_env()


# -- play_battle: the quickstart's last step, a battle to watch ------------------------


def test_play_battle_plays_a_bot_and_saves_a_battle_the_viewer_can_open(tmp_path):
    from royalegym import load_trace, make_env, play_battle

    calls = {"n": 0}

    def bot(obs):  # a trained policy's shape: one seat's obs in, an action out
        calls["n"] += 1
        return 0

    out = play_battle(make_env(engine="mock"), blue=bot, red="random", seed=3,
                      save_to=tmp_path / "battle.msgpack")
    assert out.path == tmp_path / "battle.msgpack"
    assert out.path.exists()
    trace = load_trace(out.path)
    assert trace.result is not None
    assert len(trace.frames) > 10
    assert calls["n"] > 10, "the bot was never asked for a move"
    assert out.winner in (0, 1, 2, None)
    assert out.crowns == trace.result.crowns


# -- the public API: a version, type hints a checker reads, a docstring on every name --


def test_the_package_says_its_version():
    """``__version__`` is the installed distribution's version, so it can only be held to this
    tree's pyproject when the installed royalegym IS this tree (a fresh install, CI). Run from
    another checkout of the repo, the metadata belongs to that install."""
    import json
    from importlib import metadata

    import royalegym

    with PYPROJECT.open("rb") as f:
        version = tomllib.load(f)["project"]["version"]
    dist = metadata.distribution("royalegym")
    direct = dist.read_text("direct_url.json")
    url = json.loads(direct).get("url", "") if direct else ""
    here = PYPROJECT.parent.resolve().as_posix().lower()
    if url.startswith("file:") and here not in url.lower():
        pytest.skip(f"SKIPPED, NOT PASSED: royalegym is installed from {url}, not this tree")
    editable = any("__editable__" in str(f) for f in dist.files or ())
    if editable and dist.version != version:
        # An editable install keeps the version it was installed at; a version bump in the
        # tree does not reach it until it is reinstalled. CI installs fresh and checks.
        pytest.skip(
            f"SKIPPED, NOT PASSED: this editable install says {dist.version}, made before the "
            f"tree's {version}; reinstall it (pip install -e .) to check"
        )
    assert royalegym.__version__ == version


def test_type_checkers_read_the_hints():
    """PEP 561: without py.typed a type checker ignores an installed package's hints."""
    import royalegym

    assert (Path(royalegym.__file__).parent / "py.typed").exists()
    with PYPROJECT.open("rb") as f:
        data = tomllib.load(f)["tool"]["setuptools"].get("package-data", {})
    assert "py.typed" in data.get("royalegym", []), "py.typed is not shipped in the wheel"


def test_every_public_name_says_what_it_is():
    import inspect

    import royalegym

    def own_doc(obj: object) -> str | None:
        # A class's OWN docstring: inspect.getdoc falls back to a base class's, so an enum with
        # none of its own would pass on IntEnum's "Enum where members are also ints".
        if inspect.isclass(obj):
            return obj.__dict__.get("__doc__")
        return inspect.getdoc(obj)

    bare = [n for n in royalegym.__all__
            if (inspect.isclass(getattr(royalegym, n)) or inspect.isfunction(getattr(royalegym, n)))
            and not (own_doc(getattr(royalegym, n)) or "").strip()]
    assert bare == [], f"public names with no docstring: {bare}"


def test_what_a_custom_reward_condition_or_mutator_needs_is_importable_from_royalegym():
    import royalegym

    for name in ("EntityKind", "to_own", "MatchSetup", "deck_ids", "Winner", "TowerSlot"):
        assert name in royalegym.__all__, name
        assert getattr(royalegym, name) is not None


def test_a_builder_written_from_scratch_needs_no_channel_names():
    """The two methods a builder must write are observation_space and build."""
    import numpy as np
    from gymnasium import spaces

    from royalegym import ClashParallelEnv, ObsBuilder
    from royalegym.mock_engine import MockEngine

    class Tiny(ObsBuilder):
        def observation_space(self):
            return spaces.Dict({
                "vector": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "action_mask": self.mask_space,
            })

        def build(self, state, team, action_mask):
            elixir = state.players[team].elixir_milli / 10_000
            return {"vector": np.array([elixir], dtype=np.float32), "action_mask": action_mask}

    env = ClashParallelEnv(engine=MockEngine(), obs_builder=Tiny())
    obs, _ = env.reset(seed=0)
    assert obs["blue"]["vector"].shape == (1,)
    assert Tiny().channel_names() == []


def test_make_env_deals_evolved_and_hero_forms_by_name():
    """Special forms by card name, both seats; a hero brings its ability button with it."""
    from royalegym.rust_engine import core_available

    if not core_available():
        pytest.skip("SKIPPED, NOT PASSED: the engine is not installed")
    from royalegym import make_env

    deck = ["Musketeer", "Cannon", "Knight", "Archer", "Giant", "Minions", "Fireball", "Zap"]
    env = make_env(deck=deck, evolved=["Cannon"], heroes=["Musketeer"])
    assert env.action_parser.ability_buttons, "a hero deck needs the parser's ability buttons"
    env.reset(seed=1)
    for p in env.battle_state.players:
        assert len(p.abilities) == 1, p.abilities
        assert len(p.evo) == 1, p.evo
    with pytest.raises(ValueError, match="Hog"):
        make_env(deck=deck, evolved=["Hog"])


def test_mock_engine_without_its_card_tables_says_how_to_go_on(monkeypatch, tmp_path):
    """An installed engine carries no 2018 card tables, and MockEngine reads them. Constructing
    it there must say so and name the ways on, not raise a bare missing-file error."""
    import shutil

    from royalegym import mock_engine

    # The data an installed engine ships: the ledger, the derived tables, and from the 2018
    # pack only the two tables the engine itself reads.
    real = protocol.data_dir()
    shutil.copy(real / "calibration.json", tmp_path / "calibration.json")
    shutil.copytree(real / "derived", tmp_path / "derived")
    pack = tmp_path / "raw" / mock_engine.RAW_CARD_PACK / "csv_logic"
    pack.mkdir(parents=True)
    for name in ("globals.csv", "rarities.csv"):
        shutil.copy(real / "raw" / mock_engine.RAW_CARD_PACK / "csv_logic" / name, pack / name)
    monkeypatch.setenv(protocol.DATA_DIR_ENV, str(tmp_path))
    with pytest.raises(FileNotFoundError, match="MockEngine reads the 2018 card tables") as caught:
        mock_engine.MockEngine()
    message = str(caught.value)
    assert "RustEngine" in message
    assert protocol.DATA_DIR_ENV in message


def test_the_imitate_extras_require_record_and_clone():
    """The site's basic-clone guide records, clones, and plays the clone with
    Learner.load_policy (royaleimitate 0.2.1). 0.2.3 keeps PublicLogMemory on the engine's
    refill timer and starts a bot from a saved one when installed from a wheel. 0.2.5 with its
    [replays] extra clones human players from the IL_Replay dataset (``from_replays``). 0.2.6
    stores ``spell_ids`` in its shards beside ``card_ids``."""
    extras = _extras()
    for extra in ("imitate", "all"):
        pins = [r for r in extras[extra] if r.startswith("royaleimitate")]
        assert pins == ["royaleimitate[replays]>=0.2.6"], f"[{extra}]: {pins}"

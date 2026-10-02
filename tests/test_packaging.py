"""What a user who only ran ``pip install "royalegym[all]"`` gets.

They have no RoyaleSim checkout beside this one and no environment variable set, so the
engine's data has to be found through the installed engine, and one install line has to
pull in every piece: the engine, the learner, the viewer and imitation.
"""

from __future__ import annotations

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
    with pytest.raises(FileNotFoundError, match=r'pip install "royalegym\[sim\]"'):
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
    with pytest.raises(ImportError, match=r'pip install "royalegym\[sim\]"'):
        make_env()

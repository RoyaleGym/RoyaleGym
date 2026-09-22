"""Which engine a ``gym.make`` id gives you, and whether it says so.

WHY THIS EXISTS
    ``gym.make("royalegym/ClashRoyale-v0")`` is the first line a Gymnasium user
    writes, and it used to hand back MockEngine without a word. The mock is a
    readable reference implementation, not the game -- different card table, spells
    that resolve instantly instead of travelling, no stuns or knockback -- so a bot
    trained through that id had not been trained against the game, and nothing
    anywhere said so. The result does not look wrong. It just does not transfer.

WHAT IT CATCHES
    The warning going missing, the warning firing when an engine WAS chosen, either
    named id handing back the wrong engine, and the warning naming an id that does not
    exist -- which is the same defect as a README pointer to a section nobody wrote,
    in the message a beginner reads first.

WHAT IT CANNOT CATCH
    Whether a reader acts on the warning. Python shows a UserWarning once per call
    site by default and a run with -W ignore shows none at all, which is the user's
    choice to make and not something a test can hold them to.
"""

from __future__ import annotations

import re
import warnings

import gymnasium as gym
import pytest

import royalegym
from royalegym.env import ClashParallelEnv, ClashSelfPlayVecEnv
from royalegym.mock_engine import MockEngine
from royalegym.rust_engine import CORE_IMPORT_ERROR, core_available


def made(env_id: str, **kwargs) -> tuple[object, list[warnings.WarningMessage]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        env = gym.make(env_id, **kwargs)
    return env.unwrapped, [w for w in caught if issubclass(w.category, UserWarning)]


def test_all_three_ids_are_registered() -> None:
    for env_id in royalegym.GYM_ENV_IDS:
        assert env_id in gym.registry, f"{env_id} is in GYM_ENV_IDS but was never registered"


def test_the_default_id_warns_that_it_chose_the_mock() -> None:
    env, caught = made(royalegym.GYM_ENV_ID)
    assert isinstance(env.parallel.engine, MockEngine)
    assert len(caught) == 1, f"expected one warning, got {[str(w.message) for w in caught]}"
    assert "MockEngine" in str(caught[0].message)


def test_choosing_an_engine_silences_it() -> None:
    """The warning is about not having chosen, not about the mock.

    Someone who passes MockEngine has chosen it, and warning them about their own
    argument is how a warning gets filtered out globally -- taking the one that
    matters with it.
    """
    env, caught = made(royalegym.GYM_ENV_ID, engine=MockEngine())
    assert isinstance(env.parallel.engine, MockEngine)
    assert caught == []


def test_the_mock_id_is_the_mock_and_does_not_warn() -> None:
    env, caught = made(royalegym.GYM_MOCK_ENV_ID)
    assert isinstance(env.parallel.engine, MockEngine)
    assert caught == []


@pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
def test_the_rust_id_is_the_rust_engine_and_does_not_warn() -> None:
    from royalegym.rust_engine import RustEngine

    env, caught = made(royalegym.GYM_RUST_ENV_ID)
    assert isinstance(env.parallel.engine, RustEngine)
    assert caught == []


def test_the_rust_id_refuses_rather_than_falling_back_when_the_engine_is_missing(
    monkeypatch,
) -> None:
    """An id that says Rust must not quietly hand back the mock.

    This is the path a reader hits on a fresh clone, and it is the one path the tests
    above cannot reach on a machine where the engine IS built -- so it is faked here,
    deliberately, by taking the compiled module away. What it pins is the choice: fall
    back and the id lies about what produced your numbers.
    """
    from royalegym import env as env_module
    from royalegym import rust_engine

    monkeypatch.setattr(rust_engine, "_core", None)
    monkeypatch.setattr(rust_engine, "CORE_IMPORT_ERROR", "royalesim is not built (pretend)")
    with pytest.raises(ImportError, match="not built"):
        env_module.rust_gym_env()


def test_the_self_play_env_warns_when_nobody_chose_an_engine() -> None:
    """The training entry point, so the most important place for it to be said.

    A run that reaches ``ClashSelfPlayVecEnv(n)`` with no env_fn has not chosen an
    engine and is about to train. The gym ids are where a beginner starts; this is
    where the compute gets spent.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vec = ClashSelfPlayVecEnv(2, viser=None)
    user = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user) == 1, [str(w.message) for w in caught]
    assert "MockEngine" in str(user[0].message)
    vec.close()


def test_an_env_fn_silences_the_self_play_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vec = ClashSelfPlayVecEnv(2, lambda: ClashParallelEnv(engine=MockEngine()), viser=None)
    assert [w for w in caught if issubclass(w.category, UserWarning)] == []
    vec.close()


def test_both_entry_points_give_the_same_explanation() -> None:
    """One message, so a reader does not get a different story per door.

    Two hand-written copies of an explanation is how they drift, and the one that
    drifts is always the one nobody reads.
    """
    _, from_gym = made(royalegym.GYM_ENV_ID)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ClashSelfPlayVecEnv(1, viser=None).close()
    from_vec = [w for w in caught if issubclass(w.category, UserWarning)]
    assert str(from_gym[0].message) == str(from_vec[0].message)


def test_every_id_the_warning_names_exists() -> None:
    """The message a beginner reads first has to point somewhere real.

    Same defect as a README pointer to a section nobody wrote, and it rots the same
    way: rename an id and the warning still names the old one, in the one place
    somebody is already confused.
    """
    _, caught = made(royalegym.GYM_ENV_ID)
    text = str(caught[0].message)
    named = set(re.findall(r"royalegym/[\w-]+", text))
    assert named, "the warning offers no alternative at all"
    assert named <= set(royalegym.GYM_ENV_IDS), (
        f"the warning names {sorted(named - set(royalegym.GYM_ENV_IDS))}, which is not a "
        f"registered id. Registered: {sorted(royalegym.GYM_ENV_IDS)}"
    )
    # And the alternative it offers is the one that actually differs.
    assert royalegym.GYM_RUST_ENV_ID in named

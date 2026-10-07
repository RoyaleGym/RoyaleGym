"""``CombinedReward.config()`` names each term so it can be rebuilt (Learn, 2026-10-07).

A term of this package keeps its bare class name (``"TowerHPReward"``), as every record so far
has it, so no existing config or run identity moves. A term from anywhere else is named by
its full dotted path: with only a bare name, a user's own ``TowerHPReward`` rebuilt from the
record would come back as this package's, and nothing would say so. ``CombinedReward.from_config``
rebuilds the reward from either form.
"""

from __future__ import annotations

from royalegym.reward import (
    CombinedReward,
    CrownReward,
    RewardFunction,
    TowerHPReward,
    WinLossReward,
)


class MyTerm(RewardFunction):
    """A user's own term, with a setting of its own."""

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    def config(self) -> dict[str, object]:
        return {"scale": self.scale}

    def get_reward(self, team, prev, state, results) -> float:
        return 0.0


class UserTerms:
    """A user's module of terms, one named like this package's."""

    class TowerHPReward(RewardFunction):
        def config(self) -> dict[str, object]:
            return {}

        def get_reward(self, team, prev, state, results) -> float:
            return 0.0


Shadow = UserTerms.TowerHPReward


def test_this_packages_terms_keep_their_bare_names():
    cfg = CombinedReward([(WinLossReward(), 1.0), (CrownReward(), 0.5)]).config()
    assert [t["class"] for t in cfg["terms"]] == ["WinLossReward", "CrownReward"]


def test_another_modules_term_is_named_by_its_dotted_path():
    cfg = CombinedReward([(TowerHPReward(), 1.0), (MyTerm(2.0), 0.25), (Shadow(), 1.0)]).config()
    names = [t["class"] for t in cfg["terms"]]
    assert names == ["TowerHPReward", f"{__name__}.MyTerm", f"{__name__}.UserTerms.TowerHPReward"]


def test_from_config_rebuilds_every_term_as_its_own_class():
    reward = CombinedReward([(TowerHPReward(), 1.0), (MyTerm(2.0), 0.25), (Shadow(), 3.0)])
    again = CombinedReward.from_config(reward.config())
    assert [type(t) for t, _ in again.terms] == [TowerHPReward, MyTerm, Shadow]
    assert [w for _, w in again.terms] == [1.0, 0.25, 3.0]
    assert again.config() == reward.config()
    record = {"class": "royalegym.reward.CombinedReward", "params": reward.config()}
    assert CombinedReward.from_config(record).config() == reward.config()

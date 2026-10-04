"""``reveal`` given as a mapping, so a JSON config can set it.

A training config is JSON, and RoyaleLearn passes its observation-builder kwargs through as
given: ``"reveal": {"enemy_elixir": true}`` arrives as a dict. The builder takes a mapping of
``Reveal``'s own field names to booleans and makes the ``Reveal`` from it. An unknown name or a
value that is not a boolean is refused by name, because a misspelt key would otherwise build
the fair observation while the config says otherwise. ``config()`` records the reveal as the
same mapping, so a builder rebuilt from its own config reads the same.
"""

from __future__ import annotations

import pytest

from royalegym.obs import Reveal, SpatialObsBuilder


def test_a_mapping_makes_the_reveal():
    b = SpatialObsBuilder(reveal={"enemy_elixir": True})
    assert b.reveal == Reveal(enemy_elixir=True)
    assert SpatialObsBuilder(reveal={}).reveal == Reveal()


def test_an_unknown_name_is_refused_by_name():
    with pytest.raises(ValueError, match="enemy_elxir"):
        SpatialObsBuilder(reveal={"enemy_elxir": True})


def test_a_value_that_is_not_a_boolean_is_refused():
    with pytest.raises(ValueError, match="enemy_hand"):
        SpatialObsBuilder(reveal={"enemy_hand": 1})


def test_anything_else_is_still_a_type_error():
    with pytest.raises(TypeError, match="reveal must be"):
        SpatialObsBuilder(reveal=True)


def test_a_builder_rebuilt_from_its_own_config_reads_the_same():
    b = SpatialObsBuilder(reveal=Reveal(enemy_elixir=True, enemy_spell_aim=True), heroes=True)
    again = SpatialObsBuilder(**b.config())
    assert again.reveal == b.reveal
    assert again.config() == b.config()

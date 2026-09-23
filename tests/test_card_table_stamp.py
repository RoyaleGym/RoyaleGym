"""Which card table each engine read, stamped into its config() and into every trace.

WHY IT EXISTS
    The Rust engine reads data/derived/cards.json from the RoyaleSim checkout it was
    built in, each time one is constructed. Nothing about the build says which table
    that was, and ROYALESIM_DATA_DIR does not move it, so a checkpoint or a trace that
    does not carry the table's identity cannot be told apart from one recorded on
    another table. MockEngine reads its own raw pack and states that instead.

WHAT IT CHECKS
    a. The hash is FNV-1a 64 as RoyaleSim defines it: its known answers, and an
       implementation written out here, separately, over the file's bytes.
    b. RustEngine stamps the file the engine read: the file's cards agree with the
       catalogue the engine reports, and ROYALESIM_DATA_DIR does not move the path.
       The build directory is found in an extension with text before it.
    c. Different bytes give a different stamp (a temp copy, the path lookup patched);
       a file that changes while the engine reads it is refused; an engine that
       reports its own hash is believed and says so, with the file's vintage only
       when the file has that hash.
    d. MockEngine's stamp moves with a number it read from the CSVs.
    e. The trace header carries the stamp for both engines, and a trace recorded
       before the header had it still decodes.

PLANTS (each on a green baseline)
    hash the wrong file; the wrong FNV prime; read the path from data_dir(); take the
    whole text run as the build directory; drop the before-and-after comparison;
    ignore the engine's own hash; state the file's vintage beside an engine hash the
    file does not have; the mock hashing card names only; the recorder not stamping;
    a card-table header field with no default.
"""

from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path
from unittest import mock

import msgspec
import pytest

from royalegym import mock_engine as mock_engine_module
from royalegym import protocol, rust_engine
from royalegym.mock_engine import RAW_CARD_PACK, MockEngine
from royalegym.protocol import DATA_DIR_ENV, MatchSetup, Placement, fnv1a64
from royalegym.replay import (
    CARD_TABLE_FIELDS,
    ReplayRecorder,
    Trace,
    load_trace,
    save_trace,
    verify_trace,
)
from royalegym.rust_engine import (
    CORE_IMPORT_ERROR,
    RustEngine,
    core_available,
    engine_binary_digest,
)

needs_core = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))

SHARED = ("Knight", "Minions", "Cannon", "Giant", "Archer", "Goblins", "MiniPekka", "HogRider")


def independent_fnv1a64(data: bytes) -> str:
    """FNV-1a 64, written out from the definition rather than imported."""
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) % (1 << 64)
    return format(h, "016x")


def file_vintage(path: Path) -> str:
    return json.loads(path.read_text(encoding="utf-8"))["provenance"]["vintage"]


@pytest.fixture(scope="module")
def rust() -> RustEngine:
    if not core_available():
        pytest.skip(str(CORE_IMPORT_ERROR))
    return RustEngine()


# ---------------------------------------------------------------------------
# a. the hash


def test_fnv1a64_has_the_known_answers_royalesim_pins():
    # tests/test_replay_fixture.py and tests/replay_parity.rs in RoyaleSim pin the
    # first two; "foobar" is the published FNV-1a 64 test vector.
    assert fnv1a64(b"") == "cbf29ce484222325"
    assert fnv1a64(b"a") == "af63dc4c8601ec8c"
    assert fnv1a64(b"foobar") == "85944171f73967e8"
    assert fnv1a64(b"foobar") == independent_fnv1a64(b"foobar")


# ---------------------------------------------------------------------------
# b. the file RustEngine stamps is the file the engine read


@needs_core
def test_rust_stamp_is_the_fnv1a64_of_the_file_the_engine_reads(rust):
    stamp = rust.card_table_stamp()
    path = rust.cards_json_path
    assert stamp["cards_json_fnv1a64"] == independent_fnv1a64(path.read_bytes())
    assert stamp["cards_vintage"] == file_vintage(path)
    assert stamp["cards_json_hash_source"] in ("engine", "disk_at_construction")
    config = rust.config()
    assert {k: config[k] for k in stamp} == stamp, "config() carries the stamp"


@needs_core
def test_the_stamped_file_holds_the_catalogue_the_engine_reports(rust):
    """Graded against the engine: its default catalogue is every card of the table it
    loaded, so a stamp taken from any other file shows up here. The 2018 table and a
    newer one differ in the card set and in Goblins' count, so either way round a
    wrong file fails."""
    doc = json.loads(rust.cards_json_path.read_text(encoding="utf-8"))
    by_name = {c["name"]: c for c in doc["cards"]}
    engine_cards = rust.cards()
    assert len(engine_cards) > 50, "the default catalogue is the whole table"
    missing = [c.name for c in engine_cards if c.name not in by_name]
    assert missing == [], f"the engine holds cards the stamped file does not: {missing[:5]}"
    elixir = [(c.name, c.elixir, by_name[c.name]["elixir"]) for c in engine_cards]
    assert [e for e in elixir if e[1] != e[2]] == []
    units = [c for c in engine_cards if c.placement in (Placement.TROOP, Placement.BUILDING)]
    counts = [(c.name, c.count, by_name[c.name].get("count", 1)) for c in units]
    assert [c for c in counts if c[1] != c[2]] == []


@needs_core
def test_royalesim_data_dir_does_not_move_the_engines_card_path(tmp_path, monkeypatch):
    where, _ = rust_engine.engine_cards_json_path()
    (tmp_path / "derived").mkdir()
    (tmp_path / "derived" / "cards.json").write_text('{"provenance": {"vintage": "x"}}')
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    assert protocol.data_dir() == tmp_path, "the override is live for this package"
    assert rust_engine.engine_cards_json_path()[0] == where


def test_the_build_directory_is_found_behind_other_text(tmp_path, monkeypatch):
    """The directory is preceded by whatever the linker put before it. Here a stray
    "N" and a path fragment that does not exist, the shape seen in a real build."""
    checkout = tmp_path / "Sim Checkout"
    cards = checkout / "data" / "derived" / "cards.json"
    cards.parent.mkdir(parents=True)
    cards.write_text("{}")
    # The build directory has to EXIST, as it does in a real build. The fixture used to
    # leave it out, and that is what made this test pass on Windows and return None on a
    # clean ubuntu runner: `Path(missing, "..", "..", f).is_file()` is True on Windows,
    # which normalises ".." lexically, and False on POSIX, which walks each component.
    (checkout / "crates" / "royalesim").mkdir(parents=True)
    build_dir = str(checkout / "crates" / "royalesim").encode("utf-8")
    blob = b"\x00\x01garbage/nowhere N" + build_dir + b"/../../data/derived/\x00tail"
    ext = tmp_path / "royalesim.bin"
    ext.write_bytes(blob)
    monkeypatch.setattr(rust_engine, "_extension_file", lambda: ext)
    rust_engine._cards_json_in_build_checkout.cache_clear()
    try:
        got = rust_engine._cards_json_in_build_checkout()
        # DIAGNOSTIC ON FAILURE, added 2026-09-23 after this passed on Windows and returned
        # None on a clean ubuntu runner. Windows never exercises the candidate LOOP: the
        # first regex match there is the drive letter, which is already the right start,
        # while on POSIX the first match is a slash inside the leading garbage and the
        # right answer is the second candidate. The failure said only "None", which names
        # no step, so it is reconstructed here rather than guessed at from a laptop that
        # cannot reproduce it.
        if got != cards.resolve():
            blob_read = ext.read_bytes()
            at = blob_read.find(rust_engine._CARD_PATH_TAIL)
            start = at
            while start > 0 and at - start < 4096 and 0x20 <= blob_read[start - 1] != 0x7F:
                start -= 1
            run = blob_read[start:at]
            tried = []
            for m in rust_engine._PATH_ROOT.finditer(run):
                bd = run[m.start() :].decode("utf-8", "replace")
                cand = Path(bd, "..", "..", "data", "derived", "cards.json")
                tried.append(f"{bd!r} -> is_file={cand.is_file()}")
            raise AssertionError(
                f"expected {cards.resolve()}, got {got!r}.\n"
                f"  tail found at: {at}\n"
                f"  run: {run.decode('utf-8', 'replace')!r}\n"
                f"  candidates tried, in order:\n    " + "\n    ".join(tried)
            )
    finally:
        rust_engine._cards_json_in_build_checkout.cache_clear()


# ---------------------------------------------------------------------------
# c. different bytes, a file that changes, an engine that reports its own hash


@needs_core
def test_different_bytes_give_a_different_stamp(rust, tmp_path, monkeypatch):
    real = rust.cards_json_path
    raw = real.read_bytes()
    vintage = file_vintage(real)
    edited = raw.replace(
        json.dumps(vintage).encode(), json.dumps(vintage + " (edited copy)").encode(), 1
    )
    assert edited != raw, "the edit did not land"
    copy = tmp_path / "cards.json"
    copy.write_bytes(edited)
    monkeypatch.setattr(rust_engine, "engine_cards_json_path", lambda: (copy, "test"))
    monkeypatch.setattr(rust_engine, "ENGINE_CARD_HASH_NAMES", ())
    stamp = RustEngine(card_names=SHARED).card_table_stamp()
    assert stamp["cards_json_fnv1a64"] == independent_fnv1a64(edited)
    assert stamp["cards_json_fnv1a64"] != rust.card_table_stamp()["cards_json_fnv1a64"]
    assert stamp["cards_vintage"] == vintage + " (edited copy)"
    assert stamp["cards_json_hash_source"] == "disk_at_construction"


def core_with_battle(battle_cls: type) -> types.SimpleNamespace:
    """The compiled module with its Battle class swapped for ``battle_cls``."""
    real = rust_engine._core
    names = {k: getattr(real, k) for k in dir(real) if not k.startswith("__")}
    return types.SimpleNamespace(**{**names, "Battle": battle_cls})


@needs_core
def test_a_file_that_changes_while_the_engine_reads_it_is_refused(tmp_path, monkeypatch):
    real = rust_engine._core.Battle
    copy = tmp_path / "cards.json"
    copy.write_text('{"provenance": {"vintage": "before"}}')

    class RewritingBattle:
        tower_positions = staticmethod(real.tower_positions)

        def __new__(cls, *args):
            copy.write_text('{"provenance": {"vintage": "during the read"}}')
            return real(*args)

    monkeypatch.setattr(rust_engine, "_core", core_with_battle(RewritingBattle))
    monkeypatch.setattr(rust_engine, "engine_cards_json_path", lambda: (copy, "test"))
    with pytest.raises(RuntimeError, match="changed while the engine was reading it"):
        RustEngine(card_names=SHARED)


@needs_core
@pytest.mark.parametrize("same_file", [True, False], ids=["file-matches", "file-differs"])
def test_an_engine_that_reports_its_own_hash_is_believed(rust, monkeypatch, same_file):
    """The engine's hash wins either way. The file's vintage is stated only when the
    file hashes to what the engine reported, because only then is it that table."""
    real = rust_engine._core.Battle
    on_disk = independent_fnv1a64(rust.cards_json_path.read_bytes())
    reported = int(on_disk, 16) if same_file else 0x0123456789ABCDEF

    class ReportingBattle:
        """A Battle that reports the hash of the table it loaded, as a later build may."""

        tower_positions = staticmethod(real.tower_positions)

        def __init__(self, *args):
            self._battle = real(*args)

        def __getattr__(self, name):
            return getattr(self._battle, name)

        def cards_json_fnv1a64(self):
            return reported

    monkeypatch.setattr(rust_engine, "_core", core_with_battle(ReportingBattle))
    stamp = RustEngine(card_names=SHARED).card_table_stamp()
    assert stamp == {
        "cards_json_fnv1a64": f"{reported:016x}",
        "cards_vintage": file_vintage(rust.cards_json_path) if same_file else "unknown",
        "cards_json_hash_source": "engine",
    }


def test_an_engine_hash_that_is_not_one_is_refused():
    assert rust_engine._engine_card_hash(types.SimpleNamespace()) is None
    upper = types.SimpleNamespace(card_table_fnv1a64="0123456789ABCDEF")
    assert rust_engine._engine_card_hash(upper) == "0123456789abcdef"
    with pytest.raises(RuntimeError, match="not an FNV-1a 64 hash"):
        rust_engine._engine_card_hash(types.SimpleNamespace(cards_json_fnv1a64=lambda: "xyz"))
    with pytest.raises(RuntimeError, match="not an FNV-1a 64 hash"):
        rust_engine._engine_card_hash(types.SimpleNamespace(cards_json_fnv1a64=-1))


# ---------------------------------------------------------------------------
# d. MockEngine


def test_mock_stamp_names_its_pack_and_moves_with_a_number_it_read(monkeypatch):
    base = MockEngine(card_names=["Knight", "Archer"])
    stamp = base.card_table_stamp()
    assert stamp["cards_vintage"] == RAW_CARD_PACK
    assert MockEngine(card_names=["Knight", "Archer"]).card_table_stamp() == stamp
    assert MockEngine(card_names=["Knight", "Giant"]).card_table_stamp() != stamp
    assert {k: base.config()[k] for k in stamp} == stamp, "config() carries the stamp"

    real_table = mock_engine_module._csv_table

    def knight_tougher(path, required=()):
        table = real_table(path, required)
        if "Knight" in table and "Hitpoints" in table["Knight"]:
            row = dict(table["Knight"])
            row["Hitpoints"] = str(int(row["Hitpoints"]) + 1)
            table["Knight"] = row
        return table

    monkeypatch.setattr(mock_engine_module, "_csv_table", knight_tougher)
    tougher = MockEngine(card_names=["Knight", "Archer"])
    assert tougher.cards()[0].hitpoints == base.cards()[0].hitpoints + 1, "plant did not land"
    assert tougher.card_table_stamp()["cards_loaded_fnv1a64"] != stamp["cards_loaded_fnv1a64"]


# ---------------------------------------------------------------------------
# e. the trace header


def record_short_battle(engine) -> Trace:
    n = len(engine.cards())
    deck = [i % n for i in range(8)]
    setup = MatchSetup(decks=[deck, list(reversed(deck))])
    rec = ReplayRecorder(frame_every_tick=False)
    engine.reset(5, setup)
    rec.begin(engine, 5, setup)
    engine.step([], 4)
    rec.record_step(0, 4, [], [])
    rec.record_frame(engine)
    trace = rec.end(engine)
    assert trace is not None
    return trace


def engines() -> list:
    out = [pytest.param(lambda: MockEngine(), id="mock")]
    out.append(pytest.param(lambda: RustEngine(card_names=SHARED), id="rust", marks=needs_core))
    return out


@pytest.mark.parametrize("make", engines())
@pytest.mark.parametrize("suffix", [".msgpack", ".json"])
def test_the_trace_header_carries_the_stamp(make, suffix, tmp_path):
    engine = make()
    trace = record_short_battle(engine)
    loaded = load_trace(save_trace(trace, tmp_path / f"t{suffix}"))
    header = msgspec.structs.asdict(loaded.header)
    stamp = engine.card_table_stamp()
    assert stamp, "the engine states a card table"
    assert {k: header[k] for k in stamp} == stamp
    assert verify_trace(loaded, make()) == []


@pytest.mark.parametrize("make", engines())
def test_a_trace_from_before_the_stamp_still_decodes(make):
    trace = record_short_battle(make())
    doc = json.loads(msgspec.json.encode(trace))
    for key in CARD_TABLE_FIELDS:
        del doc["header"][key]
    old = msgspec.json.decode(json.dumps(doc).encode(), type=Trace)
    assert old.header.cards_vintage == old.header.cards_json_fnv1a64 == ""
    assert old.header.cards_json_hash_source == old.header.cards_loaded_fnv1a64 == ""
    assert old.frames == trace.frames
    old_msgpack = msgspec.msgpack.decode(msgspec.msgpack.encode(doc), type=Trace)
    assert old_msgpack.header.cards_vintage == ""


# ---------------------------------------------------------------------------
# Which BINARY is loaded, as opposed to which data it was built with


def test_the_engine_binary_stamp_is_the_compiled_module_not_the_package_stub() -> None:
    """The stamp has to hash the .pyd, and the difference is not cosmetic.

    ``royalesim.__file__`` is a 119-byte ``__init__.py`` that re-exports the compiled
    submodule. Hashing THAT gives a value identical across every rebuild, so the stamp
    would read as provenance while being unable to notice a new engine: worse than no
    stamp, because it looks like evidence. The first version of this did exactly that
    and was caught by printing the value rather than trusting it.
    """
    import royalesim

    native = royalesim.royalesim
    on_disk = Path(native.__file__)
    assert on_disk.suffix.lower() not in {".py", ".pyc"}, (
        f"the compiled module resolved to {on_disk.name}, which is source, not a binary"
    )
    want = hashlib.sha256(on_disk.read_bytes()).hexdigest()[:16]
    assert engine_binary_digest() == want

    stub = Path(royalesim.__file__)
    stub_hash = hashlib.sha256(stub.read_bytes()).hexdigest()[:16]
    assert engine_binary_digest() != stub_hash, (
        "the stamp is the hash of the package's __init__.py, so it cannot change when "
        "the engine is rebuilt"
    )


def test_the_binary_stamp_notices_a_different_binary(tmp_path) -> None:
    """A stamp that never moves is the thing being guarded against, so move it.

    The cache is per-process and keyed on nothing, which is right for a file that
    cannot change under a running engine. This reaches past it deliberately.
    """
    import royalesim

    real = Path(royalesim.royalesim.__file__)
    first = engine_binary_digest()

    changed = tmp_path / real.name
    changed.write_bytes(real.read_bytes() + b"\x00not the same engine")
    stand_in = type("Native", (), {"__file__": str(changed)})()

    engine_binary_digest.cache_clear()
    try:
        with mock.patch.object(rust_engine, "_core", type("Pkg", (), {"royalesim": stand_in})()):
            assert engine_binary_digest() != first
    finally:
        engine_binary_digest.cache_clear()
    assert engine_binary_digest() == first


def test_the_binary_stamp_is_in_config(rust) -> None:
    """A checkpoint says which data an engine used; it should say which engine too."""
    config = rust.config()
    assert config["engine_binary_sha256"] == engine_binary_digest()
    assert config["build_digest"] == rust_engine.build_digest()
    # The three are different questions and must not be the same answer.
    assert len({config["engine_binary_sha256"], config["build_digest"]}) == 2


def test_a_build_directory_that_does_not_exist_is_refused_on_every_platform(
    tmp_path, monkeypatch
):
    """The cross-platform finding from the first clean-runner CI run, pinned.

    ``Path(x, "..", "..", f).is_file()`` asks a DIFFERENT QUESTION on the two platforms.
    Windows normalises ``..`` lexically and answers without the intermediate directory
    existing; POSIX walks every component and ENOENTs. Because the scanner takes the FIRST
    candidate that is a file, earliest first, the same extension could resolve to DIFFERENT
    checkouts on the two platforms -- Windows accepting a short garbage candidate that
    POSIX rejects. The consequence is silent: the card table gets a provenance label that is
    confidently wrong, which is exactly what the digest-beside-the-count discipline exists
    to prevent.

    This would have FAILED on Windows before the fix and passed on Linux, so it is the
    mirror of the failure that found it.
    """
    real = tmp_path / "real"
    (real / "data" / "derived").mkdir(parents=True)
    (real / "data" / "derived" / "cards.json").write_text("{}")
    (real / "crates" / "royalesim").mkdir(parents=True)

    # A decoy whose cards.json exists but whose BUILD DIRECTORY does not. On Windows this
    # used to win, because the lexical "..' never checks that crates/royalesim is there.
    decoy = tmp_path / "decoy"
    (decoy / "data" / "derived").mkdir(parents=True)
    (decoy / "data" / "derived" / "cards.json").write_text("{}")

    decoy_build = str(decoy / "crates" / "royalesim").encode("utf-8")
    real_build = str(real / "crates" / "royalesim").encode("utf-8")
    blob = (
        b"\x00\x01"
        + decoy_build
        + b"/../../data/derived/\x00\x01"
        + real_build
        + b"/../../data/derived/\x00tail"
    )
    ext = tmp_path / "royalesim.bin"
    ext.write_bytes(blob)
    monkeypatch.setattr(rust_engine, "_extension_file", lambda: ext)
    rust_engine._cards_json_in_build_checkout.cache_clear()
    try:
        got = rust_engine._cards_json_in_build_checkout()
    finally:
        rust_engine._cards_json_in_build_checkout.cache_clear()
    assert got == (real / "data" / "derived" / "cards.json").resolve(), (
        f"resolved to {got!r}. The decoy's cards.json exists but its build directory does "
        "not, so it is not a checkout anything was built in. Accepting it means the engine's "
        "card table is attributed to the wrong tree, silently, and differently on Windows "
        "and Linux."
    )

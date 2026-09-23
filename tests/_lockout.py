"""The opening deploy lockout, read once, without letting a stale build break collection.

WHY THIS IS NOT JUST ``RustEngine().rules().deploy_lockout_ticks``
    Several test modules need the value at IMPORT time, to put ``start_tick`` into a
    ``MatchSetup`` the whole module shares. Constructing an engine at import makes those
    modules depend on the engine being CONSTRUCTIBLE, which is a third state nobody had a
    name for until 2026-09-23:

        not installed    ``core_available()`` is False, the module skips. Fine.
        installed, fresh engine constructs. Fine.
        installed, STALE ``core_available()`` is TRUE and construction RAISES.

    In the third state a module-level constructor raises during COLLECTION, and pytest
    reports "3 errors during collection" and runs nothing at all -- including the
    MockEngine tests in those files, which do not need the engine and would have passed.
    One repo's stale data then looks like this repo's suite being broken.

    That is a real state and not a hypothetical: RoyaleSim's calibration.json is read at
    RUNTIME, so a calibration edit in another repo makes every engine here stale with no
    commit and no rebuild in this one. It happened twice on 2026-09-23.

WHAT THIS DOES INSTEAD
    Falls back to 0, which is the "no lockout" arm and a real calibration value, so a
    module still imports and its engine-free tests still run. The staleness is NOT
    swallowed: every test that actually needs the engine constructs one and gets the
    engine's own message, which names the drifted keys and says to rebuild. The report
    moves from collection, where it hides everything, to the tests it actually stops.
"""

from __future__ import annotations

from royalegym.rust_engine import RustEngine, core_available


def lockout_ticks() -> int:
    """Ticks a match refuses every deploy for, or 0 when no engine can answer."""
    if not core_available():
        return 0
    try:
        return int(RustEngine().rules().deploy_lockout_ticks)
    except Exception:
        # A stale or unbuildable engine. Deliberately broad: whatever went wrong, the
        # tests that need the engine construct one and report it themselves.
        return 0

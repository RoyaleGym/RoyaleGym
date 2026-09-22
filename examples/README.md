# Examples

Seven programs, in the order they are worth reading. Each one runs on its own and prints
something you can check. They need the engine built, which the repository README covers
under "Install".

```bash
python examples/01_one_battle.py
```

| | What it shows |
|---|---|
| [`01_one_battle.py`](01_one_battle.py) | A whole battle, both seats, random legal moves. The shortest thing that is a real match. |
| [`02_one_seat_against_a_bot.py`](02_one_seat_against_a_bot.py) | The single-agent shape: one seat learns, the other is a scripted opponent. Includes writing an opponent. |
| [`03_self_play_batch.py`](03_self_play_batch.py) | Four games as eight agent slots, and how to read an episode's statistics out of batched `final_info`. |
| [`04_your_own_reward.py`](04_your_own_reward.py) | Writing the one piece you are expected to write, and reading the per-term breakdown that says which term your bot is chasing. |
| [`05_record_and_replay.py`](05_record_and_replay.py) | Record a battle, re-run it on a fresh engine to prove it, and turn it into a page you can double-click. |
| [`06_resume_a_run.py`](06_resume_a_run.py) | Stopping a run and starting it again so it continues the same battles rather than replaying old ones. |
| [`07_is_this_bot_better.py`](07_is_this_bot_better.py) | Comparing two bots on both seats, with an error bar, so "better" is a claim the sample supports. |

Every one of them is run by the test suite, and each is checked for the line that proves
it did the thing it exists to show rather than merely reaching the end of the file
(`tests/test_examples.py`). An example that does not run is worse than no example: it is
the first thing a reader tries, and when it fails they cannot tell whether they
installed the library wrong or the library is wrong.

## Two things worth knowing before you write your own

**Name your engine.** `RustEngine()` is the game. `MockEngine()` is a readable Python
reference implementation with a different card table, spells that resolve on the spot
instead of travelling, and no stuns or knockback — right for learning the API, wrong for
believing a trained bot. Anything that defaults will tell you it defaulted.

**Pass a factory, not an instance.** Every environment needs its own engine, observation
builder, parser and reward, because each carries per-battle state. Handing one instance
to several environments does not raise; it gives you several windows onto one battle,
each stepping it again on its own turn, and a training run collecting that sees only
that learning did not work.

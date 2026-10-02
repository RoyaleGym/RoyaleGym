# RoyaleImitate

[RoyaleImitate](https://github.com/RoyaleGym/RoyaleImitate) is an optional add-on to
RoyaleLearn. It does two things:

- **Warm start.** A new bot starts from the weights of a bot you already trained, instead of
  from nothing.
- **Stay close.** While the new bot learns, a penalty keeps its moves close to that other bot's,
  so it doesn't forget what it already knew.

That's useful when you change something about your setup, such as the reward, and want to keep
what an earlier bot learned instead of starting over. Leave RoyaleImitate out and RoyaleLearn
works exactly the same without it.

## Example

Train a first bot (the teacher), save it, then start a second bot (the student) from it:

```py
from royalegym import make_env
from royaleimitate import save_actor
from royalelearn import Learner


def build_env():
    return make_env()


if __name__ == "__main__":
    teacher = Learner(build_env, save_dir="runs/teacher")
    teacher.learn(total_steps=100_000)
    digest = save_actor(teacher, "runs/teacher-actor")

    teacher_actor = {"path": "runs/teacher-actor", "sha256": digest}
    student = Learner(
        build_env,
        save_dir="runs/student",
        extensions={
            # Start from the teacher's weights.
            "warm_start": {"init": teacher_actor},
            # Keep the student's moves close to the teacher's.
            "imitation": {
                "references": {"teacher": {"kind": "snapshot", **teacher_actor}},
                "regularisers": [{
                    "kind": "reference_kl",
                    "name": "teacher",
                    "reference": "teacher",
                    "budget": {"kind": "constant", "value": 0.1},
                    "coef": {"start": 1.0},
                }],
            },
        },
    )
    student.learn(total_steps=100_000)
```

`save_actor` writes the teacher in the form RoyaleImitate reads, and returns a fingerprint of it
(`digest`), so the student is sure to load exactly that file. `budget` is how far the student may
drift from the teacher, and `coef` how hard it is pulled back; the pull adjusts itself to keep
the student inside the budget. The two numbers here are RoyaleImitate's defaults, a starting
point and not a recommendation; newer versions let you leave them out.

Every option is in the [RoyaleImitate guide](https://github.com/RoyaleGym/RoyaleImitate/blob/main/docs/guide.md).

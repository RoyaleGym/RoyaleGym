# Contributing

Thanks for helping. Bug reports, fixes, examples and docs are all welcome.

## Set up

Clone RoyaleGym and RoyaleSim into the same folder, then install both into one virtual
environment. The full steps are on the [install page](docs/site/pages/install.md).

## Before you open a pull request

    python -m pytest -q -rs
    python -m ruff check royalegym tests examples

- Add a test that fails without your change and passes with it.
- A skipped test is not a passing one. If a test skips on your machine, say which and why.
- Keep public text plain: short sentences, no internal names.

## Reporting a bug

Open an issue with the template. Include the commands you ran, the full error, your OS and
Python version, and `python -c "import royalegym; print(royalegym.__version__)"`.

## Licence

By contributing you agree your work is released under this repository's [licence](LICENSE).

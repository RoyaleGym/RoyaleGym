# Configuration Objects

A battle in RoyaleGym is built from a few parts, and each one is an object you can swap. You set
them up in `build_env` in your `quickstart.py`. Change one, train again, and see what your bot does
differently.

<div class="grid cards" markdown>

-   :material-database-export-outline:{ .lg .middle } **[State Mutators](state-mutators.md)**

    ---

    How each battle starts: the decks, evolutions and heroes, or a start partway through.

-   :material-eye-outline:{ .lg .middle } **[Observation Builders](observation-builders.md)**

    ---

    What the bot sees after every step: the arena, its hand, the elixir and the clock.

-   :material-gesture-tap:{ .lg .middle } **[Action Parsers](action-parsers.md)**

    ---

    The moves the bot can make, which card where, and which ones are legal right now.

-   :material-trophy-outline:{ .lg .middle } **[Reward Functions](reward-functions.md)**

    ---

    What the bot is paid for. The bot learns whatever makes this number big.

-   :material-flag-checkered:{ .lg .middle } **[Done Conditions](done-conditions.md)**

    ---

    When a battle ends, and the difference between ending and being cut short.

-   :material-robot-outline:{ .lg .middle } **[Opponents](opponents.md)**

    ---

    Who sits in the other seat: a scripted bot, a bot you wrote, or your own bot.

-   :material-monitor-eye:{ .lg .middle } **[Viewer](viewer.md)**

    ---

    Watch a battle live while it trains, or open a saved one later.

</div>

The [How It Fits Together](../../overview.md) page shows how they meet in one step.

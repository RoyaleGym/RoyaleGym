"""Train your first Clash Royale bot, then watch it play.

    pip install "royalegym[all]"
    python quickstart.py

It trains on your CPU and prints its progress as it goes. When it stops, it plays one
battle against a bot that makes random moves and saves it, so you can watch:

    royaleviser my_bot_battle.msgpack
"""

from royalelearn import Learner

from royalegym import TowerHPReward, make_env, play_battle


def build_env():
    # One battle: both sides get the same starter deck.
    # The reward is what the bot is paid for. Tower damage done minus tower damage taken
    # pays out on most moves, so a new bot learns quickly. Change this line to try another.
    return make_env(reward=TowerHPReward())


if __name__ == "__main__":
    learner = Learner(
        build_env,
        n_envs=4,                     # battles played at once; more is faster on more cores
        device="cpu",                 # "cuda" if you have a graphics card
        opponent="random",            # who the bot plays: "random", "noop" or "self"
        save_dir="runs/quickstart",   # checkpoints go here; run again to continue
    )
    learner.learn(total_steps=200_000)
    learner.save("runs/quickstart/bot")

    # Play one battle with the trained bot as Blue and save it to watch.
    bot = Learner.load_policy("runs/quickstart/bot")
    battle = play_battle(build_env(), blue=bot, red="random", save_to="my_bot_battle.msgpack")
    winner = {0: "your bot", 1: "the random bot", 2: "nobody (a draw)"}.get(battle.winner, "?")
    print(f"Winner: {winner}. Crowns {battle.crowns[0]}-{battle.crowns[1]}.")
    print(f"Watch it: royaleviser {battle.path}")

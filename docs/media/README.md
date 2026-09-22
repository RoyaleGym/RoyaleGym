# README media

Every file the README embeds. A *placeholder* is a generated SVG whose caption says what the
real image or screen recording must show; replace the file, keep the name.

| File | Kind | Shows / must show |
|---|---|---|
| `battle-in-viewer.png` | real still | already final |
| `family.svg` | diagram | the five repos and how they depend on each other; final |
| `five-pieces.svg` | image placeholder | Five swappable pieces - The env constructor as a diagram: obs builder, action parser, reward, state mutator, done conditions, each with its shipped defaults |
| `hidden-information.svg` | image placeholder | Hidden information, as in the game - Blue's observation with its own elixir and hand shown and the enemy's blank, next to the full engine state that has both |
| `legality-mask.svg` | image placeholder | An exact legality mask - The 691 of 2305 actions legal on the first step, drawn per hand card over the 18 x 32 tile board |
| `record-and-verify.svg` | image placeholder | Record it, re-run it, prove it - A terminal: the recorded battle re-simulated on a fresh engine, 4130 frame hashes compared, divergences: [] |
| `replay-page.svg` | image placeholder | A replay page, no server - Browser screenshot of battle.html from python -m royalegym.render: the board, the timeline scrubber, a hover tooltip on one unit |
| `self-play-batch.svg` | image placeholder | Batched self-play - ClashSelfPlayVecEnv(4): four boards, eight agent slots, one policy fed both seats' observations in their own frame |
| `start-anywhere.svg` | image placeholder | Start from any position - Three episode starts side by side: a fresh battle, a mid-game board with a tower already down, a saved snapshot resumed |
| `two-apis.svg` | image placeholder | Two APIs, one battle - Side by side: ClashParallelEnv (PettingZoo, both seats) and gymnasium.make('royalegym/ClashRoyale-v0') stepping the same board |
| `whole-battle.svg` | video placeholder | A whole battle in about a second - Screen recording: RoyaleViser playing back the Try-it battle (seed 0, two random-legal players) from kick-off to the 1-0 crown at tick 4129 |

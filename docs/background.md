# Background: why this stack is measured rather than modelled

A Clash Royale simulator is easy to start and hard to finish. The difficult part is not
writing a tick loop — it is knowing what the tick loop is supposed to do. This page records
what is publicly established about the game's rules, what is not, and why the project is
built around differential testing against recordings of the real game instead of around a
model someone reasoned out.

## Ground pathfinding cannot be derived from public description

Supercell rewrote ground pathfinding on 2025-03-31 and has changed it several times since.
Anything written about unit pathing before that date describes a different algorithm, and
some of what was written after it is already stale.

| Change | Source |
|---|---|
| 2025-03-31: troops see buildings in advance, rely much less on lanes, move diagonally | April Update release notes |
| 2025-04-01: bridge pathing fix | release notes |
| 2025-09-29: destroyed-building and Spirit Empress pathing fixes | October Update 2025 |
| 2026-02-23: troops no longer switch lanes when one Princess Tower dies | March Update 2026 |
| 2026-02-24: "various pathfinding issues" | release notes |

Sight range, which drives approach behaviour, is retuned per card by balance patches (Battle
Ram 5.5 to 6.5 and Golem 7 to 7.5 on 2026-09-08 alone). No public engine models the current
pathfinder, and no amount of reasoning about the old one recovers it.

The answer the project settled on is to treat the real game as the oracle: per-tick
recordings of real battles — entity state, targets and each unit's planned path nodes — turn
"work out the pathfinder" into a differential test. RoyaleLive, a private sibling repo, is
the client instrument that records those ground-truth traces; RoyaleSim's `tools/oracle_diff.py`
diffs the engine against one. Every constant that comes out of that process lands in
`RoyaleSim/data/calibration.json` with the client version it was measured on, because a
monthly client update can move it.

## What public sources settle, and what they do not

| Claim | Status |
|---|---|
| Arena is 18 x 32 tiles = 36 x 64 half-tiles; water at `y` in [15, 17]; bridges 2 tiles wide at `x` centres 3.5 and 14.5 | confirmed; `tilemaps/tilemap.csv` ships in the client |
| Distances are milli-tiles (1000 per tile); times are milliseconds | confirmed |
| Logic tick is 20 Hz / 50 ms | inferred, then measured: the elixir phase boundaries at ticks 2400/3600/4800/6000 land exactly on 2/3/4/5 minutes |
| Starting elixir 5; 180 s regulation plus 120 s overtime; king activation 3300 ms | confirmed |
| Globals that exist and matter: `LOGIC_DEFAULT_TARGET_USE_LANE_ID`, `LOGIC_LANE_ID_BASED_DEPLOY_SEQUENCE`, `LOGIC_XPOS_BASED_TOWER_TARGETING`, `LOGIC_PRESERVE_TARGET_IF_HIT_STARTED`, `LOGIC_RANGE_EXTENSION_TO_KEEP_TARGET = 25`, `ADD_CHARACTER_RANGE_TO_RADIUS` (range is edge to edge) | confirmed in the shipped game data |
| `PATHFINDING_DEFAULT_COST = 8`, `ROAD = 5`, `WATER = 7`, `BUILDING = 50` | **refuted**: absent from the 2018, 2022 and 2023 globals tables; the only source is one hobby repository's docstring |
| The `Speed` field divides by 60 (Medium = 1.0 tiles/s) or by 50 (1.2 tiles/s) | neither, and no public source settles it — a 20% error either way. Measured: one `Speed` unit is one milli-tile per 50 ms tick, so the raw column *is* the per-tick step |

The lesson the ledger encodes: a constant with a plausible comment next to it is
indistinguishable from a guess. Every entry in `calibration.json` therefore states how it is
known — measured, read out of the shipped game data, or an open question — and nothing reads
a number that does not say.

Independent measurements of the current pathfinder exist publicly and are useful as a
cross-check: the SQURS "Advanced Stats" per-tile Skeleton Army lane-split and damage
heatmaps changed measurably between their November 2025 and September 2026 revisions, which
is itself evidence that the pathfinder moved.

## Card and arena data

Supercell's asset CDN serves the full `csv_logic` bundle (CSV plus per-card TOML overlays)
and all 24 tilemaps. That is the source of the card table and the arena geometry; RoyaleSim's
`tools/extract_*.py` turn it into `data/derived/`. Fields that a casual merge tends to drop —
mass, flying height, jump parameters, building footprints — are exactly the ones the engine
needs, so the extractors are held to the shipped data by tests rather than trusted.

## There is no bot interface in the real game

No bot league, competition or bot API exists for Clash Royale. RLBot works in Rocket League
because Psyonix provides an offline exhibition interface and bots never touch matchmaking;
there is no equivalent here. The league this project trains against is therefore built
offline, inside these repos: self-play plus a frozen-pool ladder with confidence intervals
(RoyaleLearn), with the simulator's fidelity to the real game gated separately by the
calibration work above.

## Sizing

The target the layers are designed against is one training worker on a commodity four-core
machine with 8 GB of RAM and a 4 GB GPU. That is why the engine is integer-only Rust, why
observation building is being pushed down into it, and why the viewer is a separate process
that costs nothing when nobody is watching.

## Sources

- April Update 2025 release notes (the pathfinding rewrite, 31 Mar 2025):
  <https://supercell.com/en/games/clashroyale/blog/release-notes/april-update/>
- October Update 2025 (destroyed-building pathing fix):
  <https://supercell.com/en/games/clashroyale/blog/release-notes/october-update-2025/>
- March Update 2026 (no lane switch after tower death):
  <https://supercell.com/en/games/clashroyale/blog/release-notes/march-update-2026/>
- September 2026 balance changes (sight ranges):
  <https://supercell.com/en/games/clashroyale/blog/release-notes/september-balance-changes-2026/>
- SQURS per-tile Skeleton Army measurement:
  <https://blog.squrs.com/clashroyale/skarmy_tiles>
- The speed-unit dispute (the 1.2x frame test):
  <https://clashroyale.fandom.com/wiki/User_blog:Darknighture/the_unit_of_speed_in_cr>
- Hidden card stats (collision radius, sight range, mass, load time):
  <https://clashroyale.fandom.com/wiki/User_blog:AesDragon/Hidden_card_stats:_Collision_radius>
- CDN asset puller: <https://github.com/123456abcdef/sc-assets-download>
- Labelled frame datasets (KataCR): <https://github.com/wty-yy/Clash-Royale-Detection-Dataset>

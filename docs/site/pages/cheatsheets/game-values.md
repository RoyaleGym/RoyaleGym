# Game Values

The numbers you'll want while writing rewards, observations and done conditions. They are the
engine's values, which follow the game's.

## Arena

| Name | Value | What it is |
|---|---|---|
| Size | 18 x 32 tiles | 18 tiles wide, 32 long. Each side owns 15 rows; the river takes the 2 in the middle. |
| Tile size | 18,000 units | Positions in `state.entities` are in these units: `x` from 0 to 324,000, `y` from 0 to 576,000. |
| Blue's side | `y` below the river | Blue's king tower is at the bottom. Each seat's observation is turned so its own king is at the bottom. |
| Bridges | 2 | One in each lane. Most ground troops can cross the river only on a bridge. |

## Time

| Name | Value | What it is |
|---|---|---|
| Tick | 50 ms | The engine's step. 20 ticks make a second. `state.tick` counts them from the start. |
| Decision | 500 ms | One environment step, by default (`decision_ms`). That's 10 ticks. |
| Regular time | 3 minutes | 3,600 ticks, or 360 decisions. |
| Overtime | up to 2 minutes | Only if the crowns are level after regular time. The first crown wins. |
| Opening wait | 4.5 seconds | Nobody can play a card in the first 90 ticks. |
| Longest battle | 5 minutes | 6,000 ticks, or 600 decisions. |

## Elixir

| Name | Value | What it is |
|---|---|---|
| Starting elixir | 6 | Each side's elixir when the battle begins. |
| Most elixir | 10 | Elixir stops growing at 10. Sitting there wastes it. |
| Normal rate | 1 every 2.8 s | The first two minutes. |
| Double elixir | 1 every 1.4 s | The last minute of regular time, and the first minute of overtime. |
| Triple elixir | 1 every 0.93 s | The last minute of overtime. |
| In the state | thousandths | `state.players[team].elixir_milli`: 10 elixir is `10_000`. |

## Towers

| Name | Value | What it is |
|---|---|---|
| King tower | 4,824 hitpoints | In the middle at the back. It starts asleep, and wakes when it is hit or a princess tower falls. Destroying it wins the battle and takes all three crowns. |
| Princess towers | 3,052 hitpoints | One in front of each lane. Destroying one takes a crown. |
| Order | king, left, right | The order of `tower_hp` and `tower_max_hp`, each from its owner's side. |

Every card and tower is at level 11, the tournament standard.

## Winning

| Way | When |
|---|---|
| Three crowns | Destroy the king tower, and the battle ends at once. |
| More crowns | When regular time ends, the side with more crowns wins. |
| First crown in overtime | In overtime, the next crown wins. |
| Tiebreak | If the crowns are still level when overtime ends, the side whose weakest tower has more health left wins. If those are equal too, it's a draw. |

## Codes

| Name | Values |
|---|---|
| Seats (`team`) | `0` Blue, `1` Red |
| Agents | `"blue"`, `"red"` |
| `state.winner` | `-1` not over, `0` Blue, `1` Red, `2` draw |
| `EntityKind` | `0` troop, `1` building, `2` king tower, `3` princess tower |
| `TowerSlot` | `0` king, `1` left, `2` right |
| A play's `status` | `0` accepted; anything else means the game refused it |

## Actions

| Move | Number |
|---|---|
| Wait | `0` |
| Play hand slot `s` (0 to 3) on tile (`x`, `y`) | `1 + s * 576 + y * 18 + x` |
| Press ability button `k` (with `ability_buttons=True`) | `2305 + k` |

Tiles are counted from the acting seat's own side. See [Action Parsers](../clash-royale/configuration-objects/action-parsers.md).

## Names That Differ From the Game

RoyaleGym writes card names without spaces. Most are the game's names run together
(`HogRider`, `MiniPekka`, `GoblinBarrel`), but these are different:

| In the game | In RoyaleGym |
|---|---|
| Archers | `Archer` |
| Barbarian Barrel | `BarbLog` |
| Bandit | `Assassin` |
| Cannon Cart | `MovingCannon` |
| Dart Goblin | `BlowdartGoblin` |
| Elite Barbarians | `AngryBarbarians` |
| Elixir Collector | `Elixir Collector` (this one keeps its space) |
| Executioner | `AxeMan` |
| Fire Spirit | `FireSpirits` |
| Flying Machine | `DartBarrell` |
| Furnace | `FirespiritHut` |
| Giant Snowball | `Snowball` |
| Guards | `SkeletonWarriors` |
| Ice Golem | `IceGolemite` |
| Ice Spirit | `IceSpirits` |
| Rune Giant | `GiantBuffer` |
| Lumberjack | `RageBarbarian` |
| Magic Archer | `EliteArcher` |
| Mini P.E.K.K.A | `MiniPekka` |
| Mother Witch | `WitchMother` |
| Night Witch | `DarkWitch` |
| P.E.K.K.A | `Pekka` |
| Royal Ghost | `Ghost` |
| Skeleton Barrel | `SkeletonBalloon` |
| Sparky | `ZapMachine` |
| The Log | `Log` |
| Void | `DarkMagic` |
| X-Bow | `Xbow` |
| Zappies | `MiniSparkys` |

## Evolutions and Heroes

The cards you can play in their evolved form (`1` in `forms`), and in their hero form (`2`),
on the same engine build as the card list below. Every champion's ability works too.

**Evolutions (42):** `AngryBarbarians`, `Archer`, `AxeMan`, `BabyDragon`, `Barbarians`, `Bats`, `BattleRam`, `BlowdartGoblin`, `Bomber`, `Cannon`, `ElectroDragon`, `Firecracker`, `FirespiritHut`, `Ghost`, `GoblinBarrel`, `GoblinCage`, `GoblinDrill`, `GoblinGiant`, `Hunter`, `IceSpirits`, `InfernoDragon`, `Knight`, `MegaKnight`, `MinionHorde`, `Mortar`, `Musketeer`, `Pekka`, `Princess`, `RageBarbarian`, `RoyalGiant`, `RoyalHogs`, `RoyalRecruits`, `SkeletonArmy`, `SkeletonBalloon`, `Skeletons`, `Snowball`, `Tesla`, `Valkyrie`, `Wallbreakers`, `Witch`, `Wizard`, `Zap`

**Heroes (16):** `Balloon`, `BarbLog`, `Berserker`, `Bowler`, `DarkPrince`, `EliteArcher`, `Giant`, `Goblins`, `IceGolemite`, `Knight`, `MegaMinion`, `MiniPekka`, `Musketeer`, `Tombstone`, `Valkyrie`, `Wizard`

## Cards

The engine's card list, the same on engine build `1cb11c66cdd25ced` (RoyaleSim 0.1.1) and engine
build `ca780d18ba66da8b` (RoyaleSim 0.1.2), by the names RoyaleGym uses. Hitpoints are for one unit at level 11. A few names in the list are special
versions used in game events. To get the list for the engine you have installed:

```python
from royalegym import RustEngine

cards = RustEngine().cards()
print(len(cards))
print([card.name for card in cards[:3]])
```

```
136
['Knight', 'Archer', 'Goblins']
```

| Card | Elixir | Type | Units | Hitpoints |
|---|---|---|---|---|
| `AngryBarbarians` | 6 | Troop | 2 | 1341 |
| `Archer` | 3 | Troop | 2 | 304 |
| `ArcherQueen` | 5 | Troop | 1 | 1000 |
| `Assassin` | 3 | Troop | 1 | 906 |
| `AxeMan` | 5 | Troop | 1 | 1280 |
| `BabyDragon` | 4 | Troop | 1 | 1152 |
| `Balloon` | 5 | Troop | 1 | 1676 |
| `Barbarians` | 5 | Troop | 5 | 716 |
| `Bats` | 2 | Troop | 5 | 81 |
| `BattleHealer` | 4 | Troop | 1 | 1920 |
| `BattleRam` | 4 | Troop | 1 | 967 |
| `Berserker` | 2 | Troop | 1 | 896 |
| `BlowdartGoblin` | 3 | Troop | 1 | 261 |
| `Bomber` | 2 | Troop | 1 | 304 |
| `BossBandit` | 6 | Troop | 1 | 2624 |
| `Bowler` | 5 | Troop | 1 | 2081 |
| `DarkPrince` | 4 | Troop | 1 | 1200 |
| `DarkWitch` | 4 | Troop | 1 | 906 |
| `DartBarrell` | 4 | Troop | 1 | 614 |
| `ElectroDragon` | 5 | Troop | 1 | 1049 |
| `ElectroGiant` | 7 | Troop | 1 | 3952 |
| `ElectroSpirit` | 1 | Troop | 1 | 217 |
| `ElectroWizard` | 4 | Troop | 1 | 714 |
| `EliteArcher` | 4 | Troop | 1 | 529 |
| `ElixirGolem` | 3 | Troop | 1 | 1569 |
| `Firecracker` | 3 | Troop | 1 | 304 |
| `FirespiritHut` | 4 | Troop | 1 | 727 |
| `FireSpirits` | 1 | Troop | 1 | 215 |
| `Fisherman` | 3 | Troop | 1 | 870 |
| `Ghost` | 3 | Troop | 1 | 1210 |
| `Giant` | 5 | Troop | 1 | 3968 |
| `GiantBuffer` | 4 | Troop | 1 | 2816 |
| `GiantSkeleton` | 6 | Troop | 1 | 3361 |
| `GoblinDemolisher` | 4 | Troop | 1 | 1300 |
| `GoblinGang` | 3 | Troop | 3 | 202 |
| `GoblinGiant` | 6 | Troop | 1 | 3110 |
| `GoblinMachine` | 5 | Troop | 1 | 2265 |
| `Goblins` | 2 | Troop | 4 | 202 |
| `Goblinstein` | 5 | Troop | 1 | 2385 |
| `GoldenKnight` | 4 | Troop | 1 | 1799 |
| `Golem` | 8 | Troop | 1 | 5120 |
| `HogRider` | 4 | Troop | 1 | 1697 |
| `Hunter` | 4 | Troop | 1 | 885 |
| `IceGolemite` | 2 | Troop | 1 | 1228 |
| `IceSpirits` | 1 | Troop | 1 | 215 |
| `IceWizard` | 3 | Troop | 1 | 688 |
| `InfernoDragon` | 4 | Troop | 1 | 1295 |
| `Knight` | 3 | Troop | 1 | 1766 |
| `LavaHound` | 7 | Troop | 1 | 3581 |
| `LittlePrince` | 3 | Troop | 1 | 698 |
| `MegaKnight` | 7 | Troop | 1 | 3993 |
| `MegaMinion` | 3 | Troop | 1 | 837 |
| `MergeMaiden` | 6 | Troop | 1 | 1121 |
| `MergeMaiden_Mounted` | 6 | Troop | 1 | 1121 |
| `MergeMaiden_Normal` | 3 | Troop | 1 | 1121 |
| `MightyMiner` | 4 | Troop | 1 | 2250 |
| `Miner` | 3 | Troop | 1 | 1210 |
| `MinionGiant` | 4 | Troop | 1 | 1817 |
| `MinionHorde` | 5 | Troop | 6 | 230 |
| `Minions` | 3 | Troop | 3 | 230 |
| `MiniPekka` | 4 | Troop | 1 | 1390 |
| `MiniSparkys` | 4 | Troop | 3 | 529 |
| `Monk` | 5 | Troop | 1 | 2214 |
| `MovingCannon` | 5 | Troop | 1 | 1809 |
| `Musketeer` | 4 | Troop | 1 | 721 |
| `Pekka` | 7 | Troop | 1 | 3760 |
| `Phoenix` | 4 | Troop | 1 | 1052 |
| `Prince` | 5 | Troop | 1 | 1920 |
| `PrinceBuff` | 5 | Troop | 1 | 2304 |
| `Princess` | 3 | Troop | 1 | 261 |
| `RageBarbarian` | 4 | Troop | 1 | 1282 |
| `RamRider` | 5 | Troop | 1 | 1766 |
| `Rascals` | 5 | Troop | 1 | 1832 |
| `Ronin` | 5 | Troop | 1 | 1779 |
| `RoyalGiant` | 6 | Troop | 1 | 3164 |
| `RoyalHogs` | 5 | Troop | 4 | 837 |
| `RoyalRecruits` | 7 | Troop | 6 | 547 |
| `RoyalRecruits_Chess` | 7 | Troop | 8 | 532 |
| `SkeletonArmy` | 3 | Troop | 15 | 81 |
| `SkeletonBalloon` | 3 | Troop | 1 | 532 |
| `SkeletonDragons` | 4 | Troop | 2 | 560 |
| `SkeletonKing` | 4 | Troop | 1 | 2298 |
| `Skeletons` | 1 | Troop | 3 | 81 |
| `SkeletonWarriors` | 3 | Troop | 3 | 81 |
| `SkeletonWarriors_SpookyChess` | 7 | Troop | 8 | 81 |
| `SpearGoblins` | 2 | Troop | 3 | 133 |
| `SuperArcher` | 3 | Troop | 2 | 701 |
| `SuperHogRiderTerry` | 4 | Troop | 1 | 2300 |
| `SuperIceGolemite` | 4 | Troop | 1 | 3630 |
| `SuperKnight` | 4 | Troop | 1 | 2030 |
| `SuspiciousBush` | 2 | Troop | 1 | 81 |
| `ThreeMusketeers` | 9 | Troop | 3 | 883 |
| `TriWizards` | 7 | Troop | 1 | 755 |
| `Valkyrie` | 4 | Troop | 1 | 1907 |
| `Wallbreakers` | 2 | Troop | 2 | 330 |
| `Witch` | 5 | Troop | 1 | 839 |
| `WitchMother` | 4 | Troop | 1 | 529 |
| `Wizard` | 5 | Troop | 1 | 755 |
| `ZapMachine` | 6 | Troop | 1 | 1451 |
| `BarbarianHut` | 6 | Building | 1 | 1164 |
| `BarbarianLauncher` | 5 | Building | 1 | 1472 |
| `BombTower` | 4 | Building | 1 | 1356 |
| `Cannon` | 3 | Building | 1 | 824 |
| `Elixir Collector` | 6 | Building | 1 | 1070 |
| `GoblinCage` | 4 | Building | 1 | 780 |
| `GoblinDrill` | 4 | Building | 1 | 2560 |
| `GoblinHut` | 4 | Building | 1 | 1180 |
| `GoblinPartyHut` | 5 | Building | 1 | 1180 |
| `InfernoTower` | 5 | Building | 1 | 1748 |
| `Mortar` | 4 | Building | 1 | 1369 |
| `Tesla` | 4 | Building | 1 | 1182 |
| `Tombstone` | 3 | Building | 1 | 529 |
| `Xbow` | 6 | Building | 1 | 1600 |
| `Arrows` | 3 | Spell | - | - |
| `BarbLog` | 2 | Spell | - | - |
| `Clone` | 3 | Spell | - | - |
| `DarkMagic` | 5 | Spell | - | - |
| `Earthquake` | 3 | Spell | - | - |
| `Fireball` | 4 | Spell | - | - |
| `Freeze` | 4 | Spell | - | - |
| `GoblinBarrel` | 3 | Spell | - | - |
| `GoblinCurse` | 2 | Spell | - | - |
| `Graveyard` | 5 | Spell | - | - |
| `Heal` | 1 | Spell | - | - |
| `Lightning` | 6 | Spell | - | - |
| `Log` | 2 | Spell | - | - |
| `Mirror` | 1 + the copied card | Spell | - | - |
| `Poison` | 4 | Spell | - | - |
| `Rage` | 2 | Spell | - | - |
| `Rocket` | 6 | Spell | - | - |
| `RoyalDelivery` | 3 | Spell | - | - |
| `Snowball` | 2 | Spell | - | - |
| `Tornado` | 3 | Spell | - | - |
| `Vines` | 3 | Spell | - | - |
| `WarmSpell` | 1 | Spell | - | - |
| `Zap` | 2 | Spell | - | - |

# RoyaleViser

[RoyaleViser](https://github.com/RoyaleGym/RoyaleViser) is the viewer: RoyaleGym's RLViser. It
opens a window that shows a battle tick by tick: the arena, every unit with its health, both
hands, both elixir bars and a log of what happened. Click a unit to see everything about it.

![A battle shown in the viewer](../media/whole-battle.gif){ width="100%" }

It opens three kinds of thing:

| Command | Opens |
|---|---|
| `royaleviser` | A training run, live. |
| `royaleviser my_bot_battle.msgpack` | A saved battle. |
| `royaleviser runs/` | The newest saved battle in a folder. |

It always runs as a separate program. Training never waits for it, and when nobody is
watching, streaming costs almost nothing.

How to stream a training run, save a battle, and the keys are on the
[Viewer](../clash-royale/configuration-objects/viewer.md) page. Every option is in the
[RoyaleViser guide](https://github.com/RoyaleGym/RoyaleViser/blob/main/docs/guide.md).

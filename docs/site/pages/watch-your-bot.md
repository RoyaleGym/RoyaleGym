# Watch Your Bot

The viewer is a window that plays a battle back, so you can see what your bot actually did.
It comes with `royalegym[all]`.

## A saved battle

The Quick Start saves one battle and tells you how to open it:

```
royaleviser my_bot_battle.msgpack
```

Give it a folder instead and it opens the newest saved battle anywhere inside it:

```
royaleviser runs/
```

## A bot while it trains

Start the viewer first, with nothing after it:

```
royaleviser
```

It waits. Then start your training with `viser=True` on the `Learner`, and the battle appears in
the window as it is played. Training never slows down for the viewer, and you can close the
window and open it again at any time.

## Keys

| Key | Does |
|---|---|
| Space | play or pause |
| Left / Right | one frame back or forward (hold Shift for 20) |
| `+` / `-` | faster or slower |
| `f` | swap which side is at the bottom |
| Click a unit | pin it, to see its health and target |
| `s` | save a picture |
| `h` | list every key |
| `q` | quit |

## No window opens

On a computer with no screen (a server, or over SSH) there is nothing to draw on. Save a
picture instead: see [Troubleshooting](troubleshooting.md).

Next: [Rewards](guides/rewards.md), to change what your bot wants.

# royaleviser

The viewer. It draws a battle in its own window, and it can also write frames to a file with no
window at all.

Everything on this page is generated from the docstrings in the code.

For what the viewer is for and how to open one, read [The viewer](../pieces/viewer.md).

## Opening a window

::: royaleviser.app
    options:
      members:
        - run
        - App

## Saving pictures with no window

`capture` is how the pictures in this project's READMEs are made. It needs no display and no
clock, and the same source and arguments give you the same bytes every time.

::: royaleviser.capture

## Where frames come from

A recording, a trace saved from the engine, or a live stream from an environment stepping in
another process.

::: royaleviser.sources

## The frame model

One `Frame` is everything the viewer knows about one tick. Any other front end can draw from it.

::: royaleviser.model

## Drawing

::: royaleviser.render

::: royaleviser.theme

---
name: Bug report
about: Something does not work as the docs say
labels: bug
---

Seeing an error message? [It Doesn't Work](https://royalegym.github.io/RoyaleGym/it-doesnt-work/)
lists the common ones and what to do about each.

**Versions**
Run this and paste the one line it prints (your OS, Python and every Royale package):

```
python -c "import importlib.metadata as m, platform; print(platform.platform(), 'Python', platform.python_version(), *sorted(d.metadata['Name'] + ' ' + d.version for d in m.distributions() if d.metadata['Name'].lower().startswith('royale')))"
```

**The failing code**
The smallest program or command that shows the problem, as you ran it.

```python

```

**What happened**
The full error or output, as text rather than a screenshot.

**What you expected**

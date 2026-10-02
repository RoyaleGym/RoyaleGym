// The repository box in the header shows the stars and forks of all the public repositories in
// the RoyaleGym organisation added together, not only RoyaleGym's own. Material fills the box with
// RoyaleGym's numbers (and its latest release); this fetches the organisation's repositories once
// an hour per browser session and writes the totals over the stars and forks.
(function () {
  "use strict";

  var ORG = "RoyaleGym";
  var KEY = "royalegym.org-facts";
  var MAX_AGE_MS = 60 * 60 * 1000;
  var totals = null;

  // Material's own short form (its bundle's number formatter): 999, 1k, 1.2k, 12.3k.
  function short(n) {
    if (n > 999) {
      var digits = +((n - 950) % 1000 > 99);
      return ((n + 0.000001) / 1000).toFixed(digits) + "k";
    }
    return String(n);
  }

  function cached() {
    try {
      var c = JSON.parse(window.sessionStorage.getItem(KEY));
      if (c && Date.now() - c.at < MAX_AGE_MS) return c;
    } catch (e) {
      // No storage (private window, blocked site data): fetch instead.
    }
    return null;
  }

  function fetchTotals() {
    var url = "https://api.github.com/orgs/" + ORG + "/repos?type=public&per_page=100";
    return fetch(url, { headers: { Accept: "application/vnd.github+json" } })
      .then(function (r) {
        if (!r.ok) throw new Error("GitHub answered " + r.status);
        return r.json();
      })
      .then(function (repos) {
        var listed = repos.filter(function (r) {
          return !r.private;
        });
        var c = { at: Date.now(), repos: listed.length, stars: 0, forks: 0 };
        listed.forEach(function (r) {
          c.stars += r.stargazers_count || 0;
          c.forks += r.forks_count || 0;
        });
        try {
          window.sessionStorage.setItem(KEY, JSON.stringify(c));
        } catch (e) {
          // Not cached; the next page fetches again.
        }
        return c;
      });
  }

  function apply() {
    if (!totals) return;
    var pairs = [
      [".md-source__fact--stars", short(totals.stars)],
      [".md-source__fact--forks", short(totals.forks)],
    ];
    pairs.forEach(function (pair) {
      document.querySelectorAll(pair[0]).forEach(function (el) {
        if (el.textContent !== pair[1]) el.textContent = pair[1];
      });
    });
    var title = "Stars and forks of all " + totals.repos + " public RoyaleGym repositories";
    document.querySelectorAll(".md-source").forEach(function (a) {
      if (a.title !== title) a.title = title;
    });
  }

  // Material draws the facts either before this script runs (from its own sessionStorage cache)
  // or later, when its GitHub request returns. The apply() calls below cover the first case and
  // this observer the second. Material doesn't redraw them on instant navigation.
  new MutationObserver(apply).observe(document.body, { childList: true, subtree: true });

  var c = cached();
  if (c) {
    totals = c;
    apply();
  } else {
    fetchTotals()
      .then(function (fresh) {
        totals = fresh;
        apply();
      })
      .catch(function () {
        // Offline or rate-limited: Material's own numbers for RoyaleGym stay.
      });
  }
})();

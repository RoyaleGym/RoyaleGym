// How many people are online in the project Discord right now, from the server's widget (owner, 2026-10-03: "make use
// of the servers widget"). It fills every element with a data-discord-online attribute (its value, with {n} for the
// count; the element stays hidden until the count arrives) and the header's Discord icon's tooltip. One fetch every
// ten minutes per browser session; if Discord can't be reached, nothing shows.
(function () {
  "use strict";

  var WIDGET = "https://discord.com/api/guilds/1551699576304705647/widget.json";
  var KEY = "royalegym.discord-online";
  var MAX_AGE_MS = 10 * 60 * 1000;
  var online = null;

  function cached() {
    try {
      var c = JSON.parse(window.sessionStorage.getItem(KEY));
      if (c && Date.now() - c.at < MAX_AGE_MS) return c.online;
    } catch (e) {
      // No storage (private window, blocked site data): fetch instead.
    }
    return null;
  }

  function apply() {
    if (online === null) return;
    document.querySelectorAll("[data-discord-online]").forEach(function (el) {
      var text = el.getAttribute("data-discord-online").replace("{n}", online);
      if (el.textContent !== text) el.textContent = text;
      if (el.hidden) el.hidden = false;
    });
    var title = "The project Discord · " + online + " online now";
    document.querySelectorAll('a.rg-social__link[href*="discord"]').forEach(function (a) {
      if (a.title !== title) a.title = title;
    });
  }

  // Instant navigation swaps the page without reloading this script, so new elements are filled as they appear.
  new MutationObserver(apply).observe(document.body, { childList: true, subtree: true });

  online = cached();
  if (online !== null) {
    apply();
  } else {
    fetch(WIDGET)
      .then(function (r) {
        if (!r.ok) throw new Error("Discord answered " + r.status);
        return r.json();
      })
      .then(function (d) {
        if (typeof d.presence_count !== "number") return;
        online = d.presence_count;
        try {
          window.sessionStorage.setItem(KEY, JSON.stringify({ at: Date.now(), online: online }));
        } catch (e) {
          // Not cached; the next page fetches again.
        }
        apply();
      })
      .catch(function () {
        // Offline, or the widget is switched off: the counts stay hidden.
      });
  }
})();

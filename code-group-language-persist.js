/**
 * code-group-language-persist.js
 *
 * Remembers the language a reader picks in a code example and re-applies it on
 * every page, for BOTH of Mintlify's tab systems:
 *
 *   1. <CodeGroup> (code-only tabs): Mintlify already persists these in
 *      localStorage["code"], but CASE-SENSITIVELY by exact label.
 *   2. <Tabs>/<Tab title="..."> (content tabs, e.g. Python vs TypeScript on
 *      /weave/quickstart): these have NO cross-page persistence in Mintlify —
 *      they reset to the first tab on every page. The URL #slug hash only
 *      deep-links within a single page and is dropped on navigation.
 *
 * Content tabs are the dominant language-switch pattern in these docs (hundreds
 * of <Tab title="Python">/<Tab title="TypeScript">), so without this the reader
 * has to re-pick their language on every page. This script gives both systems
 * unified, case-insensitive, cross-page persistence.
 *
 * How it works:
 * - Remember: a capture-phase pointerdown/click listener stores the chosen
 *   tab's label in localStorage when the label is an SDK language (see
 *   LANGUAGES — only Python/TypeScript). The whitelist stops both descriptive
 *   tabs ("Web App", "Summary metrics") and incidental code tabs ("Bash",
 *   "cURL") from clobbering the saved language.
 * - Restore: on load, after hydration, and on SPA route changes (detected by
 *   a MutationObserver watching location.pathname), any tab whose label
 *   matches the saved language (compared case-insensitively) is selected.
 *   Tabs already active are skipped, so this coexists with Mintlify's own
 *   <CodeGroup> restore without fighting it.
 * - Yield: when the reader manually picks a non-language tab ("HTTP API",
 *   "Command Line", …), its group is marked overridden and restore leaves it
 *   alone for the rest of that page view. Restore also only re-runs when the
 *   pathname actually changes, never on arbitrary DOM mutations — otherwise
 *   the mutation caused by the reader's own tab switch would re-trigger a
 *   restore that instantly snaps mixed groups (e.g. Python / TypeScript /
 *   HTTP API) back to the saved language, making other tabs unselectable.
 *
 * Selection uses a real pointer sequence, not element.click(): Mintlify's tabs
 * activate on pointer/mouse-down, and a synthetic click() does not switch them.
 *
 * Depends only on the ARIA tab contract Mintlify renders (role="tab", with
 * data-state="active" for code groups or aria-selected="true" for content
 * tabs), so it is resilient to theme class-name churn.
 *
 * Environment:
 * - Loaded globally by Mintlify for every docs page (any .js in the content
 *   root is injected — see .mintignore). Wrapped in an IIFE.
 */

(function () {
  var STORAGE_KEY = 'wandb.docs.codeLanguage';

  /**
   * The SDK languages whose choice should follow the reader from page to page,
   * compared lower-cased. Deliberately limited to Python and TypeScript — the
   * two W&B SDK languages and the only genuine cross-page preference.
   *
   * Incidental code tabs (Bash, shell, cURL, YAML, JSON, …) are intentionally
   * NOT listed: persisting them would let a click on, say, a Bash tab overwrite
   * the saved Python/TypeScript choice, and a later page whose tabs are only
   * Python/TypeScript has no Bash match — so it would silently fall back to its
   * first tab, defeating the persistence. Restore is case-insensitive, so the
   * canonical labels here cover "python"/"Python"/"TypeScript" alike; short
   * aliases (py/ts) are omitted on purpose — storing "ts" would not match a
   * "TypeScript" tab and would reintroduce that same fall-back bug.
   */
  var LANGUAGES = {
    python: 1, typescript: 1
  };

  /**
   * Tab groups the reader has manually steered to a non-language tab, which
   * restore() must then leave alone. Keyed by the group's DOM node, so entries
   * evaporate naturally when a client-side route change replaces the page
   * content (and with it, the nodes).
   */
  var overridden = typeof WeakSet === 'function' ? new WeakSet() : null;

  /** Read the saved language, tolerant of disabled/throwing localStorage. */
  function read() {
    try { return localStorage.getItem(STORAGE_KEY); } catch (e) { return null; }
  }

  /** Persist the chosen language, tolerant of private-mode write failures. */
  function write(value) {
    try { localStorage.setItem(STORAGE_KEY, value); } catch (e) { /* ignore */ }
  }

  /**
   * Visible label of a tab element.
   * @param {Element} tab
   * @returns {string}
   */
  function labelOf(tab) {
    return (tab.textContent || '').trim();
  }

  /**
   * Whether a label is one of the known languages we should remember.
   * @param {string} label
   * @returns {boolean}
   */
  function isLanguage(label) {
    return !!LANGUAGES[label.toLowerCase()];
  }

  /**
   * Whether a tab is the currently-selected one in its group. Code groups use
   * data-state="active"; content tabs use aria-selected="true".
   * @param {Element} tab
   * @returns {boolean}
   */
  function isActive(tab) {
    return tab.getAttribute('data-state') === 'active' ||
           tab.getAttribute('aria-selected') === 'true';
  }

  /**
   * The tab group a tab belongs to — its ARIA tablist, or failing that its
   * parent element. Used to scope manual-override tracking per group.
   * @param {Element} tab
   * @returns {Element|null}
   */
  function groupOf(tab) {
    return tab.closest('[role="tablist"]') || tab.parentElement;
  }

  /**
   * Selects a tab the way a real click does. Mintlify's tabs activate on
   * pointer/mouse-down, not on a synthetic element.click(), so dispatch a full
   * pointer sequence.
   * @param {Element} tab
   */
  function activate(tab) {
    var opts = { bubbles: true, cancelable: true, view: window, button: 0, isPrimary: true };
    if (window.PointerEvent) tab.dispatchEvent(new PointerEvent('pointerdown', opts));
    tab.dispatchEvent(new MouseEvent('mousedown', opts));
    if (window.PointerEvent) tab.dispatchEvent(new PointerEvent('pointerup', opts));
    tab.dispatchEvent(new MouseEvent('mouseup', opts));
    tab.dispatchEvent(new MouseEvent('click', opts));
  }

  /**
   * Capture-phase selection handler. Records the chosen tab's label when it
   * names a language, so the choice survives navigation; when it names
   * anything else ("HTTP API", "Command Line", …), marks the group as
   * reader-overridden so restore() stops re-asserting the saved language in
   * it. Bound to pointerdown (when tabs actually select) and click (covers
   * keyboard activation). Only trusted events count — the synthetic clicks
   * restore() dispatches must not register as reader choices.
   * @param {Event} e
   */
  function onSelect(e) {
    if (!e.isTrusted) return;
    var tab = e.target && e.target.closest ? e.target.closest('[role="tab"]') : null;
    if (!tab) return;
    var label = labelOf(tab);
    if (!label) return;
    var group = overridden && groupOf(tab);
    if (isLanguage(label)) {
      write(label);
      if (group) overridden.delete(group);
    } else if (group) {
      overridden.add(group);
    }
  }

  /**
   * Selects the saved language in every group that isn't already on it,
   * except groups the reader has manually overridden this page view.
   * Self-terminating: tabs already active are skipped, so the synthetic clicks
   * we trigger here do not cause an observe/click feedback loop, and we never
   * fight Mintlify's own already-applied <CodeGroup> selection.
   */
  function restore() {
    var want = read();
    if (!want) return;
    want = want.toLowerCase();
    var tabs = document.querySelectorAll('[role="tab"]');
    for (var i = 0; i < tabs.length; i++) {
      var tab = tabs[i];
      if (labelOf(tab).toLowerCase() !== want || isActive(tab)) continue;
      if (overridden) {
        var group = groupOf(tab);
        if (group && overridden.has(group)) continue;
      }
      activate(tab);
    }
  }

  // Debounce restore() into a single run after a burst of DOM mutations.
  var pending = false;
  function schedule() {
    if (pending) return;
    pending = true;
    setTimeout(function () { pending = false; restore(); }, 80);
  }

  /** Restore now-ish, then again while late-hydrating content settles. */
  function restoreSoon() {
    schedule();
    [300, 800, 1500].forEach(function (ms) { setTimeout(restore, ms); });
  }

  /**
   * Mutation handler: re-applies the saved language only when the mutation
   * burst accompanies an actual client-side navigation (the pathname
   * changed). Restoring on every mutation would treat the reader's own tab
   * switch — itself a DOM mutation — as a cue to snap the group back to the
   * saved language.
   */
  var lastPathname = location.pathname;
  function onMutate() {
    if (location.pathname === lastPathname) return;
    lastPathname = location.pathname;
    restoreSoon();
  }

  function start() {
    // Capture phase so we observe the selection around Mintlify's own handling.
    document.addEventListener('pointerdown', onSelect, true);
    document.addEventListener('click', onSelect, true);

    // Initial load: restore immediately, then catch tabs that hydrate later.
    restore();
    restoreSoon();

    // Re-apply on client-side route changes.
    if (document.body) {
      new MutationObserver(onMutate).observe(document.body, {
        childList: true,
        subtree: true
      });
    }

    // Re-apply after back/forward (bfcache) restores.
    window.addEventListener('pageshow', restore);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();

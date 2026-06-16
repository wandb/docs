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
 * - Restore: on load, after hydration, and on SPA route changes (a
 *   MutationObserver), any tab whose label matches the saved language (compared
 *   case-insensitively) is selected. Tabs already active are skipped, so this
 *   coexists with Mintlify's own <CodeGroup> restore without fighting it.
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
   * names a language, so the choice survives navigation. Bound to pointerdown
   * (when tabs actually select) and click (covers keyboard activation).
   * @param {Event} e
   */
  function onSelect(e) {
    var tab = e.target && e.target.closest ? e.target.closest('[role="tab"]') : null;
    if (!tab) return;
    var label = labelOf(tab);
    if (label && isLanguage(label)) write(label);
  }

  /**
   * Selects the saved language in every group that isn't already on it.
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
      if (labelOf(tab).toLowerCase() === want && !isActive(tab)) {
        activate(tab);
      }
    }
  }

  // Debounce restore() into a single run after a burst of DOM mutations.
  var pending = false;
  function schedule() {
    if (pending) return;
    pending = true;
    setTimeout(function () { pending = false; restore(); }, 80);
  }

  function start() {
    // Capture phase so we observe the selection around Mintlify's own handling.
    document.addEventListener('pointerdown', onSelect, true);
    document.addEventListener('click', onSelect, true);

    restore();
    // Catch tabs that hydrate after first paint.
    [300, 800, 1500].forEach(function (ms) { setTimeout(restore, ms); });

    // Re-apply whenever new content renders (client-side route changes).
    if (document.body) {
      new MutationObserver(schedule).observe(document.body, {
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

/**
 * dropdown-tabs-navigate.js
 *
 * Makes clicking a dropdown-style top-nav tab navigate to the first item in
 * its dropdown, instead of only opening the dropdown.
 *
 * Why this exists:
 * - Mintlify renders top-nav tabs that have a `menu` in docs.json (e.g.
 *   "Products", "Support") as Radix UI dropdown menus. Clicking such a tab
 *   opens the dropdown but performs no navigation — users expect a click to
 *   take them somewhere, and a hover or keyboard interaction to expose items.
 * - This script intercepts clicks on those dropdown tabs and navigates to
 *   the href of the first link in the dropdown.
 *
 * How it works:
 * - A MutationObserver watches the DOM for any Radix dropdown menu opening
 *   (via hover, click, or keyboard). When one opens, it reads the first
 *   anchor inside and caches that href keyed by the trigger button's label
 *   (e.g. "Products" -> "/"). The cache populates naturally as the user
 *   hovers over tabs, and survives Radix re-mounts because the key is the
 *   label text, not the generated Radix id.
 * - A capture-phase click listener intercepts clicks on dropdown tab
 *   buttons. If a href is cached for that tab, the click is cancelled and
 *   the browser navigates to the cached href. If nothing is cached yet
 *   (e.g. first touch interaction), the click is allowed through so Radix
 *   opens the menu, and a short polling loop waits for the menu to render,
 *   captures the first href, and navigates.
 *
 * Scope:
 * - Only targets `button.nav-tabs-item[aria-haspopup="menu"]` — the Radix
 *   trigger Mintlify uses for dropdown tabs. Non-dropdown tabs (plain
 *   anchors) are untouched.
 *
 * Environment:
 * - Loaded globally by Mintlify for every docs page (any .js in the content
 *   dir is included).
 * - Wrapped in an IIFE to avoid polluting the global scope.
 */

(function () {
  /** Map from trigger label (e.g. "Products") to the first dropdown href. */
  var firstHrefByLabel = Object.create(null);

  /**
   * Whether an element is a Mintlify dropdown-tab trigger button.
   * @param {Element | null} el
   * @returns {boolean}
   */
  function isDropdownTab(el) {
    if (!el || el.tagName !== 'BUTTON') return false;
    if (!el.classList || !el.classList.contains('nav-tabs-item')) return false;
    return el.getAttribute('aria-haspopup') === 'menu';
  }

  /**
   * Stable key for a dropdown tab trigger. Radix ids are regenerated on
   * re-mount, so we key by the visible label text instead.
   * @param {HTMLElement} btn
   * @returns {string}
   */
  function keyFor(btn) {
    return (btn.textContent || '').trim();
  }

  /**
   * Returns the href of the first anchor inside an open Radix menu, or
   * null if the menu has no anchors yet.
   * @param {Element} menu
   * @returns {string | null}
   */
  function firstHrefIn(menu) {
    var a = menu.querySelector('a[href]');
    if (!a) return null;
    var href = a.getAttribute('href');
    return href && href.trim() ? href : null;
  }

  /**
   * Finds any open dropdown menus and caches their first href keyed by
   * the trigger label. Safe to call repeatedly.
   */
  function cacheOpenMenus() {
    var menus = document.querySelectorAll('[role="menu"][data-state="open"]');
    for (var i = 0; i < menus.length; i++) {
      var menu = menus[i];
      var labelledBy = menu.getAttribute('aria-labelledby');
      if (!labelledBy) continue;
      var btn = document.getElementById(labelledBy);
      if (!isDropdownTab(btn)) continue;
      var href = firstHrefIn(menu);
      if (href) firstHrefByLabel[keyFor(btn)] = href;
    }
  }

  /**
   * Returns the currently-open dropdown menu triggered by the given button,
   * or null if the menu is closed / not yet rendered.
   * @param {HTMLElement} btn
   * @returns {Element | null}
   */
  function openMenuFor(btn) {
    if (!btn.id) return null;
    // CSS.escape guards against odd characters in Radix-generated ids.
    var safeId = (typeof CSS !== 'undefined' && CSS.escape) ? CSS.escape(btn.id) : btn.id;
    return document.querySelector('[role="menu"][aria-labelledby="' + safeId + '"][data-state="open"]');
  }

  /**
   * Click interceptor. Runs in capture phase so it can preempt Radix's own
   * click handler when we already know where to go.
   * @param {MouseEvent} e
   */
  function onClick(e) {
    // Only act on primary-button clicks without modifier keys — let users
    // still open the menu with middle/right click or keyboard if they want.
    if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;

    var target = e.target;
    var btn = target && target.closest ? target.closest('button.nav-tabs-item') : null;
    if (!isDropdownTab(btn)) return;

    var href = firstHrefByLabel[keyFor(btn)];
    if (href) {
      e.preventDefault();
      e.stopPropagation();
      e.stopImmediatePropagation();
      window.location.href = href;
      return;
    }

    // Not cached yet — let Radix open the menu, then navigate as soon as
    // the first link is in the DOM. This is the fallback for touch users
    // who click before any hover has pre-populated the cache.
    var start = Date.now();
    function tick() {
      var menu = openMenuFor(btn);
      if (menu) {
        var h = firstHrefIn(menu);
        if (h) {
          firstHrefByLabel[keyFor(btn)] = h;
          window.location.href = h;
          return;
        }
      }
      if (Date.now() - start < 1000) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  /**
   * MutationObserver callback — whenever the DOM changes, re-check for open
   * menus and refresh the cache. Debounced into a microtask.
   */
  var pending = false;
  function scheduleCache() {
    if (pending) return;
    pending = true;
    Promise.resolve().then(function () {
      pending = false;
      cacheOpenMenus();
    });
  }

  function startObserving() {
    if (!document.body) return;
    var observer = new MutationObserver(scheduleCache);
    observer.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['data-state']
    });
    // Capture phase so we run before Radix's own click handler.
    document.addEventListener('click', onClick, true);
    // Initial sweep in case a menu is already open at install time.
    cacheOpenMenus();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', startObserving);
  } else {
    startObserving();
  }
})();

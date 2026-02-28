/**
 * local-links-same-tab.js
 *
 * Ensures that in-doc navigation links (e.g. sidebar "group flex" links) open in
 * the same tab when they point to local paths. Links that would otherwise have
 * target="_blank" are updated by removing the target attribute so the browser
 * uses the default same-tab behavior.
 *
 * Also swaps the external-link arrow icon (top-right) to a chevron (greater-than)
 * on local nav links, so users see expansion/navigation vs. leaving to an external URL.
 *
 * Why this exists:
 * - Some UI frameworks or themes add target="_blank" to all nav links.
 * - For internal docs paths (e.g. /products/sunk/index), opening in a new tab
 *   is usually undesirable; users expect in-site navigation to stay in the same tab.
 * - The theme may show an external-link arrow on all nav items; local links should
 *   show a chevron instead to signal in-site navigation.
 * - This script only changes links that match the "group flex" nav pattern and
 *   have a local href; external (http/https) links are left unchanged.
 *
 * How it works:
 * - Runs on DOMContentLoaded and at several delayed intervals (for slow-hydrating nav).
 * - Uses a MutationObserver to run whenever the DOM gains new nodes or a link's
 *   target attribute changes, so links are fixed as soon as the framework
 *   renders or re-applies target="_blank".
 * - Runs on pageshow so that when the user navigates back, links are fixed again.
 * - Selects <a> elements whose class list contains both "group" and "flex"
 *   (the Mintlify sidebar nav link pattern).
 * - For each such link with a local href: removes the target attribute and
 *   replaces the arrow SVG with a chevron SVG.
 *
 * Environment:
 * - Loaded globally by Mintlify for every docs page (any .js in the content dir).
 * - Wrapped in an IIFE to avoid polluting the global scope.
 */

(function () {
  /**
   * Determines whether an href should be treated as local (same-origin / in-doc).
   * Local links should open in the same tab; external links may keep target="_blank".
   *
   * @param {string | null | undefined} href - The href value from the anchor (e.g. getAttribute('href')).
   * @returns {boolean} - True if the link is local and safe to open in the same tab.
   */
  function isLocalHref(href) {
    // Missing or empty href: treat as local (e.g. placeholder or JS-handled link).
    if (!href || typeof href !== 'string') return true;

    var trimmed = href.trim();
    // Fragment-only links (e.g. #section) are in-page and always local.
    if (trimmed === '' || trimmed.startsWith('#')) return true;

    // Compare case-insensitively so HTTP: and https: are both treated as external.
    var lower = trimmed.toLowerCase();
    // Local: relative paths (/foo), path-only, or protocol-relative that we treat as same-site.
    // External: explicit http: or https: (and we do not change those).
    return !lower.startsWith('http:') && !lower.startsWith('https:');
  }

  /** Chevron (greater-than) path and dimensions matching expandable section icons. */
  var CHEVRON_PATH = 'M0 0L3 3L0 6';

  /**
   * Replaces the arrow (external-link) icon with a chevron on a nav link.
   * Matches the expandable section chevron: width="8" height="24" viewBox="0 -9 3 24".
   *
   * @param {HTMLAnchorElement} a - The nav link element.
   */
  function swapArrowToChevron(a) {
    var svg = a.querySelector('svg[viewBox="0 0 384 512"]');
    if (!svg) return;

    var path = svg.querySelector('path');
    if (!path) return;

    var d = path.getAttribute('d') || '';
    if (d.indexOf('M328 96') !== 0) return;
    if (d === CHEVRON_PATH) return;

    svg.setAttribute('viewBox', '0 -9 3 24');
    svg.setAttribute('width', '8');
    svg.setAttribute('height', '24');
    svg.setAttribute('class', 'transition-transform text-gray-400 overflow-visible group-hover:text-gray-600 dark:text-gray-600 dark:group-hover:text-gray-400 w-2 h-5 -mr-0.5 flex-shrink-0');
    path.setAttribute('d', CHEVRON_PATH);
    path.setAttribute('fill', 'none');
    path.setAttribute('stroke', 'currentColor');
    path.setAttribute('stroke-width', '1.5');
    path.setAttribute('stroke-linecap', 'round');
  }

  /**
   * Finds all "group flex" nav links with a local href, removes target="_blank"
   * so they open in the same tab, and swaps the arrow icon to a chevron.
   * Marks all processed links so hidden arrows on external links can be revealed.
   *
   * Safe to call multiple times; idempotent for already-processed links.
   */
  function stripTargetBlankFromLocalGroupFlexLinks() {
    var links = document.querySelectorAll('a[href]');

    for (var i = 0; i < links.length; i++) {
      var a = links[i];
      var cls = a.className;

      if (typeof cls !== 'string') continue;
      if (cls.indexOf('group') === -1 || cls.indexOf('flex') === -1) continue;

      var href = a.getAttribute('href');
      if (isLocalHref(href)) {
        a.removeAttribute('target');
        swapArrowToChevron(a);
      }
      a.setAttribute('data-nav-processed', '1');
    }
  }

  // Debounced run: schedule a single stripper run after mutations stop.
  var scheduleId = null;
  var debounceMs = 120;
  function scheduleStrip() {
    if (scheduleId) clearTimeout(scheduleId);
    scheduleId = setTimeout(function () {
      scheduleId = null;
      stripTargetBlankFromLocalGroupFlexLinks();
    }, debounceMs);
  }

  // Run as soon as the DOM is ready.
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
      stripTargetBlankFromLocalGroupFlexLinks();
    });
  } else {
    stripTargetBlankFromLocalGroupFlexLinks();
  }

  // Delayed runs to catch nav that appears only after framework hydration.
  [300, 800, 1500, 3000].forEach(function (ms) {
    setTimeout(stripTargetBlankFromLocalGroupFlexLinks, ms);
  });

  // Whenever the framework adds nodes or sets target on a link, run the stripper
  // (debounced) so we fix links as soon as they appear or get target="_blank".
  function startObserving() {
    if (!document.body) return;
    var observer = new MutationObserver(function () {
      scheduleStrip();
    });
    observer.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['target']
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', startObserving);
  } else {
    startObserving();
  }

  // When the user hits "back", re-run so links are fixed after bfcache or re-render.
  window.addEventListener('pageshow', function () {
    stripTargetBlankFromLocalGroupFlexLinks();
  });
})();

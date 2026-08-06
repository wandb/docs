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

  /** Chevron path matching Mintlify's sidebar group chevron (ChevronRightIcon, 18x18). */
  var CHEVRON_PATH = 'M6.5 2.75L12.75 9L6.5 15.25';

  /** Classes matching Mintlify's own sidebar group chevron styling. */
  var CHEVRON_CLASS = 'size-3 text-gray-400 group-hover:text-gray-600 dark:text-gray-600 dark:group-hover:text-gray-400 shrink-0';

  /**
   * Replaces the arrow (external-link) icon with a chevron on a sidebar nav link.
   * Matches the chevron Mintlify renders on expandable sidebar groups.
   *
   * @param {HTMLAnchorElement} a - The nav link element.
   */
  function swapArrowToChevron(a) {
    // Only sidebar entries get the swap. Heading anchors and cards also pass
    // the group/flex class check but must keep their own icons.
    if (!a.closest || !a.closest('#navigation-items')) return;

    // Every icon in Mintlify's current set renders with viewBox="0 0 18 18".
    // The trailing external-link arrow is the last such svg in the row (a
    // leading item icon, if any, comes first), so match by position rather
    // than the arrow's path geometry - the geometry has changed upstream
    // before and silently broke this swap (DOCS-3053).
    var svgs = a.querySelectorAll('svg[viewBox="0 0 18 18"]');
    if (!svgs.length) return;

    var svg = svgs[svgs.length - 1];
    if (svg.getAttribute('data-chevron') === '1') return;

    svg.setAttribute('class', CHEVRON_CLASS);
    svg.setAttribute('fill', 'none');
    svg.setAttribute('stroke-width', '2');
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    var path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', CHEVRON_PATH);
    path.setAttribute('stroke', 'currentColor');
    path.setAttribute('stroke-width', '2');
    path.setAttribute('stroke-linecap', 'round');
    path.setAttribute('stroke-linejoin', 'round');
    svg.appendChild(path);
    svg.setAttribute('data-chevron', '1');
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

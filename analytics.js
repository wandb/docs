// UserGuiding
const script1 = document.createElement('script');
script1.src = "https://ug-webapp-public-production.s3.amazonaws.com/api/js/wv-ug-ZCKXTHyLNZTV90gD.js";
script1.async = true;
document.head.appendChild(script1);

// Cookie consent, analytics, and GDPR notices run only on the production domain.
// Localhost and Mintlify preview URLs are excluded.
var isProductionDomain = window.location.hostname === 'docs.wandb.ai';

if (isProductionDomain) {
  var script2 = document.createElement('script');
  script2.src = "https://cdn.cookielaw.org/consent/29d3f242-6917-42f4-a828-bac6fba2e677/otSDKStub.js";
  script2.type = "text/javascript";
  script2.charset = "UTF-8";
  script2.setAttribute("data-domain-script", "29d3f242-6917-42f4-a828-bac6fba2e677");
  document.head.appendChild(script2);
}

function wpConsentSync() {
  if (typeof OnetrustActiveGroups === 'undefined') return;

  var activeGroups = OnetrustActiveGroups || '';
  var hasGroup = function(id) { return activeGroups.includes(',' + id + ','); };
  var isOptIn = activeGroups === ',C0001,';

  window.wp_consent_type = isOptIn ? 'optin' : 'optout';
  document.dispatchEvent(new CustomEvent('wp_consent_type_defined'));

  var consentMap = {
    'statistics': hasGroup('C0002'),
    'statistics-anonymous': hasGroup('C0002'),
    'marketing': hasGroup('C0004') || hasGroup('C0005'),
    'functional': hasGroup('C0001') || hasGroup('C0003'),
    'preferences': false
  };

  if (typeof window.wp_set_consent === 'function') {
    for (var category in consentMap) {
      if (consentMap.hasOwnProperty(category)) {
        window.wp_set_consent(category, consentMap[category] ? 'allow' : 'deny');
      }
    }
  }

  if (consentMap.marketing || consentMap.statistics) {
    loadSegment();
    loadClarity();
  }
}

function loadClarity() {
  (function(c,l,a,r,i,t,y){
    c[a]=c[a]||function(){(c[a].q=c[a].q||[]).push(arguments)};
    t=l.createElement(r);t.async=1;t.src="https://www.clarity.ms/tag/"+i;
    y=l.getElementsByTagName(r)[0];y.parentNode.insertBefore(t,y);
  })(window, document, "clarity", "script", "a6rlxhvc58");
}

function loadSegment() {
  !function(){
    var analytics = window.analytics = window.analytics || [];
    if (!analytics.initialize)
      if (analytics.invoked)
        window.console && console.error && console.error("Segment snippet included twice.");
      else {
        analytics.invoked = !0;
        analytics.methods = [
          "trackSubmit", "trackClick", "trackLink", "trackForm", "pageview", "identify",
          "reset", "group", "track", "ready", "alias", "debug", "page", "once", "off", "on",
          "addSourceMiddleware", "addIntegrationMiddleware", "setAnonymousId", "addDestinationMiddleware"
        ];
        analytics.factory = function(method){
          return function(){
            if (window.analytics.initialized) return window.analytics[method].apply(window.analytics, arguments);
            var args = Array.prototype.slice.call(arguments);
            args.unshift(method);
            analytics.push(args);
            return analytics;
          }
        };
        for (var i = 0; i < analytics.methods.length; i++) {
          var key = analytics.methods[i];
          analytics[key] = analytics.factory(key);
        }
        analytics.load = function(writeKey, options){
          var script = document.createElement("script");
          script.type = "text/javascript";
          script.async = true;
          script.src = "https://wandb.ai/sa-docs.min.js";
          var first = document.getElementsByTagName("script")[0];
          first.parentNode.insertBefore(script, first);
          analytics._loadOptions = options;
        };
        analytics._writeKey = "NYcqWZ8sgOCplYnItFyBaZ5ZRClWlVgl";
        analytics.SNIPPET_VERSION = "4.16.1";
        analytics.load("NYcqWZ8sgOCplYnItFyBaZ5ZRClWlVgl");

        analytics.ready(function() {
          analytics.page();
        });
      }
  }();
}

// OneTrust callback, only relevant on production where the SDK is loaded
function OptanonWrapper() {
  wpConsentSync();
  if (typeof Optanon !== 'undefined' && typeof Optanon.ShowBanner === 'function') {
    Optanon.ShowBanner();
  }
}

if (isProductionDomain) {
  (function() {
    var btn = document.createElement('button');
    btn.id = 'ot-sdk-btn';
    btn.className = 'ot-sdk-show-settings';
    btn.textContent = 'Cookie settings';
    document.body.appendChild(btn);
  })();
}
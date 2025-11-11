// Load first script
const script1 = document.createElement('script');
script1.src = "https://ug-webapp-public-production.s3.amazonaws.com/api/js/wv-ug-ZCKXTHyLNZTV90gD.js";
script1.async = true;
document.head.appendChild(script1);

script1.onload = () => {
  //console.log('Loaded first script!');
};

// Load second script (with extra attributes)
const script2 = document.createElement('script');
script2.src = "https://cdn.cookielaw.org/consent/29d3f242-6917-42f4-a828-bac6fba2e677/otSDKStub.js";
script2.type = "text/javascript";
script2.setAttribute("data-domain-script", "29d3f242-6917-42f4-a828-bac6fba2e677");
document.head.appendChild(script2);

script2.onload = () => {
  //console.log('Loaded cookie law script!');
};


// load Marimo config script
const script4 = document.createElement('script');
script4.type = "text/x-marimo-snippets-config";
script4.textContent = `
configureMarimoButtons({title: "Open in a marimo notebook"});
configureMarimoIframes({height: "400px"});
`;
document.body.appendChild(script4);

// Load third script (Marimo)
const script3 = document.createElement('script');
script3.src = "https://cdn.jsdelivr.net/npm/@marimo-team/marimo-snippets@1";
script3.type = "text/javascript";
document.body.appendChild(script3);

script3.onload = () => {
  console.log('Loaded Marimo script!');
  
  // Wait a bit for the page to fully render, then convert and process
  setTimeout(() => {
    // Convert data-marimo divs to marimo-iframe elements
    const marimoWrappers = document.querySelectorAll('[data-marimo="iframe"]');
    marimoWrappers.forEach(wrapper => {
      // Create marimo-iframe element
      const marimoIframe = document.createElement('marimo-iframe');
      
      // Move all children from wrapper to marimo-iframe
      while (wrapper.firstChild) {
        marimoIframe.appendChild(wrapper.firstChild);
      }
      
      // Replace the wrapper with marimo-iframe
      wrapper.parentNode.replaceChild(marimoIframe, wrapper);
    });
    
    console.log(`Converted ${marimoWrappers.length} Marimo wrappers to iframes`);
    
    // Trigger DOMContentLoaded again to make marimo process the new elements
    // The marimo script listens for this event
    const event = new Event('DOMContentLoaded', {
      bubbles: true,
      cancelable: false
    });
    document.dispatchEvent(event);
  }, 100); // Small delay to ensure everything is rendered
};

// Sync consent categories
function wpConsentSync() {
  if (typeof OnetrustActiveGroups === 'undefined') return;

  const activeGroups = OnetrustActiveGroups || '';
  const hasGroup = (id) => activeGroups.includes(`,${id},`);
  const isOptIn = activeGroups === ',C0001,';

  window.wp_consent_type = isOptIn ? 'optin' : 'optout';
  document.dispatchEvent(new CustomEvent('wp_consent_type_defined'));

  const consentMap = {
    'statistics': hasGroup('C0002'),
    'statistics-anonymous': hasGroup('C0002'),
    'marketing': hasGroup('C0004') || hasGroup('C0005'),
    'functional': hasGroup('C0001') || hasGroup('C0003'),
    'preferences': false
  };

  if (typeof window.wp_set_consent === 'function') {
    for (const [category, allowed] of Object.entries(consentMap)) {
      window.wp_set_consent(category, allowed ? 'allow' : 'deny');
    }
  }

  // 🚀 Load analytics only if marketing or statistics consent is given
  if (consentMap.marketing || consentMap.statistics) {
    loadSegment();
    loadClarity();
  }
}

// Load Microsoft Clarity Analytics
function loadClarity() {
  (function(c,l,a,r,i,t,y){
      c[a]=c[a]||function(){(c[a].q=c[a].q||[]).push(arguments)};
      t=l.createElement(r);t.async=1;t.src="https://www.clarity.ms/tag/"+i;
      y=l.getElementsByTagName(r)[0];y.parentNode.insertBefore(t,y);
  })(window, document, "clarity", "script", "a6rlxhvc58");
}

// Load Segment Analytics
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
          script.src = "https://wandb.ai/sa-docs.min.js"; // <-- Your self-hosted Segment build
          var first = document.getElementsByTagName("script")[0];
          first.parentNode.insertBefore(script, first);
          analytics._loadOptions = options;
        };
        analytics._writeKey = "NYcqWZ8sgOCplYnItFyBaZ5ZRClWlVgl"; // <-- Your real write key
        analytics.SNIPPET_VERSION = "4.16.1";
        analytics.load("NYcqWZ8sgOCplYnItFyBaZ5ZRClWlVgl");

        // 🚀 Fire analytics.page() when ready
        analytics.ready(function() {
          analytics.page();
        });
      }
  }();
}

// Main hook called automatically by OneTrust
function OptanonWrapper() {
  wpConsentSync();
  if (typeof Optanon !== 'undefined' && typeof Optanon.ShowBanner === 'function') {
    Optanon.ShowBanner();
  }
}
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


// First, process any Marimo placeholders BEFORE loading the Marimo script
async function processMarimoPlaceholders() {
  // Handle file-based Marimo elements
  const fileBasedMarimos = document.querySelectorAll('[data-marimo-file]');
  const promises = [];
  
  fileBasedMarimos.forEach(wrapper => {
    const file = wrapper.getAttribute('data-marimo-file');
    if (!file) return;
    
    const promise = fetch(file)
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return response.text();
      })
      .then(content => {
        // Re-query the element to make sure it still exists
        const currentWrapper = document.querySelector(`[data-marimo-file="${file}"]`);
        
        if (!currentWrapper) {
          console.warn(`Wrapper for ${file} no longer exists in DOM, skipping`);
          return;
        }
        
        if (!currentWrapper.parentNode) {
          console.warn(`Wrapper for ${file} has no parent node, skipping`);
          return;
        }
        
        // Create marimo-iframe with the fetched content
        const marimoIframe = document.createElement('marimo-iframe');
        marimoIframe.textContent = content;
        
        // Replace the wrapper with marimo-iframe
        currentWrapper.parentNode.replaceChild(marimoIframe, currentWrapper);
        console.log(`Created marimo-iframe from ${file} with ${content.length} characters of content`);
      })
      .catch(err => {
        console.error(`Failed to load Marimo content from ${file}:`, err);
        if (wrapper && wrapper.parentNode) {
          wrapper.innerHTML = `<div style="color: red;">Failed to load notebook from ${file}: ${err.message}</div>`;
        }
      });
    
    promises.push(promise);
  });
  
  // Wait for all file-based Marimos to load
  await Promise.all(promises);
  console.log(`Processed ${fileBasedMarimos.length} marimo-file elements`);
}

// Keep track of whether we've already initialized
let marimoInitialized = false;

// Wait for DOM to be ready, then process everything
function initializeMarimo() {
  if (marimoInitialized) {
    console.log('Marimo already initialized, skipping');
    return;
  }
  
  // Wait a bit for React components to render
  setTimeout(() => {
    const fileBasedMarimos = document.querySelectorAll('[data-marimo-file]');
    console.log(`Found ${fileBasedMarimos.length} data-marimo-file elements`);
    
    if (fileBasedMarimos.length === 0) {
      console.log('No Marimo file elements found. They may not have rendered yet.');
      // Try again in a bit (max 10 attempts)
      if (!marimoInitialized) {
        setTimeout(initializeMarimo, 500);
      }
      return;
    }
    
    marimoInitialized = true;
    processMarimoPlaceholders().then(() => {
      // load Marimo config script
      const script4 = document.createElement('script');
      script4.type = "text/x-marimo-snippets-config";
      script4.textContent = `
configureMarimoButtons({title: "Open in a marimo notebook"});
configureMarimoIframes({height: "400px"});
      `;
      document.body.appendChild(script4);
      
      // Check if Marimo script already exists
      if (document.querySelector('script[src*="marimo-snippets"]')) {
        console.log('Marimo script already loaded, triggering DOMContentLoaded');
        setTimeout(() => {
          document.dispatchEvent(new Event('DOMContentLoaded'));
        }, 100);
        return;
      }
      
      // NOW load the Marimo script - it will find our marimo-iframe elements
      const script3 = document.createElement('script');
      script3.src = "https://cdn.jsdelivr.net/npm/@marimo-team/marimo-snippets@1";
      script3.type = "text/javascript";
      document.body.appendChild(script3);
      
      script3.onload = () => {
        console.log('Loaded Marimo script!');
        
        // The Marimo script listens for DOMContentLoaded, but if that's already fired,
        // we need to manually trigger processing
        if (document.readyState === 'complete' || document.readyState === 'interactive') {
          // DOM is already loaded, manually trigger the event
          setTimeout(() => {
            console.log('Triggering DOMContentLoaded for Marimo...');
            document.dispatchEvent(new Event('DOMContentLoaded'));
            
            // Check if it worked
            setTimeout(() => {
              const frames = document.querySelectorAll('iframe[src*="marimo.app"]');
              if (frames.length > 0) {
                console.log(`Success! Marimo created ${frames.length} interactive iframe(s)`);
              } else {
                console.log('Marimo iframes not created yet. The script may process them on its own timing.');
              }
            }, 500);
          }, 100);
        }
      };
    });
  }, 500); // Wait for React to render
}

// Start the process when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    // DOM is ready but React might not have rendered yet
    setTimeout(initializeMarimo, 500);
  });
} else {
  // DOM is already loaded, but wait for React
  setTimeout(initializeMarimo, 500);
}

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
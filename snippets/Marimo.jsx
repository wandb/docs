/**
 * Marimo component - self-contained with all processing logic
 * Loads content from external file and creates interactive notebook
 * 
 * Usage:
 * <Marimo file="/marimo-examples/slider-example.txt" />
 */
export const Marimo = ({ file }) => {
  const containerRef = React.useRef(null);
  const processedRef = React.useRef(false);

  React.useEffect(() => {
    if (!file || typeof window === 'undefined' || processedRef.current) return;
    
    // Mark as processed to prevent double execution
    processedRef.current = true;

    // Initialize global state on window object
    if (typeof window !== 'undefined') {
      window.__marimoState = window.__marimoState || {
        scriptLoaded: false,
        scriptLoading: false
      };
    }

    // Function to load Marimo script if not already loaded
    const loadMarimoScript = () => {
      return new Promise((resolve) => {
        if (window.__marimoState.scriptLoaded) {
          resolve();
          return;
        }

        if (window.__marimoState.scriptLoading) {
          // Wait for the script to finish loading
          const checkInterval = setInterval(() => {
            if (window.__marimoState.scriptLoaded) {
              clearInterval(checkInterval);
              resolve();
            }
          }, 100);
          return;
        }

        window.__marimoState.scriptLoading = true;

        // Add Marimo config
        const configScript = document.createElement('script');
        configScript.type = "text/x-marimo-snippets-config";
        configScript.textContent = `
configureMarimoButtons({title: "Open in a marimo notebook"});
configureMarimoIframes({height: "400px"});
        `;
        document.body.appendChild(configScript);

        // Load Marimo script
        const script = document.createElement('script');
        script.src = "https://cdn.jsdelivr.net/npm/@marimo-team/marimo-snippets@1";
        script.type = "text/javascript";
        script.onload = () => {
          window.__marimoState.scriptLoaded = true;
          window.__marimoState.scriptLoading = false;
          console.log('Marimo script loaded successfully from CDN');
          
          // Check if the script actually loaded the functions we expect
          if (typeof configureMarimoButtons === 'function') {
            console.log('Marimo functions are available');
          } else {
            console.log('Warning: Marimo functions not found after script load');
          }
          
          // Trigger DOMContentLoaded for Marimo to process existing elements
          setTimeout(() => {
            console.log('Triggering initial DOMContentLoaded after script load');
            document.dispatchEvent(new Event('DOMContentLoaded'));
          }, 100);
          
          resolve();
        };
        script.onerror = () => {
          window.__marimoState.scriptLoading = false;
          console.error('Failed to load Marimo script');
          resolve(); // Resolve anyway to not block
        };
        document.body.appendChild(script);
      });
    };

    // Main processing function
    const processMarimo = async () => {
      try {
        // Fetch the content
        console.log(`Fetching Marimo content from: ${file}`);
        const response = await fetch(file);
        
        if (!response.ok) {
          throw new Error(`Failed to load: ${response.status} ${response.statusText}`);
        }
        
        const content = await response.text();
        console.log(`Fetched ${content.length} characters from ${file}`);
        
        // Create marimo-iframe element
        const marimoIframe = document.createElement('marimo-iframe');
        
        // Parse markdown and create HTML elements
        const codeBlockRegex = /```python\n([\s\S]*?)\n```/g;
        let match;
        let blockCount = 0;
        
        while ((match = codeBlockRegex.exec(content)) !== null) {
          const pre = document.createElement('pre');
          const code = document.createElement('code');
          code.className = 'language-python';
          code.textContent = match[1];
          pre.appendChild(code);
          marimoIframe.appendChild(pre);
          blockCount++;
        }
        
        console.log(`Created marimo-iframe with ${blockCount} code blocks`);
        
        // Load Marimo script first
        await loadMarimoScript();
        
        // Clear container and add marimo-iframe
        if (containerRef.current) {
          // Clear the loading message and remove loading styles
          containerRef.current.innerHTML = '';
          containerRef.current.style.padding = '';
          containerRef.current.style.background = '';
          containerRef.current.style.borderRadius = '';
          
          // Add the marimo-iframe
          containerRef.current.appendChild(marimoIframe);
          console.log('Added marimo-iframe to DOM, element exists:', !!document.querySelector('marimo-iframe'));
          
          // Trigger DOMContentLoaded for this new element
          setTimeout(() => {
            const marimoElements = document.querySelectorAll('marimo-iframe');
            console.log(`Found ${marimoElements.length} marimo-iframe elements before triggering DOMContentLoaded`);
            
            document.dispatchEvent(new Event('DOMContentLoaded'));
            
            // Check if successful
            setTimeout(() => {
              const frames = document.querySelectorAll('iframe[src*="marimo.app"]');
              if (frames.length > 0) {
                console.log(`Success! Marimo created ${frames.length} iframe(s)`);
              } else {
                console.log('No Marimo iframes created. Checking marimo-iframe elements still exist...');
                const stillExists = document.querySelectorAll('marimo-iframe');
                console.log(`${stillExists.length} marimo-iframe elements still in DOM`);
              }
            }, 1000);
          }, 200);
        }
      } catch (err) {
        console.error('Error processing Marimo:', err);
        if (containerRef.current) {
          containerRef.current.innerHTML = `
            <div style="padding: 20px; background: #fee; border: 1px solid #fcc; borderRadius: 8px; color: #c00">
              Failed to load Marimo notebook: ${err.message}
            </div>
          `;
        }
      }
    };

    // Start processing after a short delay to ensure component is mounted
    const timer = setTimeout(processMarimo, 100);
    
    return () => clearTimeout(timer);
  }, [file]);

  // Render container with initial loading message
  return (
    <div 
      ref={containerRef}
      className="marimo-wrapper"
      style={{ 
        padding: '20px', 
        background: '#f5f5f5', 
        borderRadius: '8px' 
      }}
    >
      Loading Marimo notebook...
    </div>
  );
};

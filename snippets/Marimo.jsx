/**
 * Marimo component - loads content from external file
 * This completely avoids Mintlify's markdown processing
 * 
 * Usage:
 * <Marimo file="/marimo-examples/slider-example.txt" />
 */
export const Marimo = ({ file, children }) => {
  // Generate unique ID for this instance
  const id = 'marimo-' + Date.now() + '-' + Math.floor(Math.random() * 1000);
  
  // Load content after mount using data attribute
  if (file) {
    return (
      <div 
        className="marimo-wrapper" 
        data-marimo-file={file}
        data-marimo-id={id}
        style={{ padding: '20px', background: '#f5f5f5', borderRadius: '8px' }}
      >
        Loading Marimo notebook...
      </div>
    );
  }
  
  // Otherwise render children normally (for backwards compatibility)
  return (
    <div className="marimo-wrapper" data-marimo="iframe">
      {children}
    </div>
  );
};

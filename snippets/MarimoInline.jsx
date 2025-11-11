/**
 * Marimo component with inline content
 * Avoids needing external file serving
 */
export const MarimoInline = ({ content, id = 'marimo-' + Date.now() }) => {
  // Store content in a data attribute for processing
  return (
    <div 
      className="marimo-wrapper" 
      data-marimo-inline="true"
      data-marimo-content={content}
      data-marimo-id={id}
      style={{ padding: '20px', background: '#f5f5f5', borderRadius: '8px' }}
    >
      Loading Marimo notebook...
    </div>
  );
};

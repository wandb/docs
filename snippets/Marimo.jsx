/**
 * Marimo component for embedding Marimo interactive code blocks
 * Wraps content in a marimo-iframe element within a div container
 */
export const Marimo = ({ children }) => {
  return (
    <div>
      <marimo-iframe>
        {children}
      </marimo-iframe>
    </div>
  );
};

/**
 * Marimo component - simplest possible implementation
 * Just wraps children in marimo-iframe element
 * https://docs.marimo.io/guides/publishing/from_code_snippets/
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

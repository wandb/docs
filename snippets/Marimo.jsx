/**
 * Marimo component for embedding Marimo interactive code blocks
 * Wraps content in a marimo-iframe element within a div container
 * and includes the Marimo snippets script
 */
export const Marimo = ({ children }) => {
  return (
    <>
      <div>
        <marimo-iframe>
          {children}
        </marimo-iframe>
      </div>
      <script src="https://cdn.jsdelivr.net/npm/@marimo-team/marimo-snippets@1"></script>
    </>
  );
};

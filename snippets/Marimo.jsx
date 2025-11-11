import React from 'react';

/**
 * Marimo component - uses data attributes to avoid custom element issues
 * The marimo script will process elements with data-marimo attribute
 */
export const Marimo = ({ children }) => {
  return (
    <div className="marimo-wrapper" data-marimo="iframe">
      {children}
    </div>
  );
};

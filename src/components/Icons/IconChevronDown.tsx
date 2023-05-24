import React from 'react';
export default function IconChevronDown({width = 20, height = 20, ...props}) {
  return (
    <svg
      viewBox="0 0 24 24"
      width={width}
      height={height}
      aria-hidden
      {...props}>
        <path fill-rule="evenodd" clip-rule="evenodd" d="M5.21967 8.96967C4.92678 9.26256 4.92678 9.73744 5.21967 10.0303L11.2197 16.0303C11.5126 16.3232 11.9874 16.3232 12.2803 16.0303L18.2803 10.0303C18.5732 9.73744 18.5732 9.26256 18.2803 8.96967C17.9874 8.67678 17.5126 8.67678 17.2197 8.96967L11.75 14.4393L6.28033 8.96967C5.98744 8.67678 5.51256 8.67678 5.21967 8.96967Z" fill="currentColor"/>
    </svg>
  );
}


import React from 'react';
import '@site/src/css/Card.css'; 

const Card = ({ href, className, children }) => {
  return (
    <a href={href} className={`card ${className}`} rel="noopener noreferrer">
      {children}
    </a>
  );
};

export default Card;

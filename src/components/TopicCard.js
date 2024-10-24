import React from 'react';

const TopicCard = ({ title, headingLevel = 'h2', docPages = [], backgroundImage }) => {
  const Heading = headingLevel;

  const cardStyle = {
    backgroundImage: `url(${backgroundImage})`,
    backgroundSize: 'cover',
    backgroundPosition: 'center',
    padding: '20px',
    borderRadius: '8px',
    color: 'white', // Adjust text color to ensure readability
    textShadow: '1px 1px 3px rgba(0,0,0,0.6)', // Add text shadow for better contrast
  };

  return (
    <div className="topic-card" style={cardStyle}>
      <Heading>{title}</Heading>
      {docPages && docPages.length > 0 && (
        <ul className="doc-pages-list">
          {docPages.map((page, index) => (
            <li key={index}>
              <a href={page.link} style={{ color: 'white' }}>{page.title}</a>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default TopicCard;
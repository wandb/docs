/**
 * ClickableCard component that mimics Hugo's card behavior
 * - Card background clicks navigate to the main href
 * - Internal links are independently clickable
 */
export const ClickableCard = ({ href, children, className = '' }) => {
  const handleCardClick = (e) => {
    // Check if the click target is an anchor or inside an anchor
    let target = e.target;
    while (target && target !== e.currentTarget) {
      if (target.tagName === 'A') {
        // Click was on a link, don't navigate
        return;
      }
      target = target.parentElement;
    }
    // Click was on the card background, navigate
    window.location.href = href;
  };

  return (
    <div
      className={`clickable-card ${className}`}
      onClick={handleCardClick}
      style={{ cursor: 'pointer' }}
    >
      {children}
    </div>
  );
};

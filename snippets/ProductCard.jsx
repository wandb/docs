/**
 * ProductCard component for the home page product grid.
 * - Icon at top-left of card, recolored to brand amber via CSS mask
 * - Card background clicks navigate to the main href
 * - Internal links are independently clickable
 *
 * iconSize defaults to 24; pass iconSize={xx} to override for a specific card.
 */
export const ProductCard = ({
  title,
  iconSrc,
  href,
  subtitle,
  children,
  iconSize = 24,
  className = '',
}) => {
  const handleCardClick = (e) => {
    // Walk up to the card root to see if the click landed on a link
    let target = e.target;
    while (target && target !== e.currentTarget) {
      if (target.tagName === 'A') return;
      target = target.parentElement;
    }
    if (href) {
      window.location.href = href;
    }
  };

  return (
    <div
      className={`product-card group flex flex-col rounded-lg p-6 transition-all ${className}`}
      onClick={handleCardClick}
      style={{ cursor: href ? 'pointer' : 'default' }}
    >
      {iconSrc && (
        <span
          className="product-card-icon"
          aria-hidden="true"
          style={{
            display: 'block',
            width: `${iconSize}px`,
            height: `${iconSize}px`,
            backgroundColor: '#D4870D',
            WebkitMaskImage: `url(${iconSrc})`,
            maskImage: `url(${iconSrc})`,
            WebkitMaskRepeat: 'no-repeat',
            maskRepeat: 'no-repeat',
            WebkitMaskSize: 'contain',
            maskSize: 'contain',
            WebkitMaskPosition: 'left center',
            maskPosition: 'left center',
            marginBottom: '8px',
            lineHeight: 1,
          }}
        />
      )}

      <h2 className="product-card-title mb-2" style={{ marginTop: 0 }}>
        {title}
      </h2>
      {subtitle && (
        <h3 className="text-base font-semibold mb-3 text-gray-700 dark:text-gray-300">
          {subtitle}
        </h3>
      )}
      <div className="product-card-body leading-relaxed">
        {children}
      </div>
    </div>
  );
};

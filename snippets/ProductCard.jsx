/**
<<<<<<< HEAD
 * ProductCard component that mimics Hugo's card styling and behavior
 * - Header row: icon (optional) on left with title and subtitle to its right
 * - Body content below spans the full card width
||||||| parent of cd5e6c7b6 (Front-end updates)
 * ProductCard component that mimics Hugo's card styling and behavior
 * - Two column layout: icon on left, content on right
=======
 * ProductCard component for the home page product grid.
 * - Icon at top-left of card, recolored to brand amber via CSS mask
>>>>>>> cd5e6c7b6 (Front-end updates)
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
<<<<<<< HEAD
||||||| parent of cd5e6c7b6 (Front-end updates)
    // Check if the click target is an anchor or inside an anchor
=======
    // Walk up to the card root to see if the click landed on a link
>>>>>>> cd5e6c7b6 (Front-end updates)
    let target = e.target;
    while (target && target !== e.currentTarget) {
<<<<<<< HEAD
      if (target.tagName === 'A') {
        return;
      }
||||||| parent of cd5e6c7b6 (Front-end updates)
      if (target.tagName === 'A') {
        // Click was on a link, don't navigate
        return;
      }
=======
      if (target.tagName === 'A') return;
>>>>>>> cd5e6c7b6 (Front-end updates)
      target = target.parentElement;
    }
    if (href) {
      window.location.href = href;
    }
  };

  return (
<<<<<<< HEAD
    <div
      className="group flex flex-col overflow-hidden rounded-lg border border-gray-200 dark:border-gray-800 hover:shadow-md transition-all px-4 sm:px-6 pt-0 pb-6"
||||||| parent of cd5e6c7b6 (Front-end updates)
    <div 
      className="group flex overflow-hidden rounded-lg border border-gray-200 dark:border-gray-800 hover:shadow-md transition-all p-6"
=======
    <div
      className={`product-card group flex flex-col rounded-lg p-6 transition-all ${className}`}
>>>>>>> cd5e6c7b6 (Front-end updates)
      onClick={handleCardClick}
      style={{ cursor: href ? 'pointer' : 'default' }}
    >
<<<<<<< HEAD
      {/* Header row: icon (optional) + title */}
      <div className="flex items-center mb-2">
        {iconSrc && (
          <div className="flex-shrink-0 mr-3">
            <img
              src={iconSrc}
              alt={title}
              className="h-10 w-auto"
            />
          </div>
        )}
        <h2
          className="text-xl font-normal flex-1"
          style={{ fontFamily: '"Source Serif 4", serif', marginTop: 0, marginBottom: 0 }}
        >
          {title}
        </h2>
      </div>

      {/* Subtitle spans full card width */}
      {subtitle && (
        <h3
          className="text-base font-semibold mb-3 text-gray-700 dark:text-gray-300"
          style={{ marginTop: '-12px' }}
        >
          {subtitle}
        </h3>
      )}

      {/* Body content spans full card width */}
      <div className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
        {children}
||||||| parent of cd5e6c7b6 (Front-end updates)
      {/* Icon column - fixed width */}
      {iconSrc && (
        <div className="flex-shrink-0 mr-4" style={{ marginTop: '-12px' }}>
          <img 
            src={iconSrc} 
            alt={title}
            width="60" 
            height="60"
            className="w-[60px] h-[60px]"
          />
        </div>
      )}
      
      {/* Content column - flexible width */}
      <div className="flex-1">
        <h2 className="text-xl font-normal mb-2" style={{ fontFamily: '"Source Serif 4", serif' }}>
          {title}
        </h2>
        {subtitle && (
          <h3 className="text-base font-semibold mb-3 text-gray-700 dark:text-gray-300">
            {subtitle}
          </h3>
        )}
        <div className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
          {children}
        </div>
=======
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
>>>>>>> cd5e6c7b6 (Front-end updates)
      </div>
    </div>
  );
};

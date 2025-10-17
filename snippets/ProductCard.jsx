/**
 * ProductCard component that mimics Hugo's card styling and behavior
 * - Two column layout: icon on left, content on right
 * - Card background clicks navigate to the main href
 * - Internal links are independently clickable
 */
export const ProductCard = ({ title, iconSrc, href, subtitle, children, className = '' }) => {
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
    if (href) {
      window.location.href = href;
    }
  };

  return (
    <div 
      className="group flex overflow-hidden rounded-lg border border-gray-200 dark:border-gray-800 hover:shadow-md transition-all p-6"
      onClick={handleCardClick}
      style={{ cursor: href ? 'pointer' : 'default' }}
    >
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
      </div>
    </div>
  );
};

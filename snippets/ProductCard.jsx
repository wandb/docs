/**
 * ProductCard component that mimics Hugo's card styling and behavior
 * - Header row: icon (optional) on left with title and subtitle to its right
 * - Body content below spans the full card width
 * - Card background clicks navigate to the main href
 * - Internal links are independently clickable
 */
export const ProductCard = ({ title, iconSrc, href, subtitle, children, className = '' }) => {
  const handleCardClick = (e) => {
    let target = e.target;
    while (target && target !== e.currentTarget) {
      if (target.tagName === 'A') {
        return;
      }
      target = target.parentElement;
    }
    if (href) {
      window.location.href = href;
    }
  };

  return (
    <div
      className="group flex flex-col overflow-hidden rounded-lg border border-gray-200 dark:border-gray-800 hover:shadow-md transition-all p-6"
      onClick={handleCardClick}
      style={{ cursor: href ? 'pointer' : 'default' }}
    >
      {/* Header row: icon (optional) + title/subtitle */}
      <div className="flex mb-3" style={{ marginTop: '-32px' }}>
        {iconSrc && (
          <div className="flex-shrink-0 mr-4">
            <img
              src={iconSrc}
              alt={title}
              width="60"
              height="60"
              className="w-[60px] h-[60px]"
            />
          </div>
        )}
        <div className="flex-1">
          <h2 className="text-xl font-normal mb-2" style={{ fontFamily: '"Source Serif 4", serif' }}>
            {title}
          </h2>
          {subtitle && (
            <h3 className="text-base font-semibold text-gray-700 dark:text-gray-300">
              {subtitle}
            </h3>
          )}
        </div>
      </div>

      {/* Body content spans full card width */}
      <div className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
        {children}
      </div>
    </div>
  );
};

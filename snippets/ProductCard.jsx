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
      className="group flex flex-col overflow-hidden rounded-lg border border-gray-200 dark:border-gray-800 hover:shadow-md transition-all px-6 pt-0 pb-6"
      onClick={handleCardClick}
      style={{ cursor: href ? 'pointer' : 'default' }}
    >
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
      </div>
    </div>
  );
};

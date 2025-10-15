export const DesktopWindow = ({ images, alt, title }) => {
  const [activeIndex, setActiveIndex] = React.useState(0);

  const imageArray = Array.isArray(images) ? images : [images];
  const showControls = imageArray.length > 1;

  const goToPrevious = () => {
    setActiveIndex((prev) => (prev === 0 ? imageArray.length - 1 : prev - 1));
  };

  const goToNext = () => {
    setActiveIndex((prev) => (prev === imageArray.length - 1 ? 0 : prev + 1));
  };

  return (
    <>
      <style>{`
        .desktop-window-container img {
          margin-top: 0 !important;
          margin-bottom: 0 !important;
        }
      `}</style>
      <div 
        className="desktop-window-container"
        style={{ 
          borderRadius: '12px',
          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
          overflow: 'hidden',
          backgroundColor: '#ffffff'
        }}>
      {/* Image container */}
      <div style={{ 
        position: 'relative',
        display: 'block'
      }}>
        <img
          src={imageArray[activeIndex]}
          alt={alt || `Image ${activeIndex + 1} of ${imageArray.length}`}
          style={{
            width: '100%',
            height: 'auto',
            display: 'block',
            padding: 0
          }}
        />

        {/* Navigation arrows */}
        {showControls && (
          <>
            <button
              onClick={goToPrevious}
              style={{
                position: 'absolute',
                left: '1rem',
                top: '50%',
                transform: 'translateY(-50%)',
                backgroundColor: 'rgba(0, 0, 0, 0.5)',
                color: 'white',
                border: 'none',
                borderRadius: '50%',
                width: '40px',
                height: '40px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                fontSize: '1.25rem',
                transition: 'background-color 0.2s'
              }}
              onMouseEnter={(e) => e.target.style.backgroundColor = 'rgba(0, 0, 0, 0.7)'}
              onMouseLeave={(e) => e.target.style.backgroundColor = 'rgba(0, 0, 0, 0.5)'}
            >
              ‹
            </button>
            <button
              onClick={goToNext}
              style={{
                position: 'absolute',
                right: '1rem',
                top: '50%',
                transform: 'translateY(-50%)',
                backgroundColor: 'rgba(0, 0, 0, 0.5)',
                color: 'white',
                border: 'none',
                borderRadius: '50%',
                width: '40px',
                height: '40px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                fontSize: '1.25rem',
                transition: 'background-color 0.2s'
              }}
              onMouseEnter={(e) => e.target.style.backgroundColor = 'rgba(0, 0, 0, 0.7)'}
              onMouseLeave={(e) => e.target.style.backgroundColor = 'rgba(0, 0, 0, 0.5)'}
            >
              ›
            </button>
          </>
        )}
      </div>

      {/* Dots indicator */}
      {showControls && (
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          gap: '0.5rem',
          padding: '1rem',
          backgroundColor: '#f8f9fa'
        }}>
          {imageArray.map((_, index) => (
            <button
              key={index}
              onClick={() => setActiveIndex(index)}
              style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                border: 'none',
                backgroundColor: index === activeIndex ? '#3b82f6' : '#d1d5db',
                cursor: 'pointer',
                padding: 0,
                transition: 'background-color 0.2s'
              }}
              aria-label={`Go to image ${index + 1}`}
            />
          ))}
        </div>
      )}
    </div>
    </>
  );
}; 
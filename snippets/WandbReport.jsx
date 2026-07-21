export const WandbReport = ({ src, title, height = 640 }) => {
  // Reports live at https://wandb.ai/<entity>/<project>/reports/<slug>--Vmlldz<id>
  // (sometimes three hyphens before the id), optionally with a ?accessToken=...
  // query when shared via a view-only ("magic") link. Only report pages allow
  // framing — project and run pages block it — so reject anything else and fall
  // back to the link-only card rather than rendering a frame that won't load.
  const isReportUrl =
    typeof src === 'string' &&
    /^https:\/\/wandb\.ai\/[^/]+\/[^/]+\/reports\/.*-{2,3}Vmlldz[A-Za-z0-9]+/.test(src);

  if (!src) {
    console.error('WandbReport: missing required `src` prop.');
    return null;
  }
  if (!isReportUrl) {
    console.warn(`WandbReport: src does not look like a W&B report URL: ${src}`);
  }

  const frameTitle = title || 'W&B report';
  if (!title) {
    console.warn('WandbReport: missing `title` prop; using a generic accessible name.');
  }
  const frameHeight = typeof height === 'number' && height > 0 ? height : 640;

  // Request W&B's streamlined embed view. `?jupyter=true` is the same parameter
  // the wandb SDK's Report/Run/Sweep `.to_html()` methods use to render a
  // cleaner frame in notebooks; it trims the top chrome the full report page
  // shows. Keep it off the human-facing caption link below, which points at the
  // normal report URL.
  const frameSrc = isReportUrl ? `${src}${src.includes('?') ? '&' : '?'}jupyter=true` : src;

  // Colors come from the CSS custom properties in scripts/css-minify/colors.css
  // (:root plus a .dark override), so the card chrome tracks light/dark mode at
  // paint time with no JavaScript and no changes to the generated stylesheet.
  return (
    <figure
      className="wandb-report-embed not-prose"
      style={{
        margin: '1.5rem 0',
        border: '1px solid var(--color-stroke)',
        borderRadius: '12px',
        overflow: 'hidden',
        backgroundColor: 'var(--color-card-rest)',
      }}
    >
      {isReportUrl && (
        <iframe
          src={frameSrc}
          title={frameTitle}
          loading="lazy"
          allow="fullscreen; clipboard-write"
          referrerPolicy="strict-origin-when-cross-origin"
          style={{ display: 'block', width: '100%', height: `${frameHeight}px`, border: 0 }}
        />
      )}
      <figcaption
        style={{
          padding: '0.6rem 1rem',
          borderTop: isReportUrl ? '1px solid var(--color-stroke)' : 'none',
          fontSize: '14px',
          color: 'var(--color-text-secondary)',
        }}
      >
        <a
          href={src}
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: 'inherit', textDecoration: 'underline' }}
        >
          View this report in the W&B app →
        </a>
      </figcaption>
    </figure>
  );
};

export default WandbReport;

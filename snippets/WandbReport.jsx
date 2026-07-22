export const WandbReport = ({ src, title, height = 640 }) => {
  // Reports live at https://wandb.ai/<entity>/<project>/reports/<slug>--Vmlldz<id>
  // (sometimes three hyphens before the id), optionally with a ?accessToken=...
  // query when shared via a view-only ("magic") link. Only report pages allow
  // framing — project and run pages block it — so reject anything else and fall
  // back to the button-only card rather than rendering a frame that won't load.
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

  // Request W&B's notebook-embed view via `?jupyter=true` — the same parameter
  // the wandb SDK appends in every `.to_html()` (reports, runs, sweeps). A
  // regular report renders in a slim embed view. A Fully Connected article
  // ignores the param and keeps its full blog chrome (nav bar, breadcrumb,
  // author, stars), so it looks broken in a frame — embed regular reports, not
  // FC articles. The View Report button below omits the param and points at the
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
          display: 'flex',
          justifyContent: 'flex-end',
          padding: '0.6rem 1rem',
          borderTop: isReportUrl ? '1px solid var(--color-stroke)' : 'none',
        }}
      >
        {/* "View Report" button matching the in-content button-links
            (.colab-link / .try-product-link in scripts/css-minify/buttons.css),
            styled inline so the component stays self-contained. It links to the
            plain report URL (no ?jupyter=true), so it doubles as the fallback
            wherever the frame can't load. */}
        <a
          href={src}
          target="_blank"
          rel="noopener noreferrer"
          aria-label={`View report: ${frameTitle}`}
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            padding: '8px 14px',
            border: '1px solid color-mix(in srgb, var(--color-stroke) 70%, transparent)',
            borderRadius: '12px',
            fontFamily: 'var(--font-sans)',
            fontSize: '14px',
            lineHeight: '20px',
            color: 'var(--color-text-secondary)',
            backgroundColor: 'color-mix(in srgb, var(--color-card-hover) 50%, transparent)',
            textDecoration: 'none',
          }}
        >
          View Report
        </a>
      </figcaption>
    </figure>
  );
};

export default WandbReport;

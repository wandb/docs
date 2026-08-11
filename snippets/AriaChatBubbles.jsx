export const AriaChatBubbles = ({ prompt, response }) => {
  const [isDark, setIsDark] = useState(false);
  const [promptCopied, setPromptCopied] = useState(false);

  // Track the dark class on <html> so inline styles update with the toggle.
  useEffect(() => {
    const root = document.documentElement;
    const sync = () => setIsDark(root.classList.contains('dark'));
    sync();
    const obs = new MutationObserver(sync);
    obs.observe(root, { attributes: true, attributeFilter: ['class'] });
    return () => obs.disconnect();
  }, []);

  useEffect(() => {
    if (!promptCopied) {
      return undefined;
    }
    const timeout = setTimeout(() => setPromptCopied(false), 2000);
    return () => clearTimeout(timeout);
  }, [promptCopied]);

  const copyText = (text) => {
    navigator.clipboard.writeText(text).then(() => setPromptCopied(true)).catch(console.error);
  };

  // Colors derived from W&B app's moon palette and ARIA composer border colors.
  // Inline styles bypass Tailwind's content-scan so arbitrary hex values render.
  //
  // User bubble:     moon-100/750 bg, moon-250/650 border (composer unfocused border)
  // Response bubble: no background, #CDF4F7 border (composer focus glow color, thin)
  const userBg      = isDark ? '#363C44'               : '#F8F8F8';
  const userBorder  = isDark ? '#4B535C'               : '#DFE0E2';
  const textColor   = isDark ? '#E8E8E9'               : '#2B3038';
  const iconColor   = isDark ? '#8F949E'               : '#79808A';
  const iconHoverBg = isDark ? 'rgba(255,255,255,0.10)' : 'rgba(0,0,0,0.05)';

  const bubbleText = {
    fontSize: '15px',
    color: textColor,
    overflowWrap: 'anywhere',
  };

  // Bubbles are capped at 85% width so each side of the conversation is visually
  // distinct — user right-aligned, ARIA left-aligned — without spanning the full row.
  const bubbleBase = { maxWidth: '85%' };

  const copyIcon = promptCopied ? (
    <svg
      aria-hidden="true"
      focusable="false"
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{ color: '#FCBC32' }}
    >
      <path d="M20 6 9 17l-5-5" />
    </svg>
  ) : (
    <svg
      aria-hidden="true"
      focusable="false"
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect width="14" height="14" x="8" y="8" rx="2" ry="2" />
      <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" />
    </svg>
  );

  return (
    <div
      aria-label="ARIA chat example"
      className="not-prose flex flex-col gap-3"
      role="group"
    >
      {/* User prompt — right-aligned with copy button */}
      <div className="flex w-full justify-end">
        <div
          className="flex w-fit items-start gap-1 rounded-2xl py-2 pl-3 pr-1.5"
          style={{ ...bubbleBase, backgroundColor: userBg, border: `1px solid ${userBorder}` }}
        >
          <div
            className="min-w-0 flex-1 hyphens-auto whitespace-pre-wrap"
            style={bubbleText}
          >
            {prompt}
          </div>
          <button
            type="button"
            className="mt-0.5 shrink-0 rounded-lg p-1.5 cursor-pointer"
            style={{ color: iconColor, background: 'transparent' }}
            onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = iconHoverBg; }}
            onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = 'transparent'; }}
            onClick={() => copyText(prompt)}
            aria-label="Copy user prompt"
          >
            {copyIcon}
          </button>
        </div>
      </div>
      {/* ARIA response — left-aligned, no copy button */}
      <div className="flex w-full justify-start">
        <div
          className="w-fit rounded-2xl py-2 px-3"
          style={{ ...bubbleBase, border: '1px solid #CDF4F7' }}
        >
          <div
            className="hyphens-auto whitespace-pre-wrap"
            style={bubbleText}
          >
            {response}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AriaChatBubbles;

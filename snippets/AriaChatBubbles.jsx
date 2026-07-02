export const AriaChatBubbles = ({ prompt, response }) => {
  const [promptCopied, setPromptCopied] = useState(false);
  const [responseCopied, setResponseCopied] = useState(false);

  useEffect(() => {
    if (!promptCopied) {
      return undefined;
    }

    const timeout = setTimeout(() => setPromptCopied(false), 2000);
    return () => clearTimeout(timeout);
  }, [promptCopied]);

  useEffect(() => {
    if (!responseCopied) {
      return undefined;
    }

    const timeout = setTimeout(() => setResponseCopied(false), 2000);
    return () => clearTimeout(timeout);
  }, [responseCopied]);

  const copyText = (text, setCopied) => {
    navigator.clipboard
      .writeText(text)
      .then(() => setCopied(true))
      .catch(console.error);
  };

  const copyButtonClassName =
    'rounded-lg p-1.5 hover:bg-gray-100 dark:hover:bg-white/10 cursor-pointer text-gray-500';

  const bubbleTextClassName =
    'hyphens-auto whitespace-pre-wrap text-base text-gray-800 [overflow-wrap:anywhere] lg:text-sm dark:text-gray-200';

  const copyIcon = (copied) =>
    copied ? (
      <svg
        aria-hidden="true"
        focusable="false"
        className="size-4 sm:size-3.5 text-primary dark:text-primary-light"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M20 6 9 17l-5-5" />
      </svg>
    ) : (
      <svg
        aria-hidden="true"
        focusable="false"
        className="size-4 sm:size-3.5"
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

  const bubbleRow = ({
    align,
    bubbleClassName,
    text,
    copied,
    setCopied,
    copyLabel,
  }) => (
    <div
      className={`flex w-full ${align === 'end' ? 'justify-end' : 'justify-start'}`}
    >
      <div
        className={`flex w-fit max-w-[min(100%,36rem)] items-start gap-1 rounded-2xl py-2 pl-3 pr-1.5 ${bubbleClassName}`}
      >
        <div className={`min-w-0 flex-1 ${bubbleTextClassName}`}>{text}</div>
        <button
          type="button"
          className={`mt-0.5 shrink-0 ${copyButtonClassName}`}
          onClick={() => copyText(text, setCopied)}
          aria-label={copyLabel}
        >
          {copyIcon(copied)}
        </button>
      </div>
    </div>
  );

  return (
    <div
      aria-label="ARIA chat example"
      className="not-prose flex flex-col gap-3"
      role="group"
    >
      {bubbleRow({
        align: 'end',
        bubbleClassName:
          'rounded-br-none bg-gray-100 dark:bg-white/5',
        text: prompt,
        copied: promptCopied,
        setCopied: setPromptCopied,
        copyLabel: 'Copy user prompt',
      })}
      {bubbleRow({
        align: 'start',
        bubbleClassName:
          'rounded-bl-none bg-primary/10 dark:bg-primary-light/10',
        text: response,
        copied: responseCopied,
        setCopied: setResponseCopied,
        copyLabel: 'Copy chat response',
      })}
    </div>
  );
};

export default AriaChatBubbles;

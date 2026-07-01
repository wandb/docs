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

  return (
    <div
      aria-label="ARIA chat example"
      className="not-prose flex flex-col gap-3"
      role="group"
    >
      <div className="flex w-full flex-col items-end gap-1">
        <div className="flex w-fit max-w-[min(100%,36rem)] items-start gap-4 rounded-2xl rounded-br-none bg-gray-100 px-3 py-2 dark:bg-white/5">
          <div className="hyphens-auto whitespace-pre-wrap text-base text-gray-800 [overflow-wrap:anywhere] lg:text-sm dark:text-gray-200">
            {prompt}
          </div>
        </div>
        <button
          type="button"
          className={copyButtonClassName}
          onClick={() => copyText(prompt, setPromptCopied)}
          aria-label="Copy user prompt"
        >
          {promptCopied ? (
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
          )}
        </button>
      </div>

      <div className="flex w-full flex-col items-start gap-1">
        <div className="flex w-fit max-w-[min(100%,36rem)] items-start gap-4 rounded-2xl rounded-bl-none bg-primary/10 px-3 py-2 text-primary dark:bg-primary-light/10 dark:text-primary-light">
          <div className="hyphens-auto whitespace-pre-wrap text-base [overflow-wrap:anywhere] lg:text-sm">
            {response}
          </div>
        </div>
        <button
          type="button"
          className={copyButtonClassName}
          onClick={() => copyText(response, setResponseCopied)}
          aria-label="Copy chat response"
        >
          {responseCopied ? (
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
          )}
        </button>
      </div>
    </div>
  );
};

export default AriaChatBubbles;

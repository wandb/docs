export const HelpQuestionForm = () => {
  const [value, setValue] = useState("");

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    setValue(params.get("assistant") || "");
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    const trimmed = value.trim();
    if (!trimmed) return;
    const url = new URL(window.location.href);
    url.searchParams.set("assistant", trimmed);
    window.location.href = url.toString();
  };

  return (
    <form onSubmit={handleSubmit} className="mb-8">
      <label htmlFor="help-question" className="block text-lg font-medium text-zinc-950 dark:text-white mb-3">
        How can we help?
      </label>
      <div className="flex gap-2 flex-wrap">
        <input
          id="help-question"
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Ask your question..."
          className="flex-1 min-w-[200px] px-4 py-2 rounded-lg border border-zinc-950/20 dark:border-white/20 bg-white dark:bg-zinc-950 text-zinc-950 dark:text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-primary-500"
          aria-label="Ask your question"
        />
        <button
          type="submit"
          className="px-4 py-2 rounded-lg bg-primary text-white font-medium hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2"
        >
          Submit
        </button>
      </div>
    </form>
  );
};

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
    <form onSubmit={handleSubmit} className="mb-8 mt-8">
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
          className="inline-flex items-center gap-2 whitespace-nowrap px-4 py-2 rounded-lg bg-primary text-white font-medium hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 18 18" class="size-4 shrink-0 text-gray-700 group-hover/ai:text-gray-800 dark:text-gray-400 dark:group-hover/ai:text-gray-200"><g fill="white"><path d="M5.658,2.99l-1.263-.421-.421-1.263c-.137-.408-.812-.408-.949,0l-.421,1.263-1.263,.421c-.204,.068-.342,.259-.342,.474s.138,.406,.342,.474l1.263,.421,.421,1.263c.068,.204,.26,.342,.475,.342s.406-.138,.475-.342l.421-1.263,1.263-.421c.204-.068,.342-.259,.342-.474s-.138-.406-.342-.474Z" fill="white" data-stroke="none" stroke="none"></path><polygon points="9.5 2.75 11.412 7.587 16.25 9.5 11.412 11.413 9.5 16.25 7.587 11.413 2.75 9.5 7.587 7.587 9.5 2.75" fill="none" stroke="white" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"></polygon></g></svg>
          Ask AI
        </button>
      </div>
    </form>
  );
};

export const KBArticlesList = ({ kbArticleMap, tag, language }) => {
  // Filter articles that contain the specified tag
  const filteredArticles = kbArticleMap.filter((article) =>
    article.tags.some((articleTag) => articleTag.toLowerCase() === tag.toLowerCase())
  );

  // Helper function to capitalize tags for display
  const capitalizeTag = (tag) => {
    return tag
      .split(" ")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  };

  const buildTagHref = (articleTag) => {
    const basePath = `/support/${articleTag.toLowerCase().replace(" ", "_")}`;
    return language ? `/${language}${basePath}` : basePath;
  };

  return (
    <div className="flex flex-col gap-4">
      {filteredArticles.map((article, index) => (
        <div
          key={index}
          className="bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4 md:p-6 transition-colors hover:border-gray-300 dark:hover:border-gray-600"
        >
          {/* Article Title with Document Icon */}
          <div className="flex items-center gap-3 mb-2">
            <svg
              className="text-teal-500 dark:text-teal-400 flex-shrink-0"
              width="16"
              height="16"
              viewBox="0 0 16 16"
              fill="currentColor"
            >
              <path d="M4 2a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2H4zm0 1h8a1 1 0 0 1 1 1v8a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z" />
              <path d="M5 4h6v1H5V4zm0 2h6v1H5V6zm0 2h4v1H5V8z" />
            </svg>
            <a
              href={article.page}
              className="text-teal-600 dark:text-teal-400 no-underline text-lg font-medium leading-snug hover:text-teal-700 dark:hover:text-teal-300 hover:underline transition-colors"
            >
              {article.title}
            </a>
          </div>

          {/* Breadcrumb */}
          <div className="text-gray-500 dark:text-gray-400 text-sm mb-4 leading-snug">
            <span className="text-gray-500 dark:text-gray-400">Support</span>
            <span className="text-gray-400 dark:text-gray-500 mx-1">&gt;</span>
            <span className="text-gray-600 dark:text-gray-300">{article.title}</span>
          </div>

          {/* Tags */}
          <div className="flex items-start gap-3 flex-wrap md:flex-nowrap">
            <span className="text-gray-700 dark:text-gray-300 font-medium text-sm whitespace-nowrap mt-0.5">
              Support:
            </span>
            <div className="flex flex-wrap gap-2">
              {article.tags.map((articleTag, tagIndex) => (
                <a
                  key={tagIndex}
                  href={buildTagHref(articleTag)}
                  className="bg-gray-200 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded px-2 md:px-3 py-1 text-xs text-teal-600 dark:text-teal-400 font-medium whitespace-nowrap hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors no-underline"
                >
                  {capitalizeTag(articleTag)}
                </a>
              ))}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};

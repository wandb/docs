---
menu:
  support:
    identifier: ko-support-kb-articles-difference_log_summary
support:
- Charts
title: What is the difference between `.log()` and `.summary`?
toc_hide: true
type: docs
url: /support/:filename
---

The summary displays in the table, while the log saves all values for future plotting.

For instance, call `run.log()` whenever accuracy changes. By default, `run.log()` updates the summary value unless set manually for that metric.

The scatterplot and parallel coordinate plots use the summary value, while the line plot shows all values recorded by `run.log`.

Some users prefer to set the summary manually to reflect the optimal accuracy instead of the most recent accuracy logged.
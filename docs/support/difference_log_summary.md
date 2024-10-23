---
title: "What is the difference between `.log()` and `.summary`?"
displayed_sidebar: support
tags:
   - None
---
The summary displays in the table, while the log saves all values for future plotting.

For instance, call `wandb.log` whenever accuracy changes. By default, `wandb.log()` updates the summary value unless set manually for that metric.

The scatterplot and parallel coordinate plots use the summary value, while the line plot shows all values recorded by `.log`.

Some users prefer to set the summary manually to reflect the optimal accuracy instead of the most recent accuracy logged.
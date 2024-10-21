---
title: "What is the difference between `.log()` and `.summary`?"
tags:
   - 
---

The summary is the value that shows in the table while the log will save all the values for plotting later.

For example, you might want to call `wandb.log` every time the accuracy changes. Usually, you can just use .log. `wandb.log()` will also update the summary value by default unless you have set the summary manually for that metric

The scatterplot and parallel coordinate plots will also use the summary value while the line plot plots all of the values set by .log

The reason we have both is that some people like to set the summary manually because they want the summary to reflect for example the optimal accuracy instead of the last accuracy logged.
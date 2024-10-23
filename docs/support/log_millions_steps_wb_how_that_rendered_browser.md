---
title: "What happens when I log millions of steps to W&B? How is that rendered in the browser?"
displayed_sidebar: support
tags:
   - experiments
---
The number of points sent affects the loading time of graphs in the UI. For lines exceeding 1,000 points, the backend samples the data down to 1,000 points before sending it to the browser. This sampling is nondeterministic, resulting in different sampled points upon page refresh.

**Guidelines**

Log fewer than 10,000 points per metric. Logging over 1 million points in a line significantly increases page load time. Explore strategies to minimize logging footprint without sacrificing accuracy in this [Colab](http://wandb.me/log-hf-colab). With more than 500 columns of config and summary metrics, only 500 display in the table.
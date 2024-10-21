---
title: "What happens when I log millions of steps to W&B? How is that rendered in the browser?"
tags:
   - experiments
---

The more points you send us, the longer it will take to load your graphs in the UI. If you have more than 1000 points on a line, we sample down to 1000 points on the backend before we send your browser the data. This sampling is nondeterministic, so if you refresh the page you'll see a different set of sampled points.

**Guidelines**

We recommend that you try to log less than 10,000 points per metric. If you log more than 1 million points in a line, it will take us while to load the page. For more on strategies for reducing logging footprint without sacrificing accuracy, check out [this Colab](http://wandb.me/log-hf-colab). If you have more than 500 columns of config and summary metrics, we'll only show 500 in the table.
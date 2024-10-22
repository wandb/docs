---
title: "Best practices to organize hyperparameter searches"
tags:
   - None
---
If the limit of 10,000 runs per project is reasonable, set unique tags in `wandb.init()` for each search. This allows efficient filtering of project runs by selecting the corresponding tag in the Project Page's Runs Table. For example, use `wandb.init(tags='your_tag')`. More documentation is available [here](../ref/python/init.md).
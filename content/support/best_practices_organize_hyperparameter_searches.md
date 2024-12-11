---
title: "Best practices to organize hyperparameter searches"
toc_hide: true
type: docs
tags:
   - hyperparameter
   - sweeps
   - runs
---
Set unique tags with `wandb.init(tags='your_tag')`. This allows efficient filtering of project runs by selecting the corresponding tag in a Project Page's Runs Table. 


For more information on wandb.int, see the [documentation](../ref/python/init.md).
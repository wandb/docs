---
title: "Best practices to organize hyperparameter searches"
tags:
   - hyperparameter
   - sweeps
   - runs table
---
Set unique tags with `wandb.init(tags='your_tag')`. This allows efficient filtering of project runs by selecting the corresponding tag in a Project Page's Runs Table. 


For more information on wandb.int, see the [documentation](../ref/python/init.md).
---
url: /support/:filename
title: "Best practices to organize hyperparameter searches"
toc_hide: true
type: docs
support:
   - hyperparameter
   - sweeps
   - runs
---
Set unique tags with `wandb.init(tags='your_tag')`. This allows efficient filtering of project runs by selecting the corresponding tag in a Project Page's Runs Table. 


For more information on `wandb.init()`, see the [`wandb.init()` reference]({{< relref "/ref/python/sdk/functions/init.md" >}}).

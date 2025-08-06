---
menu:
  support:
    identifier: ko-support-kb-articles-best_practices_organize_hyperparameter_searches
support:
- hyperparameter
- sweeps
- runs
title: Best practices to organize hyperparameter searches
toc_hide: true
type: docs
url: /support/:filename
---

Set unique tags with `wandb.init(tags='your_tag')`. This allows efficient filtering of project runs by selecting the corresponding tag in a Project Page's Runs Table. 


For more information on `wandb.init()`, see the [`wandb.init()` reference]({{< relref path="/ref/python/sdk/functions/init.md" lang="ko" >}}).
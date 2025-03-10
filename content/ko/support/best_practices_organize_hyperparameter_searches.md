---
menu:
  support:
    identifier: ko-support-best_practices_organize_hyperparameter_searches
tags:
- hyperparameter
- sweeps
- runs
title: Best practices to organize hyperparameter searches
toc_hide: true
type: docs
---

Set unique tags with `wandb.init(tags='your_tag')`. This allows efficient filtering of project runs by selecting the corresponding tag in a Project Page's Runs Table. 


For more information on wandb.int, see the [documentation]({{< relref path="/ref/python/init.md" lang="ko" >}}).
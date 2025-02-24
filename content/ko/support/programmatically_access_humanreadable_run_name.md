---
title: How do I programmatically access the human-readable run name?
menu:
  support:
    identifier: ko-support-programmatically_access_humanreadable_run_name
tags:
- experiments
toc_hide: true
type: docs
---

[`wandb.Run`]({{< relref path="/ref/python/run.md" lang="ko" >}})의 `.name` 속성은 다음과 같이 엑세스할 수 있습니다:

```python
import wandb

wandb.init()
run_name = wandb.run.name
```

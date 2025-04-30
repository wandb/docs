---
title: How do I programmatically access the human-readable run name?
menu:
  support:
    identifier: ko-support-kb-articles-programmatically_access_humanreadable_run_name
support:
- experiments
toc_hide: true
type: docs
url: /ko/support/:filename
---

[`wandb.Run`]({{< relref path="/ref/python/run.md" lang="ko" >}}) 의 `.name` 속성은 다음과 같이 액세스할 수 있습니다:

```python
import wandb

wandb.init()
run_name = wandb.run.name
```

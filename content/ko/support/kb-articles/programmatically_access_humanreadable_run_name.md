---
title: 사람이 읽을 수 있는 run 이름에 프로그래밍적으로 엑세스하려면 어떻게 해야 하나요?
menu:
  support:
    identifier: ko-support-kb-articles-programmatically_access_humanreadable_run_name
support:
- Experiments
toc_hide: true
type: docs
url: /support/:filename
---

`.name` 속성은 [`wandb.Run`]({{< relref path="/ref/python/sdk/classes/run" lang="ko" >}})에서 다음과 같이 엑세스할 수 있습니다:

```python
import wandb

with wandb.init() as run:
   run_name = run.name
   print(f"The human-readable run name is: {run_name}")
```
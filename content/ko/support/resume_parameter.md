---
title: How do I use the resume parameter when resuming a run in W&B?
menu:
  support:
    identifier: ko-support-resume_parameter
tags:
- resuming
toc_hide: true
type: docs
---

W&B 에서 `resume` 파라미터를 사용하려면, `entity`, `project`, `id` 를 명시하여 `wandb.init()` 에 `resume` 인수를 설정하세요. `resume` 인수는 `"must"` 또는 `"allow"` 값을 허용합니다.

```python
run = wandb.init(entity="your-entity", project="your-project", id="your-run-id", resume="must")
```

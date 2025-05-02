---
title: How do I use the resume parameter when resuming a run in W&B?
menu:
  support:
    identifier: ko-support-kb-articles-resume_parameter
support:
- resuming
toc_hide: true
type: docs
url: /ko/support/:filename
translationKey: resume_parameter
---
W&B에서 `resume` 파라미터를 사용하려면 `wandb.init()`에서 `resume` 인수를 설정하고 `entity`, `project`, `id`를 지정하세요. `resume` 인수는 `"must"` 또는 `"allow"` 값을 허용합니다.

```python
run = wandb.init(entity="your-entity", project="your-project", id="your-run-id", resume="must")
```
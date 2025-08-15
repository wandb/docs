---
title: W&B에서 run을 재개할 때 resume 파라미터는 어떻게 사용하나요?
menu:
  support:
    identifier: ko-support-kb-articles-resume_parameter
support:
- 재개
toc_hide: true
type: docs
url: /support/:filename
---

W&B 에서 `resume` 파라미터를 사용하려면, `wandb.init()`에서 `entity`, `project`, 그리고 `id`를 지정한 상태로 `resume` 인수를 설정하면 됩니다. `resume` 인수에는 `"must"` 또는 `"allow"` 값을 사용할 수 있습니다.

  ```python
  # run 을 이어서 실행하려면 아래와 같이 설정하세요.
  run = wandb.init(entity="your-entity", project="your-project", id="your-run-id", resume="must")
  ```
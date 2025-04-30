---
title: How do I launch multiple runs from one script?
menu:
  support:
    identifier: ko-support-kb-articles-launch_multiple_runs_one_script
support:
- experiments
toc_hide: true
type: docs
url: /ko/support/:filename
---

`wandb.init` 과 `run.finish()` 를 사용하여 단일 스크립트 내에서 여러 개의 run을 로그로 기록하세요.

1. `run = wandb.init(reinit=True)` 를 사용하여 run의 재초기화를 허용합니다.
2. 로깅을 완료하려면 각 run의 끝에서 `run.finish()` 를 호출합니다.

```python
import wandb

for x in range(10):
    run = wandb.init(reinit=True)
    for y in range(100):
        wandb.log({"metric": x + y})
    run.finish()
```

또는 Python context manager를 사용하여 로깅을 자동으로 완료합니다.

```python
import wandb

for x in range(10):
    run = wandb.init(reinit=True)
    with run:
        for y in range(100):
            run.log({"metric": x + y})
```
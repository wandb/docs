---
title: How do I launch multiple runs from one script?
menu:
  support:
    identifier: ko-support-launch_multiple_runs_one_script
tags:
- experiments
toc_hide: true
type: docs
---

`wandb.init` 과 `run.finish()` 를 사용하여 단일 스크립트 내에서 여러 개의 run을 로그할 수 있습니다:

1. `run = wandb.init(reinit=True)` 를 사용하여 run의 재 초기화를 허용합니다.
2. 각 run이 끝나면 `run.finish()` 를 호출하여 로깅을 완료합니다.

```python
import wandb

for x in range(10):
    run = wandb.init(reinit=True)
    for y in range(100):
        wandb.log({"metric": x + y})
    run.finish()
```

또는 Python 컨텍스트 관리자를 활용하여 자동으로 로깅을 완료합니다:

```python
import wandb

for x in range(10):
    run = wandb.init(reinit=True)
    with run:
        for y in range(100):
            run.log({"metric": x + y})
```
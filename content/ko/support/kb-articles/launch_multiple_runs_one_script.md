---
title: 하나의 스크립트에서 여러 run 을 실행하려면 어떻게 해야 하나요?
menu:
  support:
    identifier: ko-support-kb-articles-launch_multiple_runs_one_script
support:
- Experiments
toc_hide: true
type: docs
url: /support/:filename
---

이전 run 을 마친 후 새 run 을 시작하여 하나의 스크립트 내에서 여러 run 의 로그를 남길 수 있습니다.

이를 권장하는 방법은 `wandb.init()` 를 컨텍스트 매니저로 사용하는 것으로, 스크립트에서 예외가 발생하면 run 을 종료하고 실패로 표시해줍니다.

```python
import wandb

for x in range(10):
    with wandb.init() as run:
        for y in range(100):
            run.log({"metric": x + y})
```

직접적으로 `run.finish()` 를 호출할 수도 있습니다:

```python
import wandb

for x in range(10):
    run = wandb.init()

    try:
        for y in range(100):
            run.log({"metric": x + y})

    except Exception:
        run.finish(exit_code=1)
        raise

    finally:
        run.finish()
```

## 여러 개의 활성 run

wandb 0.19.10 버전부터는 `reinit` 설정을 `"create_new"` 로 지정하여 동시에 여러 개의 활성 run 을 만들 수 있습니다.

```python
import wandb

with wandb.init(reinit="create_new") as tracking_run:
    for x in range(10):
        with wandb.init(reinit="create_new") as run:
            for y in range(100):
                run.log({"x_plus_y": x + y})

            tracking_run.log({"x": x})
```

`reinit="create_new"` 에 대한 자세한 설명과 주의사항, 그리고 W&B 인테그레이션 관련 정보를 보려면 [Multiple runs per process]({{< relref path="guides/models/track/runs/multiple-runs-per-process.md" lang="ko" >}}) 를 참고하세요.
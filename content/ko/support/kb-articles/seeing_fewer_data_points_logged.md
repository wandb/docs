---
title: 왜 내가 로그한 데이터 포인트보다 적게 표시되나요?
menu:
  support:
    identifier: ko-support-kb-articles-seeing_fewer_data_points_logged
support:
- Experiments
- 메트릭
toc_hide: true
type: docs
url: /support/:filename
---

`Step` 이외의 X축에 대해 메트릭을 시각화할 때는 데이터 포인트가 더 적게 표시될 수 있습니다. 메트릭이 동기화된 상태를 유지하려면 반드시 같은 `Step` 에서 로그되어야 합니다. 샘플 간 보간 시에도 동일한 `Step` 에서 로그된 메트릭만 샘플링됩니다.

**가이드라인**

여러 메트릭은 하나의 `log()` 호출에서 번들로 묶어서 로그하는 것이 좋습니다. 예를 들어, 아래와 같이 하지 말고:

```python
import wandb
with wandb.init() as run:
    run.log({"Precision": precision})
    ...
    run.log({"Recall": recall})
```

다음과 같이 사용하세요:

```python
import wandb
with wandb.init() as run:
    run.log({"Precision": precision, "Recall": recall})
```

step 파라미터를 수동으로 관리하고 싶다면, 아래와 같이 코드에서 메트릭을 동기화하세요:

```python
with wandb.init() as run:
    step = 100  # 예시 step 값
    # 같은 step 에서 Precision과 Recall 로그
    run.log({"Precision": precision, "Recall": recall}, step=step)
```

메트릭이 동일한 step 에서 함께 로그되고 샘플링되기 위해서는 두 번의 `log()` 호출에서 `step` 값이 같아야 합니다. 또한, 각 호출 시 `step` 값은 반드시 단조롭게 증가해야 하며, 그렇지 않으면 `step` 값이 무시됩니다.
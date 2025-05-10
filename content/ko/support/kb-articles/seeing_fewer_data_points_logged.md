---
title: Why am I seeing fewer data points than I logged?
menu:
  support:
    identifier: ko-support-kb-articles-seeing_fewer_data_points_logged
support:
- experiments
- metrics
toc_hide: true
type: docs
url: /ko/support/:filename
translationKey: seeing_fewer_data_points_logged
---
`Step` 이외의 X축에 대해 메트릭을 시각화할 때는 표시되는 데이터 포인트 수가 더 적을 것으로 예상됩니다. 메트릭은 동기화를 유지하기 위해 동일한 `Step` 에 로그되어야 합니다. 동일한 `Step` 에 로그된 메트릭만 샘플 간 보간 중에 샘플링됩니다.

**가이드라인**

메트릭을 단일 `log()` 호출로 묶습니다. 예를 들어, 다음과 같이 하지 말고:

```python
wandb.log({"Precision": precision})
...
wandb.log({"Recall": recall})
```

다음과 같이 사용합니다:

```python
wandb.log({"Precision": precision, "Recall": recall})
```

step 파라미터를 수동으로 제어하려면 다음과 같이 코드에서 메트릭을 동기화합니다:

```python
wandb.log({"Precision": precision}, step=step)
...
wandb.log({"Recall": recall}, step=step)
```

메트릭이 동일한 단계로 로그되고 함께 샘플링되도록 하려면 `log()` 호출에서 `step` 값이 동일하게 유지되는지 확인합니다. `step` 값은 각 호출에서 단조롭게 증가해야 합니다. 그렇지 않으면 `step` 값은 무시됩니다.

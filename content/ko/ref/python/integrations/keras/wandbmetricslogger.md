---
title: WandbMetricsLogger
menu:
  reference:
    identifier: ko-ref-python-integrations-keras-wandbmetricslogger
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/callbacks/metrics_logger.py#L16-L129 >}}

시스템 메트릭을 W&B에 보내는 로거입니다.

```python
WandbMetricsLogger(
    log_freq: Union[LogStrategy, int] = "epoch",
    initial_global_step: int = 0,
    *args,
    **kwargs
) -> None
```

`WandbMetricsLogger`는 콜백 메소드가 인수로 사용하는 `logs` 사전을 자동으로 wandb에 기록합니다.

이 콜백은 다음을 W&B run 페이지에 자동으로 기록합니다.

* 시스템 (CPU/GPU/TPU) 메트릭,
* `model.compile`에 정의된 트레이닝 및 유효성 검사 메트릭,
* 학습률 (고정 값 및 학습률 스케줄러 모두)

#### 참고 사항:

`initial_epoch`를 `model.fit`에 전달하여 트레이닝을 재개하고 학습률 스케줄러를 사용하는 경우, `initial_global_step`을 `WandbMetricsLogger`에 전달해야 합니다. `initial_global_step`은 `step_size * initial_step`입니다. 여기서 `step_size`는 에포크당 트레이닝 단계 수입니다. `step_size`는 트레이닝 데이터셋의 Cardinality와 배치 크기의 곱으로 계산할 수 있습니다.

| Args |  |
| :--- | :--- |
|  `log_freq` |  ("epoch", "batch", 또는 int) "epoch"인 경우 각 에포크가 끝날 때 메트릭을 기록합니다. "batch"인 경우 각 배치 끝날 때 메트릭을 기록합니다. 정수인 경우 해당 배치 수만큼 끝날 때 메트릭을 기록합니다. 기본값은 "epoch"입니다. |
|  `initial_global_step` |  (int) `initial_epoch`에서 트레이닝을 재개하고 학습률 스케줄러를 사용하는 경우 학습률을 올바르게 기록하려면 이 인수를 사용하십시오. `step_size * initial_step`으로 계산할 수 있습니다. 기본값은 0입니다. |

## Methods

### `set_model`

```python
set_model(
    model
)
```

### `set_params`

```python
set_params(
    params
)
```

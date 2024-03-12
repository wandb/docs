
# WandbMetricsLogger

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/integration/keras/callbacks/metrics_logger.py#L23-L130' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>


W&B에 시스템 메트릭을 전송하는 로거입니다.

```python
WandbMetricsLogger(
    log_freq: Union[LogStrategy, int] = "epoch",
    initial_global_step: int = 0,
    *args,
    **kwargs
) -> None
```

`WandbMetricsLogger`는 콜백 메소드가 인수로 받는 `로그` 사전을 wandb에 자동으로 기록합니다.

이 콜백은 W&B run 페이지에 다음을 자동으로 기록합니다:

* 시스템(CPU/GPU/TPU) 메트릭,
* `model.compile`에서 정의된 트레이닝 및 검증 메트릭,
* 학습률(고정된 값이나 학습률 스케줄러 모두에 대해서)

#### 참고 사항:

`model.fit`에 `initial_epoch`을 전달하여 트레이닝을 재개하고 학습률 스케줄러를 사용하는 경우,
`WandbMetricsLogger`에 `initial_global_step`을 전달해야 합니다. `initial_global_step`은 `step_size * initial_step`입니다. 여기서
`step_size`는 에포크당 트레이닝 스텝의 수입니다. `step_size`는 트레이닝 데이터셋의 카디널리티와 배치 크기의 곱으로 계산할 수 있습니다.

| 인수 |  |
| :--- | :--- |
|  `log_freq` |  ("epoch", "batch", 또는 int) "epoch"인 경우, 각 에포크의 끝에 메트릭을 기록합니다. "batch"인 경우, 각 배치의 끝에 메트릭을 기록합니다. 정수인 경우, 그 많은 배치의 끝에 메트릭을 기록합니다. 기본값은 "epoch"입니다. |
|  `initial_global_step` |  (int) 어떤 `initial_epoch`에서 트레이닝을 재개할 때, 그리고 학습률 스케줄러가 사용될 때 학습률을 올바르게 기록하기 위해 이 인수를 사용합니다. 이는 `step_size * initial_step`로 계산할 수 있습니다. 기본값은 0입니다. |

## 메소드

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
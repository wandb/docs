
# WandbMetricsLogger

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/integration/keras/callbacks/metrics_logger.py#L23-L130' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>


시스템 메트릭을 W&B에 전송하는 로거입니다.

```python
WandbMetricsLogger(
    log_freq: Union[LogStrategy, int] = "epoch",
    initial_global_step: int = 0,
    *args,
    **kwargs
) -> None
```

`WandbMetricsLogger`는 콜백 메서드가 인수로 받는 `logs` 사전을 wandb에 자동으로 로깅합니다.

이 콜백은 다음을 W&B 실행 페이지에 자동으로 로깅합니다:

* 시스템(CPU/GPU/TPU) 메트릭,
* `model.compile`에서 정의된 학습 및 검증 메트릭,
* 학습률(고정 값 또는 학습률 스케줄러 모두에 대해)

#### 참고:

`model.fit`에 `initial_epoch`를 전달하여 학습을 재개하고 학습률 스케줄러를 사용하는 경우,
`WandbMetricsLogger`에 `initial_global_step`을 전달해야 합니다. `initial_global_step`은 `step_size * initial_step`입니다. 여기서
`step_size`는 에포크 당 학습 단계 수입니다. `step_size`는 학습 데이터세트의 카디널리티와 배치 크기의 곱으로 계산할 수 있습니다.

| 인수 |  |
| :--- | :--- |
|  `log_freq` |  ("epoch", "batch", 또는 int) "epoch"인 경우, 각 에포크의 끝에 메트릭을 로깅합니다. "batch"인 경우, 각 배치의 끝에 메트릭을 로깅합니다. 정수인 경우, 해당 배치의 끝에 메트릭을 로깅합니다. 기본값은 "epoch"입니다. |
|  `initial_global_step` |  (int) 일부 `initial_epoch`에서 학습을 재개하고 학습률 스케줄러가 사용되는 경우 학습률을 올바르게 로깅하기 위해 이 인수를 사용합니다. 이는 `step_size * initial_step`으로 계산할 수 있습니다. 기본값은 0입니다. |

## 메서드

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
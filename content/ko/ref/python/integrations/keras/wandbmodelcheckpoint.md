---
title: WandbModelCheckpoint
menu:
  reference:
    identifier: ko-ref-python-integrations-keras-wandbmodelcheckpoint
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/callbacks/model_checkpoint.py#L20-L188 >}}

주기적으로 Keras 모델 또는 모델 가중치를 저장하는 체크포인트입니다.

```python
WandbModelCheckpoint(
    filepath: StrPath,
    monitor: str = "val_loss",
    verbose: int = 0,
    save_best_only: bool = (False),
    save_weights_only: bool = (False),
    mode: Mode = "auto",
    save_freq: Union[SaveStrategy, int] = "epoch",
    initial_value_threshold: Optional[float] = None,
    **kwargs
) -> None
```

저장된 가중치는 `wandb.Artifact` 로 W&B에 업로드됩니다.

이 콜백은 `tf.keras.callbacks.ModelCheckpoint` 의 서브클래스이므로 체크포인트 로직은 상위 콜백에서 처리합니다. 자세한 내용은 https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint 에서 확인할 수 있습니다.

이 콜백은 `model.fit()` 을 사용한 트레이닝과 함께 사용하여 특정 간격으로 모델 또는 가중치 (체크포인트 파일)를 저장합니다. 모델 체크포인트는 W&B Artifacts 로 기록됩니다. 자세한 내용은 https://docs.wandb.ai/guides/artifacts 에서 확인할 수 있습니다.

이 콜백은 다음과 같은 기능을 제공합니다.
- "monitor"를 기반으로 "최고 성능"을 달성한 모델을 저장합니다.
- 성능에 관계없이 모든 에포크가 끝날 때마다 모델을 저장합니다.
- 에포크가 끝날 때 또는 고정된 수의 트레이닝 배치 후에 모델을 저장합니다.
- 모델 가중치만 저장하거나 전체 모델을 저장합니다.
- SavedModel 형식 또는 `.h5` 형식으로 모델을 저장합니다.

| Args |  |
| :--- | :--- |
|  `filepath` |  (Union[str, os.PathLike]) 모델 파일을 저장할 경로입니다. `filepath` 에는 `epoch` 의 값과 `logs` 의 키 ( `on_epoch_end` 에 전달됨)로 채워지는 명명된 형식 옵션이 포함될 수 있습니다. 예를 들어 `filepath` 가 `model-{epoch:02d}-{val_loss:.2f}` 이면 모델 체크포인트는 에포크 번호와 파일 이름의 유효성 검사 손실과 함께 저장됩니다. |
|  `monitor` |  (str) 모니터링할 메트릭 이름입니다. 기본값은 "val_loss"입니다. |
|  `verbose` |  (int) 상세 모드, 0 또는 1입니다. 모드 0은 자동, 모드 1은 콜백이 작업을 수행할 때 메시지를 표시합니다. |
|  `save_best_only` |  (bool) `save_best_only=True` 인 경우 모델이 "최고"로 간주될 때만 저장되고 모니터링되는 수량에 따라 최신 최고 모델이 덮어쓰여지지 않습니다. `filepath` 에 `{epoch}` 와 같은 형식 옵션이 포함되어 있지 않으면 `filepath` 는 각 새로운 더 나은 모델에 의해 로컬로 덮어쓰여집니다. 아티팩트로 기록된 모델은 여전히 올바른 `monitor` 와 연결됩니다. Artifacts 는 새로운 최고 모델이 발견되면 지속적으로 업로드되고 버전이 분리됩니다. |
|  `save_weights_only` |  (bool) True인 경우 모델의 가중치만 저장됩니다. |
|  `mode` |  (Mode) {'auto', 'min', 'max'} 중 하나입니다. `val_acc` 의 경우 `max` 여야 하고 `val_loss` 의 경우 `min` 여야 합니다. |
|  `save_freq` |  (Union[SaveStrategy, int]) `epoch` 또는 정수입니다. `'epoch'` 를 사용하면 콜백은 각 에포크 후에 모델을 저장합니다. 정수를 사용하면 콜백은 이 많은 배치가 끝날 때 모델을 저장합니다. `val_acc` 또는 `val_loss` 와 같은 유효성 검사 메트릭을 모니터링할 때 save_freq는 해당 메트릭이 에포크가 끝날 때만 사용할 수 있으므로 "epoch" 로 설정해야 합니다. |
|  `initial_value_threshold` |  (Optional[float]) 모니터링할 메트릭의 부동 소수점 초기 "최고" 값입니다. |

| Attributes |  |
| :--- | :--- |

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

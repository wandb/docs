# WandbModelCheckpoint

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/integration/keras/callbacks/model_checkpoint.py#L27-L195' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

Keras 모델 또는 모델 가중치를 주기적으로 저장하는 체크포인트입니다.

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

저장된 가중치는 `wandb.Artifact`로 W&B에 업로드됩니다.

이 콜백은 `tf.keras.callbacks.ModelCheckpoint`를 서브클래스로 하여, 체크포인트 로직은 부모 콜백에 의해 처리됩니다. 더 많은 정보를 확인하려면 여기로 이동하세요: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint

이 콜백은 `model.fit()`을 사용하여 모델이나 가중치를(체크포인트 파일에) 일정 간격으로 저장할 때 사용됩니다. 모델 체크포인트는 W&B Artifacts로 로그됩니다. 더 알아보려면 여기를 참조하세요: /guides/artifacts

이 콜백은 다음과 같은 기능을 제공합니다:
- "모니터" 기준으로 "최고 성능"을 달성한 모델 저장
- 성능에 상관없이 각 에포크 종료 시 모델 저장
- 에포크 종료 또는 일정 훈련 배치 수 후에 모델 저장
- 모델 가중치만 저장하거나 전체 모델 저장
- 모델을 SavedModel 형식이나 `.h5` 형식으로 저장

| 인수 |  |
| :--- | :--- |
|  `filepath` |  (Union[str, os.PathLike]) 모델 파일을 저장할 경로. `filepath`는 `epoch`와 `logs`의 키( `on_epoch_end`에 전달된)에 의해 채워질 수 있는 서식 옵션을 포함할 수 있습니다. 예: `filepath`가 `model-{epoch:02d}-{val_loss:.2f}`인 경우, 에포크 번호와 검증 손실이 파일 이름에 포함되어 모델 체크포인트가 저장됩니다. |
|  `monitor` |  (str) 모니터링할 메트릭 이름. 기본값은 "val_loss"입니다. |
|  `verbose` |  (int) 상세 모드, 0 또는 1. 모드 0은 무음, 모드 1은 콜백이 작업을 수행할 때 메시지를 표시합니다. |
|  `save_best_only` |  (bool) `save_best_only=True`일 경우, 모델이 "최고"로 간주될 때만 저장됩니다. 모니터링된 양에 따라 최신의 최고 모델은 덮어쓰지 않습니다. 만약 `filepath`가 `{epoch}`와 같은 서식 옵션을 포함하지 않으면, 각 새 더 나은 모델에 의해 `filepath`는 로컬에서 덮어씌워집니다. 아티팩트로 로깅된 모델은 여전히 올바른 `monitor`와 연관됩니다. Artifacts은 지속적으로 업로드되고 새로운 최고 모델이 발견될 때 별도로 버전이 매겨집니다. |
|  `save_weights_only` |  (bool) 참일 경우, 모델의 가중치만 저장됩니다. |
|  `mode` |  (Mode) {'auto', 'min', 'max'} 중 하나. `val_acc`일 경우 `max`로 설정해야 하며, `val_loss`일 경우 `min`으로 설정해야 합니다. |
|  `save_freq` |  (Union[SaveStrategy, int]) `epoch` 또는 정수. `'epoch'`을 사용하면 콜백은 각 에포크 후에 모델을 저장합니다. 정수를 사용하면 이 배치 수만큼 끝날 때 모델을 저장합니다. `val_acc` 또는 `val_loss` 같은 검증 메트릭을 모니터링할 때는 save_freq는 에포크 종료 시점에만 사용 가능하므로 "epoch"으로 설정해야 합니다. |
|  `initial_value_threshold` |  (Optional[float]) 모니터링할 메트릭의 초기 "최고" 값입니다. |

| 속성 |  |
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
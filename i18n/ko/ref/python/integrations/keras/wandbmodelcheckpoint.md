
# WandbModelCheckpoint

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/integration/keras/callbacks/model_checkpoint.py#L27-L200' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>


주기적으로 Keras 모델이나 모델 가중치를 저장하는 체크포인트입니다.

```python
WandbModelCheckpoint(
    filepath: StrPath,
    monitor: str = "val_loss",
    verbose: int = 0,
    save_best_only: bool = (False),
    save_weights_only: bool = (False),
    mode: Mode = "auto",
    save_freq: Union[SaveStrategy, int] = "epoch",
    options: Optional[str] = None,
    initial_value_threshold: Optional[float] = None,
    **kwargs
) -> None
```

저장된 가중치는 W&B에 `wandb.Artifact`로 업로드됩니다.

이 콜백은 `tf.keras.callbacks.ModelCheckpoint`에서 서브클래스화되었기 때문에,
체크포인팅 로직은 부모 콜백에 의해 처리됩니다. 여기에서 더 알아볼 수 있습니다: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint

이 콜백은 `model.fit()`을 사용한 트레이닝과 함께 사용되며, 일정 간격으로 모델이나 가중치(체크포인트 파일에)를 저장합니다. 모델 체크포인트는 W&B Artifacts로 로그됩니다. 여기에서 더 알아볼 수 있습니다:
https://docs.wandb.ai/guides/artifacts

이 콜백은 다음과 같은 기능을 제공합니다:
- "monitor"를 기반으로 "최고 성능"을 달성한 모델 저장
- 성능에 상관없이 매 에포크마다 모델 저장
- 매 에포크 끝이나 일정 수의 트레이닝 배치 후에 모델 저장
- 모델 가중치만 저장하거나 전체 모델을 저장
- 모델을 SavedModel 포맷 또는 `.h5` 포맷으로 저장

| 인수 |  |
| :--- | :--- |
|  `filepath` |  (Union[str, os.PathLike]) 모델 파일을 저장할 경로. `filepath`은 `epoch`와 `logs`의 키 값( `on_epoch_end`에 전달됨)으로 채워질 수 있는 명명된 포맷 옵션을 포함할 수 있습니다. 예를 들어, `filepath`이 `model-{epoch:02d}-{val_loss:.2f}`인 경우, 모델 체크포인트는 에포크 번호와 검증 손실을 파일명에 포함하여 저장됩니다. |
|  `monitor` |  (str) 모니터링할 메트릭 이름. 기본값은 "val_loss". |
|  `verbose` |  (int) 출력 모드, 0 또는 1. 모드 0은 조용한 모드이고, 모드 1은 콜백이 작업을 수행할 때 메시지를 표시합니다. |
|  `save_best_only` |  (bool) `save_best_only=True`인 경우, 모델이 "최고"로 간주될 때만 저장합니다. 그리고 최신의 최고 모델은 모니터링되는 수량에 따라 덮어쓰지 않습니다. `filepath`이 `{epoch}` 같은 포맷 옵션을 포함하지 않으면 `filepath`은 로컬에서 새로운 더 나은 모델마다 덮어쓰게 됩니다. 아티팩트로 로그된 모델은 여전히 올바른 `monitor`와 연관됩니다. 아티팩트는 지속적으로 업로드되며 새로운 최고 모델이 발견될 때마다 별도로 버전 관리됩니다. |
|  `save_weights_only` |  (bool) True인 경우, 모델의 가중치만 저장됩니다. |
|  `mode` |  (Mode) {'auto', 'min', 'max'} 중 하나. `val_acc`의 경우 `max`, `val_loss`의 경우 `min` 등이어야 합니다. |
|  `save_freq` |  (Union[SaveStrategy, int]) `epoch` 또는 정수. `'epoch'`를 사용할 때, 콜백은 각 에포크 후에 모델을 저장합니다. 정수를 사용할 때, 콜백은 이 많은 배치의 끝에 모델을 저장합니다. `val_acc`나 `val_loss` 같은 검증 메트릭을 모니터링할 때, save_freq는 "epoch"으로 설정되어야 합니다. 그렇지 않으면 해당 메트릭은 에포크 끝에서만 사용할 수 있습니다. |
|  `options` |  (Optional[str]) `save_weights_only`가 참인 경우 선택적 `tf.train.CheckpointOptions` 오브젝트 또는 `save_weights_only`가 거짓인 경우 선택적 `tf.saved_model.SaveOptions` 오브젝트입니다. |
|  `initial_value_threshold` |  (Optional[float]) 모니터링되는 메트릭의 초기 "최고" 값의 부동 소수점입니다. |

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
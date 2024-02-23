
# WandbModelCheckpoint

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/fa4423647026d710e3780287b4bac2ee9494e92b/wandb/integration/keras/callbacks/model_checkpoint.py#L27-L200' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>


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
    options: Optional[str] = None,
    initial_value_threshold: Optional[float] = None,
    **kwargs
) -> None
```

저장된 가중치는 `wandb.Artifact`로 W&B에 업로드됩니다.

이 콜백은 `tf.keras.callbacks.ModelCheckpoint`에서 파생되었기 때문에, 
체크포인트 로직은 부모 콜백에 의해 처리됩니다. 여기에서 더 자세히 알아볼 수 있습니다: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint

이 콜백은 `model.fit()`을 사용한 학습과 함께 사용되어, 일정 간격으로 모델 또는 가중치(체크포인트 파일에서)를 저장합니다. 모델 체크포인트는 W&B 아티팩트로 로그됩니다. 여기에서 더 자세히 알아볼 수 있습니다:
https://docs.wandb.ai/guides/artifacts

이 콜백은 다음과 같은 기능을 제공합니다:
- "monitor"를 기반으로 "최고 성능"을 달성한 모델 저장
- 성능에 관계없이 매 에포크마다 모델 저장
- 에포크마다 또는 정해진 수의 학습 배치 후에 모델 저장
- 모델 가중치만 저장하거나 전체 모델 저장
- 모델을 SavedModel 형식 또는 `.h5` 형식으로 저장

| 인수 |  |
| :--- | :--- |
|  `filepath` |  (Union[str, os.PathLike]) 모델 파일을 저장할 경로. `filepath`은 `epoch`의 값과 `logs`의 키( `on_epoch_end`에서 전달됨)에 의해 채워질 수 있는 명명된 포맷 옵션을 포함할 수 있습니다. 예를 들어, `filepath`이 `model-{epoch:02d}-{val_loss:.2f}`인 경우, 모델 체크포인트는 에포크 번호와 검증 손실을 파일 이름에 포함하여 저장됩니다. |
|  `monitor` |  (str) 모니터링할 메트릭 이름. 기본값은 "val_loss". |
|  `verbose` |  (int) 상세 모드, 0 또는 1. 모드 0은 조용하며, 모드 1은 콜백이 작업을 수행할 때 메시지를 표시합니다. |
|  `save_best_only` |  (bool) `save_best_only=True`이면, 모델이 "최고"로 간주될 때만 저장됩니다. `filepath`이 `{epoch}`과 같은 포맷 옵션을 포함하지 않으면 `filepath`은 로컬에서 새로운 좋은 모델로 매번 덮어씌워집니다. 아티팩트로 로그된 모델은 여전히 올바른 `monitor`와 연결될 것입니다. 아티팩트는 지속적으로 업로드되며 새로운 최고 모델이 발견될 때마다 별도로 버전이 지정됩니다. |
|  `save_weights_only` |  (bool) True이면, 모델의 가중치만 저장됩니다. |
|  `mode` |  (Mode) {'auto', 'min', 'max'} 중 하나. `val_acc`의 경우 `max`여야 하며, `val_loss`의 경우 `min`이어야 합니다. |
|  `save_freq` |  (Union[SaveStrategy, int]) `epoch` 또는 정수. `'epoch'`를 사용할 때, 콜백은 매 에포크 후에 모델을 저장합니다. 정수를 사용할 때, 콜백은 이 수많은 배치의 끝에 모델을 저장합니다. `val_acc` 또는 `val_loss`와 같은 검증 메트릭을 모니터링할 때는 save_freq를 "epoch"로 설정해야 합니다. |
|  `options` |  (Optional[str]) `save_weights_only`가 true인 경우 선택적 `tf.train.CheckpointOptions` 개체 또는 `save_weights_only`가 false인 경우 선택적 `tf.saved_model.SaveOptions` 개체. |
|  `initial_value_threshold` |  (Optional[float]) 모니터링할 메트릭의 초기 "최고" 값의 부동 소수점. |

| 속성 |  |
| :--- | :--- |

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
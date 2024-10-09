# WandbCallback

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/integration/keras/keras.py#L291-L1091' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

`WandbCallback`은 keras와 wandb를 자동으로 통합합니다.

```python
WandbCallback(
    monitor="val_loss", verbose=0, mode="auto", save_weights_only=(False),
    log_weights=(False), log_gradients=(False), save_model=(True),
    training_data=None, validation_data=None, labels=None, predictions=36,
    generator=None, input_type=None, output_type=None, log_evaluation=(False),
    validation_steps=None, class_colors=None, log_batch_frequency=None,
    log_best_prefix="best_", save_graph=(True), validation_indexes=None,
    validation_row_processor=None, prediction_row_processor=None,
    infer_missing_processors=(True), log_evaluation_frequency=0,
    compute_flops=(False), **kwargs
)
```

#### 예시:

```python
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    callbacks=[WandbCallback()],
)
```

`WandbCallback`은 keras에서 수집한 모든 메트릭의 기록 데이터를 자동으로 로그합니다: 손실 및 `keras_model.compile()`에 전달된 항목들.

`WandbCallback`은 "최고" 트레이닝 단계에 대한 런의 요약 메트릭을 설정합니다. 여기서 "최고"는 `monitor`와 `mode` 속성에 의해 정의되며, 기본적으로는 최소 `val_loss`를 가진 에포크입니다. `WandbCallback`은 기본적으로 최고의 `epoch`와 연관된 모델을 저장합니다.

`WandbCallback`은 옵션으로 그레이디언트 및 파라미터 히스토그램을 로그할 수 있습니다.

`WandbCallback`은 wandb가 시각화할 수 있도록 트레이닝 및 검증 데이터를 저장할 수 있습니다.

| 인수 |  |
| :--- | :--- |
|  `monitor` |  (문자열) 모니터링할 메트릭의 이름입니다. 기본값은 `val_loss`. |
|  `mode` |  (문자열) {`auto`, `min`, `max`} 중 하나입니다. `min` - 모니터가 최소화될 때 모델을 저장합니다. `max` - 모니터가 최대화될 때 모델을 저장합니다. `auto` - 모델을 저장할 시기를 예측하려고 시도합니다 (기본값). |
|  `save_model` |  True - 모니터가 이전 에포크를 초과할 때 모델을 저장합니다. False - 모델을 저장하지 않습니다. |
|  `save_graph` |  (부울) True일 경우 모델 그래프를 wandb에 저장합니다 (기본값은 True). |
|  `save_weights_only` |  (부울) True일 경우, 모델의 가중치만 저장됩니다 (`model.save_weights(filepath)`). 그렇지 않으면 전체 모델이 저장됩니다 (`model.save(filepath)`). |
|  `log_weights` |  (부울) True일 경우, 모델의 레이어 가중치의 히스토그램을 저장합니다. |
|  `log_gradients` |  (부울) True일 경우, 트레이닝 그레이디언트의 히스토그램을 로그합니다. |
|  `training_data` |  (튜플) `model.fit`에 전달된 `(X,y)` 형식과 동일합니다. 그레이디언트를 계산하기 위해 필요합니다. `log_gradients`가 `True`인 경우 필수입니다. |
|  `validation_data` |  (튜플) `model.fit`에 전달된 `(X,y)` 형식과 동일합니다. wandb가 시각화할 데이터 세트입니다. 이 설정이 있으면, 매 에포크마다 wandb는 약간의 예측값을 만들어 나중에 시각화할 결과를 저장합니다. 이미지 데이터를 사용할 경우, 올바르게 로그하기 위해 `input_type`과 `output_type`도 설정하세요. |
|  `generator` |  (생성기) wandb가 시각화할 검증 데이터를 반환하는 생성기입니다. 이 생성기는 `(X,y)` 튜플을 반환해야 합니다. `validate_data` 또는 생성기가 wandb가 특정 데이터 예제를 시각화하도록 설정되어야 합니다. 이미지 데이터를 사용할 경우, `input_type`과 `output_type`도 설정하여 올바르게 로그하세요. |
|  `validation_steps` |  (정수) `validation_data`가 생성기일 경우, 전체 검증 세트를 위해 생성기를 실행할 단계 수입니다. |
|  `labels` |  (목록) wandb로 데이터를 시각화할 경우, 이 레이블 목록은 다중 클래스 분류기를 구축할 때 수치 출력을 이해 가능한 문자열로 변환합니다. 이진 분류기를 만들고 있는 경우, 두 레이블 ["거짓에 대한 레이블", "참에 대한 레이블"] 목록을 전달할 수 있습니다. `validate_data`와 생성기가 모두 거짓인 경우, 아무 효과도 없습니다. |
|  `predictions` |  (정수) 매 에포크마다 시각화를 위한 예측 횟수, 최대 100입니다. |
|  `input_type` |  (문자열) 시각화를 돕기 위한 모델 입력 타입입니다. 다음 중 하나일 수 있습니다: (`image`, `images`, `segmentation_mask`, `auto`). |
|  `output_type` |  (문자열) 시각화를 돕기 위한 모델 출력 타입입니다. 다음 중 하나일 수 있습니다: (`image`, `images`, `segmentation_mask`, `label`). |
|  `log_evaluation` |  (부울) True면 매 에포크에 검증 데이터와 모델의 예측을 포함하는 테이블을 저장합니다. 추가 세부정보는 `validation_indexes`, `validation_row_processor`, 및 `output_row_processor`를 참조하세요. |
|  `class_colors` |  ([실수, 실수, 실수]) 입력 또는 출력이 세분화 마스크일 경우, 각 클래스에 대한 RGB 튜플(범위 0-1)을 포함하는 배열입니다. |
|  `log_batch_frequency` |  (정수) None일 경우, 콜백은 매 에포크마다 로그합니다. 정수로 설정된 경우, 콜백은 `log_batch_frequency` 배치마다 트레이닝 메트릭을 로그합니다. |
|  `log_best_prefix` |  (문자열) None일 경우, 추가 요약 메트릭이 저장되지 않습니다. 문자열로 설정된 경우, 모니터링하는 메트릭과 에포크는 이 값으로 시작하여 요약 메트릭으로 저장됩니다. |
|  `validation_indexes` |  ([wandb.data_types._TableLinkMixin]) 각 검증 예제와 연결할 인덱스 키의 정렬된 목록입니다. log_evaluation이 True고 `validation_indexes`가 제공된 경우, 검증 데이터의 테이블은 생성되지 않으며 각 예측은 `TableLinkMixin`이 나타내는 행과 연결됩니다. 이 키를 얻는 가장 일반적인 방법은 `Table.get_index()`를 사용하여 행 키의 목록을 반환하는 것입니다. |
|  `validation_row_processor` |  (Callable) 검증 데이터에 적용할 함수로, 데이터를 시각화하는 데 자주 사용됩니다. 함수는 `ndx` (정수)와 `row` (딕셔너리)를 받습니다. 모델에 단일 입력이 있는 경우, `row["input"]`은 행의 입력 데이터가 됩니다. 그렇지 않은 경우, 입력 슬롯의 이름을 기준으로 키가 지정됩니다. 맞춤 함수가 단일 목표를 포함할 경우, `row["target"]`는 행의 목표 데이터입니다. 그렇지 않으면 출력 슬롯의 이름을 기준으로 키가 지정됩니다. 예를 들어, 입력 데이터가 단일 ndarray인 경우, 데이터를 이미지로 시각화하려면 `lambda ndx, row: {"img": wandb.Image(row["input"])}`와 같은 프로세서를 제공할 수 있습니다. log_evaluation이 False거나 `validation_indexes`가 있는 경우 무시됩니다. |
|  `output_row_processor` |  (Callable) `validation_row_processor`와 동일하나, 모델 출력에 적용됩니다. `row["output"]`는 모델 출력의 결과를 포함합니다. |
|  `infer_missing_processors` |  (bool) `validation_row_processor`와 `output_row_processor`가 누락된 경우 추론할지 여부를 결정합니다. 기본값은 True입니다. `labels`가 제공된 경우, 적절한 곳에서 분류 유형 프로세서를 추론하려고 시도합니다. |
|  `log_evaluation_frequency` |  (정수) 평가 결과를 로그할 빈도를 결정합니다. 기본값 0 (트레이닝 끝에만 한 번). 1로 설정하면 매 에포크마다 로그하고, 2로 설정하면 매 에포크 마다 로그합니다. log_evaluation이 False면 아무 효과도 없습니다. |
|  `compute_flops` |  (bool) Keras Sequential 또는 Functional 모델의 FLOPs를 GigaFLOPs 단위로 계산합니다. |

## 메소드

### `get_flops`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/integration/keras/keras.py#L1045-L1091)

```python
get_flops() -> float
```

추론 모드에서 tf.keras.Model 또는 tf.keras.Sequential 모델에 대한 FLOPS [GFLOPs]를 계산합니다.

본질적으로 tf.compat.v1.profiler를 사용합니다.

### `set_model`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/integration/keras/keras.py#L565-L574)

```python
set_model(
    model
)
```

### `set_params`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.18.0/wandb/integration/keras/keras.py#L562-L563)

```python
set_params(
    params
)
```
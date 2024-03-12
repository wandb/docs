
# WandbCallback

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/integration/keras/keras.py#L291-L1080' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>GitHub에서 소스 보기</a></button></p>

`WandbCallback`은 keras를 자동으로 wandb와 통합합니다.

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

#### 예제:

```python
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    callbacks=[WandbCallback()],
)
```

`WandbCallback`은 keras가 수집한 모든 메트릭의 히스토리 데이터를 자동으로 기록합니다: loss 및 `keras_model.compile()`로 전달된 모든 것.

`WandbCallback`은 `monitor`와 `mode` 속성에 의해 정의된 "최고"의 트레이닝 스텝과 연관된 run의 요약 메트릭을 설정합니다. 이는 기본적으로 `val_loss`가 최소인 에포크입니다. `WandbCallback`은 기본적으로 최고의 `epoch`와 연관된 모델을 저장합니다.

`WandbCallback`은 선택적으로 그레이디언트와 파라미터 히스토그램을 기록할 수 있습니다.

`WandbCallback`은 선택적으로 트레이닝 및 검증 데이터를 저장하여 wandb에서 시각화할 수 있습니다.

| 인수 |  |
| :--- | :--- |
|  `monitor` |  (str) 모니터할 메트릭의 이름. 기본값은 `val_loss`. |
|  `mode` |  (str) {`auto`, `min`, `max`} 중 하나. `min` - 모니터가 최소화될 때 모델 저장 `max` - 모니터가 최대화될 때 모델 저장 `auto` - 모델을 저장할 때를 추측하려고 시도합니다 (기본값). |
|  `save_model` |  True - 모니터가 이전 에포크를 모두 초과할 때 모델 저장 False - 모델 저장하지 않음 |
|  `save_graph` |  (boolean) True이면 wandb에 모델 그래프 저장 (기본값 True). |
|  `save_weights_only` |  (boolean) True이면 모델의 가중치만 저장됩니다 (`model.save_weights(filepath)`), 그렇지 않으면 전체 모델이 저장됩니다 (`model.save(filepath)`). |
|  `log_weights` |  (boolean) True이면 모델 레이어의 가중치 히스토그램을 저장합니다. |
|  `log_gradients` |  (boolean) True이면 트레이닝 그레이디언트의 히스토그램을 기록합니다 |
|  `training_data` |  (tuple) `model.fit`에 전달된 것과 동일한 형식 `(X,y)`. 이는 그레이디언트를 계산하기 위해 필요하며, `log_gradients`가 `True`인 경우 필수입니다. |
|  `validation_data` |  (tuple) `model.fit`에 전달된 것과 동일한 형식 `(X,y)`. wandb가 시각화할 데이터 세트입니다. 이 값이 설정되면, 매 에포크마다 wandb는 소수의 예측을 수행하고 결과를 나중에 시각화를 위해 저장합니다. 이미지 데이터를 다루는 경우, 올바르게 기록하기 위해 `input_type` 및 `output_type`도 설정해야 합니다. |
|  `generator` |  (generator) wandb가 시각화할 검증 데이터를 반환하는 제너레이터. 이 제너레이터는 튜플 `(X,y)`를 반환해야 합니다. `validate_data` 또는 제너레이터 중 하나가 설정되어 있어야 wandb가 특정 데이터 예시를 시각화할 수 있습니다. 이미지 데이터를 다루는 경우, 올바르게 기록하기 위해 `input_type` 및 `output_type`도 설정해야 합니다. |
|  `validation_steps` |  (int) `validation_data`가 제너레이터인 경우, 전체 검증 세트에 대해 제너레이터를 실행할 단계 수입니다. |
|  `labels` |  (list) 데이터를 wandb로 시각화하는 경우, 이 레이블 목록은 숫자 출력을 이해하기 쉬운 문자열로 변환합니다. 이진 분류기를 만들고 있다면 ["false에 대한 레이블", "true에 대한 레이블"]의 두 레이블 목록을 전달할 수 있습니다. `validate_data`와 제너레이터가 모두 거짓이면 아무 작용도 하지 않습니다. |
|  `predictions` |  (int) 각 에포크마다 시각화를 위해 수행할 예측의 수, 최대 100입니다. |
|  `input_type` |  (string) 시각화를 돕기 위한 모델 입력의 유형. 다음 중 하나일 수 있습니다: (`image`, `images`, `segmentation_mask`, `auto`). |
|  `output_type` |  (string) 시각화를 돕기 위한 모델 출력의 유형. 다음 중 하나일 수 있습니다: (`image`, `images`, `segmentation_mask`, `label`). |
|  `log_evaluation` |  (boolean) True이면, 각 에포크에서 검증 데이터와 모델의 예측값을 포함하는 테이블을 저장합니다. 추가 세부 사항은 `validation_indexes`, `validation_row_processor`, 및 `output_row_processor`를 참조하세요. |
|  `class_colors` |  ([float, float, float]) 입력 또는 출력이 세그멘테이션 마스크인 경우, 각 클래스에 대한 rgb 튜플(범위 0-1)을 포함하는 배열입니다. |
|  `log_batch_frequency` |  (integer) None이면, 콜백은 매 에포크마다 로그를 기록합니다. 정수로 설정된 경우, 콜백은 `log_batch_frequency` 배치마다 트레이닝 메트릭을 기록합니다. |
|  `log_best_prefix` |  (string) None이면, 추가 요약 메트릭이 저장되지 않습니다. 문자열로 설정된 경우, 모니터링된 메트릭과 에포크는 이 값으로 시작하며 요약 메트릭으로 저장됩니다. |
|  `validation_indexes` |  ([wandb.data_types._TableLinkMixin]) 각 검증 예제와 연관된 인덱스 키의 정렬된 목록입니다. log_evaluation이 True이고 `validation_indexes`가 제공되면 검증 데이터의 테이블이 생성되지 않고 대신 각 예측이 `TableLinkMixin`에 의해 표현된 행과 연관됩니다. 이러한 키를 얻는 가장 일반적인 방법은 `Table.get_index()`를 사용하는 것으로, 행 키 목록을 반환합니다. |
|  `validation_row_processor` |  (Callable) 검증 데이터에 적용할 함수로, 일반적으로 데이터를 시각화하는 데 사용됩니다. 함수는 `ndx` (int)와 `row` (dict)를 받습니다. 모델의 입력이 하나만 있는 경우, `row["input"]`은 행의 입력 데이터가 됩니다. 그렇지 않으면 입력 슬롯의 이름을 기준으로 키가 지정됩니다. fit 함수가 단일 대상을 사용하는 경우, `row["target"]`은 행의 대상 데이터가 됩니다. 그렇지 않으면 출력 슬롯의 이름을 기준으로 키가 지정됩니다. 예를 들어, 입력 데이터가 단일 ndarray이지만 데이터를 Image로 시각화하려는 경우, 프로세서로 `lambda ndx, row: {"img": wandb.Image(row["input"])}`를 제공할 수 있습니다. log_evaluation이 False이거나 `validation_indexes`가 있으면 무시됩니다. |
|  `output_row_processor` |  (Callable) `validation_row_processor`와 동일하지만 모델의 출력에 적용됩니다. `row["output"]`은 모델 출력의 결과를 포함할 것입니다. |
|  `infer_missing_processors` |  (bool) `validation_row_processor` 및 `output_row_processor`가 누락된 경우 추론할지 여부를 결정합니다. 기본값은 True입니다. `labels`가 제공되면, 우리는 적절한 경우 분류 유형 프로세서를 추론하려고 시도할 것입니다. |
|  `log_evaluation_frequency` |  (int) 평가 결과가 기록되는 빈도를 결정합니다. 기본값은 0입니다(트레이닝이 끝날 때만). 매 에포크마다 기록하려면 1로 설정하고, 격 에포크마다 기록하려면 2로 설정하고, 그 이상으로 설정합니다. log_evaluation이 False일 때는 영향이 없습니다. |
|  `compute_flops` |  (bool) Keras Sequential 또는 Functional 모델의 FLOPs를 GigaFLOPs 단위로 계산합니다. |

## 메소드

### `get_flops`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/integration/keras/keras.py#L1034-L1080)

```python
get_flops() -> float
```

추론 모드에서 tf.keras.Model 또는 tf.keras.Sequential 모델의 FLOPS [GFLOPs]를 계산합니다.

내부적으로 tf.compat.v1.profiler를 사용합니다.

### `set_model`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/integration/keras/keras.py#L554-L563)

```python
set_model(
    model
)
```

### `set_params`

[소스 보기](https://www.github.com/wandb/wandb/tree/v0.16.4/wandb/integration/keras/keras.py#L551-L552)

```python
set_params(
    params
)
```
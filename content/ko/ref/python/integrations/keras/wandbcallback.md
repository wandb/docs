---
title: WandbCallback
menu:
  reference:
    identifier: ko-ref-python-integrations-keras-wandbcallback
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/keras.py#L291-L1091 >}}

`WandbCallback`은 keras를 wandb와 자동으로 통합합니다.

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

`WandbCallback`은 keras에서 수집한 모든 메트릭에서 히스토리 데이터를 자동으로 로깅합니다. 손실 및 `keras_model.compile()`에 전달된 모든 것.

`WandbCallback`은 "최고" 트레이닝 단계와 관련된 run에 대한 요약 메트릭을 설정합니다.
여기서 "최고"는 `monitor` 및 `mode` 속성에 의해 정의됩니다. 이것은 기본적으로 최소 `val_loss`를 갖는 에포크입니다. `WandbCallback`은 기본적으로 최고의 `epoch`와 관련된 모델을 저장합니다.

`WandbCallback`은 선택적으로 그레이디언트 및 파라미터 히스토그램을 로깅할 수 있습니다.

`WandbCallback`은 시각화를 위해 트레이닝 및 검증 데이터를 wandb에 선택적으로 저장할 수 있습니다.

| Args |  |
| :--- | :--- |
|  `monitor` | (str) 모니터링할 메트릭 이름. 기본값은 `val_loss`입니다. |
|  `mode` | (str) {`auto`, `min`, `max`} 중 하나입니다. `min` - 모니터가 최소화될 때 모델 저장 `max` - 모니터가 최대화될 때 모델 저장 `auto` - 모델을 저장할 시기를 추측하려고 시도합니다(기본값). |
|  `save_model` | True - 모니터가 이전의 모든 에포크보다 좋을 때 모델을 저장합니다. False - 모델을 저장하지 않습니다. |
|  `save_graph` | (boolean) True인 경우 모델 그래프를 wandb에 저장합니다 (기본값은 True). |
|  `save_weights_only` | (boolean) True인 경우 모델의 가중치만 저장됩니다 (`model.save_weights(filepath)`), 그렇지 않으면 전체 모델이 저장됩니다 (`model.save(filepath)`). |
|  `log_weights` | (boolean) True인 경우 모델 레이어 가중치의 히스토그램을 저장합니다. |
|  `log_gradients` | (boolean) True인 경우 트레이닝 그레이디언트의 히스토그램을 로깅합니다. |
|  `training_data` | `model.fit`에 전달된 것과 동일한 형식 `(X,y)`입니다. 이것은 그레이디언트를 계산하는 데 필요합니다. `log_gradients`가 `True`인 경우 필수입니다. |
|  `validation_data` | `model.fit`에 전달된 것과 동일한 형식 `(X,y)`입니다. wandb가 시각화할 데이터 세트입니다. 이것이 설정되면 매 에포크마다 wandb는 적은 수의 예측값을 만들고 나중에 시각화할 수 있도록 결과를 저장합니다. 이미지 데이터를 사용하는 경우 올바르게 로깅하려면 `input_type` 및 `output_type`도 설정하십시오. |
|  `generator` | wandb가 시각화할 검증 데이터를 반환하는 제너레이터입니다. 이 제너레이터는 튜플 `(X,y)`를 반환해야 합니다. wandb가 특정 데이터 예제를 시각화하려면 `validate_data` 또는 제너레이터를 설정해야 합니다. 이미지 데이터를 사용하는 경우 올바르게 로깅하려면 `input_type` 및 `output_type`도 설정하십시오. |
|  `validation_steps` | `validation_data`가 제너레이터인 경우 전체 검증 세트에 대해 제너레이터를 실행할 단계 수입니다. |
|  `labels` | wandb로 데이터를 시각화하는 경우 이 레이블 목록은 다중 클래스 분류기를 구축하는 경우 숫자 출력을 이해 가능한 문자열로 변환합니다. 이진 분류기를 만드는 경우 두 개의 레이블 목록 ["false에 대한 레이블", "true에 대한 레이블"]을 전달할 수 있습니다. `validate_data`와 제너레이터가 모두 false인 경우 아무 작업도 수행하지 않습니다. |
|  `predictions` | 각 에포크에서 시각화를 위해 만들 예측 수이며 최대값은 100입니다. |
|  `input_type` | 시각화를 돕기 위한 모델 입력 유형입니다. 다음 중 하나일 수 있습니다. (`image`, `images`, `segmentation_mask`, `auto`). |
|  `output_type` | 시각화를 돕기 위한 모델 출력 유형입니다. 다음 중 하나일 수 있습니다. (`image`, `images`, `segmentation_mask`, `label`). |
|  `log_evaluation` | True인 경우 각 에포크에서 검증 데이터와 모델의 예측값을 포함하는 Table을 저장합니다. 자세한 내용은 `validation_indexes`, `validation_row_processor` 및 `output_row_processor`를 참조하십시오. |
|  `class_colors` | 입력 또는 출력이 세분화 마스크인 경우 각 클래스에 대한 rgb 튜플(범위 0-1)을 포함하는 배열입니다. |
|  `log_batch_frequency` | None인 경우 콜백은 모든 에포크를 로깅합니다. 정수로 설정하면 콜백은 `log_batch_frequency` 배치마다 트레이닝 메트릭을 로깅합니다. |
|  `log_best_prefix` | None인 경우 추가 요약 메트릭이 저장되지 않습니다. 문자열로 설정하면 모니터링되는 메트릭과 에포크가 이 값으로 시작하여 요약 메트릭으로 저장됩니다. |
|  `validation_indexes` | 각 검증 예제와 연결할 인덱스 키의 정렬된 목록입니다. log_evaluation이 True이고 `validation_indexes`가 제공되면 검증 데이터의 Table이 생성되지 않고 대신 각 예측이 `TableLinkMixin`으로 표시되는 행과 연결됩니다. 이러한 키를 얻는 가장 일반적인 방법은 행 키 목록을 반환하는 `Table.get_index()`를 사용하는 것입니다. |
|  `validation_row_processor` | 검증 데이터에 적용할 함수로, 일반적으로 데이터를 시각화하는 데 사용됩니다. 함수는 `ndx`(int)와 `row`(dict)를 받습니다. 모델에 단일 입력이 있는 경우 `row["input"]`은 행에 대한 입력 데이터가 됩니다. 그렇지 않으면 입력 슬롯의 이름을 기반으로 키가 지정됩니다. 적합 함수가 단일 대상을 사용하는 경우 `row["target"]`은 행에 대한 대상 데이터가 됩니다. 그렇지 않으면 출력 슬롯의 이름을 기반으로 키가 지정됩니다. 예를 들어 입력 데이터가 단일 ndarray이지만 데이터를 이미지로 시각화하려는 경우 `lambda ndx, row: {"img": wandb.Image(row["input"])}`를 프로세서로 제공할 수 있습니다. log_evaluation이 False이거나 `validation_indexes`가 있는 경우 무시됩니다. |
|  `output_row_processor` | `validation_row_processor`와 동일하지만 모델의 출력에 적용됩니다. `row["output"]`에는 모델 출력 결과가 포함됩니다. |
|  `infer_missing_processors` | 누락된 경우 `validation_row_processor` 및 `output_row_processor`를 추론해야 하는지 여부를 결정합니다. 기본값은 True입니다. `labels`가 제공되면 적절한 분류 유형 프로세서를 추론하려고 시도합니다. |
|  `log_evaluation_frequency` | 평가 결과를 로깅할 빈도를 결정합니다. 기본값은 0입니다(트레이닝 종료 시에만). 모든 에포크를 로깅하려면 1로 설정하고, 다른 모든 에포크를 로깅하려면 2로 설정하는 식입니다. log_evaluation이 False인 경우에는 효과가 없습니다. |
|  `compute_flops` | Keras Sequential 또는 Functional 모델의 FLOP 수를 GigaFLOPs 단위로 계산합니다. |

## Methods

### `get_flops`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/keras.py#L1045-L1091)

```python
get_flops() -> float
```

추론 모드에서 tf.keras.Model 또는 tf.keras.Sequential 모델에 대한 FLOPS [GFLOPs]를 계산합니다.

내부적으로 tf.compat.v1.profiler를 사용합니다.

### `set_model`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/keras.py#L567-L576)

```python
set_model(
    model
)
```

### `set_params`

[View source](https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/integration/keras/keras.py#L564-L565)

```python
set_params(
    params
)
```

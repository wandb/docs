---
displayed_sidebar: default
---

# Keras

[**여기에서 Colab 노트북으로 시도해보세요 →**](http://wandb.me/intro-keras)

## Weights & Biases Keras 콜백

우리는 Keras와 TensorFlow 사용자를 위해 세 가지 새로운 콜백을 추가했으며, `wandb` v0.13.4부터 사용할 수 있습니다. 기존의 `WandbCallback`에 대해서는 아래로 스크롤하세요.

**`WandbMetricsLogger`** : [실험 추적](https://docs.wandb.ai/guides/track)을 위해 이 콜백을 사용하세요. 이 콜백은 학습 및 검증 메트릭과 함께 시스템 메트릭을 Weights and Biases에 로그합니다.

**`WandbModelCheckpoint`** : 모델 체크포인트를 Weights and Biases [아티팩트](https://docs.wandb.ai/guides/data-and-model-versioning)에 로그하기 위해 이 콜백을 사용하세요.

**`WandbEvalCallback`**: 이 기본 콜백은 모델 예측값을 Weights and Biases [테이블](https://docs.wandb.ai/guides/tables)에 로그하여 상호 작용 가능한 시각화를 제공합니다.

이 새로운 콜백들은,

* Keras 디자인 철학을 준수합니다
* 단일 콜백(`WandbCallback`)을 사용하여 모든 것을 처리하는 인지 부하를 줄입니다,
* Keras 사용자가 자신의 특정 사용 사례를 지원하기 위해 콜백을 서브클래싱하여 수정하기 쉽게 합니다.

## `WandbMetricsLogger`를 사용한 실험 추적

[**여기에서 Colab 노트북으로 시도해보세요 →**](https://github.com/wandb/examples/blob/master/colabs/keras/Use\_WandbMetricLogger\_in\_your\_Keras\_workflow.ipynb)

`WandbMetricsLogger`는 `on_epoch_end`, `on_batch_end` 등의 콜백 메서드가 인수로 사용하는 Keras의 `logs` 사전을 자동으로 로그합니다.

이를 사용하면 다음을 제공합니다:

* `model.compile`에서 정의한 학습 및 검증 메트릭
* 시스템(CPU/GPU/TPU) 메트릭
* 학습률(고정 값이나 학습률 스케줄러 모두에 대해)

```python
import wandb
from wandb.keras import WandbMetricsLogger

# 새로운 W&B 실행 초기화
wandb.init(config={"bs": 12})

# model.fit에 WandbMetricsLogger를 전달
model.fit(
    X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbMetricsLogger()]
)
```

**`WandbMetricsLogger` 참조**


| 파라미터 | 설명 | 
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `log_freq`            | ("epoch", "batch", 혹은 int): "epoch"인 경우, 각 에포크의 끝에 메트릭을 로그합니다. "batch"인 경우, 각 배치의 끝에 메트릭을 로그합니다. int인 경우, 그 많은 배치의 끝에 메트릭을 로그합니다. 기본값은 "epoch"입니다.                                 |
| `initial_global_step` | (int): 어떤 initial_epoch에서 학습을 재개할 때 학습률 스케줄러가 사용되는 경우 학습률을 올바르게 로그하기 위해 이 인수를 사용하세요. 이는 step_size * initial_step으로 계산할 수 있습니다. 기본값은 0입니다. |

## `WandbModelCheckpoint`를 사용한 모델 체크포인트

[**여기에서 Colab 노트북으로 시도해보세요 →**](https://github.com/wandb/examples/blob/master/colabs/keras/Use\_WandbModelCheckpoint\_in\_your\_Keras\_workflow.ipynb)

`WandbModelCheckpoint` 콜백을 사용하여 Keras 모델(`SavedModel` 형식) 또는 모델 가중치를 주기적으로 저장하고 이를 W&B에 `wandb.Artifact`로 업로드하여 모델 버전 관리를 합니다.

이 콜백은 [`tf.keras.callbacks.ModelCheckpoint`](https://www.tensorflow.org/api\_docs/python/tf/keras/callbacks/ModelCheckpoint)에서 서브클래스화되었으므로, 체크포인트 로직은 부모 콜백에 의해 처리됩니다.

이 콜백은 다음 기능을 제공합니다:

* "모니터"를 기반으로 "최고 성능"을 달성한 모델 저장.
* 성능에 상관없이 매 에포크마다 모델 저장.
* 매 에포크 끝이나 일정한 훈련 배치 수 후에 모델 저장.
* 모델 가중치만 저장하거나 전체 모델 저장.
* 모델을 SavedModel 형식 또는 `.h5` 형식으로 저장.

이 콜백은 `WandbMetricsLogger`와 함께 사용해야 합니다.

```python
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

# 새로운 W&B 실행 초기화
wandb.init(config={"bs": 12})

# model.fit에 WandbModelCheckpoint를 전달
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    callbacks=[
        WandbMetricsLogger(),
        WandbModelCheckpoint("models"),
    ],
)
```

**`WandbModelCheckpoint` 참조**

| 파라미터 | 설명 | 
| ------------------------- |  ---- | 
| `filepath`   | (str): 모델 파일을 저장할 경로.|  
| `monitor`                 | (str): 모니터링할 메트릭 이름.         |
| `verbose`                 | (int): 메시지 모드, 0 또는 1. 모드 0은 조용하고, 모드 1은 콜백이 작업을 수행할 때 메시지를 표시합니다.   |
| `save_best_only`          | (bool): `save_best_only=True`인 경우, 모델이 "최고"로 간주될 때만 저장합니다. 모니터링되는 수량(`monitor`)에 따라 최신 최고 모델이 덮어쓰여지지 않습니다.     |
| `save_weights_only`       | (bool): True인 경우, 모델의 가중치만 저장됩니다.                                            |
| `mode`                    | ("auto", "min", 혹은 "max"): val_acc의 경우 ‘max’, val_loss의 경우 ‘min’ 등이어야 합니다.  |
| `save_weights_only`       | (bool): True인 경우, 모델의 가중치만 저장됩니다.                                            |
| `save_freq`               | ("epoch" 혹은 int): ‘epoch’을 사용할 때, 콜백은 각 에포크 후에 모델을 저장합니다. 정수를 사용할 때, 콜백은 이 많은 배치의 끝에 모델을 저장합니다. `val_acc`나 `val_loss`와 같은 검증 메트릭을 모니터링할 때, `save_freq`는 "epoch"으로 설정해야 합니다. 왜냐하면 이러한 메트릭은 에포크의 끝에서만 사용할 수 있기 때문입니다. |
| `options`                 | (str): `save_weights_only`가 true인 경우 선택적 `tf.train.CheckpointOptions` 개체 또는 `save_weights_only`가 false인 경우 선택적 `tf.saved_model.SaveOptions` 개체.    |
| `initial_value_threshold` | (float): 모니터링할 메트릭의 초기 "최고" 값의 부동 소수점입니다.       |

### N 에포크 후에 체크포인트를 로그하는 방법은?

기본값(`save_freq="epoch"`)으로 콜백은 각 에포크 후에 체크포인트를 생성하고 아티팩트로 업로드합니다. `save_freq`에 정수를 전달하면 그 많은 배치 후에 체크포인트가 생성됩니다. `N` 에포크 후에 체크포인트를 생성하려면, 학습 데이터로더의 카디널리티를 계산하고 `save_freq`에 전달하세요:

```
WandbModelCheckpoint(
    filepath="models/",
    save_freq=int((trainloader.cardinality()*N).numpy())
)
```

### TPU 노드 아키텍처에서 효율적으로 체크포인트를 로그하는 방법은?

TPU에서 체크포인트를 생성할 때 `UnimplementedError: File system scheme '[local]' not implemented` 오류 메시지를 마주칠 수 있습니다. 이는 모델 디렉터리(`filepath`)가 클라우드 스토리지 버킷 경로(`gs://bucket-name/...`)를 사용해야 하며, 이 버킷은 TPU 서버에서 접근 가능해야 합니다. 하지만, 로컬 경로를 사용하여 체크포인트를 생성한 후 아티팩트로 업로드할 수 있습니다.

```
checkpoint_options = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")

WandbModelCheckpoint(
    filepath="models/,
    options=checkpoint_options,
)
```

## `WandbEvalCallback`을 사용한 모델 예측 시각화

[**여기에서 Colab 노트북으로 시도해보세요 →**](https://github.com/wandb/examples/blob/e66f16fbe7ae7a2e636d59350a50059d3f7e5494/colabs/keras/Use_WandbEvalCallback_in_your_Keras_workflow.ipynb)

`WandbEvalCallback`은 주로 모델 예측 및 부차적으로 데이터셋 시각화를 위해 Keras 콜백을 구축하기 위한 추상 기본 클래스입니다.

이 추상 콜백은 데이터셋 및 작업과 관련하여 중립적입니다. 이를 사용하려면, 이 기본 `WandbEvalCallback` 콜백 클래스에서 상속받아 `add_ground_truth` 및 `add_model_prediction` 메서드를 구현하세요.

`WandbEvalCallback`은 유용한 메서드를 제공하는 유틸리티 클래스로, 다음을 수행할 수 있습니다:

* 데이터 및 예측 `wandb.Table` 인스턴스 생성,
* 데이터 및 예측 테이블을 `wandb.Artifact`로 로그
* 데이터 테이블을 `on_train_begin`에서 로그
* 예측 테이블을 `on_epoch_end`에서 로그

예를 들어, 아래에서 이미지 분류 작업에 대해 구현한 `WandbClfEvalCallback`를 보여줍니다. 이 예제 콜백은:

* 검증 데이터(`data_table`)를 W&B에 로그,
* 매 에포크 끝에 추론을 수행하고 예측(`pred_table`)을 W&B에 로그.

```python
import wandb
from wandb.keras import WandbMetricsLogger, WandbEvalCallback


# 모델 예측 시각화 콜백 구현
class WandbClfEvalCallback(WandbEvalCallback):
    def __init__(
        self, validation_data, data_table_columns, pred_table_columns, num_samples=100
    ):
        super().__init__(data_table_columns, pred_table_columns)

        self.x = validation_data[0]
        self.y = validation_data[1]

    def add_ground_truth(self, logs=None):
        for idx, (image, label) in enumerate(zip(self.x, self.y)):
            self.data_table.add_data(idx, wandb.Image(image), label)

    def add_model_predictions(self, epoch, logs=None):
        preds = self.model.predict(self.x, verbose=0)
        preds = tf.argmax(preds, axis=-1)

        table_idxs = self.data_table_ref.get_index()

        for idx in table_idxs:
            pred = preds[idx]
            self.pred_table.add_data(
                epoch,
                self.data_table_ref.data[idx][0],
                self.data_table_ref.data[idx][1],
                self.data_table_ref.data[idx][2],
                pred,
            )


# ...

# 새로운 W&B 실행 초기화
wandb.init(config={"hyper": "parameter"})

# Model.fit에 콜백 추가
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    callbacks=[
        WandbMetricsLogger(),
        WandbClfEvalCallback(
            validation_data=(X_test, y_test),
            data_table_columns=["idx", "image", "label"],
            pred_table_columns=["epoch", "idx", "image", "label", "pred"],
        ),
    ],
)
```

:::info
💡 테이블은 기본적으로 W&B [아티팩트 페이지](https://docs.wandb.ai/ref/app/pages/project-page#artifacts-tab)에 로그되며, [워크스페이스](https://docs.wandb.ai/ref/app/pages/workspaces) 페이지에는 로그되지 않습니다.
:::

**`WandbEvalCallback` 참조**

| 파라미터            | 설명                                      |
| -------------------- | ------------------------------------------------ |
| `data_table_columns` | (list) `data_table`의 열 이름 목록 |
| `pred_table_columns` | (list) `pred_table`의 열 이름 목록 |

### 메모리 사용량이 어떻게 줄어드나요?

`on_train_begin` 메서드가 호출될 때 `data_table`을 W&B에 로그합니다. 일단 W&B 아티팩트로 업로드되면, 이 테이블에 대한 참조를 `data_table_ref` 클래스 변수를 사용하여 얻을 수 있습니다. `data_table_ref`는 `self.data_table_ref[idx][n]`처럼 인덱싱할 수 있는 2D 리스트입니다. 여기서 `idx`는 행 번호이고 `n`은 열 번호입니다. 아래 예제에서 사용법을 확인해보세요.

### 콜백을 더 맞춤화하기

더 세밀한 제어를 원한다면 `on_train_begin` 또는 `on_epoch_end` 메서드를 오버라이드할 수 있습니다. `N` 배치 후에 샘플을 로그하고 싶다면 `on_train_batch_end` 메서드를 구현할 수 있습니다.

:::info
💡 `WandbEvalCallback`을 상속하여 모델 예측 시각화 콜백을 구현하는 경우, 명확히 하거나 수정해야 할 사항이 있다면 [이슈](https://github.com/wandb/wandb/issues)를 통해 알려주세요.
:::

## WandbCallback [레거시]

W&B 라이브러리 [`WandbCallback`](https://docs.wandb.ai/ref/python/integrations/keras/wandbcallback) 클래스를 사용하여 `model.fit`에서 추적된 모든 메트릭과 손실 값을 자동으로 저장하세요.

```python
import wandb
from wandb.keras import WandbCallback

wandb.init(config={"hyper": "parameter"})

...  # Keras에서 모델 설정 코드

# 모델.fit에 콜백 전달
model.fit(
    X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbCallback()]
)
```

**사용 예시**

W&B와 Keras를 처음 통합하는 경우 이 분 단위 단계별 동영상을 참조하세요: [1분 미만으로 Keras 및 Weights & Biases 시작하기](https://www.youtube.com/watch?ab_channel=Weights&Biases&v=4FjDIJ-vO_M)

더 자세한 비디오는 [Keras와 Weights & Biases 통합하기](https://www.youtube.com/watch?v=Bsudo7jbMow\&ab\_channel=Weights%26Biases)를 참고하세요. 사용된 노트북 예제는 여기에서 찾을 수 있습니다: [Colab Jupyter 노트북](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Keras\_pipeline\_with\_Weights\_and\_Biases.ipynb).

:::info
위 동영상의 W&B와 Keras 통합 예제를 [colab 노트북](http://wandb.me/keras-colab)에서 시도해 보세요. 또는 스크립트를 포함한 [예제 저장소](https://github.com/wandb/examples)를 확인해 보세요. 예를 들어 [Fashion MNIST 예제](https://github.com/wandb/examples/blob/master/examples/keras/keras-cnn-fashion/train.py)와 이에 대한 [W&B 대시보드](https://wandb.ai/wandb/keras-fashion-mnist/runs/5z1d85qs)가 있습니다.
:::

`WandbCallback` 클래스는 메트릭 모니터링 지정, 가중치 및 그레이디언트 추적, 학습\_데이터 및 검증\_데이터에 대한 예측 로깅 등 다양한 로깅 구성 옵션을 지원합니다.

`keras.WandbCallback`에 대한 [참고 문서](../../ref/python/integrations/keras/wandbcallback.md)에서 전체 세부 사항을 확인하세요.

`WandbCallback`은

* keras가 수집한 모든 메트릭의 기록 데이터를 자동으로 로그합니다: 손실 및 `keras_model.compile()`에 전달된 모든 것
* '최상' 학습 단계와 관련된 실행에 대한 요약 메트릭을 설정합니다. 여기서 "최상"은 `monitor` 및 `mode` 속성에 의해 정의됩니다. 이는 기본적으로 최소 `val_loss`를 가진 에포크입니다. `WandbCallback`은 기본적으로 최상의 `epoch`와 관련된 모델을 저장합니다.
* 선택적으로 그레이디언트 및 파라미터 히스토그램을 로그할 수 있습니다.
* 선택적으로 wandb가 시각화할 학습 및 검증 데이터를 저장할 수 있습니다.

**`WandbCallback` 참조**

| 인수                      | 설명                                                                                         |
| -------------------------- | -------------------------------------------------------------------------------------------------- |
| `monitor`                  | (str) 모니터할 메트릭의 이름. 기본값은 `val_loss`입니다.                                                                             |
| `mode`                     | (str) {`auto`, `min`, `max`} 중 하나. `min` - 모니터 최소화 시 모델 저장 `max` - 모니터 최대화 시 모델 저장 `auto` - 모델 저장 시점 추측 (기본값).                                                                                                                                                     |
| `save_model`               | True - 모니터가 이전 에포크를 모두 초과할 때 모델 저장 False - 모델 저장하지 않음                                                 |
| `save_graph`               | (boolean) True이면 wandb에 모델 그래프 저장 (기본값 True).                                                                     |
| `save_weights_only`        | (boolean) True이면 모델의 가중치만 저장 (`model.save_weights(filepath)`), 그렇지 않으면 전체 모델 저장 (`model.save(filepath)`).     |
| `log_weights`              | (boolean) True이면 모델 레이어의 가중치 히스토그램 저장.                                                                  |
| `log_gradients`            | (boolean) True이면 학습 그레이디언트의 히스토그램 로그.                                                                     |
| `training_data`            | (tuple) `model.fit`에 전달된 것과 동일한 형식 `(X,y)`. 그레이디언트 계산을 위해 필요 - `log_gradients`가 `True`이면 필수.             |
| `validation_data`          | (tuple) `model.fit`에 전달된 것과 동일한 형식 `(X,y)`. wandb가 시각화할 데이터 세트. 이것이 설정되면, 매 에포크마다 wandb는 소수의 예측을 수행하고 결과를 나중에 시각화하기 위해 저장합니다.    |
| `generator`                | (generator) wandb가 시각화할 검증 데이터를 반환하는 생성기. 이 생성기는 `(X,y)` 튜플을 반환해야 합니다. `validate_data` 또는 생성기 중 하나가 wandb가 특정 데이터 예제를 시각화하기 위해 설정되어야 합니다.       |
| `validation_steps`         | (int) `validation_data`가 생성기인 경우 전체 검증 세트에 대해 생성기를 실행할 단계 수.         |
| `labels`                   | (list) 데이터를 wandb와 함께 시각화하는 경우 이 레이블 목록은 숫자 출력을 이해하기 쉬운 문자열로 변환합니다. 다중 클래스 분류기를 구축하는 경우입니다. 이진 분류기를 만들고 있다면 두 레이블의 목록 \["false에 대한 레이블", "true에 대한 레이블"]을 전달할 수 있습니다. `validate_data`와 생성기가 모두 거짓이면 아무런 작용을 하지 않습니다. |
| `predictions`              | (int) 매 에포크마다 시각화를 위해 수행할 예측 수, 최대 100개.  |
| `input_type`               | (string) 시각화를 돕기 위한 모델 입력 유형. 다음 중 하나일 수 있습니다: (`image`, `images`, `segmentation_mask`).    |
| `output_type`              | (string) 시각화를 돕기 위한 모델 출력 유형. 다음 중 하나일 수 있습니다: (`image`, `images`, `segmentation_mask`).      |
| `log_evaluation`           | (boolean) True이면 각 에포크에서 검증 데이터와 모델의 예측을 포함하는 테이블을 저장합니다. `validation_indexes`, `validation_row_processor`, 및 `output_row_processor`에 대한 추가 세부 사항을 참조하세요.       |
| `class_colors`             | (\[float, float, float]) 입력 또는 출력이 세분화 마스크인 경우 각 클래스에 대한 rgb 튜플(범위 0-1)을 포함하는 배열.                    |
| `log_batch_frequency`      | (integer) None이면 콜백은 매 에포크마다 로그를 기록합니다. 정수로 설정된 경우 콜백은 `log_batch_frequency` 배치마다 학습 메트릭을 로그합니다.            |
| `log_best_prefix`          | (string) None이면 추가 요약 메트릭이 저장되지 않습니다. 문자열로 설정된 경우, 모니터링되는 메트릭과 에포크가 이 값으로 시작되어 요약 메트릭으로 저장됩니다. |
| `validation_indexes`       | (\[wandb.data\_types.\_TableLinkMixin]) 각 검증 예제와 연관된 인덱스 키의 순서 있는 목록. log\_evaluation이 True이고 `validation_indexes`가 제공되면 검증 데이터의 테이블이 생성되지 않고 대신 각 예측이 `TableLinkMixin`에 의해 표현된 행과 연관됩니다. 이러한 키를 얻는 가장 일반적인 방법은 `Table.get_index()`를 사용하는 것이며, 이는 행 키 목록을 반환할 것입니다.            |
| `validation_row_processor` | (Callable) 검증 데이터에 적용할 함수로, 일반적으로 데이터를 시각화하는 데 사용됩니다. 함수는 `ndx`(int)와 `row`(dict)를 받게 됩니다. 모델 입력이 단일 항목이면, `row["input"]`은 해당 행의 입력 데이터가 됩니다. 그렇지 않으면 입력 슬롯의 이름에 따라 키가 지정됩니다. fit 함수가 단일 대상을 취한다면, `row["target"]`은 해당 행의 대상 데이터가 됩니다. 그렇지 않으면 출력 슬롯의 이름에 따라 키가 지정됩니다. 예를 들어, 입력 데이터가 단일 ndarray이지만 데이터를 이미지로 시각화하고 싶다면, 프로세서로 `lambda ndx, row: {"img": wandb.Image(row["input"])}`을 제공할 수 있습니다. log\_evaluation이 False이거나 `validation_indexes`가 있는 경우 무시됩니다. |
| `output_row_processor`     | (Callable) `validation_row_processor`와 동일하지만 모델 출력에 적용됩니다. `row["output"]`은 모델 출력의 결과를 포함할 것입니다.            |
| `infer_missing_processors` | (bool) `validation_row_processor` 및 `output_row_processor`가 없는 경우 추론할지 여부를 결정합니다. 기본값은 True입니다. `labels`가 제공되면 적절한 경우 분류 유형 프로세서를 추론하려고 시도합니다.        |
| `log_evaluation_frequency` | (int) 평가 결과 로그 빈도를 결정합니다. 기본값은 0(학습 종료 시에만)입니다. 매 에포크마다 로그하려면 1로 설정하고, 격 에포크마다 로그하려면 2로 설정하고, 그 이후로 계속합니다. log\_evaluation이 False일 때는 효과가 없습니다.  |

## 자주 묻는 질문

### `Keras`의 멀티프로세싱을 `wandb`와 어떻게 사용하나요?

`use_multiprocessing=True`를 설정하고 다음과 같은 오류가 발생하는 경우:

```python
Error("wandb.init()을 호출하기 전에 wandb.config.batch_size를 호출해야 합니다")
```

다음을 시도하세요:

1. `Sequence` 클래스 구성에 `wandb.init(group='...')`를 추가합니다.
2. 메인 프로그램에서 `if __name__ == "__main__":`을 사용하고 나머지 스크립트 로직을 그 안에 넣습니다.
---
title: Keras
displayed_sidebar: default
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases_keras.ipynb"></CTAButtons>

## Weights & Biases Keras 콜백

Keras와 TensorFlow 사용자를 위한 세 가지 새로운 콜백이 `wandb` v0.13.4에서 제공됩니다. 이전 `WandbCallback`을 보려면 아래로 스크롤하십시오.

**`WandbMetricsLogger`** : [Experiment Tracking](/guides/track)을 위해 이 콜백을 사용하세요. 트레이닝과 검증 메트릭을 시스템 메트릭과 함께 Weights & Biases에 로그합니다.

**`WandbModelCheckpoint`** : Weights & Biases [Artifacts](/guides/artifacts)에 모델 체크포인트를 로그하기 위해 이 콜백을 사용하세요.

**`WandbEvalCallback`**: 이 기본 콜백은 Weights & Biases [Tables](/guides/tables)에 모델 예측값을 로그하여 대화형 시각화를 제공합니다.

이 새로운 콜백은,

* Keras 디자인 철학에 부합하고,
* 모든 것을 하나의 콜백(`WandbCallback`)으로 사용하는 인지 부하를 줄여주며,
* Keras 사용자가 특정 유스 케이스를 지원하기 위해 콜백을 서브클래스화하여 쉽게 수정할 수 있도록 합니다.

## `WandbMetricsLogger`를 사용한 Experiment Tracking

<CTAButtons colabLink="https://github.com/wandb/examples/blob/master/colabs/keras/Use_WandbMetricLogger_in_your_Keras_workflow.ipynb"></CTAButtons>

`WandbMetricsLogger`는 `on_epoch_end`, `on_batch_end` 등과 같은 콜백 메소드에서 인수로 받는 Keras의 `logs` 사전을 자동으로 로그합니다.

이것을 사용하면 다음을 제공합니다:

* `model.compile`에 정의된 트레이닝 및 검증 메트릭
* 시스템(CPU/GPU/TPU) 메트릭
* 학습률(고정 값 및 학습률 스케줄러 모두에 대해)

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger

# 새로운 W&B run을 초기화
wandb.init(config={"bs": 12})

# WandbMetricsLogger를 model.fit에 전달
model.fit(
    X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbMetricsLogger()]
)
```

**`WandbMetricsLogger` 참조**

| 파라미터 | 설명 |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `log_freq`            | ("epoch", "batch", 또는 int): "epoch"일 경우 각 에포크의 끝에서 메트릭을 로그합니다. "batch"일 경우 각 배치의 끝에서 메트릭을 로그합니다. int인 경우 해당 수의 배치 후 메트릭을 로그합니다. 기본값은 "epoch"입니다. |
| `initial_global_step` | (int): 초기 에포크에서 트레이닝을 재개하고 학습률 스케줄러가 사용될 때 학습률을 올바르게 로그하기 위해 이 인수를 사용하세요.  기본값은 0입니다. |

## `WandbModelCheckpoint`를 사용한 모델 체크포인트 생성

<CTAButtons colabLink="https://github.com/wandb/examples/blob/master/colabs/keras/Use_WandbModelCheckpoint_in_your_Keras_workflow.ipynb"></CTAButtons>

`WandbModelCheckpoint` 콜백을 사용하여 Keras 모델(`SavedModel`) 또는 모델 가중치를 주기적으로 저장하고 이를 모델 버전 관리용 `wandb.Artifact`로 W&B에 업로드합니다.

이 콜백은 [`tf.keras.callbacks.ModelCheckpoint`](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint)에서 서브클래스화되어, 체크포인트 로직은 부모 콜백에서 처리됩니다.

이 콜백은 다음과 같은 기능을 제공합니다:

* "모니터"를 기반으로 "최고 성능"을 달성한 모델 저장
* 성능에 관계없이 각 에포크 끝에서 모델 저장
* 에포크가 끝난 후 또는 고정된 수의 트레이닝 배치 후 모델 저장
* 모델 가중치만 저장하거나 전체 모델 저장
* SavedModel 형식이나 `.h5` 형식으로 모델 저장

이 콜백은 `WandbMetricsLogger`와 함께 사용해야 합니다.

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

# 새로운 W&B run을 초기화
wandb.init(config={"bs": 12})

# WandbModelCheckpoint를 model.fit에 전달
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
| `filepath`   | (str): 모드 파일을 저장할 경로입니다.|  
| `monitor`                 | (str): 모니터링할 메트릭 이름입니다.         |
| `verbose`                 | (int): 설명 모드, 0 또는 1. 모드 0은 조용하고, 모드 1은 콜백이 작업을 수행할 때 메시지를 표시합니다.   |
| `save_best_only`          | (bool): `save_best_only=True`이면 모델이 "최고"로 간주될 때만 저장하고, 모니터링하는 양(량)에 따라 최신 최고의 모델이 덮어쓰이지 않습니다.     |
| `save_weights_only`       | (bool): True이면 모델의 가중치만 저장됩니다.                                            |
| `mode`                    | ("auto", "min", 또는 "max"): `val_acc`의 경우 'max'이며, `val_loss`의 경우 'min'이어야 합니다.  |
| `save_weights_only`       | (bool): True이면 모델의 가중치만 저장됩니다.                                            |
| `save_freq`               | ("epoch" 또는 int): ‘epoch’를 사용할 때, 콜백은 각 에포크가 끝날 때 모델을 저장합니다. 정수를 사용할 때, 콜백은 그만큼의 배치가 끝날 때 모델을 저장합니다. `val_acc` 또는 `val_loss`와 같은 검증 메트릭을 모니터링할 때는 `save_freq`를 "epoch"로 설정해야만 합니다. 이러한 메트릭은 에포크가 끝날 때만 이용할 수 있습니다. |
| `options`                 | (str): `save_weights_only`가 true인 경우 선택적 `tf.train.CheckpointOptions` 객체 또는 `save_weights_only`가 거짓일 경우 선택적 `tf.saved_model.SaveOptions` 객체입니다.    |
| `initial_value_threshold` | (float): 모니터할 메트릭의 초기 "최고" 값입니다.       |

### N 에포크 후 체크포인트를 로그하는 방법?

기본적으로(`save_freq="epoch"`) 콜백은 각 에포크 후 체크포인트를 생성하고 이를 아티팩트로 업로드합니다. `save_freq`에 정수를 전달하면 그만큼의 배치 이후에 체크포인트가 생성됩니다. `N` 에포크 후 체크포인트를 생성하려면 트레이닝 데이터로더의 카디널리티를 계산하여 `save_freq`에 전달하십시오:

```
WandbModelCheckpoint(
    filepath="models/",
    save_freq=int((trainloader.cardinality()*N).numpy())
)
```

### TPU 노드 아키텍처에서 효율적으로 체크포인트를 로그하는 방법?

TPU에서 체크포인트를 생성할 때 `UnimplementedError: File system scheme '[local]' not implemented` 오류 메시지가 발생할 수 있습니다. 이는 모델 디렉토리(`filepath`)가 클라우드 스토리지 버킷 경로(`gs://bucket-name/...`)를 사용해야 하며, 이 버킷은 TPU 서버에서 엑세스할 수 있어야 하기 때문입니다. 로컬 경로를 사용한 체크포인트 생성은 가능하며, 이는 다시 Artifacts로 업로드됩니다.

```
checkpoint_options = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")

WandbModelCheckpoint(
    filepath="models/,
    options=checkpoint_options,
)
```

## `WandbEvalCallback`을 사용한 모델 예측 시각화

<CTAButtons colabLink="https://github.com/wandb/examples/blob/e66f16fbe7ae7a2e636d59350a50059d3f7e5494/colabs/keras/Use_WandbEvalCallback_in_your_Keras_workflow.ipynb"></CTAButtons>

`WandbEvalCallback`은 주로 모델 예측을 위한 Keras 콜백을 구축하기 위한 추상 기본 클래스이며, 두 번째로 데이터셋 시각화를 위한 것입니다.

이 추상 콜백은 데이터셋과 작업에 관계없이 사용할 수 있습니다. 이를 사용하려면 이 기본 `WandbEvalCallback` 콜백 클래스를 상속하고 `add_ground_truth` 및 `add_model_prediction` 메소드를 구현하세요.

`WandbEvalCallback`은 다음을 위해 유용한 메소드를 제공하는 유틸리티 클래스입니다:

* 데이터 및 예측 `wandb.Table` 인스턴스 생성,
* 데이터 및 예측 Table을 `wandb.Artifact`로 로그
* `on_train_begin`시 데이터 테이블 로그
* `on_epoch_end`시 예측 테이블 로그

예를 들어, 우리는 다음과 같이 이미지 분류 작업을 위해 `WandbClfEvalCallback`을 구현했습니다. 이 예시 콜백은:

* 검증 데이터(`data_table`)를 W&B에 로그하고,
* 추론을 수행하며 각 에포크 끝에서 예측(`pred_table`)을 W&B에 로그합니다.

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbEvalCallback

# 모델 예측 시각화 콜백을 구현하세요.
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

# 새로운 W&B run을 초기화
wandb.init(config={"hyper": "parameter"})

# 콜백을 Model.fit에 추가
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
💡 테이블은 기본적으로 W&B [Artifact 페이지](/guides/artifacts/explore-and-traverse-an-artifact-graph)에 로그되며, 워크스페이스 페이지에는 로그되지 않습니다.
:::

**`WandbEvalCallback` 참조**

| 파라미터 | 설명 |
| -------------------- | ------------------------------------------------ |
| `data_table_columns` | (list) `data_table`의 열 이름 목록 |
| `pred_table_columns` | (list) `pred_table`의 열 이름 목록 |

### 메모리 사용량이 줄어드는 방법?

`on_train_begin` 메소드가 호출되면 `data_table`을 W&B에 로그합니다. 이것이 W&B 아티팩트로 업로드되면, 이 테이블을 참조할 수 있으며, 이는 `data_table_ref` 클래스 변수로 엑세스할 수 있습니다. `data_table_ref`는 인덱스가 가능한 2D 목록이며, `self.data_table_ref[idx][n]`과 같이 인덱싱할 수 있습니다. 여기서 `idx`는 행 수이며, `n`은 열 수입니다. 예를 보겠습니다.

### 콜백을 추가로 맞춤 설정

더 세부적인 제어를 원하시면, `on_train_begin` 또는 `on_epoch_end` 메소드를 재정의할 수 있습니다. `N` 배치 후에 샘플을 로그하고 싶다면, `on_train_batch_end` 메소드를 구현할 수 있습니다.

:::info
💡 `WandbEvalCallback`을 상속해서 모델 예측 시각화 콜백을 구현하고, 수정이나 명확한 설명이 필요하면 [issue](https://github.com/wandb/wandb/issues)를 열어서 알려주세요.
:::

## WandbCallback [이전 버전]

W&B 라이브러리의 [`WandbCallback`](/ref/python/integrations/keras/wandbcallback) 클래스를 사용하여 `model.fit`에서 추적되는 모든 메트릭 및 손실 값을 자동으로 저장하십시오.

```python
import wandb
from wandb.integration.keras import WandbCallback

wandb.init(config={"hyper": "parameter"})

...  # Keras에서 모델 설정을 위한 코드

# 콜백을 model.fit에 전달
model.fit(
    X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbCallback()]
)
```

**사용 예시**

Keras와 W&B를 처음 통합하는 경우 이 1분짜리 단계별 비디오를 참조하세요: [Get Started with Keras and Weights & Biases in Less Than a Minute](https://www.youtube.com/watch?ab_channel=Weights&Biases&v=4FjDIJ-vO_M)

더 자세한 비디오는 [Integrate Weights & Biases with Keras](https://www.youtube.com/watch?v=Bsudo7jbMow\&ab_channel=Weights%26Biases)를 참조하세요. 사용된 노트북 예시는 여기에서 찾을 수 있습니다: [Colab Jupyter Notebook](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Keras_pipeline_with_Weights_and_Biases.ipynb).

:::info
스크립트, 예를 들어 [Fashion MNIST example](https://github.com/wandb/examples/blob/master/examples/keras/keras-cnn-fashion/train.py)와 생성된 [W&B Dashboard](https://wandb.ai/wandb/keras-fashion-mnist/runs/5z1d85qs)를 포함한 [example repo](https://github.com/wandb/examples)를 참조하세요.
:::

`WandbCallback` 클래스는 다양한 로그 설정 옵션을 지원합니다: 모니터링할 메트릭 지정, 가중치 및 그레이디언트 추적, 트레이닝 데이터 및 검증 데이터에 대한 예측 로그, 등등.

전체 세부사항은 [keras.WandbCallback에 대한 참조 문서](../../ref/python/integrations/keras/wandbcallback.md)를 참조하세요.

`WandbCallback`

* Keras에서 수집된 메트릭: 손실 및 `keras_model.compile()`에 전달된 모든 메트릭의 히스토리 데이터를 자동으로 로그
* "최고" 트레이닝 단계와 관련된 run에 대한 요약 메트릭을 설정합니다. "최고"는 `monitor` 및 `mode` 속성에 의해 정의됩니다. 기본값은 `val_loss`의 최소값을 가지는 에포크입니다. `WandbCallback`은 기본적으로 최고의 `epoch`와 관련된 모델을 저장합니다.
* 선택적으로 그레이디언트 및 파라미터 히스토그램을 로그할 수 있습니다.
* wandb가 시각화할 수 있도록 트레이닝 및 검증 데이터를 선택적으로 저장할 수 있습니다.

**`WandbCallback` 참조**

| 인수                       | 설명                                    |
| -------------------------- | ------------------------------------------- |
| `monitor`                  | (str) 모니터링할 메트릭의 이름입니다. 기본값은 `val_loss`입니다.                                                                   |
| `mode`                     | (str) {`auto`, `min`, `max`} 중 하나입니다. `min` - 모니터가 최소화될 때 모델을 저장합니다 `max` - 모니터가 최대화될 때 모델을 저장합니다 `auto` - 모델을 저장 시점을 추측하려고 시도합니다 (기본값).                                                                                                                                                |
| `save_model`               | True - 모니터가 이전의 모든 에포크를 이길 때 모델을 저장합니다. False - 모델을 저장하지 않습니다.                                       |
| `save_graph`               | (boolean) True일 경우, 모델 그래프를 wandb에 저장합니다 (기본값은 True).                                                           |
| `save_weights_only`        | (boolean) True일 경우, 모델의 가중치만 저장됩니다 (`model.save_weights(filepath)`), 그렇지 않으면 전체 모델이 저장됩니다 (`model.save(filepath)`).   |
| `log_weights`              | (boolean) True일 경우, 모델 계층의 가중치 히스토그램을 저장합니다.                                                |
| `log_gradients`            | (boolean) True일 경우, 트레이닝 그레이디언트의 히스토그램을 로그합니다.                                                       |
| `training_data`            | (tuple) `model.fit`에 전달된 형식 `(X,y)`과 동일합니다. 이는 그레이디언트를 계산하기 위해 필요합니다 - `log_gradients`가 `True`인 경우 필수입니다.       |
| `validation_data`          | (tuple) `model.fit`에 전달된 형식 `(X,y)`과 동일합니다. wandb가 시각화할 데이터 세트입니다. 이 설정을 하면, 매 에포크마다, wandb는 소량의 예측을 만들고 결과를 저장하여 나중에 시각화할 수 있습니다.          |
| `generator`                | (generator) wandb가 시각화할 검증 데이터를 반환하는 생성자입니다. 이 생성자는 튜플 `(X,y)`를 반환해야 합니다. wandb가 특정 데이터 예시를 시각화하려면 `validate_data` 또는 generator가 설정되어야 합니다.     |
| `validation_steps`          | (int) `validation_data`가 생성자인 경우, 전체 검증 세트를 위해 실행할 생성자 스텝 수입니다.       |
| `labels`                   | (list) wandb를 사용하여 데이터를 시각화할 경우, 이 레이블 목록은 숫자 출력을 알아볼 수 있는 문자열로 변환합니다. 다중 클래스 분류기를 구축하는 경우, 이 목록을 사용하세요. 이진 분류기를 만드는 경우, \["거짓에 대한 레이블", "참에 대한 레이블"]과 같은 두 개의 레이블 목록을 전달할 수 있습니다. `validate_data` 및 생성자가 모두 false인 경우에는 아무런 효과가 없습니다.    |
| `predictions`              | (int) 각 에포크마다 시각화를 위해 예측할 예측 수입니다, 최대는 100입니다.    |
| `input_type`               | (string) 시각화 헬프를 위한 모델 입력의 타입입니다. 다음 중 하나일 수 있습니다: (`image`, `images`, `segmentation_mask`).  |
| `output_type`              | (string) 시각화를 위한 모델 출력의 타입입니다. 다음 중 하나일 수 있습니다: (`image`, `images`, `segmentation_mask`).    |
| `log_evaluation`           | (boolean) True일 경우, 검증 데이터와 각 에포크에서 모델의 예측을 포함한 테이블을 저장합니다. 추가 사항은 `validation_indexes`, `validation_row_processor`, `output_row_processor`를 참조하세요.     |
| `class_colors`             | (\[float, float, float]) 입력 또는 출력이 세그멘테이션 마스크인 경우, 각 클래스에 대한 rgb 튜플(범위 0-1)을 포함하는 배열을 의미합니다.                  |
| `log_batch_frequency`      | (integer) None일 경우, 콜백은 매 에포크마다 로그합니다. 정수로 설정할 경우, 콜백은 매 `log_batch_frequency` 배치마다 트레이닝 메트릭을 로그합니다.          |
| `log_best_prefix`          | (string) None인 경우, 추가 요약 메트릭이 저장되지 않습니다. 문자열로 설정되는 경우, 모니터된 메트릭과 에포크는 이 값을 선행하고 요약 메트릭으로 저장됩니다.   |
| `validation_indexes`       | (\[wandb.data_types._TableLinkMixin]) 각 검증 예시와 연관된 인덱스 키의 정렬된 리스트입니다. log_evaluation이 True이고 `validation_indexes`가 제공되면, 검증 데이터의 테이블이 생성되지 않으며, 대신, 각 예측은 `TableLinkMixin`이 나타내는 행과 연관됩니다. 이러한 키를 얻는 가장 일반적인 방법은 `Table.get_index()`를 사용하는 것이며, 이는 행 키의 목록을 반환합니다.          |
| `validation_row_processor` | (Callable) 검증 데이터에 적용할 함수로, 일반적으로 데이터 시각화에 사용됩니다. 함수는 `ndx` (int) 및 `row` (dict)를 받습니다. 모델에 단일 입력이 있는 경우, `row["input"]`는 해당 행의 입력 데이터입니다. 그렇지 않으면, 입력 슬롯의 이름에 기반한 키입니다. fit 함수가 단일 목표를 수용하는 경우, `row["target"]`은 해당 행의 목표 데이터입니다. 출력 슬롯의 이름에 기반한 키입니다. 예를 들어, 입력 데이터가 단일 ndarray인 경우에도 데이터를 이미지로 시각화하고자 한다면, `lambda ndx, row: {"img": wandb.Image(row["input"])}`를 프로세서로 제공할 수 있습니다. log_evaluation이 False 또는 `validation_indexes`가 있는 경우 무시됩니다. |
| `output_row_processor`     | (Callable) `validation_row_processor`와 동일하되, 모델의 출력에 적용됩니다. `row["output"]`는 모델 출력의 결과를 포함합니다.          |
| `infer_missing_processors` | (bool) 누락된 경우에 `validation_row_processor`와 `output_row_processor`를 추론해야 하는지를 결정합니다. 기본값은 True입니다. `labels`가 제공된 경우, 적절한 위치에서 분류 유형 프로세서를 추론하려고 시도합니다.      |
| `log_evaluation_frequency` | (int) 평가 결과를 로그할 빈도를 결정합니다. 기본값은 0 (트레이닝 종료 시)입니다. 매 에포크마다 로그를 기록하려면 1로 설정하고, 매 두 번째 에포크마다 로그를 기록하려면 2로 설정하는 식입니다. log_evaluation이 False일 때는 아무런 영향을 주지 않습니다.    |

## 자주 묻는 질문

### `Keras` 멀티프로세싱을 `wandb`와 함께 사용하는 방법은?

`use_multiprocessing=True`를 설정하고 다음과 같은 오류가 발생하는 경우:

```python
Error("You must call wandb.init() before wandb.config.batch_size")
```

다음과 같이 시도하십시오:

1. `Sequence` 클래스 생성 시, `wandb.init(group='...')`을 추가하세요.
2. 메인 프로그램에서, `if __name__ == "__main__":`을 사용하고 나머지 스크립트 로직을 그 안에 넣으세요.
---
title: Keras
menu:
  default:
    identifier: ko-guides-integrations-keras
    parent: integrations
weight: 160
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases_keras.ipynb" >}}

## Keras 콜백

W&B는 `wandb` v0.13.4부터 사용할 수 있는 Keras용 콜백 세 가지를 제공합니다. 기존 `WandbCallback`은 아래로 스크롤하세요.

- **`WandbMetricsLogger`** : [Experiment 추적]({{< relref path="/guides/models/track" lang="ko" >}})에 이 콜백을 사용하세요. 트레이닝 및 검증 메트릭과 시스템 메트릭을 Weights & Biases에 기록합니다.

- **`WandbModelCheckpoint`** : 모델 체크포인트를 Weight and Biases [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ko" >}})에 기록하려면 이 콜백을 사용하세요.

- **`WandbEvalCallback`**: 이 기본 콜백은 모델 예측값을 대화형 시각화를 위해 Weights and Biases [Tables]({{< relref path="/guides/models/tables/" lang="ko" >}})에 기록합니다.

이 새로운 콜백은 다음과 같은 특징이 있습니다.

* Keras 디자인 철학을 준수합니다.
* 모든 작업에 단일 콜백(`WandbCallback`)을 사용하는 데 따른 인지적 부담을 줄입니다.
* Keras 사용자가 콜백을 서브클래싱하여 특정 유스 케이스를 지원하도록 수정하기 쉽습니다.

## `WandbMetricsLogger`로 Experiments 추적

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/keras/Use_WandbMetricLogger_in_your_Keras_workflow.ipynb" >}}

`WandbMetricsLogger`는 `on_epoch_end`, `on_batch_end` 등과 같은 콜백 메서드가 인수로 사용하는 Keras의 `logs` 사전을 자동으로 기록합니다.

다음 항목을 추적합니다.

* `model.compile`에 정의된 트레이닝 및 검증 메트릭.
* 시스템(CPU/GPU/TPU) 메트릭.
* 학습률(고정값 및 학습률 스케줄러 모두).

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger

# 새로운 W&B run 초기화
wandb.init(config={"bs": 12})

# WandbMetricsLogger를 model.fit에 전달
model.fit(
    X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbMetricsLogger()]
)
```

### `WandbMetricsLogger` 참조

| 파라미터 | 설명 |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `log_freq`            | (`epoch`, `batch` 또는 `int`): `epoch`인 경우 각 에포크가 끝날 때 메트릭을 기록합니다. `batch`인 경우 각 배치가 끝날 때 메트릭을 기록합니다. `int`인 경우 해당 배치 수만큼 끝날 때 메트릭을 기록합니다. 기본값은 `epoch`입니다. |
| `initial_global_step` | (int): 학습률 스케줄러가 사용되고 일부 initial_epoch에서 트레이닝을 재개할 때 학습률을 올바르게 기록하려면 이 인수를 사용하세요. 이는 step_size * initial_step으로 계산할 수 있습니다. 기본값은 0입니다. |

## `WandbModelCheckpoint`를 사용하여 모델 체크포인트

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/keras/Use_WandbModelCheckpoint_in_your_Keras_workflow.ipynb" >}}

`WandbModelCheckpoint` 콜백을 사용하여 Keras 모델(`SavedModel` 형식) 또는 모델 가중치를 주기적으로 저장하고 모델 버전 관리를 위해 `wandb.Artifact`로 W&B에 업로드합니다.

이 콜백은 [`tf.keras.callbacks.ModelCheckpoint`](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint)에서 서브클래싱되므로 체크포인트 로직은 상위 콜백에서 처리합니다.

이 콜백은 다음을 저장합니다.

* 모니터에 따라 최상의 성능을 달성한 모델.
* 성능에 관계없이 모든 에포크가 끝날 때의 모델.
* 에포크가 끝날 때 또는 고정된 수의 트레이닝 배치 후의 모델.
* 모델 가중치만 또는 전체 모델.
* `SavedModel` 형식 또는 `.h5` 형식의 모델.

`WandbMetricsLogger`와 함께 이 콜백을 사용하세요.

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

# 새로운 W&B run 초기화
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

### `WandbModelCheckpoint` 참조

| 파라미터 | 설명 |
| ------------------------- |  ---- |
| `filepath`   | (str): 모드 파일을 저장할 경로입니다.|
| `monitor`                 | (str): 모니터링할 메트릭 이름입니다. |
| `verbose`                 | (int): 상세 모드, 0 또는 1. 모드 0은 자동이고 모드 1은 콜백이 작업을 수행할 때 메시지를 표시합니다. |
| `save_best_only`          | (Boolean): `save_best_only=True`인 경우 `monitor` 및 `mode` 속성으로 정의된 대로 최신 모델 또는 가장 적합하다고 간주되는 모델만 저장합니다. |
| `save_weights_only`       | (Boolean): True인 경우 모델의 가중치만 저장합니다. |
| `mode`                    | (`auto`, `min` 또는 `max`): `val_acc`의 경우 `max`로 설정하고 `val_loss`의 경우 `min`으로 설정하는 식입니다. |
| `save_freq`               | ("epoch" 또는 int): ‘epoch’를 사용하는 경우 콜백은 각 에포크 후에 모델을 저장합니다. 정수를 사용하는 경우 콜백은 이만큼의 배치가 끝날 때 모델을 저장합니다. `val_acc` 또는 `val_loss`와 같은 검증 메트릭을 모니터링할 때 이러한 메트릭은 에포크가 끝날 때만 사용할 수 있으므로 `save_freq`를 "epoch"로 설정해야 합니다. |
| `options`                 | (str): `save_weights_only`가 true인 경우 선택적 `tf.train.CheckpointOptions` 오브젝트이거나 `save_weights_only`가 false인 경우 선택적 `tf.saved_model.SaveOptions` 오브젝트입니다. |
| `initial_value_threshold` | (float): 모니터링할 메트릭의 부동 소수점 초기 "최상" 값입니다. |

### N 에포크 후에 체크포인트 기록

기본적으로(`save_freq="epoch"`) 콜백은 각 에포크 후에 체크포인트를 만들고 아티팩트로 업로드합니다. 특정 수의 배치 후에 체크포인트를 만들려면 `save_freq`를 정수로 설정합니다. `N` 에포크 후에 체크포인트를 만들려면 `train` 데이터 로더의 카디널리티를 계산하여 `save_freq`에 전달합니다.

```python
WandbModelCheckpoint(
    filepath="models/",
    save_freq=int((trainloader.cardinality()*N).numpy())
)
```

### TPU 아키텍처에서 체크포인트를 효율적으로 기록

TPU에서 체크포인트를 만드는 동안 `UnimplementedError: File system scheme '[local]' not implemented` 오류 메시지가 발생할 수 있습니다. 이는 모델 디렉토리(`filepath`)가 클라우드 스토리지 버킷 경로(`gs://bucket-name/...`)를 사용해야 하고 이 버킷에 TPU 서버에서 엑세스할 수 있어야 하기 때문에 발생합니다. 그러나 체크포인트를 만드는 데 로컬 경로를 사용할 수 있으며, 이는 Artifacts로 업로드됩니다.

```python
checkpoint_options = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")

WandbModelCheckpoint(
    filepath="models/,
    options=checkpoint_options,
)
```

## `WandbEvalCallback`을 사용하여 모델 예측 시각화

{{< cta-button colabLink="https://github.com/wandb/examples/blob/e66f16fbe7ae7a2e636d59350a50059d3f7e5494/colabs/keras/Use_WandbEvalCallback_in_your_Keras_workflow.ipynb" >}}

`WandbEvalCallback`은 주로 모델 예측을 위해, 그리고 부차적으로 데이터셋 시각화를 위해 Keras 콜백을 빌드하는 데 사용되는 추상 기본 클래스입니다.

이 추상 콜백은 데이터셋 및 작업과 관련하여 독립적입니다. 이를 사용하려면 이 기본 `WandbEvalCallback` 콜백 클래스에서 상속하고 `add_ground_truth` 및 `add_model_prediction` 메서드를 구현합니다.

`WandbEvalCallback`은 다음과 같은 메서드를 제공하는 유틸리티 클래스입니다.

* 데이터 및 예측 `wandb.Table` 인스턴스를 만듭니다.
* 데이터 및 예측 Tables를 `wandb.Artifact`로 기록합니다.
* `on_train_begin`에 데이터 테이블을 기록합니다.
* `on_epoch_end`에 예측 테이블을 기록합니다.

다음 예제에서는 이미지 분류 작업에 `WandbClfEvalCallback`을 사용합니다. 이 예제 콜백은 검증 데이터(`data_table`)를 W&B에 기록하고 추론을 수행한 다음 모든 에포크가 끝날 때 예측(`pred_table`)을 W&B에 기록합니다.

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbEvalCallback


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

# 새로운 W&B run 초기화
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

{{% alert %}}
W&B [Artifact 페이지]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph" lang="ko" >}})에는 **Workspace** 페이지가 아닌 Table 로그가 기본적으로 포함되어 있습니다.
{{% /alert %}}

### `WandbEvalCallback` 참조

| 파라미터 | 설명 |
| -------------------- | ------------------------------------------------ |
| `data_table_columns` | (list) `data_table`의 열 이름 목록입니다. |
| `pred_table_columns` | (list) `pred_table`의 열 이름 목록입니다. |

### 메모리 공간 세부 정보

`on_train_begin` 메서드가 호출되면 `data_table`을 W&B에 기록합니다. W&B Artifact로 업로드되면 `data_table_ref` 클래스 변수를 사용하여 엑세스할 수 있는 이 테이블에 대한 참조가 생성됩니다. `data_table_ref`는 `self.data_table_ref[idx][n]`과 같이 인덱싱할 수 있는 2D 목록이며, 여기서 `idx`는 행 번호이고 `n`은 열 번호입니다. 아래 예에서 사용법을 살펴보겠습니다.

### 콜백 사용자 정의

`on_train_begin` 또는 `on_epoch_end` 메서드를 재정의하여 더 세분화된 제어를 할 수 있습니다. `N` 배치 후에 샘플을 기록하려면 `on_train_batch_end` 메서드를 구현하면 됩니다.

{{% alert %}}
💡 `WandbEvalCallback`을 상속하여 모델 예측 시각화를 위한 콜백을 구현하고 있으며 설명이 필요하거나 수정해야 할 사항이 있는 경우 [문제](https://github.com/wandb/wandb/issues)를 열어 알려주시기 바랍니다.
{{% /alert %}}

## `WandbCallback` [기존]

W&B 라이브러리 [`WandbCallback`]({{< relref path="/ref/python/integrations/keras/wandbcallback" lang="ko" >}}) 클래스를 사용하여 `model.fit`에서 추적된 모든 메트릭 및 손실 값을 자동으로 저장합니다.

```python
import wandb
from wandb.integration.keras import WandbCallback

wandb.init(config={"hyper": "parameter"})

...  # Keras에서 모델을 설정하는 코드

# 콜백을 model.fit에 전달
model.fit(
    X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbCallback()]
)
```

짧은 비디오 [1분 이내에 Keras 및 Weights & Biases 시작하기](https://www.youtube.com/watch?ab_channel=Weights&Biases&v=4FjDIJ-vO_M)를 시청할 수 있습니다.

자세한 내용은 비디오 [Weights & Biases를 Keras와 통합하기](https://www.youtube.com/watch?v=Bsudo7jbMow\&ab_channel=Weights%26Biases)를 시청하세요. [Colab Jupyter Notebook](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Keras_pipeline_with_Weights_and_Biases.ipynb)을 검토할 수 있습니다.

{{% alert %}}
스크립트(예: [Fashion MNIST 예제](https://github.com/wandb/examples/blob/master/examples/keras/keras-cnn-fashion/train.py)) 및 생성되는 [W&B 대시보드](https://wandb.ai/wandb/keras-fashion-mnist/runs/5z1d85qs)를 보려면 [예제 리포지토리](https://github.com/wandb/examples)를 참조하세요.
{{% /alert %}}

`WandbCallback` 클래스는 모니터링할 메트릭 지정, 가중치 및 그레이디언트 추적, training_data 및 validation_data에 대한 예측 기록 등 다양한 로깅 구성 옵션을 지원합니다.

자세한 내용은 [`keras.WandbCallback`에 대한 참조 문서]({{< relref path="/ref/python/integrations/keras/wandbcallback.md" lang="ko" >}})를 확인하세요.

`WandbCallback`

* Keras에서 수집한 모든 메트릭(손실 및 `keras_model.compile()`에 전달된 항목)에서 기록 데이터를 자동으로 기록합니다.
* `monitor` 및 `mode` 속성으로 정의된 대로 "최상" 트레이닝 단계와 연결된 run에 대한 요약 메트릭을 설정합니다. 기본값은 최소 `val_loss`가 있는 에포크입니다. `WandbCallback`은 기본적으로 최상의 `epoch`와 연결된 모델을 저장합니다.
* 선택적으로 그레이디언트 및 파라미터 히스토그램을 기록합니다.
* 선택적으로 시각화를 위해 트레이닝 및 검증 데이터를 wandb에 저장합니다.

### `WandbCallback` 참조

| 인수 | |
| -------------------------- | ------------------------------------------- |
| `monitor`                  | (str) 모니터링할 메트릭 이름입니다. 기본값은 `val_loss`입니다. |
| `mode`                     | (str) {`auto`, `min`, `max`} 중 하나입니다. `min` - 모니터가 최소화될 때 모델 저장 `max` - 모니터가 최대화될 때 모델 저장 `auto` - 모델을 저장할 시기를 추측하려고 시도합니다(기본값). |
| `save_model`               | True - 모니터가 이전의 모든 에포크보다 나을 때 모델 저장 False - 모델을 저장하지 않음 |
| `save_graph`               | (boolean) True인 경우 모델 그래프를 wandb에 저장합니다(기본값은 True). |
| `save_weights_only`        | (boolean) True인 경우 모델의 가중치(`model.save_weights(filepath)`)만 저장합니다. 그렇지 않으면 전체 모델을 저장합니다. |
| `log_weights`              | (boolean) True인 경우 모델 레이어의 가중치 히스토그램을 저장합니다. |
| `log_gradients`            | (boolean) True인 경우 트레이닝 그레이디언트의 히스토그램을 기록합니다. |
| `training_data`            | (tuple) `model.fit`에 전달된 것과 동일한 형식 `(X,y)`입니다. 그레이디언트 계산에 필요합니다. `log_gradients`가 `True`인 경우 필수입니다. |
| `validation_data`          | (tuple) `model.fit`에 전달된 것과 동일한 형식 `(X,y)`입니다. wandb가 시각화할 데이터 세트입니다. 이 필드를 설정하면 모든 에포크에서 wandb는 적은 수의 예측을 수행하고 나중에 시각화할 수 있도록 결과를 저장합니다. |
| `generator`                | (generator) wandb가 시각화할 검증 데이터를 반환하는 생성기입니다. 이 생성기는 튜플 `(X,y)`를 반환해야 합니다. wandb가 특정 데이터 예제를 시각화하려면 `validate_data` 또는 생성기를 설정해야 합니다. |
| `validation_steps`         | (int) `validation_data`가 생성기인 경우 전체 검증 세트에 대해 생성기를 실행할 단계 수입니다. |
| `labels`                   | (list) wandb로 데이터를 시각화하는 경우 이 레이블 목록은 여러 클래스로 분류기를 빌드하는 경우 숫자 출력을 이해하기 쉬운 문자열로 변환합니다. 이진 분류기의 경우 두 개의 레이블 목록([`false 레이블`, `true 레이블`])을 전달할 수 있습니다. `validate_data` 및 `generator`가 모두 false인 경우 아무 작업도 수행하지 않습니다. |
| `predictions`              | (int) 각 에포크에서 시각화를 위해 수행할 예측 수이며, 최대값은 100입니다. |
| `input_type`               | (string) 시각화를 돕기 위한 모델 입력 유형입니다. (`image`, `images`, `segmentation_mask`) 중 하나일 수 있습니다. |
| `output_type`              | (string) 시각화를 돕기 위한 모델 출력 유형입니다. (`image`, `images`, `segmentation_mask`) 중 하나일 수 있습니다. |
| `log_evaluation`           | (boolean) True인 경우 각 에포크에서 검증 데이터와 모델의 예측을 포함하는 Table을 저장합니다. 자세한 내용은 `validation_indexes`, `validation_row_processor` 및 `output_row_processor`를 참조하세요. |
| `class_colors`             | ([float, float, float]) 입력 또는 출력이 분할 마스크인 경우 각 클래스에 대한 rgb 튜플(범위 0-1)을 포함하는 배열입니다. |
| `log_batch_frequency`      | (integer) None인 경우 콜백은 모든 에포크를 기록합니다. 정수로 설정된 경우 콜백은 `log_batch_frequency` 배치마다 트레이닝 메트릭을 기록합니다. |
| `log_best_prefix`          | (string) None인 경우 추가 요약 메트릭을 저장하지 않습니다. 문자열로 설정된 경우 모니터링된 메트릭과 에포크 앞에 접두사를 붙이고 결과를 요약 메트릭으로 저장합니다. |
| `validation_indexes`       | ([wandb.data_types._TableLinkMixin]) 각 검증 예제와 연결할 인덱스 키의 정렬된 목록입니다. `log_evaluation`이 True이고 `validation_indexes`를 제공하는 경우 검증 데이터 Table을 만들지 않습니다. 대신 각 예측을 `TableLinkMixin`으로 표시된 행과 연결합니다. 행 키 목록을 가져오려면 `Table.get_index()`를 사용하세요. |
| `validation_row_processor` | (Callable) 검증 데이터에 적용할 함수로, 일반적으로 데이터를 시각화하는 데 사용됩니다. 이 함수는 `ndx`(int)와 `row`(dict)를 받습니다. 모델에 입력이 하나 있는 경우 `row["input"]`에 행에 대한 입력 데이터가 포함됩니다. 그렇지 않으면 입력 슬롯의 이름이 포함됩니다. 적합 함수가 대상이 하나인 경우 `row["target"]`에 행에 대한 대상 데이터가 포함됩니다. 그렇지 않으면 출력 슬롯의 이름이 포함됩니다. 예를 들어 입력 데이터가 단일 배열인 경우 데이터를 이미지로 시각화하려면 `lambda ndx, row: {"img": wandb.Image(row["input"])}`를 프로세서로 제공합니다. `log_evaluation`이 False이거나 `validation_indexes`가 있는 경우 무시됩니다. |
| `output_row_processor`     | (Callable) `validation_row_processor`와 동일하지만 모델의 출력에 적용됩니다. `row["output"]`에 모델 출력 결과가 포함됩니다. |
| `infer_missing_processors` | (Boolean) 누락된 경우 `validation_row_processor` 및 `output_row_processor`를 유추할지 여부를 결정합니다. 기본값은 True입니다. `labels`를 제공하는 경우 W&B는 적절한 분류 유형 프로세서를 유추하려고 시도합니다. |
| `log_evaluation_frequency` | (int) 평가 결과를 기록하는 빈도를 결정합니다. 기본값은 트레이닝이 끝날 때만 기록하는 `0`입니다. 모든 에포크를 기록하려면 1로 설정하고, 다른 모든 에포크를 기록하려면 2로 설정하는 식입니다. `log_evaluation`이 False인 경우 아무런 효과가 없습니다. |

## 자주 묻는 질문

### `wandb`로 `Keras` 멀티프로세싱을 어떻게 사용합니까?

`use_multiprocessing=True`를 설정하면 다음 오류가 발생할 수 있습니다.

```python
Error("You must call wandb.init() before wandb.config.batch_size")
```

해결 방법:

1. `Sequence` 클래스 구성에서 `wandb.init(group='...')`를 추가합니다.
2. `main`에서 `if __name__ == "__main__":`를 사용하고 있는지 확인하고 스크립트 로직의 나머지 부분을 그 안에 넣습니다.
```
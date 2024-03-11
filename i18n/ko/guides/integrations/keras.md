---
displayed_sidebar: default
---

# Keras

[**여기서 Colab 노트북에서 시도해 보세요 →**](http://wandb.me/intro-keras)

## Weights & Biases Keras 콜백

Keras와 TensorFlow 사용자를 위한 세 가지 새로운 콜백을 `wandb` v0.13.4부터 사용할 수 있습니다. 레거시 `WandbCallback`에 대해서는 아래로 스크롤하세요.


**`WandbMetricsLogger`** : [실험 추적](https://docs.wandb.ai/guides/track)을 위해 이 콜백을 사용하세요. 트레이닝 및 검증 메트릭과 시스템 메트릭을 Weights and Biases에 로그합니다.

**`WandbModelCheckpoint`** : 모델 체크포인트를 Weights and Biases [아티팩트](https://docs.wandb.ai/guides/data-and-model-versioning)에 로그하기 위해 이 콜백을 사용하세요.

**`WandbEvalCallback`**: 이 기본 콜백은 모델 예측값을 Weights and Biases [테이블](https://docs.wandb.ai/guides/tables)에 로그하여 대화형 시각화를 할 수 있습니다.

이 새로운 콜백은,

* Keras 설계 철학을 준수합니다
* 하나의 콜백(`WandbCallback`)을 사용하여 모든 것을 처리하는 인지 부담을 줄입니다,
* Keras 사용자가 자신의 특정 유스 케이스를 지원하기 위해 콜백을 수정하기 쉽게 만듭니다.

## `WandbMetricsLogger`를 사용한 실험 추적

[**여기서 Colab 노트북에서 시도해 보세요 →**](https://github.com/wandb/examples/blob/master/colabs/keras/Use\_WandbMetricLogger\_in\_your\_Keras\_workflow.ipynb)

`WandbMetricsLogger`는 `on_epoch_end`, `on_batch_end` 등과 같은 콜백 메소드가 인수로 취하는 Keras의 `logs` 사전을 자동으로 로그합니다.

이를 사용하면 다음을 제공합니다:

* `model.compile`에서 정의된 트레이닝 및 검증 메트릭
* 시스템(CPU/GPU/TPU) 메트릭
* 학습률(고정 값이나 학습률 스케줄러를 사용하는 경우 모두)

```python
import wandb
from wandb.keras import WandbMetricsLogger

# 새로운 W&B run을 초기화합니다
wandb.init(config={"bs": 12})

# WandbMetricsLogger를 model.fit에 전달합니다
model.fit(
    X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbMetricsLogger()]
)
```

**`WandbMetricsLogger` 참조**


| 파라미터 | 설명 | 
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `log_freq`            | ("epoch", "batch", 또는 int): "epoch"이면, 각 에포크의 끝에 메트릭을 로그합니다. "batch"이면, 각 배치의 끝에 메트릭을 로그합니다. int이면, 그 많은 배치의 끝에 메트릭을 로그합니다. 기본값은 "epoch"입니다.                                 |
| `initial_global_step` | (int): 일부 initial_epoch에서 트레이닝을 재개할 때 학습률 스케줄러가 사용되는 경우 학습률을 올바르게 로그하기 위해 이 인수를 사용합니다. 이는 step_size * initial_step으로 계산할 수 있습니다. 기본값은 0입니다. |

## `WandbModelCheckpoint`를 사용한 모델 체크포인트

[**여기서 Colab 노트북에서 시도해 보세요 →**](https://github.com/wandb/examples/blob/master/colabs/keras/Use\_WandbModelCheckpoint\_in\_your\_Keras\_workflow.ipynb)

`WandbModelCheckpoint` 콜백을 사용하여 Keras 모델(`SavedModel` 형식) 또는 모델 가중치를 주기적으로 저장하고 `wandb.Artifact`로 W&B에 업로드하여 모델 버전 관리를 수행하세요.

이 콜백은 [`tf.keras.callbacks.ModelCheckpoint`](https://www.tensorflow.org/api\_docs/python/tf/keras/callbacks/ModelCheckpoint)에서 서브클래스로 생성되므로, 체크포인트 로직은 부모 콜백에 의해 처리됩니다.

이 콜백은 다음과 같은 기능을 제공합니다:

* "best performance"를 달성한 모델을 저장합니다.
* 모든 에포크의 끝에서 성능에 상관없이 모델을 저장합니다.
* 에포크의 끝이나 고정된 수의 트레이닝 배치 후에 모델을 저장합니다.
* 모델 가중치만 저장하거나 전체 모델을 저장합니다.
* 모델을 SavedModel 형식이나 `.h5` 형식으로 저장합니다.

이 콜백은 `WandbMetricsLogger`와 함께 사용해야 합니다.

```python
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

# 새로운 W&B run을 초기화합니다
wandb.init(config={"bs": 12})

# WandbModelCheckpoint를 model.fit에 전달합니다
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
| `monitor`                 | (str): 모니터할 메트릭 이름.         |
| `verbose`                 | (int): 상세 모드, 0 또는 1. 모드 0은 조용하고, 모드 1은 콜백이 작업을 수행할 때 메시지를 표시합니다.   |
| `save_best_only`          | (bool): `save_best_only=True`인 경우, "best"로 간주되는 모델만 저장하며 최신 최고 모델은 (`monitor`)에 따라 덮어쓰기 되지 않습니다.     |
| `save_weights_only`       | (bool): True이면, 모델의 가중치만 저장됩니다.                                            |
| `mode`                    | ("auto", "min", 또는 "max"): val\_acc의 경우 ‘max’여야 하고, val\_loss의 경우 ‘min’이어야 합니다.  |
| `save_weights_only`       | (bool): True이면, 모델의 가중치만 저장됩니다.                                            |
| `save_freq`               | ("epoch" 또는 int): ‘epoch’를 사용할 때, 콜백은 각 에포크 후 모델을 저장합니다. 정수를 사용할 때, 콜백은 이 많은 배치의 끝에 모델을 저장합니다. `val_acc` 또는 `val_loss`와 같은 검증 메트릭을 모니터링하는 경우, `save_freq`는 "epoch"으로 설정되어야 합니다. |
| `options`                 | (str): `save_weights_only`가 true인 경우 선택적 `tf.train.CheckpointOptions` 객체 또는 `save_weights_only`가 false인 경우 선택적 `tf.saved_model.SaveOptions` 객체.    |
| `initial_value_threshold` | (float): 모니터링할 메트릭의 초기 "best" 값의 부동 소수점입니다.       |

### N 에포크 후 체크포인트를 로그하는 방법은?

기본값(`save_freq="epoch"`)으로 콜백은 각 에포크 후 체크포인트를 생성하고 아티팩트로 업로드합니다. `save_freq`에 정수를 전달하면 그 많은 배치 후에 체크포인트가 생성됩니다. `N` 에포크 후에 체크포인트를 하려면 트레인 데이터로더의 카디널리티를 계산하고 `save_freq`에 전달하세요:

```
WandbModelCheckpoint(
    filepath="models/",
    save_freq=int((trainloader.cardinality()*N).numpy())
)
```

### TPU 노드 아키텍처에서 효율적으로 체크포인트를 로그하는 방법은?

TPU에서 체크포인트를 할 때 `UnimplementedError: File system scheme '[local]' not implemented` 오류 메시지를 만날 수 있습니다. 이는 모델 디렉토리(`filepath`)가 클라우드 스토리지 버킷 경로(`gs://bucket-name/...`)를 사용해야 하며 이 버킷은 TPU 서버에서 접근할 수 있어야 합니다. 그러나 로컬 경로를 사용하여 체크포인트를 할 수 있으며, 이는 차례로 아티팩트로 업로드됩니다.

```
checkpoint_options = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")

WandbModelCheckpoint(
    filepath="models/,
    options=checkpoint_options,
)
```

## `WandbEvalCallback`를 사용한 모델 예측 시각화

[**여기서 Colab 노트북에서 시도해 보세요 →**](https://github.com/wandb/examples/blob/e66f16fbe7ae7a2e636d59350a50059d3f7e5494/colabs/keras/Use_WandbEvalCallback_in_your_Keras_workflow.ipynb)

`WandbEvalCallback`은 주로 모델 예측을 위해, 그리고 부차적으로 데이터셋 시각화를 위해 Keras 콜백을 구축하기 위한 추상 기본 클래스입니다.

이 추상 콜백은 데이터셋과 작업에 대해 불가지한 상태입니다. 이를 사용하려면 이 기본 `WandbEvalCallback` 콜백 클래스에서 상속하고 `add_ground_truth` 및 `add_model_prediction` 메소드를 구현하세요.

`WandbEvalCallback`은 유용한 메소드를 제공하는 유틸리티 클래스입니다:

* 데이터와 예측 `wandb.Table` 인스턴스를 생성합니다,
* 데이터와 예측 테이블을 `wandb.Artifact`로 로그합니다
* 데이터 테이블은 `on_train_begin`에 로그합니다
* 예측 테이블은 `on_epoch_end`에 로그합니다

예를 들어, 아래에서 이미지 분류 작업을 위한 `WandbClfEvalCallback`을 구현했습니다. 이 예제 콜백은:

* 검증 데이터(`data_table`)를 W&B에 로그합니다,
* 각 에포크의 끝에서 추론을 수행하고 예측(`pred_table`)을 W&B에 로그합니다.

```python
import wandb
from wandb.keras import WandbMetricsLogger, WandbEvalCallback


# 모델 예측 시각화 콜백을 구현합니다
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

# 새로운 W&B run을 초기화합니다
wandb.init(config={"hyper": "parameter"})

# Model.fit에 콜백을 추가합니다
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
💡 테이블은 기본적으로 W&B [아티팩트 페이지](https://docs.wandb.ai/ref/app/pages/project-page#artifacts-tab)에 로그되며 [워크스페이스](https://docs.wandb.ai/ref/app/pages/workspaces) 페이지에는 로그되지 않습니다.
:::

**`WandbEvalCallback` 참조**

| 파라미터            | 설명                                      |
| -------------------- | ------------------------------------------------ |
| `data_table_columns` | (`data_table`에 대한 열 이름 목록 |
| `pred_table_columns` | (`pred_table`에 대한 열 이름 목록 |

### 메모리 사용량이 어떻게 줄어드나요?

`on_train_begin` 메소드가 호출될 때 `data_table`을 W&B에 로그합니다. W&B 아티팩트로 업로드되면 이 테이블에 대한 참조를 얻을 수 있으며, 이는 `data_table_ref` 클래스 변수를 사용하여 액세스할 수 있습니다. `data_table_ref`는 `self.data_table_ref[idx][n]`처럼 인덱싱할 수 있는 2D 리스트입니다. 여기서 `idx`는 행 번호이고 `n`은 열 번호입니다. 아래 예제에서 사용법을 확인하세요.

### 콜백을 더 자세히 사용자 정의하는 방법

`on_train_begin` 또는 `on_epoch_end` 메소드를 오버라이드하여 더 세밀한 제어를 할 수 있습니다. `N` 배치 후 샘플을 로그하려면 `on_train_batch_end` 메소드를 구현할 수 있습니다.

:::info
💡 `WandbEvalCallback`을 상속하여 모델 예측 시각화 콜백을 구현하는 경우 무엇인가 명확히 하거나 수정해야 한다면, [이슈](https://github.com/wandb/wandb/issues)를 열어서 알려주세요.
:::

## WandbCallback [레거시]

`model.fit`에서 추적된 모든 메트릭과 손실 값을 자동으로 저장하기 위해 W&B 라이브러리 [`WandbCallback`](https://docs.wandb.ai/ref/python/integrations/keras/wandbcallback) 클래스를 사용하세요.

```python
import wandb
from wandb.keras import WandbCallback

wandb.init(config={"hyper": "parameter"})

...  # Keras에서 모델 설정 코드

# model.fit에 콜백을 전달합니다
model.fit(
    X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbCallback()]
)
```

**사용 예제**

W&B와 Keras를 처음 통합하는 경우 이 한 분짜리 단계별 비디오를 보세요: [Keras와 Weights & Biases로 1분 만에 시작하기](https://www.youtube.com/watch?ab_channel=Weights&Biases&v=4FjDIJ-vO_M)

더 자세한 비디오는 [Keras와 Weights & Bi
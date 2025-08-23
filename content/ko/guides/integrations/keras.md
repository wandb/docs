---
title: Keras
menu:
  default:
    identifier: ko-guides-integrations-keras
    parent: integrations
weight: 160
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases_keras.ipynb" >}}

## Keras 콜백(callbacks)

W&B에는 `wandb` v0.13.4부터 사용할 수 있는 Keras 콜백 3가지가 있습니다. 기존의 `WandbCallback`에 대한 내용은 아래에서 확인하세요.

- **`WandbMetricsLogger`** : 이 콜백은 [실험 추적]({{< relref path="/guides/models/track" lang="ko" >}})에 사용합니다. 트레이닝 및 검증 메트릭과 시스템 메트릭을 함께 W&B에 기록합니다.

- **`WandbModelCheckpoint`** : 이 콜백은 모델 체크포인트를 W&B [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ko" >}})에 기록할 때 사용합니다.

- **`WandbEvalCallback`** : 이 베이스 콜백은 모델 예측값을 W&B [Tables]({{< relref path="/guides/models/tables/" lang="ko" >}})에 기록하여 대화형 시각화를 지원합니다.

이 새로운 콜백들은 다음과 같은 특징을 갖습니다.

* Keras의 설계 철학을 따릅니다.
* 모든 기능을 단일 콜백(`WandbCallback`)으로 처리할 때의 인지적 부담을 줄여줍니다.
* Keras 사용자가 자신만의 유스 케이스에 맞게 콜백을 서브클래싱(subclassing)하여 쉽게 수정할 수 있도록 해줍니다.

## `WandbMetricsLogger`로 실험 추적하기

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/keras/Use_WandbMetricLogger_in_your_Keras_workflow.ipynb" >}}

`WandbMetricsLogger`는 `on_epoch_end`, `on_batch_end` 등과 같이 콜백 메소드에서 인수로 전달받는 Keras의 `logs` 사전을 자동으로 기록합니다.

이를 통해 다음을 추적할 수 있습니다:

* `model.compile`에 정의된 트레이닝 및 검증 메트릭
* 시스템 메트릭(CPU/GPU/TPU 사용량 등)
* 학습률(고정 값 또는 스케줄러 모두 가능)

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger

# 새로운 W&B Run 시작
wandb.init(config={"bs": 12})

# model.fit에 WandbMetricsLogger 전달
model.fit(
    X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbMetricsLogger()]
)
```

### `WandbMetricsLogger` 참고

| 파라미터 | 설명 | 
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `log_freq`            | (`epoch`, `batch`, 또는 `int`): `epoch`로 설정하면 각 에포크가 끝날 때 메트릭이 기록됩니다. `batch`는 각 배치가 끝날 때 기록합니다. `int`는 해당 배치 수마다 한 번씩 기록합니다. 기본값은 `epoch`입니다.                                 |
| `initial_global_step` | (int): 트레이닝을 재개할 때(`initial_epoch` 사용)와 학습률 스케줄러가 적용될 때 올바른 학습률 기록을 위해 사용합니다. 보통 step_size * initial_step로 계산하며 기본값은 0입니다. |

## `WandbModelCheckpoint`로 모델 체크포인트 저장

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/keras/Use_WandbModelCheckpoint_in_your_Keras_workflow.ipynb" >}}

`WandbModelCheckpoint` 콜백을 사용하면 Keras 모델(`SavedModel` 포맷) 또는 모델 가중치를 주기적으로 저장하고, 이를 W&B의 `wandb.Artifact`로 업로드하여 모델 버전 관리를 할 수 있습니다.

이 콜백은 [`tf.keras.callbacks.ModelCheckpoint`](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint)에서 상속받았으며, 체크포인트 관련 로직은 부모 콜백에서 처리됩니다.

이 콜백이 저장하는 경우:

* 모니터링 기준에서 가장 좋은 성능을 보인 모델
* 각 에포크가 끝날 때마다(성능과 상관 없이)
* 에포크가 끝나거나, 지정한 배치 수마다
* 전체 모델 또는 가중치만
* `SavedModel` 포맷 또는 `.h5` 포맷

`WandbMetricsLogger` 콜백과 함께 사용하는 것이 좋습니다.

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

# 새로운 W&B Run 시작
wandb.init(config={"bs": 12})

# model.fit에 WandbModelCheckpoint 전달
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

### `WandbModelCheckpoint` 참고

| 파라미터 | 설명 | 
| ------------------------- |  ---- | 
| `filepath`   | (str): 모델 파일이 저장될 경로 |  
| `monitor`                 | (str): 모니터링할 메트릭 이름         |
| `verbose`                 | (int): 출력 모드, 0 또는 1. 0은 메시지 비표시, 1은 콜백이 동작할 때 메시지 표시   |
| `save_best_only`          | (Boolean): `save_best_only=True`이면 모니터 기준 최고/최신 모델만 저장. 기준은 `monitor`와 `mode` 속성이 결정함   |
| `save_weights_only`       | (Boolean): True일 경우 모델의 가중치만 저장                                          |
| `mode`                    | (`auto`, `min`, `max`): `val_acc`는 `max`, `val_loss`는 `min` 등으로 설정  |                     
| `save_freq`               | ("epoch" 또는 int):  ‘epoch’ 사용 시 매 에포크마다 저장, int 사용 시 지정한 배치마다 저장. 검증 메트릭 등을 모니터링한다면 반드시 "epoch"로 설정해야 함(해당 메트릭은 에포크 마지막에만 사용 가능) |
| `options`                 | (str): `save_weights_only`가 True면 `tf.train.CheckpointOptions` 객체, False면 `tf.saved_model.SaveOptions` 객체를 전달할 수 있음 |
| `initial_value_threshold` | (float): 모니터링할 메트릭의 "best" 값 초기값    |

### N 에포크마다 체크포인트 기록

기본적으로(`save_freq="epoch"`), 콜백은 매 에포크가 끝날 때마다 체크포인트를 생성하고 이를 아티팩트로 업로드합니다. 특정 배치마다 체크포인트를 생성하려면 `save_freq`에 정수를 지정하세요. `N` 에포크마다 체크포인트를 만들고 싶다면 `train` dataloader의 데이터 개수를 구해 `save_freq`에 전달하세요.

```python
WandbModelCheckpoint(
    filepath="models/",
    save_freq=int((trainloader.cardinality()*N).numpy())
)
```

### TPU 아키텍처에서 체크포인트 효율적으로 기록하기

TPU 환경에서 체크포인트를 기록할 때 `UnimplementedError: File system scheme '[local]' not implemented` 에러가 발생할 수 있습니다. 이는 모델 디렉토리(`filepath`)가 클라우드 스토리지 버킷 경로(`gs://bucket-name/...`)를 사용해야 하고, 해당 버킷이 TPU 서버에서 엑세스 가능해야 함을 의미합니다. 하지만 로컬 경로에 체크포인트를 저장한 뒤 이를 Artifacts로 업로드하여 사용할 수도 있습니다.

```python
checkpoint_options = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")

WandbModelCheckpoint(
    filepath="models/,
    options=checkpoint_options,
)
```

## `WandbEvalCallback`으로 모델 예측값 시각화하기

{{< cta-button colabLink="https://github.com/wandb/examples/blob/e66f16fbe7ae7a2e636d59350a50059d3f7e5494/colabs/keras/Use_WandbEvalCallback_in_your_Keras_workflow.ipynb" >}}

`WandbEvalCallback`은 주로 모델 예측, 그리고 부차적으로 데이터셋 시각화를 위해 생성된 Keras 콜백의 추상 베이스 클래스입니다.

이 추상 콜백은 데이터셋과 태스크에 영향을 받지 않습니다. 사용하려면 이 베이스 콜백을 상속받아 `add_ground_truth`와 `add_model_prediction` 메소드를 구현하세요.

`WandbEvalCallback`이 제공하는 주요 메소드:

* 데이터 및 예측값 `wandb.Table` 인스턴스 생성
* 데이터 및 예측값 Tables를 `wandb.Artifact`로 기록
* 트레이닝 시작시(`on_train_begin`) 데이터 테이블 기록
* 에포크 종료시(`on_epoch_end`) 예측 테이블 기록

아래 예시는 이미지 분류 작업에 `WandbClfEvalCallback`을 사용하는 코드입니다. 검증 데이터(`data_table`)를 W&B에 기록하고 예측값(`pred_table`)을 매 에포크 끝에 W&B에 기록합니다.

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

# 새로운 W&B Run 시작
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

{{% alert %}}
W&B의 [Artifact 페이지]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph" lang="ko" >}})에서는 기본적으로 Table 로그가 표시됩니다(Workspace 페이지가 아니라).
{{% /alert %}}

### `WandbEvalCallback` 참고

| 파라미터            | 설명                                      |
| -------------------- | ------------------------------------------------ |
| `data_table_columns` | (list) `data_table`의 컬럼명 리스트 |
| `pred_table_columns` | (list) `pred_table`의 컬럼명 리스트 |

### 메모리 사용량 상세

`on_train_begin` 메서드가 호출될 때 `data_table`을 W&B에 업로드합니다. 아티팩트가 업로드되면 이 테이블에 대한 참조(`data_table_ref`)를 사용하여 클래스 변수에서 접근할 수 있습니다. `data_table_ref`는 2차원 리스트로, `self.data_table_ref[idx][n]`에서 `idx`는 행 번호, `n`은 열 번호입니다. 아래 예시에서 사용법을 볼 수 있습니다.

### 콜백 커스터마이징

더 세밀하게 제어하고 싶다면 `on_train_begin`이나 `on_epoch_end` 메소드를 오버라이드할 수 있습니다. 샘플을 `N` 배치마다 기록하고 싶다면 `on_train_batch_end`를 구현하면 됩니다.

{{% alert %}}
`WandbEvalCallback`을 상속받아 모델 예측 시각화 콜백을 구현하다가 궁금하거나 개선할 점이 있다면, [issue](https://github.com/wandb/wandb/issues)를 남겨주세요.
{{% /alert %}}

## `WandbCallback` [레거시]

W&B의 `WandbCallback` 클래스를 사용하면 `model.fit`을 통해 추적되는 모든 메트릭과 손실 값들을 자동으로 기록할 수 있습니다.

```python
import wandb
from wandb.integration.keras import WandbCallback

wandb.init(config={"hyper": "parameter"})

...  # Keras에서 모델을 설정하는 코드

# model.fit에 콜백 전달
model.fit(
    X_train, y_train, validation_data=(X_test, y_test), callbacks=[WandbCallback()]
)
```

짧은 영상 [Keras와 W&B 빠르게 시작하기](https://www.youtube.com/watch?ab_channel=Weights&Biases&v=4FjDIJ-vO_M)를 참고하세요.

더 자세한 내용은 [W&B와 Keras 통합하기](https://www.youtube.com/watch?v=Bsudo7jbMow\&ab_channel=Weights%26Biases) 영상과 [Colab 주피터 노트북](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Keras_pipeline_with_Weights_and_Biases.ipynb) 자료도 확인하실 수 있습니다.

{{% alert %}}
[Fashion MNIST 예시](https://github.com/wandb/examples/blob/master/examples/keras/keras-cnn-fashion/train.py)를 비롯한 다양한 스크립트는 [예제 저장소](https://github.com/wandb/examples)와, 해당으로 생성된 [W&B Dashboard](https://wandb.ai/wandb/keras-fashion-mnist/runs/5z1d85qs)에서 확인할 수 있습니다.
{{% /alert %}}

`WandbCallback` 클래스는 다양한 로깅 설정이 가능합니다. 특정 메트릭 모니터링, 가중치 및 그레이디언트 추적, 트레이닝/검증 데이터에 대한 예측값 기록 등 다양한 기능을 지원합니다.

더 상세한 내용은 `keras.WandbCallback`의 레퍼런스 문서를 참고해주세요.

`WandbCallback`은

* Keras에서 수집된 모든 메트릭의 히스토리를 자동으로 기록합니다: 손실 및 `keras_model.compile()`에 전달된 값 등.
* `monitor`와 `mode` 속성(기본값은 `val_loss` 최소값)에 따라 "best" 트레이닝 스텝에 대한 summary 메트릭도 설정합니다. 기본적으로 최고 성능을 보인 에포크 모델을 저장합니다.
* 선택적으로 그레이디언트 및 파라미터 히스토그램을 기록할 수 있습니다.
* 선택적으로 트레이닝, 검증 데이터를 시각화용으로 저장할 수 있습니다.

### `WandbCallback` 참고

| 인자                  | 설명                                    |
| -------------------------- | ------------------------------------------- |
| `monitor`                  | (str) 모니터링할 메트릭 이름. 기본값은 `val_loss`                                                                   |
| `mode`                     | (str) {`auto`, `min`, `max`} 중 하나. `min` - 메트릭 값이 감소할 때 저장, `max` - 메트릭 값이 증가할 때 저장, `auto` - 자동 결정(기본값)                                                                                |
| `save_model`               | True - 이전 에포크보다 좋은 결과의 모델을 저장, False - 모델 저장 안 함                                       |
| `save_graph`               | (boolean) True일 경우 모델 그래프를 wandb에 저장(기본값 True)                                                         |
| `save_weights_only`        | (boolean) True이면 모델의 가중치만 저장(`model.save_weights(filepath)`). 아니라면 전체 모델 저장    |
| `log_weights`              | (boolean) True일 경우 각 레이어의 가중치 히스토그램 저장                                                |
| `log_gradients`            | (boolean) True일 경우 트레이닝 그레이디언트 히스토그램 기록                                                       |
| `training_data`            | (tuple) `(X,y)` 형식, 그레이디언트 계산에 필요. `log_gradients`가 True면 필수       |
| `validation_data`          | (tuple) `(X,y)` 형식, 시각화용 검증 데이터. 설정하면 매 에포크마다 W&B가 일부 예측값을 저장           |
| `generator`                | (generator) W&B에서 시각화용 검증 데이터를 반환하는 제너레이터. `(X,y)` 튜플 반환.  검증 데이터를 시각화하려면 `validate_data` 또는 generator 중 하나를 반드시 설정해야 함  |
| `validation_steps`         | (int) 검증 데이터가 제너레이터일 경우, 전체 검증 세트에 대해 몇 번 반복할지       |
| `labels`                   | (list) 데이터 시각화 시, 숫자 출력을 사람이 읽을 문자열로 바꿔줌. 바이너리 분류기는 두 개의 레이블 \[`false`일 때, `true`일 때\]. `validate_data`와 generator가 모두 false면 아무 효과 없음.    |
| `predictions`              | (int) 매 에포크마다 시각화용으로 예측할 샘플 개수, 최대 100         |
| `input_type`               | (string) 시각화를 돕기 위한 모델 입력 데이터의 타입. (`image`, `images`, `segmentation_mask`) 중 선택 가능  |
| `output_type`              | (string) 시각화를 돕기 위한 모델 출력 데이터의 타입. (`image`, `images`, `segmentation_mask`) 중 선택 가능    |
| `log_evaluation`           | (boolean) True일 경우, 각 에포크마다 검증 데이터와 모델 예측을 Table로 저장. 추가 상세는 `validation_indexes`, `validation_row_processor`, `output_row_processor` 참고     |
| `class_colors`             | (\[float, float, float]) 입력 또는 출력이 segmentation mask일 경우, 각 클래스별로 rgb 튜플(0-1 범위)로 표시       |
| `log_batch_frequency`      | (integer) None이면 각 에포크마다 기록, 정수로 지정하면 해당 배치마다 트레이닝 메트릭 기록          |
| `log_best_prefix`          | (string) None이면 summary 메트릭 저장 안 함. 문자열 지정 시, 모니터 메트릭과 에포크 앞에 붙여 저장   |
| `validation_indexes`       | (\[wandb.data_types._TableLinkMixin]) 각 검증 예제와 연결할 인덱스 키들의 순서 있는 리스트. `log_evaluation`이 True이고 `validation_indexes` 제공 시, 검증 데이터 Table을 생성하지 않음. 대신 각 예측과 인덱스 키를 연결. 인덱스 키 리스트는 `Table.get_index()`로 얻을 수 있음 |
| `validation_row_processor` | (Callable) 검증 데이터에 적용할 함수. 시각화 목적으로 자주 사용. 함수는 `ndx`(int)와 `row`(dict)를 인자로 받음. 단일 입력이면 `row["input"]`, 여러 입력이면 입럭 슬롯명. 타깃도 마찬가지. 예) 단일 array 이미지를 시각화하려면 `lambda ndx, row: {"img": wandb.Image(row["input"])}` 전달. `log_evaluation`이 False거나 validation_indexes 제공 시 무시됨. |
| `output_row_processor`     | (Callable) `validation_row_processor`와 유사하지만 모델 출력에 적용. `row["output"]`에 출력 결과 포함          |
| `infer_missing_processors` | (Boolean) 미지정시 processor 함수 추론 여부. 기본값 True. `labels`을 제공하는 경우 분류용 프로세서를 자동으로 생성함      |
| `log_evaluation_frequency` | (int) 평가 결과 기록 빈도. 기본값 0(트레이닝 마지막만 기록). 1이면 모든 에포크, 2이면 2 에포크마다 등. `log_evaluation`이 False면 영향 없음.    |

## 자주 묻는 질문

### `wandb`와 함께 Keras 멀티프로세싱 사용법은?

`use_multiprocessing=True`로 설정했을 때 다음과 같은 에러가 발생할 수 있습니다:

```python
Error("You must call wandb.init() before wandb.config.batch_size")
```

해결 방법:

1. `Sequence` 클래스 생성자에서 `wandb.init(group='...')`을 추가하세요.
2. `main`에서는 반드시 `if __name__ == "__main__":`를 사용하고, 나머지 스크립트 로직은 그 아래에 두세요.
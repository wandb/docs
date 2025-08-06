---
title: Keras 테이블
menu:
  tutorials:
    identifier: ko-tutorials-integration-tutorials-keras_tables
    parent: integration-tutorials
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Use_WandbEvalCallback_in_your_Keras_workflow.ipynb" >}}
W&B를 사용하여 기계학습 실험 추적, 데이터셋 버전 관리, 프로젝트 협업을 손쉽게 할 수 있습니다.

{{< img src="/images/tutorials/huggingface-why.png" alt="W&B 사용의 이점" >}}

이 Colab 노트북에서는 `WandbEvalCallback`을 소개합니다. 이는 추상 콜백으로, 모델 예측값 시각화 및 데이터셋 시각화에 유용한 콜백을 확장하여 만들 수 있습니다. 

## 셋업 및 설치

먼저, W&B의 최신 버전을 설치합시다. 이후 이 colab 인스턴스에서 W&B를 사용할 수 있도록 인증 과정을 진행합니다.

```shell
pip install -qq -U wandb
```

```python
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow_datasets as tfds

# W&B 관련 라이브러리 임포트
import wandb
from wandb.integration.keras import WandbMetricsLogger
from wandb.integration.keras import WandbModelCheckpoint
from wandb.integration.keras import WandbEvalCallback
```

W&B를 처음 사용하거나 아직 로그인하지 않은 경우, `wandb.login()` 실행 이후 나타나는 링크를 클릭하면 회원가입/로그인 페이지로 이동합니다. [무료 계정](https://wandb.ai/signup) 가입은 클릭 몇 번이면 끝납니다.

```python
wandb.login()
```

## 하이퍼파라미터

재현 가능한 기계학습을 위해 올바른 config 시스템을 사용하는 것이 권장됩니다. W&B를 사용하면 각 실험의 하이퍼파라미터를 추적할 수 있습니다. 이 colab에서는 간단한 Python `dict`로 config를 관리합니다.

```python
configs = dict(
    num_classes=10,
    shuffle_buffer=1024,
    batch_size=64,
    image_size=28,
    image_channels=1,
    earlystopping_patience=3,
    learning_rate=1e-3,
    epochs=10,
)
```

## 데이터셋

이 colab에서는 TensorFlow Dataset 카탈로그의 [CIFAR100](https://www.tensorflow.org/datasets/catalog/cifar100) 데이터셋을 사용할 예정입니다. TensorFlow/Keras를 활용해 간단한 이미지 분류 파이프라인을 만들어볼 것입니다.

```python
train_ds, valid_ds = tfds.load("fashion_mnist", split=["train", "test"])
```

```
AUTOTUNE = tf.data.AUTOTUNE

def parse_data(example):
    # 이미지 추출
    image = example["image"]
    # image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # 라벨 추출
    label = example["label"]
    label = tf.one_hot(label, depth=configs["num_classes"])

    return image, label

def get_dataloader(ds, configs, dataloader_type="train"):
    dataloader = ds.map(parse_data, num_parallel_calls=AUTOTUNE)

    if dataloader_type=="train":
        dataloader = dataloader.shuffle(configs["shuffle_buffer"])
      
    dataloader = (
        dataloader
        .batch(configs["batch_size"])
        .prefetch(AUTOTUNE)
    )

    return dataloader
```

```python
trainloader = get_dataloader(train_ds, configs)
validloader = get_dataloader(valid_ds, configs, dataloader_type="valid")
```

## 모델

```python
def get_model(configs):
    backbone = tf.keras.applications.mobilenet_v2.MobileNetV2(
        weights="imagenet", include_top=False
    )
    backbone.trainable = False

    inputs = layers.Input(
        shape=(configs["image_size"], configs["image_size"], configs["image_channels"])
    )
    resize = layers.Resizing(32, 32)(inputs)
    neck = layers.Conv2D(3, (3, 3), padding="same")(resize)
    preprocess_input = tf.keras.applications.mobilenet.preprocess_input(neck)
    x = backbone(preprocess_input)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(configs["num_classes"], activation="softmax")(x)

    return models.Model(inputs=inputs, outputs=outputs)
```

```python
tf.keras.backend.clear_session()
model = get_model(configs)
model.summary()
```

## 모델 컴파일

```python
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top@5_accuracy"),
    ],
)
```

## `WandbEvalCallback`

`WandbEvalCallback`은 주로 모델 예측값 시각화, 그 다음으로 데이터셋 시각화를 위한 Keras 콜백을 만들기 위한 추상 베이스 클래스입니다.

이 콜백은 데이터셋과 태스크에 상관없이 사용할 수 있는 추상 콜백입니다. 사용하려면 이 베이스 콜백을 상속해서 `add_ground_truth`와 `add_model_prediction` 메소드를 구현하면 됩니다.

`WandbEvalCallback`은 다음과 같은 유용한 메소드들을 제공합니다:

- 데이터 및 예측값을 가진 `wandb.Table` 객체 생성
- 데이터와 예측값 Table을 `wandb.Artifact`로 로그
- `on_train_begin`에서 데이터 테이블 로그
- `on_epoch_end`에서 예측값 테이블 로그

예시로, 아래에 이미지 분류 태스크를 위한 `WandbClfEvalCallback` 구현이 있습니다. 이 콜백은
- 검증 데이터를 W&B에 로그(`data_table`)
- 추론을 수행하고 매 에포크 종료 시 예측값을 W&B에 로그(`pred_table`)

## 메모리 사용량이 어떻게 줄어드는가

`on_train_begin` 메소드가 호출될 때, `data_table`을 W&B에 로그합니다. W&B Artifact로 업로드되면 이 테이블에 대한 참조를 받아와 `data_table_ref` 클래스 변수로 엑세스할 수 있습니다. `data_table_ref`는 2차원 리스트이며, `self.data_table_ref[idx][n]`처럼 인덱싱 할 수 있습니다 (`idx`는 행 번호, `n`은 열 번호). 아래 예제에서 어떻게 사용하는지 확인해보세요.

```python
class WandbClfEvalCallback(WandbEvalCallback):
    def __init__(
        self, validloader, data_table_columns, pred_table_columns, num_samples=100
    ):
        super().__init__(data_table_columns, pred_table_columns)

        self.val_data = validloader.unbatch().take(num_samples)

    def add_ground_truth(self, logs=None):
        for idx, (image, label) in enumerate(self.val_data):
            self.data_table.add_data(idx, wandb.Image(image), np.argmax(label, axis=-1))

    def add_model_predictions(self, epoch, logs=None):
        # 예측 결과 가져오기
        preds = self._inference()
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

    def _inference(self):
        preds = []
        for image, label in self.val_data:
            pred = self.model(tf.expand_dims(image, axis=0))
            argmax_pred = tf.argmax(pred, axis=-1).numpy()[0]
            preds.append(argmax_pred)

        return preds
```

## 학습

```python
# W&B Run 초기화
run = wandb.init(project="intro-keras", config=configs)

# 모델 학습
model.fit(
    trainloader,
    epochs=configs["epochs"],
    validation_data=validloader,
    callbacks=[
        WandbMetricsLogger(log_freq=10),
        WandbClfEvalCallback(
            validloader,
            data_table_columns=["idx", "image", "ground_truth"],
            pred_table_columns=["epoch", "idx", "image", "ground_truth", "prediction"],
        ),  # 여기서 WandbEvalCallback이 사용됨을 참고하세요
    ],
)

# W&B Run 종료
run.finish()
```
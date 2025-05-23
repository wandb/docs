---
title: Keras tables
menu:
  tutorials:
    identifier: ko-tutorials-integration-tutorials-keras_tables
    parent: integration-tutorials
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Use_WandbEvalCallback_in_your_Keras_workflow.ipynb" >}}
Weights & Biases를 사용하여 기계 학습 실험 추적, 데이터셋 버전 관리 및 프로젝트 협업을 수행하세요.

{{< img src="/images/tutorials/huggingface-why.png" alt="" >}}

이 Colab 노트북은 모델 예측 시각화 및 데이터셋 시각화를 위한 유용한 콜백을 구축하기 위해 상속될 수 있는 추상 콜백인 `WandbEvalCallback`을 소개합니다.

## 설정 및 설치

먼저, Weights & Biases의 최신 버전을 설치해 보겠습니다. 그런 다음 이 Colab 인스턴스를 인증하여 W&B를 사용합니다.

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

# Weights and Biases 관련 import
import wandb
from wandb.integration.keras import WandbMetricsLogger
from wandb.integration.keras import WandbModelCheckpoint
from wandb.integration.keras import WandbEvalCallback
```

W&B를 처음 사용하거나 로그인하지 않은 경우, `wandb.login()`을 실행한 후 나타나는 링크를 통해 가입/로그인 페이지로 이동합니다. 몇 번의 클릭만으로 [무료 계정](https://wandb.ai/signup)에 가입할 수 있습니다.

```python
wandb.login()
```

## 하이퍼파라미터

재현 가능한 기계 학습을 위해서는 적절한 구성 시스템을 사용하는 것이 좋습니다. W&B를 사용하여 모든 실험에 대한 하이퍼파라미터를 추적할 수 있습니다. 이 Colab에서는 간단한 Python `dict`를 구성 시스템으로 사용합니다.

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

이 Colab에서는 TensorFlow 데이터셋 카탈로그의 [CIFAR100](https://www.tensorflow.org/datasets/catalog/cifar100) 데이터셋을 사용합니다. TensorFlow/Keras를 사용하여 간단한 이미지 분류 파이프라인을 구축하는 것을 목표로 합니다.

```python
train_ds, valid_ds = tfds.load("fashion_mnist", split=["train", "test"])
```

```
AUTOTUNE = tf.data.AUTOTUNE


def parse_data(example):
    # 이미지 가져오기
    image = example["image"]
    # image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # 레이블 가져오기
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

`WandbEvalCallback`은 주로 모델 예측 시각화를 위해, 부차적으로는 데이터셋 시각화를 위해 Keras 콜백을 구축하는 추상 기본 클래스입니다.

이는 데이터셋 및 작업에 구애받지 않는 추상 콜백입니다. 이를 사용하려면 이 기본 콜백 클래스에서 상속하고 `add_ground_truth` 및 `add_model_prediction` 메소드를 구현합니다.

`WandbEvalCallback`은 다음과 같은 유용한 메소드를 제공하는 유틸리티 클래스입니다.

- 데이터 및 예측 `wandb.Table` 인스턴스 생성,
- 데이터 및 예측 Tables를 `wandb.Artifact`로 기록,
- 데이터 테이블을 `on_train_begin`에 기록,
- 예측 테이블을 `on_epoch_end`에 기록.

예를 들어, 아래에 이미지 분류 작업을 위한 `WandbClfEvalCallback`을 구현했습니다. 이 예제 콜백은 다음과 같습니다.
- 검증 데이터(`data_table`)를 W&B에 기록,
- 모든 에포크 종료 시 추론을 수행하고 예측(`pred_table`)을 W&B에 기록.

## 메모리 공간을 줄이는 방법

`on_train_begin` 메소드가 호출될 때 `data_table`을 W&B에 기록합니다. W&B Artifact로 업로드되면 `data_table_ref` 클래스 변수를 사용하여 엑세스할 수 있는 이 테이블에 대한 참조를 얻습니다. `data_table_ref`는 `self.data_table_ref[idx][n]`과 같이 인덱싱할 수 있는 2D 목록입니다. 여기서 `idx`는 행 번호이고 `n`은 열 번호입니다. 아래 예에서 사용법을 살펴보겠습니다.

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
        # 예측값 가져오기
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
# W&B run 초기화
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
        ),  # 여기에서 WandbEvalCallback 사용에 주목하세요.
    ],
)

# W&B run 닫기
run.finish()
```
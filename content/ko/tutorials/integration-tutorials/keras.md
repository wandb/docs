---
title: Keras
menu:
  tutorials:
    identifier: ko-tutorials-integration-tutorials-keras
    parent: integration-tutorials
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Use_WandbMetricLogger_in_your_Keras_workflow.ipynb" >}}
W&B를 사용하여 기계학습 실험 추적, 데이터셋 버전 관리, 프로젝트 협업을 시작하세요.

{{< img src="/images/tutorials/huggingface-why.png" alt="W&B 사용의 이점" >}}

이 Colab 노트북에서는 `WandbMetricsLogger` 콜백에 대해 소개합니다. 이 콜백을 [실험 추적]({{< relref path="/guides/models/track" lang="ko" >}})에 활용할 수 있습니다. 트레이닝 및 검증 메트릭과 시스템 메트릭까지 모두 W&B에 함께 로그합니다.

## 설치 및 환경 세팅

먼저, W&B의 최신 버전을 설치합니다. 이후 이 colab 인스턴스에서 W&B를 사용할 수 있도록 인증하겠습니다.

```shell
pip install -qq -U wandb
```

```python
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow_datasets as tfds

# W&B 관련 import
import wandb
from wandb.integration.keras import WandbMetricsLogger
```

W&B를 처음 사용하거나 로그인이 되어 있지 않다면, `wandb.login()` 실행 후 나타나는 링크를 눌러 회원가입/로그인 페이지로 이동하세요. [무료 계정](https://wandb.ai/signup) 가입은 몇 번의 클릭만으로 완료됩니다.

```python
wandb.login()
```

## 하이퍼파라미터

재현 가능한 기계학습을 위해서는 제대로 된 config 시스템 사용이 권장되는 모범 사례입니다. W&B를 통해 모든 실험의 하이퍼파라미터를 추적할 수 있습니다. 이 colab에서는 간단한 Python `dict`로 config 시스템을 구성합니다.

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

이 colab에서는 TensorFlow Dataset의 [CIFAR100](https://www.tensorflow.org/datasets/catalog/cifar100) 데이터셋을 사용할 예정입니다. TensorFlow/Keras를 이용해 간단한 이미지 분류 파이프라인을 구축하는 것이 목표입니다.

```python
train_ds, valid_ds = tfds.load("fashion_mnist", split=["train", "test"])
```

```python
AUTOTUNE = tf.data.AUTOTUNE

def parse_data(example):
    # 이미지 가져오기
    image = example["image"]
    # image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # 라벨 가져오기
    label = example["label"]
    label = tf.one_hot(label, depth=configs["num_classes"])

    return image, label

def get_dataloader(ds, configs, dataloader_type="train"):
    dataloader = ds.map(parse_data, num_parallel_calls=AUTOTUNE)

    if dataloader_type == "train":
        dataloader = dataloader.shuffle(configs["shuffle_buffer"])

    dataloader = dataloader.batch(configs["batch_size"]).prefetch(AUTOTUNE)

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

## 트레이닝

```python
# W&B Run을 초기화합니다.
run = wandb.init(project="intro-keras", config=configs)

# 모델 트레이닝
model.fit(
    trainloader,
    epochs=configs["epochs"],
    validation_data=validloader,
    callbacks=[
        WandbMetricsLogger(log_freq=10)
    ],  # 여기서 WandbMetricsLogger 콜백을 사용함을 주목하세요
)

# W&B Run을 종료합니다.
run.finish()
```
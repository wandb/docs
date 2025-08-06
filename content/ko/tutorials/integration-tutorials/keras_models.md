---
title: Keras 모델
menu:
  tutorials:
    identifier: ko-tutorials-integration-tutorials-keras_models
    parent: integration-tutorials
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Use_WandbModelCheckpoint_in_your_Keras_workflow.ipynb" >}}
W&B 를 사용하여 기계학습 실험 추적, 데이터셋 버전 관리, 그리고 프로젝트 협업을 시작하세요.

{{< img src="/images/tutorials/huggingface-why.png" alt="W&B 사용의 장점" >}}

이 Colab 노트북에서는 `WandbModelCheckpoint` 콜백을 소개합니다. 이 콜백을 사용하면 모델 체크포인트를 W&B [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ko" >}})에 기록할 수 있습니다.

## 환경 설정 및 설치

먼저, W&B 의 최신 버전을 설치합니다. 그리고 이 colab 인스턴스를 W&B 와 연동하기 위해 인증을 진행합니다.

```python
!pip install -qq -U wandb
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
from wandb.integration.keras import WandbModelCheckpoint
```

W&B 를 처음 사용하거나 로그인하지 않은 경우, `wandb.login()` 실행 후 표시되는 링크로 이동하여 간단히 회원가입 또는 로그인을 할 수 있습니다. [무료 계정](https://wandb.ai/signup) 가입은 클릭 몇 번이면 완료됩니다.

```python
wandb.login()
```

## 하이퍼파라미터

재현 가능한 기계학습을 위해서는 적절한 config 시스템 사용이 권장됩니다. W&B 를 사용하면 각 실험의 하이퍼파라미터를 추적할 수 있습니다. 이 colab에서는 간단한 Python `dict` 를 config 시스템으로 활용합니다.

```python
configs = dict(
    num_classes = 10,
    shuffle_buffer = 1024,
    batch_size = 64,
    image_size = 28,
    image_channels = 1,
    earlystopping_patience = 3,
    learning_rate = 1e-3,
    epochs = 10
)
```

## 데이터셋

이 colab에서는 TensorFlow Dataset 카탈로그에 있는 [CIFAR100](https://www.tensorflow.org/datasets/catalog/cifar100) 데이터셋을 사용합니다. TensorFlow/Keras 를 이용하여 간단한 이미지 분류 파이프라인을 구축하는 것이 목표입니다.

```python
train_ds, valid_ds = tfds.load('fashion_mnist', split=['train', 'test'])
```

```python
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
    backbone = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False)
    backbone.trainable = False

    inputs = layers.Input(shape=(configs["image_size"], configs["image_size"], configs["image_channels"]))
    resize = layers.Resizing(32, 32)(inputs)
    neck = layers.Conv2D(3, (3,3), padding="same")(resize)
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
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = ["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top@5_accuracy')]
)
```

## 학습

```python
# W&B Run 초기화
run = wandb.init(
    project = "intro-keras",
    config = configs
)

# 모델 학습
model.fit(
    trainloader,
    epochs = configs["epochs"],
    validation_data = validloader,
    callbacks = [
        WandbMetricsLogger(log_freq=10),
        WandbModelCheckpoint(filepath="models/") # 여기서 WandbModelCheckpoint 사용 예시
    ]
)

# W&B Run 종료
run.finish()
```
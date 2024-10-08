---
title: Keras tables
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

<CTAButtons colabLink='https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Use_WandbEvalCallback_in_your_Keras_workflow.ipynb'/>

Weights & Biases를 사용하여 기계학습 실험 추적, 데이터셋 버전 관리 및 프로젝트 협업을 수행하세요.

![](/images/tutorials/huggingface-why.png)

이 Colab 노트북은 `WandbEvalCallback`을 소개합니다. 이는 모델 예측 시각화 및 데이터셋 시각화를 위한 유용한 콜백을 구축하기 위해 상속할 수 있는 추상 콜백입니다.

## 🌴 설치 및 설정

먼저, 최신 버전의 Weights and Biases를 설치합니다. 그런 다음 이 Colab 인스턴스를 인증하여 W&B를 사용합니다.

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

# Weights and Biases 관련 모듈 임포트
import wandb
from wandb.integration.keras import WandbMetricsLogger
from wandb.integration.keras import WandbModelCheckpoint
from wandb.integration.keras import WandbEvalCallback
```

W&B를 처음 사용하거나 로그인되어 있지 않다면, `wandb.login()` 실행 후 나타나는 링크가 가입/로그인 페이지로 안내합니다. [무료 계정](https://wandb.ai/signup)으로 가입하는 것은 몇 번의 클릭만으로 가능합니다.

```python
wandb.login()
```

## 🌳 하이퍼파라미터

적절한 구성 시스템의 사용은 재현 가능한 기계학습을 위해 권장되는 모범 사례입니다. W&B를 사용하여 각 실험의 하이퍼파라미터를 추적할 수 있습니다. 이 colab에서는 간단한 Python `dict`를 우리의 구성 시스템으로 사용할 것입니다.

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

## 🍁 데이터셋

이 colab에서는 TensorFlow 데이터셋 카탈로그에서 [CIFAR100](https://www.tensorflow.org/datasets/catalog/cifar100) 데이터셋을 사용할 것입니다. TensorFlow/Keras를 사용하여 간단한 이미지 분류 파이프라인을 구축하는 것이 목표입니다.

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

## 🎄 모델

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

## 🌿 모델 컴파일

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

## 💫 `WandbEvalCallback`

`WandbEvalCallback`은 주로 모델 예측 시각화와 데이터셋 시각화를 위한 Keras 콜백을 구축하기 위한 추상 기반 클래스입니다.

이는 데이터셋과 작업에 종속되지 않은 추상 콜백입니다. 이를 사용하려면, 이 기본 콜백 클래스를 상속하여 `add_ground_truth` 및 `add_model_prediction` 메소드를 구현하십시오.

`WandbEvalCallback`은 다음과 같은 유용한 메소드를 제공하는 유틸리티 클래스입니다:

- 데이터 및 예측 `wandb.Table` 인스턴스 생성,
- 데이터 및 예측 테이블을 `wandb.Artifact`로 로그,
- `on_train_begin`시에 데이터 테이블 로그,
- `on_epoch_end`마다 예측 테이블 로그.

예로서, 이미지 분류 작업의 `WandbClfEvalCallback`을 아래와 같이 구현했습니다. 이 콜백 예제는:
- 검증 데이터를 W&B에 로그(`data_table`),
- 추론을 수행하고 에포크가 끝날 때마다 W&B에 예측을 로그(`pred_table`).

## ✨ 메모리 사용량이 어떻게 줄어드나요?

`on_train_begin` 메소드가 호출될 때 `data_table`을 W&B에 로그합니다. 일단 W&B Artifact로 업로드되면, 이 테이블에 대한 참고를 얻을 수 있으며, 이는 `data_table_ref` 클래스 변수를 통해 엑세스할 수 있습니다. `data_table_ref`는 2D 리스트이며, `self.data_table_ref[idx][n]`처럼 인덱싱할 수 있습니다. 여기서 `idx`는 행 번호이고 `n`은 열 번호입니다. 아래 예에서 사용법을 살펴보겠습니다.

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
        # 예측값 얻기
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

## 🌻 학습

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
        ),  # 여기서 WandbEvalCallback이 사용됨을 주목하세요
    ],
)

# W&B run 종료
run.finish()
```
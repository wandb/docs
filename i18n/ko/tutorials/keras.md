
# Keras

[**여기에서 Colab 노트북으로 시도해보세요 →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Use_WandbMetricLogger_in_your_Keras_workflow.ipynb)

기계학습 실험 추적, 데이터셋 버전 관리 및 프로젝트 협업을 위해 Weights & Biases를 사용하세요.

<img src="http://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />

이 Colab 노트북은 `WandbMetricsLogger` 콜백을 소개합니다. 이 콜백을 [실험 추적](https://docs.wandb.ai/guides/track)에 사용하세요. 트레이닝 및 검증 메트릭과 시스템 메트릭을 Weights and Biases에 로그합니다.

# 🌴 설치 및 설정

먼저, Weights & Biases의 최신 버전을 설치합니다. 그런 다음 이 Colab 인스턴스를 W&B에 인증합니다.

```python
!pip install -qq -U wandb
```

```python
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow_datasets as tfds

# Weights and Biases 관련 임포트
import wandb
from wandb.keras import WandbMetricsLogger
```

W&B를 처음 사용하거나 로그인하지 않은 경우, `wandb.login()`을 실행한 후 나타나는 링크가 회원가입/로그인 페이지로 이동합니다. 몇 번의 클릭으로 [무료 계정](https://wandb.ai/signup)에 가입하는 것은 매우 간단합니다.

```python
wandb.login()
```

# 🌳 하이퍼파라미터

재현 가능한 기계학습을 위해 적절한 설정 시스템 사용을 권장하는 모범 사례입니다. W&B를 사용하여 모든 실험의 하이퍼파라미터를 추적할 수 있습니다. 이 Colab에서는 간단한 Python `dict`을 설정 시스템으로 사용할 것입니다.

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

# 🍁 데이터셋

이 Colab에서는 TensorFlow 데이터셋 카탈로그에서 [CIFAR100](https://www.tensorflow.org/datasets/catalog/cifar100) 데이터셋을 사용할 것입니다. TensorFlow/Keras를 사용하여 간단한 이미지 분류 파이프라인을 구축하는 것을 목표로 합니다.

```python
train_ds, valid_ds = tfds.load('fashion_mnist', split=['train', 'test'])
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

# 🎄 모델

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

# 🌿 모델 컴파일

```python
model.compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = ["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top@5_accuracy')]
)
```

# 🌻 트레이닝

```python
# W&B run 초기화
run = wandb.init(
    project = "intro-keras",
    config = configs
)

# 모델 트레이닝
model.fit(
    trainloader,
    epochs = configs["epochs"],
    validation_data = validloader,
    callbacks = [WandbMetricsLogger(log_freq=10)] # 여기서 WandbMetricsLogger 사용에 주목하세요
)

# W&B run 종료
run.finish()
```
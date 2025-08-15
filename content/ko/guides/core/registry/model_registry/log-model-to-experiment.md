---
title: 모델 추적하기
description: W&B Python SDK를 사용하여 모델, 해당 모델의 종속성, 그리고 모델과 관련된 기타 정보를 추적하세요.
menu:
  default:
    identifier: ko-guides-core-registry-model_registry-log-model-to-experiment
    parent: model-registry
weight: 3
---

W&B Python SDK를 사용하여 모델, 모델의 의존성, 그리고 해당 모델과 관련된 기타 정보를 추적할 수 있습니다.

W&B는 내부적으로 [model artifact]({{< relref path="./model-management-concepts.md#model-artifact" lang="ko" >}})의 계보를 생성하며, 이 계보는 W&B App에서 또는 W&B Python SDK를 통해 프로그래밍 방식으로 확인할 수 있습니다. 자세한 내용은 [Create model lineage map]({{< relref path="./model-lineage.md" lang="ko" >}})을 참고하세요.

## 모델을 log 하는 방법

`run.log_model` API를 사용하여 모델을 로그할 수 있습니다. `path` 파라미터에는 모델 파일이 저장된 경로를 입력하세요. 이 경로는 로컬 파일, 디렉토리, 또는 `s3://bucket/path`와 같은 외부 버킷의 [reference URI]({{< relref path="/guides/core/artifacts/track-external-files.md#amazon-s3--gcs--azure-blob-storage-references" lang="ko" >}})가 될 수 있습니다.

선택적으로, `name` 파라미터에 모델 artifact의 이름을 지정할 수 있습니다. 만약 `name`을 지정하지 않으면, W&B는 입력 경로의 basename 앞에 run ID를 붙여 사용합니다.

아래 코드조각을 복사해서 사용하세요. `< >`로 감싸진 값은 여러분의 환경에 맞게 수정해야 합니다.

```python
import wandb

# W&B run 초기화
run = wandb.init(project="<project>", entity="<entity>")

# 모델 로그
run.log_model(path="<path-to-model>", name="<name>")
```

<details>

<summary>예시: Keras 모델을 W&B에 로그하기</summary>

아래 코드 예시는 합성곱 신경망(CNN) 모델을 W&B에 로그하는 방법을 보여줍니다.

```python
import os
import wandb
from tensorflow import keras
from tensorflow.keras import layers

config = {"optimizer": "adam", "loss": "categorical_crossentropy"}

# W&B run 초기화
run = wandb.init(entity="charlie", project="mnist-project", config=config)

# 트레이닝 알고리즘
loss = run.config["loss"]
optimizer = run.config["optimizer"]
metrics = ["accuracy"]
num_classes = 10
input_shape = (28, 28, 1)

model = keras.Sequential(
    [
        layers.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

# 모델 저장
model_filename = "model.h5"
local_filepath = "./"
full_path = os.path.join(local_filepath, model_filename)
model.save(filepath=full_path)

# 모델 로그
run.log_model(path=full_path, name="MNIST")

# W&B run 종료
run.finish()
```
</details>
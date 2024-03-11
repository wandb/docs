---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 모델 추적하기

W&B Python SDK로 모델, 모델의 의존성 및 해당 모델과 관련된 기타 정보를 추적하세요.

내부적으로, W&B는 [모델 아티팩트](./model-management-concepts.md#model-artifact)의 계보를 생성하며, 이를 W&B App UI나 W&B Python SDK로 프로그래매틱하게 볼 수 있습니다. 더 많은 정보는 [모델 이력 맵 생성하기](./model-lineage.md)를 참조하세요.

## 모델 로깅하는 방법

`run.log_model` API를 사용하여 모델을 로깅하세요. 모델 파일이 저장된 경로를 `path` 파라미터에 제공하세요. 경로는 로컬 파일, 디렉토리 또는 `s3://bucket/path`와 같은 외부 버킷으로의 [참조 URI](../artifacts/track-external-files.md#amazon-s3--gcs--azure-blob-storage-references)일 수 있습니다.

선택적으로 모델 아티팩트의 이름을 `name` 파라미터에 제공하세요. `name`이 지정되지 않은 경우, W&B는 입력 경로의 기본 이름에 run ID를 앞에 붙여 사용합니다.

다음 코드 조각을 복사하고 붙여넣으세요. `<>`로 묶인 값들을 귀하의 것으로 교체하세요.

```python
import wandb

# W&B run 초기화
run = wandb.init(project="<project>", entity="<entity>")

# 모델 로깅
run.log_model(path="<path-to-model>", name="<name>")
```

<details>

<summary>예시: Keras 모델을 W&B에 로깅하기</summary>

다음 코드 예시는 컨볼루션 신경망(CNN) 모델을 W&B에 로깅하는 방법을 보여줍니다.

```python showLineNumbers
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

# 모델 로깅
# highlight-next-line
run.log_model(path=full_path, name="MNIST")

# W&B에 run이 끝났음을 명시적으로 알림.
run.finish()
```
</details>
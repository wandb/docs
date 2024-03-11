---
description: Learn how to use W&B for Model Management
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 워크스루

다음 워크스루는 모델을 W&B에 로깅하는 방법을 보여줍니다. 워크스루가 끝나면 다음을 수행할 수 있습니다:

* MNIST 데이터셋과 Keras 프레임워크로 모델을 생성하고 트레이닝합니다.
* 트레이닝한 모델을 W&B 프로젝트에 로깅합니다.
* 생성한 모델에 사용된 데이터셋을 의존성으로 표시합니다.
* 모델을 W&B 레지스트리에 연결합니다.
* 레지스트리에 연결한 모델의 성능을 평가합니다.
* 모델 버전을 프로덕션 준비 완료로 표시합니다.

:::note
* 이 가이드에서 제시된 순서대로 코드조각을 복사하세요.
* 모델 레지스트리에 고유하지 않은 코드는 접을 수 있는 셀에 숨겨져 있습니다.
:::

## 설정하기

시작하기 전에, 이 워크스루에 필요한 파이썬 의존성을 불러오세요:

```python
import wandb
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from wandb.keras import WandbCallback
from sklearn.model_selection import train_test_split
```

`entity` 변수에 W&B 엔티티를 제공하세요:

```python
entity = "<entity>"
```

### 데이터셋 아티팩트 생성하기

먼저, 데이터셋을 생성합니다. 다음 코드조각은 MNIST 데이터셋을 다운로드하는 함수를 생성합니다:
```python
def generate_raw_data(train_size=6000):
    eval_size = int(train_size / 6)
    (x_train, y_train), (x_eval, y_eval) = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255
    x_eval = x_eval.astype("float32") / 255
    x_train = np.expand_dims(x_train, -1)
    x_eval = np.expand_dims(x_eval, -1)

    print("Generated {} rows of training data.".format(train_size))
    print("Generated {} rows of eval data.".format(eval_size))

    return (x_train[:train_size], y_train[:train_size]), (
        x_eval[:eval_size],
        y_eval[:eval_size],
    )

# 데이터셋 생성하기
(x_train, y_train), (x_eval, y_eval) = generate_raw_data()
```

다음으로, 생성한 데이터셋을 W&B에 업로드합니다. 이를 위해 [아티팩트](../artifacts/intro.md) 객체를 생성하고 그 아티팩트에 데이터셋을 추가하세요.

```python
project = "model-registry-dev"

model_use_case_id = "mnist"
job_type = "build_dataset"

# W&B run 초기화
run = wandb.init(entity=entity, project=project, job_type=job_type)

# 트레이닝 데이터용 W&B 테이블 생성
train_table = wandb.Table(data=[], columns=[])
train_table.add_column("x_train", x_train)
train_table.add_column("y_train", y_train)
train_table.add_computed_columns(lambda ndx, row: {"img": wandb.Image(row["x_train"])})

# 평가 데이터용 W&B 테이블 생성
eval_table = wandb.Table(data=[], columns=[])
eval_table.add_column("x_eval", x_eval)
eval_table.add_column("y_eval", y_eval)
eval_table.add_computed_columns(lambda ndx, row: {"img": wandb.Image(row["x_eval"])})

# 아티팩트 객체 생성
artifact_name = "{}_dataset".format(model_use_case_id)
artifact = wandb.Artifact(name=artifact_name, type="dataset")

# 아티팩트에 wandb.WBValue 오브젝트 추가.
artifact.add(train_table, "train_table")
artifact.add(eval_table, "eval_table")

# 아티팩트에 대한 변경사항을 저장합니다.
artifact.save()

# 이 W&B run이 완료되었음을 알립니다.
run.finish()
```

:::tip
데이터셋과 같은 파일을 아티팩트로 저장하는 것은 모델을 로깅할 때 유용합니다. 왜냐하면 모델의 의존성을 추적할 수 있기 때문입니다.
:::

## 모델 트레이닝하기
이전 단계에서 생성한 아티팩트 데이터셋으로 모델을 트레이닝합니다.

### run에 데이터셋 아티팩트를 입력으로 선언하기

이전 단계에서 생성한 데이터셋 아티팩트를 W&B run의 입력으로 선언하세요. 이는 모델을 로깅하는 맥락에서 특히 유용하며, run에 아티팩트를 입력으로 선언하면 특정 모델을 트레이닝하는데 사용된 데이터셋(및 데이터셋의 버전)을 추적할 수 있게 합니다. W&B는 수집된 정보를 사용하여 [계보 맵](./model-lineage.md)을 생성합니다.

`use_artifact` API를 사용하여 데이터셋 아티팩트를 run의 입력으로 선언하고 아티팩트 자체를 검색하세요.

```python
job_type = "train_model"
config = {
    "옵티마이저": "adam",
    "batch_size": 128,
    "에포크": 5,
    "validation_split": 0.1,
}

# W&B run 초기화
run = wandb.init(project=project, job_type=job_type, config=config)

# 데이터셋 아티팩트 검색
version = "latest"
name = "{}:{}".format("{}_dataset".format(model_use_case_id), version)
artifact = run.use_artifact(artifact_or_name=name)

# 데이터프레임에서 특정 내용 가져오기
train_table = artifact.get("train_table")
x_train = train_table.get_column("x_train", convert_to="numpy")
y_train = train_table.get_column("y_train", convert_to="numpy")
```

모델의 입력과 출력을 추적하는 방법에 대한 자세한 내용은 [모델 이력 맵 생성하기](./model-lineage.md)를 참조하세요.

### 모델 정의 및 트레이닝하기

이 워크스루에서는 Keras를 사용하여 MNIST 데이터셋에서 이미지를 분류하는 2D Convolutional Neural Network(CNN)을 정의합니다.

<details>
<summary>MNIST 데이터에서 CNN 트레이닝하기</summary>

```python
# config 사전에서 변수로 값 저장하기
num_classes = 10
input_shape = (28, 28, 1)
loss = "categorical_crossentropy"
optimizer = run.config["옵티마이저"]
metrics = ["accuracy"]
batch_size = run.config["batch_size"]
epochs = run.config["에포크"]
validation_split = run.config["validation_split"]

# 모델 아키텍처 생성
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

# 트레이닝 데이터에 대한 라벨 생성
y_train = keras.utils.to_categorical(y_train, num_classes)

# 트레이닝 및 테스트 세트 생성
x_t, x_v, y_t, y_v = train_test_split(x_train, y_train, test_size=0.33)
```
다음으로, 모델을 트레이닝합니다:

```python
# 모델 트레이닝
model.fit(
    x=x_t,
    y=y_t,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_v, y_v),
    callbacks=[WandbCallback(log_weights=True, log_evaluation=True)],
)
```

마지막으로, 모델을 로컬 컴퓨터에 저장하세요:

```python
# 모델 로컬에 저장
path = "model.h5"
model.save(path)
```
</details>

## 모델을 모델 레지스트리에 로깅하고 연결하기
[`link_model`](../../ref/python/run.md#link_model) API를 사용하여 하나 이상의 파일을 W&B run에 로그하고 [W&B 모델 레지스트리](./intro.md)에 연결하세요.

```python
path = "./model.h5"
registered_model_name = "MNIST-dev"

run.link_model(path=path, registered_model_name=registered_model_name)
run.finish()
```

`registered-model-name`에 지정한 이름이 이미 존재하지 않는 경우 W&B가 당신을 위해 등록된 모델을 생성합니다.

API 참조 가이드에서 [`link_model`](../../ref/python/run.md#link_model)에 대한 추가 매개변수에 대한 정보를 확인하세요.

## 모델의 성능 평가하기
하나 이상의 모델의 성능을 평가하는 것은 일반적인 관행입니다.

먼저, 이전 단계에서 W&B에 저장된 평가 데이터셋 아티팩트를 가져옵니다.

```python
job_type = "evaluate_model"

# run 초기화
run = wandb.init(project=project, entity=entity, job_type=job_type)

model_use_case_id = "mnist"
version = "latest"

# 데이터셋 아티팩트 가져오기, 의존성으로 표시
artifact = run.use_artifact(
    "{}:{}".format("{}_dataset".format(model_use_case_id), version)
)

# 원하는 데이터프레임 가져오기
eval_table = artifact.get("eval_table")
x_eval = eval_table.get_column("x_eval", convert_to="numpy")
y_eval = eval_table.get_column("y_eval", convert_to="numpy")
```

평가하고자 하는 W&B에서 [모델 버전](./model-management-concepts.md#model-version)을 다운로드하세요. `use_model` API를 사용하여 모델에 액세스하고 다운로드하세요.

```python
alias = "latest"  # 에일리어스
name = "mnist_model"  # 모델 아티팩트의 이름

# 모델에 액세스하고 다운로드합니다. 다운로드된 아티팩트의 경로를 반환합니다
downloaded_model_path = run.use_model(name=f"{name}:{alias}")
```

Keras 모델을 로드하고 손실을 계산하세요:

```python
model = keras.models.load_model(downloaded_model_path)

y_eval = keras.utils.to_categorical(y_eval, 10)
(loss, _) = model.evaluate(x_eval, y_eval)
score = (loss, _)
```

마지막으로, 손실 메트릭을 W&B run에 로그하세요:

```python
# 메트릭, 이미지, 테이블 또는 평가에 유용한 모든 데이터를 로그합니다.
run.log(data={"loss": (loss, _)})
```

## 모델 버전 승진시키기
[*모델 에일리어스*](./model-management-concepts.md#model-alias)를 사용하여 모델 버전을 기계학습 워크플로우의 다음 단계 준비 완료로 표시하세요. 각 등록된 모델은 하나 이상의 모델 에일리어스를 가질 수 있습니다. 모델 에일리어스는 한 번에 하나의 모델 버전에만 속할 수 있습니다.

예를 들어, 모델의 성능을 평가한 후 해당 모델이 프로덕션 준비가 되었다고 확신하는 경우, 해당 모델 버전에 `production` 에일리어스를 추가하여 승진시킬 수 있습니다.

:::tip
`production` 에일리어스는 모델을 프로덕션 준비 완료로 표시하는 가장 일반적인 에일리어스 중 하나입니다.
:::

W&B 모델 레지스트리 앱이나 파이썬 SDK를 사용하여 모델 버전에 에일리어스를 추가할 수 있습니다. 다음 단계는 W&B 모델 레지스트리 앱을 사용하여 에일리어스를 추가하는 방법을 보여줍니다:


1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model)에서 모델 레지스트리 앱으로 이동하세요.
2. 등록된 모델의 이름 옆에 있는 **View details**를 클릭하세요.
3. **Versions** 섹션에서 승진시키고 싶은 모델 버전 옆에 있는 **View** 버튼을 클릭하세요.
4. **Aliases** 필드 옆에 있는 플러스 아이콘(**+**)을 클릭하세요.
5. 나타나는 필드에 `production`을 입력하세요.
6. 키보드의 Enter를 누르세요.


![](/images/models/promote_model_production.gif)
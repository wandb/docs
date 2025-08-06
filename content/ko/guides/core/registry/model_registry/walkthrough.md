---
title: '튜토리얼: W&B로 Model Management 수행하기'
description: W&B를 사용한 Model Management 활용 방법 알아보기
menu:
  default:
    identifier: ko-guides-core-registry-model_registry-walkthrough
    parent: model-registry
weight: 1
---

다음 워크스루에서는 W&B에 모델을 로그하는 방법을 보여줍니다. 이 워크스루를 마치면 다음을 할 수 있습니다.

* MNIST 데이터셋과 Keras 프레임워크를 사용해 모델을 만들고 트레이닝합니다.
* 트레이닝한 모델을 W&B 프로젝트에 로그합니다.
* 사용한 데이터셋을 생성한 모델의 의존성으로 표시합니다.
* 모델을 W&B Registry와 연결합니다.
* Registry에 연결한 모델의 성능을 평가합니다.
* 모델 버전을 프로덕션 준비 상태로 표시합니다.

{{% alert %}}
* 본 가이드에 제시된 순서대로 코드조각을 복사해 사용하세요.
* Model Registry와 직접 관련 없는 코드는 접을 수 있는 셀에 숨겨져 있습니다.
{{% /alert %}}

## 환경 설정

시작하기 전에, 본 워크스루를 위해 필요한 Python 의존성을 임포트하세요:

```python
import wandb
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from wandb.integration.keras import WandbMetricsLogger
from sklearn.model_selection import train_test_split
```

W&B entity를 `entity` 변수에 입력하세요:

```python
entity = "<entity>"
```

### 데이터셋 아티팩트 생성하기

먼저, 데이터셋을 생성합니다. 다음 코드조각은 MNIST 데이터셋을 다운로드하는 함수를 만듭니다:
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

# 데이터셋 생성
(x_train, y_train), (x_eval, y_eval) = generate_raw_data()
```

다음으로, 데이터셋을 W&B에 업로드합니다. 이를 위해 [artifact]({{< relref path="/guides/core/artifacts/" lang="ko" >}}) 오브젝트를 생성하고 데이터셋을 그 아티팩트에 추가하세요.

```python
project = "model-registry-dev"

model_use_case_id = "mnist"
job_type = "build_dataset"

# W&B run 초기화
run = wandb.init(entity=entity, project=project, job_type=job_type)

# 트레이닝 데이터용 W&B Table 생성
train_table = wandb.Table(data=[], columns=[])
train_table.add_column("x_train", x_train)
train_table.add_column("y_train", y_train)
train_table.add_computed_columns(lambda ndx, row: {"img": wandb.Image(row["x_train"])})

# 평가 데이터용 W&B Table 생성
eval_table = wandb.Table(data=[], columns=[])
eval_table.add_column("x_eval", x_eval)
eval_table.add_column("y_eval", y_eval)
eval_table.add_computed_columns(lambda ndx, row: {"img": wandb.Image(row["x_eval"])})

# artifact 오브젝트 생성
artifact_name = "{}_dataset".format(model_use_case_id)
artifact = wandb.Artifact(name=artifact_name, type="dataset")

# wandb.WBValue 오브젝트를 artifact에 추가
artifact.add(train_table, "train_table")
artifact.add(eval_table, "eval_table")

# artifact에 한 모든 변경사항 저장
artifact.save()

# W&B에 run 완료 알림
run.finish()
```

{{% alert %}}
파일(예: 데이터셋)을 아티팩트에 저장하는 것은 모델을 로그할 때 유용합니다. 모델의 의존성을 추적할 수 있기 때문입니다.
{{% /alert %}}

## 모델 트레이닝
이전 단계에서 생성한 artifact 데이터셋으로 모델을 트레이닝하세요.

### Run의 입력으로 dataset artifact 선언하기

이전 단계에서 생성한 dataset artifact를 W&B run의 입력으로 선언하세요. 이는 모델을 로그할 때 특히 유용합니다. 아티팩트를 run의 입력으로 선언하면 특정 모델을 트레이닝할 때 사용된 데이터셋(과 해당 버전)을 추적할 수 있습니다. W&B는 수집한 정보를 토대로 [계보 맵]({{< relref path="./model-lineage.md" lang="ko" >}})을 만듭니다.

`use_artifact` API를 사용하면 dataset artifact를 run의 입력으로 선언하고, 동시에 해당 artifact 자체를 가져올 수 있습니다.

```python
job_type = "train_model"
config = {
    "optimizer": "adam",
    "batch_size": 128,
    "epochs": 5,
    "validation_split": 0.1,
}

# W&B run 초기화
run = wandb.init(project=project, job_type=job_type, config=config)

# 데이터셋 artifact 가져오기
version = "latest"
name = "{}:{}".format("{}_dataset".format(model_use_case_id), version)
artifact = run.use_artifact(artifact_or_name=name)

# 데이터프레임에서 특정 데이터 얻기
train_table = artifact.get("train_table")
x_train = train_table.get_column("x_train", convert_to="numpy")
y_train = train_table.get_column("y_train", convert_to="numpy")
```

모델의 입력 및 출력을 추적하는 자세한 내용은 [모델 계보 만들기]({{< relref path="./model-lineage.md" lang="ko" >}}) 맵을 참고하세요.

### 모델 정의 및 트레이닝

이번 워크스루에서는 MNIST 데이터셋 이미지를 분류하기 위해 Keras로 2D 합성곱 신경망(CNN)을 정의합니다.

<details>
<summary>MNIST 데이터로 CNN 트레이닝</summary>

```python
# config 사전에서 변수 가져오기 (접근 용이하게)
num_classes = 10
input_shape = (28, 28, 1)
loss = "categorical_crossentropy"
optimizer = run.config["optimizer"]
metrics = ["accuracy"]
batch_size = run.config["batch_size"]
epochs = run.config["epochs"]
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

# 트레이닝 데이터용 라벨 생성
y_train = keras.utils.to_categorical(y_train, num_classes)

# 트레이닝 및 테스트 세트 생성
x_t, x_v, y_t, y_v = train_test_split(x_train, y_train, test_size=0.33)
```
이제 모델을 트레이닝하세요:

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

마지막으로, 로컬에 모델을 저장하세요:

```python
# 모델을 로컬에 저장
path = "model.h5"
model.save(path)
```
</details>

## 모델 로그 및 Model Registry에 연결하기
[`link_model`]({{< relref path="/ref/python/sdk/classes/run.md#link_model" lang="ko" >}}) API를 사용해 하나 이상의 모델 파일을 W&B run에 로그하고, [W&B Model Registry]({{< relref path="./" lang="ko" >}})에 연결하세요.

```python
path = "./model.h5"
registered_model_name = "MNIST-dev"

run.link_model(path=path, registered_model_name=registered_model_name)
run.finish()
```

`registered-model-name`에 지정한 이름의 등록된 모델이 아직 없으면, W&B가 자동으로 등록 모델을 생성합니다.

API Reference 가이드의 [`link_model`]({{< relref path="/ref/python/sdk/classes/run.md#link_model" lang="ko" >}})에서 선택 가능한 파라미터를 확인할 수 있습니다.

## 모델 성능 평가하기
하나 이상의 모델 성능을 평가하는 것은 일반적인 관행입니다.

먼저, 이전 단계에서 W&B에 저장한 평가 데이터셋 artifact를 가져오세요.

```python
job_type = "evaluate_model"

# run 초기화
run = wandb.init(project=project, entity=entity, job_type=job_type)

model_use_case_id = "mnist"
version = "latest"

# 데이터셋 artifact 가져오기, 의존성으로 표시
artifact = run.use_artifact(
    "{}:{}".format("{}_dataset".format(model_use_case_id), version)
)

# 원하는 데이터프레임 불러오기
eval_table = artifact.get("eval_table")
x_eval = eval_table.get_column("x_eval", convert_to="numpy")
y_eval = eval_table.get_column("y_eval", convert_to="numpy")
```

W&B에서 평가할 [model version]({{< relref path="./model-management-concepts.md#model-version" lang="ko" >}})을 다운로드합니다. `use_model` API를 사용해 모델에 엑세스하고 다운로드할 수 있습니다.

```python
alias = "latest"  # 에일리어스
name = "mnist_model"  # 모델 artifact 이름

# 모델 엑세스 및 다운로드, 다운로드된 artifact의 경로 반환
downloaded_model_path = run.use_model(name=f"{name}:{alias}")
```

Keras 모델을 로드하고 loss를 계산합니다:

```python
model = keras.models.load_model(downloaded_model_path)

y_eval = keras.utils.to_categorical(y_eval, 10)
(loss, _) = model.evaluate(x_eval, y_eval)
score = (loss, _)
```

마지막으로 loss 메트릭을 W&B run에 로그하세요:

```python
# # 메트릭, 이미지, 테이블 등 평가에 유용한 데이터를 로그
run.log(data={"loss": (loss, _)})
```

## 모델 버전 프로모트하기
[*model alias*]({{< relref path="./model-management-concepts.md#model-alias" lang="ko" >}})로 머신러닝 워크플로우의 다음 단계로 모델 버전을 준비된 상태로 표시하세요. 각 등록 모델(registered model)은 하나 이상의 모델 에일리어스를 가질 수 있습니다. 모델 에일리어스는 한 번에 하나의 모델 버전에만 할당됩니다.

예를 들어, 모델의 성능을 평가한 후, 해당 모델이 프로덕션에 적합하다고 판단된다면, 해당 모델 버전에 `production` 에일리어스를 추가하여 프로모트합니다.

{{% alert %}}
`production` 에일리어스는 모델을 프로덕션 준비 상태로 표시할 때 가장 많이 사용되는 에일리어스입니다.
{{% /alert %}}

W&B App UI에서 인터랙티브하게 또는 Python SDK로 프로그래밍 방식으로 모델 버전에 에일리어스를 추가할 수 있습니다. 다음 단계는 W&B Model Registry App에서 에일리어스를 추가하는 방법을 보여줍니다:

1. [Model Registry App](https://wandb.ai/registry/model)으로 이동합니다.
2. 등록된 모델 이름 옆의 **View details**를 클릭합니다.
3. **Versions** 섹션에서 프로모트 할 모델 버전 이름 옆의 **View** 버튼을 클릭합니다.
4. **Aliases** 필드 옆의 플러스 아이콘(**+**)을 클릭합니다.
5. 나타나는 필드에 `production`을 입력합니다.
6. 키보드에서 Enter를 누릅니다.

{{< img src="/images/models/promote_model_production.gif" alt="Promote model to production" >}}
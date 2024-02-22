---
description: Learn how to use W&B for Model Management
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 안내

다음 안내는 모델을 W&B에 기록하는 방법을 보여줍니다. 안내를 마치면 다음을 할 수 있습니다:

* MNIST 데이터세트와 Keras 프레임워크를 사용하여 모델을 생성하고 학습합니다.
* 학습한 모델을 W&B 프로젝트에 기록합니다.
* 생성한 모델에 사용된 데이터세트를 의존성으로 표시합니다.
* 모델을 W&B 레지스트리에 연결합니다.
* 레지스트리에 연결된 모델의 성능을 평가합니다.
* 모델 버전을 프로덕션 준비 완료로 표시합니다.

:::note
* 이 가이드에 제시된 순서대로 코드 조각을 복사하십시오.
* 모델 레지스트리에 고유하지 않은 코드는 접을 수 있는 셀에 숨겨져 있습니다.
:::

## 설정하기

시작하기 전에, 이 안내를 위해 필요한 Python 의존성을 가져오십시오:

```python
import wandb
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from wandb.keras import WandbCallback
from sklearn.model_selection import train_test_split
```

`entity` 변수에 W&B 엔티티를 제공하십시오:

```python
entity = "<entity>"
```

### 데이터세트 아티팩트 생성하기

먼저, 데이터세트를 생성하십시오. 다음 코드 조각은 MNIST 데이터세트를 다운로드하는 함수를 생성합니다:
```python
def generate_raw_data(train_size=6000):
    eval_size = int(train_size / 6)
    (x_train, y_train), (x_eval, y_eval) = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255
    x_eval = x_eval.astype("float32") / 255
    x_train = np.expand_dims(x_train, -1)
    x_eval = np.expand_dims(x_eval, -1)

    print("학습 데이터 {}행 생성됨.".format(train_size))
    print("평가 데이터 {}행 생성됨.".format(eval_size))

    return (x_train[:train_size], y_train[:train_size]), (
        x_eval[:eval_size],
        y_eval[:eval_size],
    )

# 데이터세트 생성
(x_train, y_train), (x_eval, y_eval) = generate_raw_data()
```

다음으로, W&B에 데이터세트를 업로드하십시오. 이를 위해, [아티팩트](../artifacts/intro.md) 객체를 생성하고 데이터세트를 해당 아티팩트에 추가하십시오.

```python
project = "model-registry-dev"

model_use_case_id = "mnist"
job_type = "build_dataset"

# W&B 실행 초기화
run = wandb.init(entity=entity, project=project, job_type=job_type)

# 학습 데이터용 W&B 테이블 생성
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

# wandb.WBValue 개체를 아티팩트에 추가합니다.
artifact.add(train_table, "train_table")
artifact.add(eval_table, "eval_table")

# 아티팩트에 대한 변경사항을 유지합니다.
artifact.save()

# W&B에 이 실행이 종료되었음을 알립니다.
run.finish()
```

:::tip
모델의 의존성을 추적할 수 있게 해주므로, 아티팩트에 파일(예: 데이터세트)을 저장하는 것이 모델을 기록하는 맥락에서 유용합니다.
:::

## 모델 학습하기
이전 단계에서 생성한 아티팩트 데이터세트로 모델을 학습하십시오.

### 실행에 데이터세트 아티팩트를 입력으로 선언하기

이전 단계에서 생성한 데이터세트 아티팩트를 W&B 실행의 입력으로 선언하십시오. 이는 특정 모델을 학습하는 데 사용된 데이터세트(및 데이터세트의 버전)를 추적할 수 있게 해주므로, 모델을 기록하는 맥락에서 특히 유용합니다. W&B는 수집된 정보를 사용하여 [계보 맵](./model-lineage.md)을 생성합니다.

`use_artifact` API를 사용하여 데이터세트 아티팩트를 실행의 입력으로 선언하고 아티팩트 자체를 검색하십시오.

```python
job_type = "train_model"
config = {
    "옵티마이저": "adam",
    "배치 크기": 128,
    "에포크": 5,
    "검증 분할": 0.1,
}

# W&B 실행 초기화
run = wandb.init(project=project, job_type=job_type, config=config)

# 데이터세트 아티팩트 검색
version = "latest"
name = "{}:{}".format("{}_dataset".format(model_use_case_id), version)
artifact = run.use_artifact(artifact_or_name=name)

# 데이터프레임에서 특정 내용 가져오기
train_table = artifact.get("train_table")
x_train = train_table.get_column("x_train", convert_to="numpy")
y_train = train_table.get_column("y_train", convert_to="numpy")
```

모델의 입력 및 출력을 추적하는 데 대한 자세한 정보는 [모델 계보 생성](./model-lineage.md) 맵을 참조하십시오.

### 모델 정의 및 학습하기

이 안내에서는 MNIST 데이터세트의 이미지를 분류하기 위해 Keras로 2D 컨볼루션 신경망(CNN)을 정의하십시오.

<details>
<summary>MNIST 데이터에 CNN 학습하기</summary>

```python
# 우리의 설정 사전에서 값들을 변수로 저장하여 쉽게 접근
num_classes = 10
input_shape = (28, 28, 1)
loss = "categorical_crossentropy"
optimizer = run.config["옵티마이저"]
metrics = ["정확도"]
batch_size = run.config["배치 크기"]
epochs = run.config["에포크"]
validation_split = run.config["검증 분할"]

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

# 학습 데이터에 대한 레이블 생성
y_train = keras.utils.to_categorical(y_train, num_classes)

# 학습 및 테스트 세트 생성
x_t, x_v, y_t, y_v = train_test_split(x_train, y_train, test_size=0.33)
```
다음으로, 모델을 학습하십시오:

```python
# 모델 학습
model.fit(
    x=x_t,
    y=y_t,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_v, y_v),
    callbacks=[WandbCallback(log_weights=True, log_evaluation=True)],
)
```

마지막으로, 모델을 로컬 컴퓨터에 저장하십시오:

```python
# 모델 로컬에 저장
path = "model.h5"
model.save(path)
```
</details>

## 모델을 모델 레지스트리에 기록하고 연결하기
[`link_model`](../../ref/python/run.md#link_model) API를 사용하여 모델 하나 이상의 파일을 W&B 실행에 기록하고 [W&B 모델 레지스트리](./intro.md)에 연결하십시오.

```python
path = "./model.h5"
registered_model_name = "MNIST-dev"

run.link_model(path=path, registered_model_name=registered_model_name)
run.finish()
```

`registered-model-name`에 지정한 이름이 이미 존재하지 않는 경우 W&B가 등록된 모델을 생성합니다.

API 참조 가이드에서 [`link_model`](../../ref/python/run.md#link_model)에 대한 자세한 정보와 선택적 파라미터를 확인하십시오.

## 모델의 성능 평가하기
하나 이상의 모델의 성능을 평가하는 것은 일반적인 관행입니다.

먼저, 이전 단계에서 W&B에 저장된 평가 데이터세트 아티팩트를 가져오십시오.

```python
job_type = "evaluate_model"

# 실행 초기화
run = wandb.init(project=project, entity=entity, job_type=job_type)

model_use_case_id = "mnist"
version = "latest"

# 데이터세트 아티팩트 가져오기, 의존성으로 표시
artifact = run.use_artifact(
    "{}:{}".format("{}_dataset".format(model_use_case_id), version)
)

# 원하는 데이터프레임 가져오기
eval_table = artifact.get("eval_table")
x_eval = eval_table.get_column("x_eval", convert_to="numpy")
y_eval = eval_table.get_column("y_eval", convert_to="numpy")
```

평가하려는 W&B에서 [모델 버전](./model-management-concepts.md#model-version)을 다운로드하십시오. `use_model` API를 사용하여 모델에 접근하고 다운로드하십시오.

```python
alias = "latest"  # 별칭
name = "mnist_model"  # 모델 아티팩트의 이름

# 모델에 접근하고 다운로드합니다. 다운로드된 아티팩트의 경로를 반환
downloaded_model_path = run.use_model(name=f"{name}:{alias}")
```

Keras 모델을 로드하고 손실을 계산하십시오:

```python
model = keras.models.load_model(downloaded_model_path)

y_eval = keras.utils.to_categorical(y_eval, 10)
(loss, _) = model.evaluate(x_eval, y_eval)
score = (loss, _)
```

마지막으로, 손실 메트릭을 W&B 실행에 기록하십시오:

```python
# 메트릭, 이미지, 테이블 또는 평가에 유용한 모든 데이터를 기록합니다.
run.log(data={"loss": (loss, _)})
```

## 모델 버전 승격하기
머신 러닝 워크플로의 다음 단계에 대해 준비된 모델 버전을 [*모델 별칭*](./model-management-concepts.md#model-alias)으로 표시하십시오. 각 등록된 모델은 하나 이상의 모델 별칭을 가질 수 있습니다. 모델 별칭은 한 번에 하나의 모델 버전에만 속할 수 있습니다.

예를 들어, 모델의 성능을 평가한 후 해당 모델이 프로덕션 준비가 되었다고 확신하는 경우, 해당 모델 버전에 `production` 별칭을 추가하십시오.

:::tip
`production` 별칭은 모델을 프로덕션 준비가 완료되었다고 표시하는 데 가장 일반적으로 사용되는 별칭 중 하나입니다.
:::

별칭을 모델 버전에 추가하는 것은 W&B App UI를 통해 대화식으로 또는 Python SDK를 사용하여 프로그래매틱하게 수행할 수 있습니다. 다음 단계는 W&B 모델 레지스트리 앱을 사용하여 별칭을 추가하는 방법을 보여줍니다:


1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model)에서 모델 레지스트리 앱으로 이동하십시오.
2. 등록된 모델의 이름 옆에 있는 **세부 정보 보기**를 클릭하십시오.
3. **버전** 섹션에서 승격하려는 모델 버전 이름 옆에 있는 **보기** 버튼을 클릭하십시오.
4. **별칭** 필드 옆에 있는 플러스 아이콘(**+**)을 클릭하십시오.
5. 나타나는 필드에 `production`을 입력하십시오.
6. 키보드에서 Enter를 누르십시오.


![](/images/models/promote_model_production.gif)
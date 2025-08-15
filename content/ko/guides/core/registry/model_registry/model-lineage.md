---
title: 모델 계보 맵 생성
description: ''
menu:
  default:
    identifier: ko-guides-core-registry-model_registry-model-lineage
    parent: model-registry
weight: 7
---

이 페이지는 기존 W&B Model Registry에서 계보 그래프(lineage graph)를 생성하는 방법을 설명합니다. W&B Registry의 계보 그래프에 대한 내용은 [계보 맵 생성 및 확인하기]({{< relref path="../lineage.md" lang="ko" >}})를 참고하세요.

{{% alert %}}
W&B는 기존 [W&B Model Registry]({{< relref path="/guides/core/registry/model_registry/" lang="ko" >}})의 자산들을 새로운 [W&B Registry]({{< relref path="./" lang="ko" >}})로 이전할 예정입니다. 이 마이그레이션은 W&B에서 전적으로 관리 및 실행되며, 사용자 측에서는 별도의 조치가 필요하지 않습니다. 이전 프로세스는 최대한 원활하게 이루어져 기존 워크플로우 interruption을 최소화하도록 설계되었습니다. 자세한 내용은 [기존 Model Registry에서 마이그레이션하기]({{< relref path="../model_registry_eol.md" lang="ko" >}})를 참고하세요.
{{% /alert %}}

W&B에 모델 아티팩트를 기록하는 데 있어 유용한 기능 중 하나가 계보 그래프입니다. 계보 그래프는 한 run에서 기록된 아티팩트와, 특정 run에서 활용된 아티팩트를 함께 보여줍니다.

즉, 모델 아티팩트를 기록하면 최소한 해당 모델 아티팩트를 사용하거나 생성한 W&B run을 볼 수 있습니다. 또한 [의존성 추적]({{< relref path="#track-an-artifact-dependency" lang="ko" >}})을 하면, 모델 아티팩트에서 사용한 입력값도 함께 확인할 수 있습니다.

예를 들어, 아래 이미지는 하나의 ML experiment 전반에서 생성 및 사용된 아티팩트들을 보여줍니다:

{{< img src="/images/models/model_lineage_example.png" alt="Model lineage graph" >}}

좌측에서 우측으로 이미지는 아래와 같이 진행됩니다:
1. `jumping-monkey-1` W&B run이 `mnist_dataset:v0` 데이터셋 아티팩트를 생성합니다.
2. `vague-morning-5` W&B run이 `mnist_dataset:v0` 데이터셋 아티팩트를 사용하여 모델을 트레이닝합니다. 이 run의 출력으로 `mnist_model:v0`이라는 모델 아티팩트가 생성됩니다.
3. `serene-haze-6`라는 run이 이 모델 아티팩트(`mnist_model:v0`)를 이용해 모델을 평가합니다.

## 아티팩트 의존성 추적하기

데이터셋 아티팩트를 W&B run의 입력(input)으로 선언하고 의존성을 추적하려면, `use_artifact` API를 사용하세요.

아래 코드조각은 `use_artifact` API 사용 예시입니다:

```python
# run 초기화
run = wandb.init(project=project, entity=entity)

# 아티팩트 가져오기 및 의존성 표시
artifact = run.use_artifact(artifact_or_name="name", aliases="<alias>")
```

아티팩트를 가져왔다면, 이를 활용해 예를 들어 모델 성능 평가 등 다양한 작업에 사용할 수 있습니다.

<details>

<summary>예시: 모델을 학습시키고 데이터셋을 입력값으로 추적하기</summary>

```python
job_type = "train_model"

config = {
    "optimizer": "adam",
    "batch_size": 128,
    "epochs": 5,
    "validation_split": 0.1,
}

run = wandb.init(project=project, job_type=job_type, config=config)

version = "latest"
name = "{}:{}".format("{}_dataset".format(model_use_case_id), version)

artifact = run.use_artifact(name)

train_table = artifact.get("train_table")
x_train = train_table.get_column("x_train", convert_to="numpy")
y_train = train_table.get_column("y_train", convert_to="numpy")

# config 사전에서 값을 변수로 뽑아두기
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

# 트레이닝 데이터에 대한 레이블 생성
y_train = keras.utils.to_categorical(y_train, num_classes)

# 트레이닝 및 테스트 세트 생성
x_t, x_v, y_t, y_v = train_test_split(x_train, y_train, test_size=0.33)

# 모델 트레이닝
model.fit(
    x=x_t,
    y=y_t,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_v, y_v),
    callbacks=[WandbCallback(log_weights=True, log_evaluation=True)],
)

# 모델을 로컬에 저장
path = "model.h5"
model.save(path)

path = "./model.h5"
registered_model_name = "MNIST-dev"
name = "mnist_model"

run.link_model(path=path, registered_model_name=registered_model_name, name=name)
run.finish()
```

</details>
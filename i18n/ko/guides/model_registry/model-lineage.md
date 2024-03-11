---
displayed_sidebar: default
---

# 모델 이력 맵 생성
W&B에 모델 아티팩트를 로그할 때 유용한 기능 중 하나는 계보 그래프입니다. 계보 그래프는 특정 run에 의해 사용되거나 로그된 아티팩트를 보여줍니다.

이는 모델 아티팩트를 로그할 때, 최소한 해당 모델 아티팩트를 사용하거나 생성한 W&B run을 볼 수 있음을 의미합니다. [의존성을 추적](#track-an-artifact-dependency)하면, 모델 아티팩트에 의해 사용된 입력도 볼 수 있습니다.

예를 들어, 다음 이미지는 ML 실험 전반에 걸쳐 생성 및 사용된 아티팩트를 보여줍니다:

![](/images/models/model_lineage_example.png)

왼쪽에서 오른쪽으로, 이미지는 다음을 보여줍니다:
1. `jumping-monkey-1` W&B run이 `mnist_dataset:v0` 데이터셋 아티팩트를 생성했습니다.
2. `vague-morning-5` W&B run이 `mnist_dataset:v0` 데이터셋 아티팩트를 사용하여 모델을 트레이닝했습니다. 이 W&B run의 출력은 `mnist_model:v0`라는 모델 아티팩트였습니다.
3. `serene-haze-6`이라는 run이 모델 아티팩트(`mnist_model:v0`)를 사용하여 모델을 평가했습니다.

## 아티팩트 의존성 추적하기

W&B run에 입력으로 데이터셋 아티팩트를 선언하고 `use_artifact` API를 사용하여 의존성을 추적합니다.

다음 코드조각은 `use_artifact` API 사용 방법을 보여줍니다:

```python
# run을 초기화합니다
run = wandb.init(project=project, entity=entity)

# 아티팩트를 가져와서 의존성으로 표시합니다
artifact = run.use_artifact(artifact_or_name="name", aliases="<alias>")
```

아티팩트를 가져온 후, 해당 아티팩트를 사용하여 (예를 들어) 모델의 성능을 평가할 수 있습니다.

<details>

<summary>예시: 모델을 트레이닝하고 데이터셋을 모델의 입력으로 추적하기</summary>

```python
job_type = "train_model"

config = {
    "옵티마이저": "adam",
    "batch_size": 128,
    "에포크": 5,
    "validation_split": 0.1,
}

run = wandb.init(project=project, job_type=job_type, config=config)

version = "latest"
name = "{}:{}".format("{}_dataset".format(model_use_case_id), version)

# highlight-start
artifact = run.use_artifact(name)
# highlight-end

train_table = artifact.get("train_table")
x_train = train_table.get_column("x_train", convert_to="numpy")
y_train = train_table.get_column("y_train", convert_to="numpy")

# config 사전에서 값을 변수로 쉽게 접근하기 위해 저장합니다
num_classes = 10
input_shape = (28, 28, 1)
loss = "categorical_crossentropy"
옵티마이저 = run.config["옵티마이저"]
메트릭 = ["accuracy"]
batch_size = run.config["batch_size"]
에포크 = run.config["에포크"]
validation_split = run.config["validation_split"]

# 모델 아키텍처를 생성합니다
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
model.compile(loss=loss, optimizer=옵티마이저, metrics=메트릭)

# 트레이닝 데이터를 위한 레이블을 생성합니다
y_train = keras.utils.to_categorical(y_train, num_classes)

# 트레이닝 및 테스트 세트를 생성합니다
x_t, x_v, y_t, y_v = train_test_split(x_train, y_train, test_size=0.33)

# 모델을 트레이닝합니다
model.fit(
    x=x_t,
    y=y_t,
    batch_size=batch_size,
    에포크=에포크,
    validation_data=(x_v, y_v),
    콜백=[WandbCallback(log_weights=True, log_evaluation=True)],
)

# 모델을 로컬에 저장합니다
path = "model.h5"
model.save(path)

path = "./model.h5"
registered_model_name = "MNIST-dev"
name = "mnist_model"

# highlight-start
run.link_model(path=path, registered_model_name=registered_model_name, name=name)
# highlight-end
run.finish()
```

</details>
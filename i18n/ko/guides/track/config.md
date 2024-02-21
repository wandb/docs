---
description: Use a dictionary-like object to save your experiment configuration
displayed_sidebar: default
---

# 실험 설정하기

<head>
  <title>머신 러닝 실험 설정하기</title>
</head>

[**Colab 노트북에서 시도해보기**](http://wandb.me/config-colab)

`wandb.config` 객체를 사용하여 다음과 같은 학습 구성을 저장합니다:
- 하이퍼파라미터
- 데이터세트 이름이나 모델 유형과 같은 입력 설정
- 실험에 대한 다른 독립 변수들

`wandb.config` 속성을 사용하면 실험을 쉽게 분석하고 미래에 작업을 재현할 수 있습니다. W&B 앱에서 구성 값별로 그룹화하고, 다른 W&B 실행의 설정을 비교하고, 다양한 학습 구성이 출력에 어떤 영향을 미치는지 확인할 수 있습니다. 실행의 `config` 속성은 사전과 유사한 객체이며, 많은 사전과 유사한 객체에서 구성될 수 있습니다.

:::info
종속 변수(예: 손실 및 정확도) 또는 출력 메트릭은 `wandb.log`를 사용하여 저장해야 합니다.
:::

## 실험 구성 설정하기
구성은 일반적으로 학습 스크립트의 시작 부분에서 정의됩니다. 그러나 머신 러닝 워크플로는 다양할 수 있으므로 학습 스크립트의 시작 부분에서 구성을 정의할 필요는 없습니다.

:::caution
구성 변수 이름에 점을 사용하는 것을 피하십시오. 대신 대시나 밑줄을 사용하십시오. 스크립트가 루트 아래의 `wandb.config` 키에 엑세스하는 경우 속성 엑세스 구문 `config.key.foo` 대신 사전 엑세스 구문 `["key"]["foo"]`를 사용하십시오.
:::


다음 섹션에서는 실험 구성을 정의하는 다양한 일반적인 시나리오를 개요합니다.

### 초기화할 때 구성 설정하기
스크립트 시작 시 `wandb.init()` API를 호출할 때 사전을 전달하여 W&B 실행을 생성하고 데이터를 동기화 및 로깅하는 백그라운드 프로세스를 생성합니다.

다음 코드 조각은 구성 값이 있는 Python 사전을 정의하는 방법과 W&B 실행을 초기화할 때 해당 사전을 인수로 전달하는 방법을 보여줍니다.

```python
import wandb

# 구성 사전 객체 정의
config = {
    "hidden_layer_sizes": [32, 64],
    "kernel_sizes": [3],
    "activation": "ReLU",
    "pool_sizes": [2],
    "dropout": 0.5,
    "num_classes": 10,
}

# W&B를 초기화할 때 구성 사전 전달
run = wandb.init(project="config_example", config=config)
```

:::info
중첩된 사전을 `wandb.config()`에 전달할 수 있습니다. W&B 백엔드에서 점을 사용하여 이름을 평탄화합니다.
:::

다른 Python 사전에서처럼 키를 인덱스 값으로 사용하여 사전에서 값을 액세스합니다:

```python
# 키를 인덱스 값으로 사용하여 값에 액세스
hidden_layer_sizes = wandb.config["hidden_layer_sizes"]
kernel_sizes = wandb.config["kernel_sizes"]
activation = wandb.config["activation"]

# Python 사전 get() 메서드
hidden_layer_sizes = wandb.config.get("hidden_layer_sizes")
kernel_sizes = wandb.config.get("kernel_sizes")
activation = wandb.config.get("activation")
```

:::note
개발자 가이드와 예제에서는 가독성을 위해 구성 값을 별도의 변수로 복사합니다. 이 단계는 선택 사항입니다.
:::

### argparse를 사용하여 구성 설정하기
argparse 객체를 사용하여 구성을 설정할 수 있습니다. [argparse](https://docs.python.org/3/library/argparse.html)는 Python 3.2 이상에서 표준 라이브러리 모듈로, 커맨드 라인 인수의 모든 유연성과 파워를 활용하여 스크립트를 작성하기 쉽게 만듭니다.

이는 커맨드 라인에서 시작된 스크립트의 결과를 추적하는 데 유용합니다.

다음 Python 스크립트는 실험 구성을 정의하고 설정하기 위해 파서 객체를 정의하는 방법을 보여줍니다. 이 데모를 위한 `train_one_epoch` 및 `evaluate_one_epoch` 함수는 학습 루프를 시뮬레이션하기 위해 제공됩니다:

```python
# config_experiment.py
import wandb
import argparse
import numpy as np
import random


# 학습 및 평가 데모 코드
def train_one_epoch(epoch, lr, bs):
    acc = 0.25 + ((epoch / 30) + (random.random() / 10))
    loss = 0.2 + (1 - ((epoch - 1) / 10 + random.random() / 5))
    return acc, loss


def evaluate_one_epoch(epoch):
    acc = 0.1 + ((epoch / 20) + (random.random() / 10))
    loss = 0.25 + (1 - ((epoch - 1) / 10 + random.random() / 6))
    return acc, loss


def main(args):
    # W&B 실행 시작
    run = wandb.init(project="config_example", config=args)

    # 구성 사전에서 값을 액세스하고
    # 가독성을 위해 변수에 저장
    lr = wandb.config["learning_rate"]
    bs = wandb.config["batch_size"]
    epochs = wandb.config["epochs"]

    # 학습 시뮬레이션 및 W&B에 값 로깅
    for epoch in np.arange(1, epochs):
        train_acc, train_loss = train_one_epoch(epoch, lr, bs)
        val_acc, val_loss = evaluate_one_epoch(epoch)

        wandb.log(
            {
                "epoch": epoch,
                "train_acc": train_acc,
                "train_loss": train_loss,
                "val_acc": val_acc,
                "val_loss": val_loss,
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-b", "--batch_size", type=int, default=32, help="배치 크기")
    parser.add_argument(
        "-e", "--epochs", type=int, default=50, help="학습 에포크 수"
    )
    parser.add_argument(
        "-lr", "--learning_rate", type=int, default=0.001, help="학습률"
    )

    args = parser.parse_args()
    main(args)
```

### 스크립트 전체에서 구성 설정하기
스크립트 전체에서 구성 객체에 더 많은 파라미터를 추가할 수 있습니다. 다음 코드 조각은 구성 객체에 새로운 키-값 쌍을 추가하는 방법을 보여줍니다:

```python
import wandb

# 구성 사전 객체 정의
config = {
    "hidden_layer_sizes": [32, 64],
    "kernel_sizes": [3],
    "activation": "ReLU",
    "pool_sizes": [2],
    "dropout": 0.5,
    "num_classes": 10,
}

# W&B를 초기화할 때 구성 사전 전달
run = wandb.init(project="config_example", config=config)

# W&B를 초기화한 후 구성 업데이트
wandb.config["dropout"] = 0.2
wandb.config.epochs = 4
wandb.config["batch_size"] = 32
```
한 번에 여러 값을 업데이트할 수 있습니다:

```python
wandb.init(config={"epochs": 4, "batch_size": 32})
# 나중에
wandb.config.update({"lr": 0.1, "channels": 16})
```

### 실행이 끝난 후 구성 설정하기
실행이 완료된 후 구성(또는 실행에 대한 다른 모든 것)을 업데이트하려면 [W&B 공개 API](../../ref/python/public-api/README.md)를 사용하십시오. 실행 중에 값을 로그하지 않은 경우 특히 유용합니다.

실행이 끝난 후 구성을 업데이트하려면 `entity`, `프로젝트 이름`, 그리고 `실행 ID`를 제공하십시오. 이 값들은 `wandb.run` 자체나 [W&B 앱 UI](../app/intro.md)에서 직접 찾을 수 있습니다:

```python
api = wandb.Api()

# 실행 객체나 W&B 앱에서
# 속성에 직접 액세스
username = wandb.run.entity
project = wandb.run.project
run_id = wandb.run.id

run = api.run(f"{username}/{project}/{run_id}")
run.config["bar"] = 32
run.update()
```

## `absl.FLAGS`


[`absl` 플래그](https://abseil.io/docs/python/guides/flags)를 전달할 수도 있습니다.

```python
flags.DEFINE_string("model", None, "실행할 모델")  # 이름, 기본값, 도움말

wandb.config.update(flags.FLAGS)  # absl 플래그를 구성에 추가
```

## 파일 기반 구성
`config-defaults.yaml`이라는 파일을 생성하면 키-값 쌍이 자동으로 `wandb.config`에 전달됩니다.

다음 코드 조각은 `config-defaults.yaml` YAML 파일의 샘플을 보여줍니다:

```yaml
# config-defaults.yaml
# 샘플 구성 기본값 파일
epochs:
  desc: 학습을 진행할 에포크 수
  value: 100
batch_size:
  desc: 각 미니-배치의 크기
  value: 32
```
`config-defaults.yaml`에 의해 자동으로 전달된 값을 덮어쓸 수 있습니다. 이를 위해 `wandb.init`의 `config` 인수에 값을 전달합니다.

명령줄 인수 `--configs`를 사용하여 다른 구성 파일을 로드할 수도 있습니다.

### 파일 기반 구성의 예제 사용 사례
실행에 대한 일부 메타데이터가 포함된 YAML 파일이 있고, 그리고 Python 스크립트에 하이퍼파라미터의 사전이 있다고 가정해 보겠습니다. 중첩된 `config` 객체에 둘 다 저장할 수 있습니다:

```python
hyperparameter_defaults = dict(
    dropout=0.5,
    batch_size=100,
    learning_rate=0.001,
)

config_dictionary = dict(
    yaml=my_yaml_file,
    params=hyperparameter_defaults,
)

wandb.init(config=config_dictionary)
```

## TensorFlow v1 플래그

`wandb.config` 객체에 직접 TensorFlow 플래그를 전달할 수 있습니다.

```python
wandb.init()
wandb.config.epochs = 4

flags = tf.app.flags
flags.DEFINE_string("data_dir", "/tmp/data")
flags.DEFINE_integer("batch_size", 128, "배치 크기.")
wandb.config.update(flags.FLAGS)  # tensorflow 플래그를 구성에 추가
```
---
title: Configure experiments
description: 사전과 같은 오브젝트를 사용하여 실험 설정을 저장하세요
displayed_sidebar: default
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Configs_in_W%26B.ipynb"></CTAButtons>

`wandb.config` 오브젝트를 사용하여 다음과 같은 트레이닝 설정을 저장하세요:
- 하이퍼파라미터
- 데이터셋 이름이나 모델 유형과 같은 입력 설정
- 실험의 다른 독립 변수들

`wandb.config` 속성은 실험을 분석하고 미래에 작업을 재현하는 것을 쉽게 만듭니다. W&B 앱에서 설정 값을 기준으로 그룹화하고, 다양한 W&B Runs의 설정을 비교하며, 서로 다른 트레이닝 설정이 출력에 어떻게 영향을 미치는지 볼 수 있습니다. Run의 `config` 속성은 사전과 유사한 오브젝트이며, 다수의 사전과 유사한 오브젝트로 구성될 수 있습니다.

:::info
종속 변수(예: 손실과 정확도)나 출력 메트릭은 대신 `wandb.log`로 저장해야 합니다.
:::



## 실험 설정 준비하기
설정은 일반적으로 트레이닝 스크립트의 시작 부분에서 정의됩니다. 기계학습 워크플로우는 다양할 수 있으므로, 트레이닝 스크립트의 시작 부분에서 설정을 정의할 필요는 없습니다.

:::caution
설정 변수 이름에서 점 사용을 피하는 것을 권장합니다. 대신 대시나 밑줄을 사용하세요. 스크립트가 `wandb.config` 키에 루트 이하로 엑세스하는 경우 속성 엑세스 구문 `config.key.foo` 대신 사전 엑세스 구문 `["key"]["foo"]`를 사용하세요.
:::


다음 섹션에서는 실험 설정을 정의하는 다양한 일반적인 시나리오를 설명합니다.

### 초기화 시점에 설정하기
스크립트의 시작 부분에서 사전을 전달하여 `wandb.init()` API를 호출하여 W&B Run으로 데이터 동기화 및 로그를 위한 백그라운드 프로세스를 생성하세요.

다음 코드조각은 설정 값을 가진 파이썬 사전을 정의하고 W&B Run을 초기화할 때 그 사전을 인수로 전달하는 방법을 설명합니다.

```python
import wandb

# 설정 사전 오브젝트 정의
config = {
    "hidden_layer_sizes": [32, 64],
    "kernel_sizes": [3],
    "activation": "ReLU",
    "pool_sizes": [2],
    "dropout": 0.5,
    "num_classes": 10,
}

# W&B 초기화 시 설정 사전 전달
run = wandb.init(project="config_example", config=config)
```

:::info
`wandb.config()`에 중첩된 사전을 전달할 수 있습니다. W&B는 백엔드에서 이름을 점을 사용하여 평탄화합니다.
:::

사전의 값을 다른 파이썬 사전에서 엑세스하는 것과 비슷하게 엑세스하세요:

```python
# 인덱스 값으로 키를 사용하여 값 엑세스
hidden_layer_sizes = wandb.config["hidden_layer_sizes"]
kernel_sizes = wandb.config["kernel_sizes"]
activation = wandb.config["activation"]

# 파이썬 사전 get() 메소드 사용
hidden_layer_sizes = wandb.config.get("hidden_layer_sizes")
kernel_sizes = wandb.config.get("kernel_sizes")
activation = wandb.config.get("activation")
```

:::note
개발자 가이드와 예제 전반에서 설정 값을 별개의 변수로 복사합니다. 이 단계는 선택 사항입니다. 가독성을 위해 수행됩니다.
:::

### argparse로 설정하기
argparse 오브젝트로 설정을 정의할 수 있습니다. [argparse](https://docs.python.org/3/library/argparse.html)는 인수 파서를 줄인 말로, Python 3.2 이상에서 사용할 수 있는 표준 라이브러리 모듈로, 커맨드라인 인수의 유연성과 강력함을 활용하는 스크립트를 쉽게 작성할 수 있도록 합니다.

이는 커맨드라인에서 실행된 스크립트의 결과 추적에 유용합니다.

다음 파이썬 스크립트는 실험 설정을 정의하고 설정하기 위한 파서 오브젝트를 정의하는 방법을 설명합니다. `train_one_epoch`와 `evaluate_one_epoch` 함수는 이 데모를 위한 트레이닝 루프를 시뮬레이션하기 위해 제공됩니다:

```python
# config_experiment.py
import wandb
import argparse
import numpy as np
import random


# 트레이닝 및 평가 데모 코드
def train_one_epoch(epoch, lr, bs):
    acc = 0.25 + ((epoch / 30) + (random.random() / 10))
    loss = 0.2 + (1 - ((epoch - 1) / 10 + random.random() / 5))
    return acc, loss


def evaluate_one_epoch(epoch):
    acc = 0.1 + ((epoch / 20) + (random.random() / 10))
    loss = 0.25 + (1 - ((epoch - 1) / 10 + random.random() / 6))
    return acc, loss


def main(args):
    # W&B Run 시작
    run = wandb.init(project="config_example", config=args)

    # 설정 사전에서 값을 엑세스하고 가독성을 위해 변수에 저장
    lr = wandb.config["learning_rate"]
    bs = wandb.config["batch_size"]
    epochs = wandb.config["epochs"]

    # 트레이닝을 시뮬레이트하고 W&B에 값을 로그
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

    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "-e", "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "-lr", "--learning_rate", type=int, default=0.001, help="Learning rate"
    )

    args = parser.parse_args()
    main(args)
```
### 스크립트 전반에 설정하기
스크립트 전반에 걸쳐 config 오브젝트에 더 많은 파라미터를 추가할 수 있습니다. 다음 코드조각은 config 오브젝트에 새로운 키-값 쌍을 추가하는 방법을 보여줍니다:

```python
import wandb

# 설정 사전 오브젝트 정의
config = {
    "hidden_layer_sizes": [32, 64],
    "kernel_sizes": [3],
    "activation": "ReLU",
    "pool_sizes": [2],
    "dropout": 0.5,
    "num_classes": 10,
}

# W&B 초기화 시 설정 사전 전달
run = wandb.init(project="config_example", config=config)

# W&B 초기화 후 설정 업데이트
wandb.config["dropout"] = 0.2
wandb.config.epochs = 4
wandb.config["batch_size"] = 32
```
여러 값을 한 번에 업데이트 할 수 있습니다:

```python
wandb.init(config={"epochs": 4, "batch_size": 32})
# 나중에
wandb.config.update({"lr": 0.1, "channels": 16})
```

### Run이 완료된 후 설정하기
Run 이후에 설정을 업데이트하려면 [W&B Public API](../../ref/python/public-api/README.md)를 사용하세요. 이는 Run 중에 값을 로그하는 것을 잊은 경우 특히 유용합니다.

`entity`, `project name`, `Run ID`를 제공하여 Run이 완료된 후 설정을 업데이트하세요. 이러한 값은 Run 오브젝트 자체인 `wandb.run`에서 직접 또는 [W&B App UI](../app/intro.md)에서 찾을 수 있습니다:

```python
api = wandb.Api()

# Run 오브젝트 또는 W&B 앱에서 속성 엑세스
username = wandb.run.entity
project = wandb.run.project
run_id = wandb.run.id

run = api.run(f"{username}/{project}/{run_id}")
run.config["bar"] = 32
run.update()
```

## `absl.FLAGS`

[`absl` flags](https://abseil.io/docs/python/guides/flags)를 전달할 수도 있습니다.

```python
flags.DEFINE_string("model", None, "model to run")  # name, default, help

wandb.config.update(flags.FLAGS)  # absl 플래그를 config에 추가
```

## 파일 기반 설정
`config-defaults.yaml`이라는 파일을 만들면 키-값 쌍이 자동으로 `wandb.config`로 전달됩니다.

다음 코드조각은 샘플 `config-defaults.yaml` YAML 파일을 보여줍니다:

```yaml
# config-defaults.yaml
# 샘플 설정 기본값 파일
epochs:
  desc: Train over할 에포크 수
  value: 100
batch_size:
  desc: 각 미니배치의 크기
  value: 32
```
자동으로 `config-defaults.yaml`에 의해 전달된 것을 덮어쓸 수 있습니다. 이렇게 하려면 `wandb.init`의 `config` 인수에 값을 전달하세요.

`--configs`라는 커맨드라인 인수로 다른 설정 파일을 로드할 수 있습니다.

### 파일 기반 설정의 예시 사용 사례
메타데이터가 포함된 YAML 파일과 파이썬 스크립트에 하이퍼파라미터 사전이 있다고 가정합니다. 두 가지를 중첩된 `config` 오브젝트에 저장할 수 있습니다:

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

TensorFlow 플래그를 직접 `wandb.config` 오브젝트에 전달할 수 있습니다.

```python
wandb.init()
wandb.config.epochs = 4

flags = tf.app.flags
flags.DEFINE_string("data_dir", "/tmp/data")
flags.DEFINE_integer("batch_size", 128, "Batch size.")
wandb.config.update(flags.FLAGS)  # tensorflow 플래그를 config에 추가
```
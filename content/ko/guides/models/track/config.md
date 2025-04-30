---
title: Configure experiments
description: 사전 과 유사한 오브젝트를 사용하여 실험 설정을 저장합니다.
menu:
  default:
    identifier: ko-guides-models-track-config
    parent: experiments
weight: 2
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Configs_in_W%26B.ipynb" >}}

트레이닝 설정을 저장하려면 run의 `config` 속성을 사용하세요.
- 하이퍼파라미터
- 데이터셋 이름 또는 모델 유형과 같은 입력 설정
- 모든 다른 독립 변수들

`run.config` 속성을 사용하면 실험을 쉽게 분석하고 향후 작업을 재현할 수 있습니다. W&B 앱에서 설정 값으로 그룹화하고, 서로 다른 W&B run의 설정을 비교하고, 각 트레이닝 설정이 결과에 미치는 영향을 평가할 수 있습니다. `config` 속성은 여러 사전과 유사한 오브젝트로 구성될 수 있는 사전과 유사한 오브젝트입니다.

{{% alert %}}
손실 및 정확도와 같은 출력 메트릭 또는 종속 변수를 저장하려면 `run.config` 대신 `run.log`를 사용하세요.
{{% /alert %}}

## 실험 설정 구성하기
일반적으로 설정은 트레이닝 스크립트의 시작 부분에 정의됩니다. 그러나 기계 학습 워크플로우는 다를 수 있으므로 트레이닝 스크립트의 시작 부분에 설정을 정의할 필요는 없습니다.

config 변수 이름에는 마침표 (`.`) 대신 대시 (`-`) 또는 밑줄 (`_`)을 사용하세요.

스크립트가 루트 아래의 `run.config` 키에 엑세스하는 경우 속성 엑세스 구문 `config.key.value` 대신 사전 엑세스 구문 `["key"]["value"]`를 사용하세요.

다음 섹션에서는 실험 설정을 정의하는 다양한 일반적인 시나리오에 대해 설명합니다.

### 초기화 시 설정 구성하기
W&B Run으로 데이터를 동기화하고 기록하는 백그라운드 프로세스를 생성하기 위해 `wandb.init()` API를 호출할 때 스크립트 시작 부분에 사전을 전달합니다.

다음 코드 조각은 설정 값으로 Python 사전을 정의하는 방법과 W&B Run을 초기화할 때 해당 사전을 인수로 전달하는 방법을 보여줍니다.

```python
import wandb

# config 사전 오브젝트 정의
config = {
    "hidden_layer_sizes": [32, 64],
    "kernel_sizes": [3],
    "activation": "ReLU",
    "pool_sizes": [2],
    "dropout": 0.5,
    "num_classes": 10,
}

# W&B를 초기화할 때 config 사전 전달
with wandb.init(project="config_example", config=config) as run:
    ...
```

중첩된 사전을 `config`로 전달하면 W&B는 점을 사용하여 이름을 평면화합니다.

Python에서 다른 사전에 엑세스하는 방법과 유사하게 사전에서 값에 엑세스합니다.

```python
# 키를 인덱스 값으로 사용하여 값에 엑세스
hidden_layer_sizes = run.config["hidden_layer_sizes"]
kernel_sizes = run.config["kernel_sizes"]
activation = run.config["activation"]

# Python 사전 get() 메소드
hidden_layer_sizes = run.config.get("hidden_layer_sizes")
kernel_sizes = run.config.get("kernel_sizes")
activation = run.config.get("activation")
```

{{% alert %}}
개발자 가이드 및 예제 전체에서 설정 값을 별도의 변수로 복사합니다. 이 단계는 선택 사항입니다. 가독성을 위해 수행됩니다.
{{% /alert %}}

### argparse로 설정 구성하기
argparse 오브젝트로 설정을 구성할 수 있습니다. argument parser의 약자인 [argparse](https://docs.python.org/3/library/argparse.html)는 커맨드라인 인수의 모든 유연성과 성능을 활용하는 스크립트를 쉽게 작성할 수 있도록 하는 Python 3.2 이상의 표준 라이브러리 모듈입니다.

이는 커맨드라인에서 실행되는 스크립트의 결과를 추적하는 데 유용합니다.

다음 Python 스크립트는 파서 오브젝트를 정의하여 실험 설정을 정의하고 구성하는 방법을 보여줍니다. 함수 `train_one_epoch` 및 `evaluate_one_epoch`는 이 데모의 목적으로 트레이닝 루프를 시뮬레이션하기 위해 제공됩니다.

```python
# config_experiment.py
import argparse
import random

import numpy as np
import wandb


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
    with wandb.init(project="config_example", config=args) as run:
        # config 사전에서 값에 엑세스하고 가독성을 위해 변수에 저장합니다.
        lr = run.config["learning_rate"]
        bs = run.config["batch_size"]
        epochs = run.config["epochs"]

        # W&B에 값을 시뮬레이션하여 트레이닝하고 기록합니다.
        for epoch in np.arange(1, epochs):
            train_acc, train_loss = train_one_epoch(epoch, lr, bs)
            val_acc, val_loss = evaluate_one_epoch(epoch)

            run.log(
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
        "-e", "--epochs", type=int, default=50, help="트레이닝 에포크 수"
    )
    parser.add_argument(
        "-lr", "--learning_rate", type=int, default=0.001, help="학습률"
    )

    args = parser.parse_args()
    main(args)
```
### 스크립트 전체에서 설정 구성하기
스크립트 전체에서 config 오브젝트에 더 많은 파라미터를 추가할 수 있습니다. 다음 코드 조각은 config 오브젝트에 새 키-값 쌍을 추가하는 방법을 보여줍니다.

```python
import wandb

# config 사전 오브젝트 정의
config = {
    "hidden_layer_sizes": [32, 64],
    "kernel_sizes": [3],
    "activation": "ReLU",
    "pool_sizes": [2],
    "dropout": 0.5,
    "num_classes": 10,
}

# W&B를 초기화할 때 config 사전 전달
with wandb.init(project="config_example", config=config) as run:
    # W&B를 초기화한 후 config 업데이트
    run.config["dropout"] = 0.2
    run.config.epochs = 4
    run.config["batch_size"] = 32
```

여러 값을 한 번에 업데이트할 수 있습니다.

```python
run.config.update({"lr": 0.1, "channels": 16})
```

### Run이 완료된 후 설정 구성하기
[W&B Public API]({{< relref path="/ref/python/public-api/" lang="ko" >}})를 사용하여 완료된 run의 구성을 업데이트합니다.

API에 엔티티, 프로젝트 이름 및 run의 ID를 제공해야 합니다. Run 오브젝트 또는 [W&B 앱 UI]({{< relref path="/guides/models/track/workspaces.md" lang="ko" >}})에서 이러한 세부 정보를 찾을 수 있습니다.

```python
with wandb.init() as run:
    ...

# 현재 스크립트 또는 노트북에서 시작된 경우 Run 오브젝트에서 다음 값을 찾거나 W&B 앱 UI에서 복사할 수 있습니다.
username = run.entity
project = run.project
run_id = run.id

# api.run()은 wandb.init()과 다른 유형의 오브젝트를 반환합니다.
api = wandb.Api()
api_run = api.run(f"{username}/{project}/{run_id}")
api_run.config["bar"] = 32
api_run.update()
```

## `absl.FLAGS`

[`absl` flags](https://abseil.io/docs/python/guides/flags)를 전달할 수도 있습니다.

```python
flags.DEFINE_string("model", None, "실행할 모델")  # name, default, help

run.config.update(flags.FLAGS)  # absl flags를 config에 추가
```

## 파일 기반 Configs
`config-defaults.yaml`이라는 파일을 run 스크립트와 동일한 디렉토리에 배치하면 run이 파일에 정의된 키-값 쌍을 자동으로 선택하여 `run.config`에 전달합니다.

다음 코드 조각은 샘플 `config-defaults.yaml` YAML 파일을 보여줍니다.

```yaml
batch_size:
  desc: 각 미니배치의 크기
  value: 32
```

`wandb.init`의 `config` 인수에 업데이트된 값을 설정하여 `config-defaults.yaml`에서 자동으로 로드된 기본값을 재정의할 수 있습니다. 예:

```python
import wandb

# 사용자 지정 값을 전달하여 config-defaults.yaml 재정의
with wandb.init(config={"epochs": 200, "batch_size": 64}) as run:
    ...
```

`config-defaults.yaml` 이외의 구성 파일을 로드하려면 `--configs 커맨드라인` 인수를 사용하고 파일 경로를 지정합니다.

```bash
python train.py --configs other-config.yaml
```

### 파일 기반 configs의 유스 케이스 예제
Run에 대한 일부 메타데이터가 있는 YAML 파일과 Python 스크립트에 하이퍼파라미터 사전이 있다고 가정합니다. 중첩된 `config` 오브젝트에 둘 다 저장할 수 있습니다.

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

with wandb.init(config=config_dictionary) as run:
    ...
```

## TensorFlow v1 flags

TensorFlow flags를 `wandb.config` 오브젝트에 직접 전달할 수 있습니다.

```python
with wandb.init() as run:
    run.config.epochs = 4

    flags = tf.app.flags
    flags.DEFINE_string("data_dir", "/tmp/data")
    flags.DEFINE_integer("batch_size", 128, "배치 크기.")
    run.config.update(flags.FLAGS)  # tensorflow flags를 config로 추가
```
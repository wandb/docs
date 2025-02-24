---
title: Configure experiments
description: 사전 과 같은 오브젝트를 사용하여 실험 설정을 저장하세요
menu:
  default:
    identifier: ko-guides-models-track-config
    parent: experiments
weight: 2
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Configs_in_W%26B.ipynb" >}}

`wandb.config` 오브젝트를 사용하여 다음과 같은 트레이닝 설정을 저장합니다:
- 하이퍼파라미터
- 데이터셋 이름 또는 모델 유형과 같은 입력 설정
- 실험을 위한 다른 독립 변수

`wandb.config` 속성을 사용하면 실험을 쉽게 분석하고 향후 작업을 재현할 수 있습니다. W&B App에서 설정 값으로 그룹화하고, 다른 W&B Runs의 설정을 비교하고, 다양한 트레이닝 설정이 출력에 미치는 영향을 확인할 수 있습니다. Run의 `config` 속성은 사전과 유사한 오브젝트이며, 다양한 사전과 유사한 오브젝트에서 빌드할 수 있습니다.

{{% alert %}}
종속 변수 (손실 및 정확도와 같은) 또는 출력 메트릭은 `wandb.log`를 사용하여 저장해야 합니다.
{{% /alert %}}

## 실험 설정 구성
설정은 일반적으로 트레이닝 스크립트의 시작 부분에 정의됩니다. 그러나 기계 학습 워크플로우는 다를 수 있으므로 트레이닝 스크립트의 시작 부분에서 설정을 정의할 필요는 없습니다.

{{% alert color="secondary" %}}
config 변수 이름에 점을 사용하지 않는 것이 좋습니다. 대신 대시 또는 밑줄을 사용하십시오. 스크립트가 루트 아래의 `wandb.config` 키에 엑세스하는 경우 속성 엑세스 구문 `config.key.foo` 대신 사전 엑세스 구문 `["key"]["foo"]`를 사용하십시오.
{{% /alert %}}

다음 섹션에서는 실험 설정을 정의하는 다양한 일반적인 시나리오를 간략하게 설명합니다.

### 초기화 시 설정 구성
스크립트 시작 시 `wandb.init()` API를 호출할 때 사전을 전달하여 백그라운드 프로세스를 생성하여 데이터를 동기화하고 W&B Run으로 기록합니다.

다음 코드 조각은 구성 값으로 Python 사전을 정의하고 W&B Run을 초기화할 때 해당 사전을 인수로 전달하는 방법을 보여줍니다.

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
run = wandb.init(project="config_example", config=config)
```

{{% alert %}}
중첩된 사전을 `wandb.config()`에 전달할 수 있습니다. W&B는 W&B 백엔드에서 점을 사용하여 이름을 평면화합니다.
{{% /alert %}}

Python에서 다른 사전에 엑세스하는 방법과 유사하게 사전에서 값에 엑세스합니다:

```python
# 키를 인덱스 값으로 사용하여 값에 엑세스
hidden_layer_sizes = wandb.config["hidden_layer_sizes"]
kernel_sizes = wandb.config["kernel_sizes"]
activation = wandb.config["activation"]

# Python 사전 get() 메소드
hidden_layer_sizes = wandb.config.get("hidden_layer_sizes")
kernel_sizes = wandb.config.get("kernel_sizes")
activation = wandb.config.get("activation")
```

{{% alert %}}
개발자 가이드와 예제 전체에서 구성 값을 별도의 변수에 복사합니다. 이 단계는 선택 사항입니다. 가독성을 위해 수행됩니다.
{{% /alert %}}

### argparse로 설정 구성
argparse 오브젝트로 구성을 설정할 수 있습니다. [argparse](https://docs.python.org/3/library/argparse.html)(argument parser의 약자)는 커맨드라인 인수의 모든 유연성과 기능을 활용하는 스크립트를 쉽게 작성할 수 있도록 하는 Python 3.2 이상의 표준 라이브러리 모듈입니다.

이는 커맨드라인에서 시작된 스크립트의 결과를 추적하는 데 유용합니다.

다음 Python 스크립트는 실험 config를 정의하고 설정하기 위해 parser 오브젝트를 정의하는 방법을 보여줍니다. 함수 `train_one_epoch` 및 `evaluate_one_epoch`는 이 데모의 목적으로 트레이닝 루프를 시뮬레이션하기 위해 제공됩니다.

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

    # config 사전에서 값에 엑세스하고 가독성을 위해 변수에 저장
    lr = wandb.config["learning_rate"]
    bs = wandb.config["batch_size"]
    epochs = wandb.config["epochs"]

    # W&B에 트레이닝 및 로깅 값 시뮬레이션
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
        "-e", "--epochs", type=int, default=50, help="트레이닝 에포크 수"
    )
    parser.add_argument(
        "-lr", "--learning_rate", type=int, default=0.001, help="학습률"
    )

    args = parser.parse_args()
    main(args)
```
### 스크립트 전체에서 설정 구성
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
run = wandb.init(project="config_example", config=config)

# W&B를 초기화한 후 config 업데이트
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

### Run이 완료된 후 설정 구성
[W&B Public API]({{< relref path="/ref/python/public-api/" lang="ko" >}})를 사용하여 Run 완료 후 config (또는 완료된 Run의 다른 모든 항목)를 업데이트합니다. 이는 Run 중에 값을 기록하는 것을 잊은 경우 특히 유용합니다.

Run 완료 후 구성을 업데이트하려면 `entity`, `프로젝트 이름` 및 `Run ID`를 제공합니다. 이러한 값은 Run 오브젝트 자체 `wandb.run` 또는 [W&B App UI]({{< relref path="/guides/models/track/workspaces.md" lang="ko" >}})에서 직접 찾으십시오.

```python
api = wandb.Api()

# Run 오브젝트 또는 W&B App에서 직접 속성에 엑세스
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
flags.DEFINE_string("model", None, "실행할 모델")  # 이름, 기본값, 도움말

wandb.config.update(flags.FLAGS)  # absl flags를 config에 추가
```

## 파일 기반 Configs
`config-defaults.yaml`이라는 파일을 run 스크립트와 동일한 디렉토리에 배치하면 run은 파일에 정의된 키-값 쌍을 자동으로 선택하여 `wandb.config`에 전달합니다.

다음 코드 조각은 샘플 `config-defaults.yaml` YAML 파일을 보여줍니다:

```yaml
batch_size:
  desc: 각 미니배치의 크기
  value: 32
```

`wandb.init`의 `config` 인수에 업데이트된 값을 설정하여 `config-defaults.yaml`에서 자동으로 로드된 기본값을 재정의할 수 있습니다. 예:

```python
import wandb
# 사용자 지정 값을 전달하여 config-defaults.yaml 재정의
wandb.init(config={"epochs": 200, "batch_size": 64})
```

`config-defaults.yaml`이 아닌 다른 구성 파일을 로드하려면 `--configs 커맨드라인` 인수를 사용하고 파일 경로를 지정합니다:

```bash
python train.py --configs other-config.yaml
```

### 파일 기반 configs의 유스 케이스 예
Run에 대한 일부 메타데이터가 포함된 YAML 파일과 Python 스크립트에 하이퍼파라미터 사전이 있다고 가정합니다. 중첩된 `config` 오브젝트에 둘 다 저장할 수 있습니다:

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

## TensorFlow v1 flags

TensorFlow flags를 `wandb.config` 오브젝트에 직접 전달할 수 있습니다.

```python
wandb.init()
wandb.config.epochs = 4

flags = tf.app.flags
flags.DEFINE_string("data_dir", "/tmp/data")
flags.DEFINE_integer("batch_size", 128, "배치 크기.")
wandb.config.update(flags.FLAGS)  # tensorflow flags를 config로 추가
```
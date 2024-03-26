---
description: Use a dictionary-like object to save your experiment configuration
displayed_sidebar: default
---

# 실험 설정하기

<head>
  <title>머신러닝 실험 설정하기</title>
</head>

[**여기에있는 Colab 노트북으로 시도해보세요**](http://wandb.me/config-colab)

`wandb.config` 객체를 사용하여 다음과 같은 트레이닝 설정을 저장하세요:
- 하이퍼파라미터
- 데이터셋 이름 또는 모델 유형과 같은 입력 설정
- 실험을 위한 다른 독립 변수

`wandb.config` 속성을 사용하면 쉽게 실험을 분석하고 향후에 재현할 수 있습니다. W&B 앱에서 구성 값별로 그룹화하고, 다른 W&B Run의 설정과 비교하고, 다른 트레이닝 설정이 출력에 어떤 영향을 미치는지 확인할 수 있습니다. Run의 `config` 속성은 딕셔너리와 유사한 객체로, 여러개의 딕셔너리 같은 객체로부터 구축할 수 있습니다.

:::info
종속 변수(예: 손실과 정확도) 또는 출력 메트릭은 `wandb.log`를 통해 저장해야 합니다.
:::

## 실험 구성 설정하기
구성은 일반적으로 트레이닝 스크립트의 시작 부분에서 정의됩니다. 그러나 머신러닝 워크플로우는 다양할 수 있으므로, 꼭 트레이닝 스크립트의 시작 부분에서 구성을 정의할 필요는 없습니다.

:::caution
구성 변수 이름에 점을 사용하는 것을 피해주세요. 대신 대시나 밑줄을 사용하세요. 스크립트가 `wandb.config` 키를 루트 아래에서 엑세스할 경우에는 속성 엑세스 신택스 `config.key.foo` 대신 딕셔너리 엑세스 신택스 `["key"]["foo"]`를  사용하세요.
:::


다음 섹션에서는 실험 구성을 정의하는 방법에 대한 여러 가지 일반적인 시나리오를 간략하게 설명합니다.

### 초기화 시 구성 설정하기
W&B Run으로 데이터를 동기화하고 기록하는 백그라운드 프로세스를 생성하기 위해 `wandb.init()` API를 호출할 때 스크립트의 시작 부분에 딕셔너리를 넣어주세요.

다음 코드조각은 W&B Run을 초기화할 때 구성 값이 포함된 Python 딕셔너리를 정의하는 방법과 해당 딕셔너리를 인수로 전달하는 방법을 보여줍니다.

```python
import wandb

# Config 딕셔너리 객체 정의하기
config = {
    "hidden_layer_sizes": [32, 64],
    "kernel_sizes": [3],
    "activation": "ReLU",
    "pool_sizes": [2],
    "dropout": 0.5,
    "num_classes": 10,
}

# W&B 초기화 시 config 딕셔너리 전달하기
run = wandb.init(project="config_example", config=config)
```

:::info
`wandb.config()`에 중첩 딕셔너리를 전달할 수 있습니다. W&B 백엔드에서는 점을 사용하여 이름을 플래튼하게 만듭니다.
:::

딕셔너리에서 값을 접근하는 방법은 파이썬에서 다른 딕셔너리를 접근하는 방법과 유사합니다:

```python
# 키를 인덱스 값으로 사용하여 값에 접근하기
hidden_layer_sizes = wandb.config["hidden_layer_sizes"]
kernel_sizes = wandb.config["kernel_sizes"]
activation = wandb.config["activation"]

# 파이썬 딕셔너리 get() 메소드
hidden_layer_sizes = wandb.config.get("hidden_layer_sizes")
kernel_sizes = wandb.config.get("kernel_sizes")
activation = wandb.config.get("activation")
```

:::note
개발자 가이드와 예제에서는 가독성을 위해 구성 값을 별도의 변수로 복사합니다. 이 단계는 선택 사항입니다.
:::

### argparse를 사용하여 구성 설정하기
argparse 객체를 사용하여 구성을 설정할 수 있습니다. [argparse](https://docs.python.org/3/library/argparse.html), 즉 인수 파서는 Python 3.2 이상에서 사용할 수 있는 표준 라이브러리 모듈로, 커맨드라인 인수의 모든 유연성과 강력한 기능을 활용하여 스크립트를 쉽게 작성할 수 있도록 해줍니다.

이는 커맨드라인에서 실행되는 스크립트의 결과를 추적하는 데 유용합니다.

이어지는 파이썬 스크립트는 파서 객체를 정의하여 실험 구성을 정의하고 설정하는 방법을 보여줍니다. 이 데모의 목적에 따라 훈련 루프를 시뮬레이션하기 위해 `train_one_epoch` 및 `evaluate_one_epoch` 함수가 제공됩니다:

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
    # W&B Run 시작하기
    run = wandb.init(project="config_example", config=args)

    # config 딕셔너리에서 값에 접근하여
    # 가독성을 위해 변수에 저장하기
    lr = wandb.config["learning_rate"]
    bs = wandb.config["batch_size"]
    epochs = wandb.config["epochs"]

    # 트레이닝 시뮬레이션 및 W&B에 값 로깅하기
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

### 스크립트 전체에서 구성 설정하기
스크립트 전체에서 config 오브젝트에 더 많은 파라미터를 추가할 수 있습니다. 다음 코드조각은 config 오브젝트에 새로운 키-값 페어를 추가하는 방법을 보여줍니다:

```python
import wandb

# Config 딕셔너리 오브젝트 정의하기
config = {
    "hidden_layer_sizes": [32, 64],
    "kernel_sizes": [3],
    "activation": "ReLU",
    "pool_sizes": [2],
    "dropout": 0.5,
    "num_classes": 10,
}

# W&B 초기화 시 config 딕셔너리 패스하기
run = wandb.init(project="config_example", config=config)

# W&B 초기화 후 config 업데이트하기
wandb.config["dropout"] = 0.2
wandb.config.epochs = 4
wandb.config["batch_size"] = 32
```
여러 값을 한 번에 업데이트할 수 있습니다:

```python
wandb.init(config={"epochs": 4, "batch_size": 32})
# 나중에
wandb.config.update({"lr": 0.1, "channels": 16})
```

### Run이 끝난 후 구성 설정하기
Run이 완료된 후에 [W&B Public API](../../ref/python/public-api/README.md)를 사용하여 구성(또는 완료된 Run과 다른 내용)을 업데이트할 수 있습니다. 이 기능은 Run 중에 값을 로깅하는 것을 잊어버린 경우에 특히 유용합니다.

`entity`, `프로젝트 이름`, 그리고 `Run ID`를 통해 Run이 완료된 후 구성을 업데이트할 수 있습니다. 이러한 값은 `wandb.run` 자체나 [W&B 앱 UI](../app/intro.md)에서 직접 찾아볼 수 있습니다:

```python
api = wandb.Api()

# run 오브젝트나 W&B 앱에서
# 속성에 직접 접근하기
username = wandb.run.entity
project = wandb.run.project
run_id = wandb.run.id

run = api.run(f"{username}/{project}/{run_id}")
run.config["bar"] = 32
run.update()
```

## `absl.FLAGS`

[`absl` 플래그](https://abseil.io/docs/python/guides/flags)도 패스할 수 있습니다.

```python
flags.DEFINE_string("model", None, "실행할 모델")  # 이름, 기본값, 도움말

wandb.config.update(flags.FLAGS)  # absl 플래그를 config에 추가
```

## 파일 기반 구성
`config-defaults.yaml`이라는 파일을 만들면 키-값 페어가 자동으로 `wandb.config`에 전달됩니다.

다음 코드조각은 `config-defaults.yaml` YAML 파일의 예시를 보여줍니다:

```yaml
# config-defaults.yaml
# 샘플 기본 구성 파일
epochs:
  desc: 트레이닝할 에포크 수
  value: 100
batch_size:
  desc: 각 미니 배치의 크기
  value: 32
```
`config-defaults.yaml`에 자동으로 전달된 값을 덮어쓸 수 있습니다. 이를 위해 `wandb.init`의 `config` 인수에 값을 패스하세요.

커맨드라인 인수 `--configs`를 사용하여 다른 구성 파일을 로드할 수도 있습니다.

### 파일 기반 구성의 예제 유스 케이스
Run에 대한 일부 메타데이터가 있는 YAML 파일이 있고, 파이썬 스크립트에 하이퍼파라미터의 사전이 있다고 가정해 보겠습니다. 중첩된 `config` 객체에 둘 다 저장할 수 있습니다:

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

`wandb.config` 오브젝트로 직접 TensorFlow 플래그를 전달할 수 있습니다.

```python
wandb.init()
wandb.config.epochs = 4

flags = tf.app.flags
flags.DEFINE_string("data_dir", "/tmp/data")
flags.DEFINE_integer("batch_size", 128, "배치 크기.")
wandb.config.update(flags.FLAGS)  # tensorflow 플래그를 config로 추가
```
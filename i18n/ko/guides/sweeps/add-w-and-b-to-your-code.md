---
description: Add W&B to your Python code script or Jupyter Notebook.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# W&B를 코드에 추가하기

<head>
  <title>W&B를 파이썬 코드에 추가하기</title>
</head>

W&B Python SDK를 스크립트나 Jupyter Notebook에 추가하는 방법은 여러 가지가 있습니다. 아래에는 W&B Python SDK를 자신의 코드에 통합하는 "최선의 방법" 예제를 설명합니다.

### 원본 트레이닝 스크립트

Jupyter Notebook 셀이나 Python 스크립트에 다음과 같은 코드가 있다고 가정해 봅시다. 우리는 일반적인 트레이닝 루프를 모방하는 `main`이라는 함수를 정의합니다. 각 에포크마다, 트레이닝 및 검증 데이터 세트에 대한 정확도와 손실이 계산됩니다. 값은 이 예제의 목적을 위해 무작위로 생성됩니다.

하이퍼파라미터 값이 저장되는 `config`라는 사전을 정의했습니다(15번 줄). 셀의 마지막에는 모의 트레이닝 코드를 실행하기 위해 `main` 함수를 호출합니다.

```python showLineNumbers
# train.py
import random
import numpy as np


def train_one_epoch(epoch, lr, bs):
    acc = 0.25 + ((epoch / 30) + (random.random() / 10))
    loss = 0.2 + (1 - ((epoch - 1) / 10 + random.random() / 5))
    return acc, loss


def evaluate_one_epoch(epoch):
    acc = 0.1 + ((epoch / 20) + (random.random() / 10))
    loss = 0.25 + (1 - ((epoch - 1) / 10 + random.random() / 6))
    return acc, loss


config = {"lr": 0.0001, "bs": 16, "epochs": 5}


def main():
    # `wandb.config`에서 값을 정의하는 것에 유의하세요
    # 대신 하드코딩된 값을 정의하지 않습니다
    lr = config["lr"]
    bs = config["bs"]
    epochs = config["epochs"]

    for epoch in np.arange(1, epochs):
        train_acc, train_loss = train_one_epoch(epoch, lr, bs)
        val_acc, val_loss = evaluate_one_epoch(epoch)

        print("epoch: ", epoch)
        print("training accuracy:", train_acc, "training loss:", train_loss)
        print("validation accuracy:", val_acc, "training loss:", val_loss)


# main 함수를 호출합니다.
main()
```

### W&B Python SDK가 포함된 트레이닝 스크립트

다음 코드 예제는 W&B Python SDK를 코드에 추가하는 방법을 보여줍니다. CLI에서 W&B 스윕 작업을 시작하려면 CLI 탭을 탐색합니다. Jupyter notebook이나 Python 스크립트 내에서 W&B 스윕 작업을 시작하려면 Python SDK 탭을 탐색합니다.

<Tabs
  defaultValue="script"
  values={[
    {label: 'Python 스크립트 또는 Jupyter Notebook', value: 'script'},
    {label: 'CLI', value: 'cli'},
  ]}>
  <TabItem value="script">
  W&B 스윕을 생성하기 위해, 코드 예제에 다음을 추가했습니다:

1. 1번 줄: Weights & Biases Python SDK를 가져옵니다.
2. 6번 줄: 스윕 구성을 정의하는 키-값 쌍이 포함된 사전 오브젝트를 생성합니다. 다음 예제에서는 각 스윕에서 배치 크기(`batch_size`), 에포크(`epochs`), 학습률(`lr`) 하이퍼파라미터가 변화됩니다. 스윕 구성을 생성하는 방법에 대한 자세한 정보는 [스윕 구성 정의](./define-sweep-configuration.md)를 참조하세요.
3. 19번 줄: 스윕 구성 사전을 [`wandb.sweep`](../../ref/python/sweep)에 전달합니다. 이는 스윕을 초기화합니다. 이는 스윕 ID(`sweep_id`)를 반환합니다. 스윕을 초기화하는 방법에 대한 자세한 정보는 [스윕 초기화](./initialize-sweeps.md)를 참조하세요.
4. 33번 줄: [`wandb.init()`](../../ref/python/init.md) API를 사용하여 데이터를 동기화하고 로그하는 백그라운드 프로세스를 생성합니다([W&B Run](../../ref/python/run.md) 참조).
5. 37-39번 줄: (선택사항) 하드코딩된 값을 정의하는 대신 `wandb.config`에서 값을 정의합니다.
6. 45번 줄: [`wandb.log`](../../ref/python/log.md)로 최적화하려는 메트릭을 로그합니다. 설정에서 정의한 메트릭을 반드시 로그해야 합니다. 이 예제에서의 구성 사전(`sweep_configuration`) 내에서 `val_acc` 값을 최대화하도록 스윕을 정의했습니다).
7. 54번 줄: [`wandb.agent`](../../ref/python/agent.md) API 호출로 스윕을 시작합니다. 스윕 ID(19번 줄), 스윕이 실행할 함수의 이름(`function=main`), 시도할 최대 실행 횟수를 네 번(`count=4`)으로 설정하여 제공합니다. W&B 스윕을 시작하는 방법에 대한 자세한 정보는 [스윕 에이전트 시작](./start-sweep-agents.md)을 참조하세요.


```python showLineNumbers
import wandb
import numpy as np
import random

# 스윕 구성 정의
sweep_configuration = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "val_acc"},
    "parameters": {
        "batch_size": {"values": [16, 32, 64]},
        "epochs": {"values": [5, 10, 15]},
        "lr": {"max": 0.1, "min": 0.0001},
    },
}

# 구성을 전달하여 스윕 초기화.
# (선택사항) 프로젝트의 이름을 제공합니다.
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")


# 하이퍼파라미터 값을 `wandb.config`에서 가져와
# 모델을 트레이닝하고 메트릭을 반환하는 트레이닝 함수 정의
def train_one_epoch(epoch, lr, bs):
    acc = 0.25 + ((epoch / 30) + (random.random() / 10))
    loss = 0.2 + (1 - ((epoch - 1) / 10 + random.random() / 5))
    return acc, loss


def evaluate_one_epoch(epoch):
    acc = 0.1 + ((epoch / 20) + (random.random() / 10))
    loss = 0.25 + (1 - ((epoch - 1) / 10 + random.random() / 6))
    return acc, loss


def main():
    run = wandb.init()

    # 하드코딩된 값을 정의하는 대신 `wandb.config`에서 값을 정의하는 것에 유의하세요
    lr = wandb.config.lr
    bs = wandb.config.batch_size
    epochs = wandb.config.epochs

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


# 스윕 작업을 시작합니다.
wandb.agent(sweep_id, function=main, count=4)
```
  </TabItem>
  <TabItem value="cli">

  W&B 스윕을 생성하기 위해, 먼저 YAML 구성 파일을 생성합니다. 구성 파일에는 스윕이 탐색할 하이퍼파라미터가 포함됩니다. 다음 예제에서는 각 스윕에서 배치 크기(`batch_size`), 에포크(`epochs`), 학습률(`lr`) 하이퍼파라미터가 변화됩니다.
  
```yaml
# config.yaml
program: train.py
method: random
name: sweep
metric:
  goal: maximize
  name: val_acc
parameters:
  batch_size: 
    values: [16,32,64]
  lr:
    min: 0.0001
    max: 0.1
  epochs:
    values: [5, 10, 15]
```

W&B 스윕 구성을 생성하는 방법에 대한 자세한 정보는 [스윕 구성 정의](./define-sweep-configuration.md)를 참조하세요.

YAML 파일에서 `program` 키에 대해 파이썬 스크립트 이름을 반드시 제공해야 합니다.

다음으로, 코드 예제에 다음을 추가합니다:

1. 1-2번 줄: Weights & Biases Python SDK(`wandb`)와 PyYAML(`yaml`)을 가져옵니다. PyYAML은 YAML 구성 파일을 읽는 데 사용됩니다.
2. 18번 줄: 구성 파일을 읽어옵니다.
3. 21번 줄: [`wandb.init()`](../../ref/python/init.md) API를 사용하여 데이터를 동기화하고 로그하는 백그라운드 프로세스를 생성합니다. config 오브젝트를 config 파라미터로 전달합니다.
4. 25 - 27번 줄: 하드코딩된 값을 사용하는 대신 `wandb.config`에서 하이퍼파라미터 값을 정의합니다.
5. 33-39번 줄: 최적화하려는 메트릭을 [`wandb.log`](../../ref/python/log.md)로 로그합니다. 설정에서 정의한 메트릭을 반드시 로그해야 합니다. 이 예제에서 구성 사전(`sweep_configuration`) 내에서 `val_acc` 값을 최대화하도록 스윕을 정의했습니다.


```python showLineNumbers
import wandb
import yaml
import random
import numpy as np


def train_one_epoch(epoch, lr, bs):
    acc = 0.25 + ((epoch / 30) + (random.random() / 10))
    loss = 0.2 + (1 - ((epoch - 1) / 10 + random.random() / 5))
    return acc, loss


def evaluate_one_epoch(epoch):
    acc = 0.1 + ((epoch / 20) + (random.random() / 10))
    loss = 0.25 + (1 - ((epoch - 1) / 10 + random.random() / 6))
    return acc, loss


def main():
    # 기본 하이퍼파라미터 설정
    with open("./config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    run = wandb.init(config=config)

    # 하드코딩된 값을 정의하는 대신 `wandb.config`에서 값을 정의하는 것에 유의하세요
    lr = wandb.config.lr
    bs = wandb.config.batch_size
    epochs = wandb.config.epochs

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


# main 함수를 호출합니다.
main()
```


CLI로 이동합니다. CLI 내에서, 스윕 에이전트가 시도해야 할 최대 실행 횟수를 설정합니다. 이 단계는 선택사항입니다. 다음 예제에서는 최대 숫자를 다섯으로 설정했습니다.

```bash
NUM=5
```

다음으로, [`wandb sweep`](../../ref/cli/wandb-sweep.md) 명령어로 스윕을 초기화합니다. YAML 파일의 이름을 제공합니다. 선택적으로 프로젝트 플래그(`--project`)에 대한 프로젝트 이름을 제공합니다:

```bash
wandb sweep --project sweep-demo-cli config.yaml
```

이는 스윕 ID를 반환합니다. 스윕을 초기화하는 방법에 대한 자세한 정보는 [스윕 초기화](./initialize-sweeps.md)를 참조하세요.

스윕 ID를 복사하고, [`wandb agent`](../../ref/cli/wandb-agent.md) 명령어로 스윕 작업을 시작하는 다음 코드 스니펫에서 `sweepID`를 대체합니다:

```bash
wandb agent --count $NUM your-entity/sweep-demo-cli/sweepID
```

스윕 작업을 시작하는 방법에 대한 자세한 정보는 [스윕 작업 시작](./start-sweep-agents.md)을 참조하세요.
  </TabItem>
</Tabs>

## 메트릭 로깅 시 고려사항 

스윕 구성에서 명시한 메트릭을 W&B에 명시적으로 로그해야 합니다. 스윕의 메트릭을 하위 디렉토리 내에 로그하지 마세요.

예를 들어, 다음 의사코드를 고려해 보세요. 사용자는 검증 손실(`"val_loss": loss`)을 로그하려고 합니다. 먼저 값들을 사전에 전달합니다(16번 줄). 그러나 `wandb.log`에 전달된 사전은 사전 내의 키-값 쌍에 명시적으로 접근하지 않습니다:

```python title="train.py" showLineNumbers
# W&B Python 라이브러리를 가져오고 W&B에 로그인합니다.
import wandb
import random


def train():
    offset = random.random() / 5
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset

    val_metrics = {"val_loss": loss, "val_acc": acc}
    return val_metrics


def main():
    wandb.init(entity="<entity>", project="my-first-sweep")
    val_metrics = train()
    # highlight-next-line
    wandb.log({"val_loss": val_metrics})


sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")

wandb.agent(sweep_id, function=main, count=10)
```

대신, Python 사전 내에서 키-값 쌍에 명시적으로 접근하세요. 예를 들어, 사전을 생성한 후 `wandb.log` 메소드에 사전을 전달할 때 키-값 쌍을 명시적으로 지정합니다:

```python title="train.py" showLineNumbers
# W&B Python 라이브러리를 가져오고 W&B에 로그인합니다.
import wandb
import random


def train():
    offset = random.random() / 5
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset

    val_metrics = {"val_loss": loss, "val_acc": acc}
    return val_metrics


def main():
    wandb.init(entity="<entity>", project="my-first-sweep")
    val_metrics = train()
    # highlight-next-line
    wandb.log({"val_loss", val_metrics["val_loss"]})


sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")

wandb.agent(sweep_id, function=main, count=10)
```
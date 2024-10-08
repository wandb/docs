---
title: Add W&B (wandb) to your code
description: Python 코드 스크립트나 Jupyter 노트북에 W&B를 추가하세요.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

W&B Python SDK를 스크립트나 Jupyter 노트북에 추가하는 방법은 여러 가지가 있습니다. 아래에는 W&B Python SDK를 코드에 통합하는 "최적의 방법" 예시가 설명되어 있습니다.

### 원본 트레이닝 스크립트

다음과 같은 코드가 Jupyter 노트북 셀이나 Python 스크립트에 있다고 가정해 보세요. 우리는 `main`이라는 함수를 정의했고, 이것은 일반적인 트레이닝 루프를 모방합니다. 각 에포크마다, 트레이닝과 검증 데이터 세트에서 정확도와 손실이 계산됩니다. 이 예시에서는 값이 무작위로 생성됩니다.

15번째 줄에, 하이퍼파라미터 값을 저장하는 `config`라는 사전을 정의했습니다. 셀의 마지막 부분에서, 모의 트레이닝 코드를 실행하기 위해 `main` 함수를 호출합니다.

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
    # 여기에, `wandb.config`에서 값을 정의합니다.
    # 하드코딩된 값을 정의하는 대신에
    lr = config["lr"]
    bs = config["bs"]
    epochs = config["epochs"]

    for epoch in np.arange(1, epochs):
        train_acc, train_loss = train_one_epoch(epoch, lr, bs)
        val_acc, val_loss = evaluate_one_epoch(epoch)

        print("epoch: ", epoch)
        print("training accuracy:", train_acc, "training loss:", train_loss)
        print("validation accuracy:", val_acc, "training loss:", val_loss)


# 메인 함수를 호출합니다.
main()
```

### W&B Python SDK와 함께하는 트레이닝 스크립트

다음 코드 예제는 W&B Python SDK를 코드에 추가하는 방법을 보여줍니다. CLI에서 W&B Sweep 작업을 시작하는 경우, CLI 탭을 탐색합니다. Jupyter 노트북이나 Python 스크립트 내에서 W&B Sweep 작업을 시작하는 경우, Python SDK 탭을 탐색합니다.

<Tabs
  defaultValue="script"
  values={[
    {label: 'Python script or Jupyter Notebook', value: 'script'},
    {label: 'CLI', value: 'cli'},
  ]}>
  <TabItem value="script">
  
W&B Sweep을 생성하기 위해, 코드 예제에 다음을 추가했습니다:

1. 1번째 줄에서: Weights & Biases Python SDK를 가져옵니다.
2. 6번째 줄에서: 키-값 쌍이 스윕 구성을 정의하는 사전 객체를 만듭니다. 다음 예제에서, 배치 크기 (`batch_size`), 에포크 (`epochs`), 그리고 학습률 (`lr`) 하이퍼파라미터가 각 스윕 동안 변경됩니다. 스윕 구성을 생성하는 방법에 대한 자세한 내용은 [스윕 구성 정의하기](./define-sweep-configuration.md)를 참조하십시오.
3. 19번째 줄에서: 스윕 구성 사전을 [`wandb.sweep`](../../ref/python/sweep)에 전달합니다. 이것은 스윕을 초기화합니다. 스윕을 초기화하는 방법에 대한 자세한 내용은 [스윕 초기화하기](./initialize-sweeps.md)를 참조하십시오.
4. 33번째 줄에서: [`wandb.init()`](../../ref/python/init.md) API를 사용하여 백그라운드 프로세스를 생성하고 데이터를 [W&B Run](../../ref/python/run.md)으로 동기화하고 기록합니다.
5. 37-39번째 줄에서: (선택사항) `wandb.config`에서 값을 정의하여 하드 코딩된 값을 정의하는 대신에 사용합니다.
6. 45번째 줄에서: [`wandb.log`](../../ref/python/log.md)를 사용하여 최적화하고자 하는 메트릭을 로그합니다. 당신의 설정에 정의한 메트릭을 로그해야 합니다. 설정 사전 (`sweep_configuration` 예제에서) 내에서 우리는 `val_acc` 값을 최대화하도록 스윕을 정의했습니다.
7. 54번째 줄에서: [`wandb.agent`](../../ref/python/agent.md) API 호출을 사용하여 스윕을 시작합니다. 스윕 ID (19번째 줄), 스윕이 실행할 함수의 이름 (`function=main`), 그리고 시도할 최대 실행 횟수를 네 번으로 설정합니다 (`count=4`). W&B Sweep을 시작하는 방법에 대한 자세한 내용은 [스윕 에이전트 시작하기](./start-sweep-agents.md)를 참조하십시오.


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

# 설정을 전달하여 스윕 초기화
# (선택 사항) 프로젝트의 이름을 제공합니다.
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")


# 하이퍼파라미터를 가져와서 사용하는
# 트레이닝 함수를 정의합니다.
# 모델을 트레이닝하고 메트릭을 반환하도록 합니다.
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

    # `wandb.config`에서 값을 정의합니다.
    # 하드 코딩된 값을 정의하는 대신에
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


# 스윕 작업 시작.
wandb.agent(sweep_id, function=main, count=4)
```
  </TabItem>
  <TabItem value="cli">

W&B Sweep을 생성하기 위해, 먼저 YAML 설정 파일을 만듭니다. 설정 파일에는 스윕에서 탐색하고자 하는 하이퍼파라미터가 포함되어 있습니다. 다음 예제에서, 배치 크기 (`batch_size`), 에포크 (`epochs`), 그리고 학습률 (`lr`) 하이퍼파라미터가 각 스윕 동안 변경됩니다.  
  
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

W&B Sweep 구성을 생성하는 방법에 대한 자세한 내용은 [스윕 구성 정의하기](./define-sweep-configuration.md)를 참조하십시오.

당신의 YAML 파일의 `program` 키에는 Python 스크립트의 이름을 제공해야 합니다.

다음으로, 코드 예제에 다음을 추가합니다:

1. 1-2번째 줄에서: Wieghts & Biases Python SDK (`wandb`)와 PyYAML (`yaml`)을 가져옵니다. PyYAML은 우리의 YAML 설정 파일을 읽는 데 사용됩니다.
2. 18번째 줄에서: 설정 파일을 읽습니다.
3. 21번째 줄에서: [`wandb.init()`](../../ref/python/init.md) API를 사용하여 백그라운드 프로세스를 생성하고 데이터를 [W&B Run](../../ref/python/run.md)으로 동기화하고 기록합니다. 구성 오브젝트를 config 파라미터에 전달합니다.
4. 25-27번째 줄에서: 하드 코딩된 값을 사용하는 대신 `wandb.config`에서 하이퍼파라미터 값을 정의합니다.
5. 33-39번째 줄에서: [`wandb.log`](../../ref/python/log.md)를 사용하여 최적화하고자 하는 메트릭을 로그합니다. 설정에 정의된 메트릭을 로그해야 합니다. 설정 사전 (`sweep_configuration` 예제에서) 내에서 우리는 `val_acc` 값을 최대화하도록 스윕을 정의했습니다.


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

    # `wandb.config`에서 값을 정의합니다.
    # 하드 코딩된 값을 정의하는 대신에
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


# 메인 함수를 호출합니다.
main()
```

CLI로 이동하십시오. CLI 내에서 스윕 에이전트가 시도할 최대 실행 횟수를 설정합니다. 이 단계는 선택 사항입니다. 다음 예제에서는 최대 실행 횟수를 다섯으로 설정합니다.

```bash
NUM=5
```

다음으로, [`wandb sweep`](../../ref/cli/wandb-sweep.md) 코맨드를 사용하여 스윕을 초기화합니다. YAML 파일의 이름을 제공하십시오. 선택적으로 프로젝트 플래그 (`--project`)에 프로젝트 이름을 제공할 수 있습니다:

```bash
wandb sweep --project sweep-demo-cli config.yaml
```

이것은 스윕 ID를 반환합니다. 스윕을 초기화하는 방법에 대한 자세한 내용은 [스윕 초기화하기](./initialize-sweeps.md)를 참조하십시오.

스윕 ID를 복사하여 다음 코드조각에서 `sweepID`를 대체하여 [`wandb agent`](../../ref/cli/wandb-agent.md) 코맨드를 사용하여 스윕 작업을 시작하십시오:

```bash
wandb agent --count $NUM your-entity/sweep-demo-cli/sweepID
```

스윕 작업을 시작하는 방법에 대한 자세한 내용은 [스윕 작업 시작하기](./start-sweep-agents.md)를 참조하십시오.
  </TabItem>
</Tabs>

## 메트릭을 로그할 때 고려사항

스윕 구성에서 명시적으로 정의한 메트릭을 W&B에 기록하십시오. 하위 디렉토리 내에 스윕의 메트릭을 기록하지 마십시오.

예를 들어, 다음의 의사코드를 고려하십시오. 사용자가 검증 손실 (`"val_loss": loss`)을 로그하고 싶어 합니다. 먼저 값을 사전으로 전달합니다 (16번째 줄). 그러나 `wandb.log`에 전달된 사전은 사전에서 키-값 쌍에 명시적으로 엑세스하지 않습니다:

```python title="train.py" showLineNumbers
# Import the W&B Python Library and log into W&B
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

대신, Python 사전 내에서 키-값 쌍에 명시적으로 엑세스하십시오. 예를 들어, 다음 코드는 사전을 생성한 후 한 줄에서, 사전을 `wandb.log` 메소드에 전달할 때 키-값 쌍을 명확히 지정합니다:

```python title="train.py" showLineNumbers
# Import the W&B Python Library and log into W&B
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
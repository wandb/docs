---
title: Add W&B (wandb) to your code
description: Python 코드 스크립트 또는 Jupyter 노트북에 W&B를 추가하세요.
menu:
  default:
    identifier: ko-guides-models-sweeps-add-w-and-b-to-your-code
    parent: sweeps
weight: 2
---

스크립트 또는 Jupyter Notebook에 W&B Python SDK를 추가하는 방법은 다양합니다. 아래에는 W&B Python SDK를 자신의 코드에 통합하는 "모범 사례" 예제가 나와 있습니다.

### 원본 트레이닝 스크립트

Jupyter Notebook 셀 또는 Python 스크립트에 다음과 같은 코드가 있다고 가정합니다. 일반적인 트레이닝 루프를 모방하는 `main`이라는 함수를 정의합니다. 각 에포크마다 트레이닝 및 검증 데이터 세트에서 정확도와 손실이 계산됩니다. 값은 이 예제의 목적을 위해 무작위로 생성됩니다.

`config`라는 사전을 정의하여 하이퍼파라미터 값을 저장합니다 (15행). 셀이 끝나면 `main` 함수를 호출하여 모의 트레이닝 코드를 실행합니다.

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
    # `wandb.config`에서 값을 정의합니다.
    # 하드 코딩된 값을 정의하는 대신
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

### W&B Python SDK를 사용한 트레이닝 스크립트

다음 코드 예제에서는 W&B Python SDK를 코드에 추가하는 방법을 보여줍니다. CLI에서 W&B Sweep 작업을 시작하는 경우 CLI 탭을 살펴보는 것이 좋습니다. Jupyter Notebook 또는 Python 스크립트 내에서 W&B Sweep 작업을 시작하는 경우 Python SDK 탭을 살펴보십시오.

{{< tabpane text=true >}}
    {{% tab header="Python 스크립트 또는 노트북" %}}
  W&B Sweep을 생성하기 위해 코드 예제에 다음을 추가했습니다.

1. 1행: Weights & Biases Python SDK를 가져옵니다.
2. 6행: 키-값 쌍이 스윕 구성을 정의하는 사전 오브젝트를 만듭니다. 진행되는 예제에서 배치 크기 (`batch_size`), 에포크 (`epochs`) 및 학습률 (`lr`) 하이퍼파라미터는 각 스윕 중에 다양합니다. 스윕 구성을 만드는 방법에 대한 자세한 내용은 [스윕 구성 정의]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ko" >}})을 참조하십시오.
3. 19행: 스윕 구성 사전을 [`wandb.sweep`]({{< relref path="/ref/python/sweep.md" lang="ko" >}})에 전달합니다. 이렇게 하면 스윕이 초기화됩니다. 그러면 스윕 ID (`sweep_id`)가 반환됩니다. 스윕을 초기화하는 방법에 대한 자세한 내용은 [스윕 초기화]({{< relref path="./initialize-sweeps.md" lang="ko" >}})을 참조하십시오.
4. 33행: [`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ko" >}}) API를 사용하여 [W&B Run]({{< relref path="/ref/python/run.md" lang="ko" >}})으로 데이터를 동기화하고 기록하는 백그라운드 프로세스를 생성합니다.
5. 37-39행: (선택 사항) 하드 코딩된 값을 정의하는 대신 `wandb.config`에서 값을 정의합니다.
6. 45행: [`wandb.log`]({{< relref path="/ref/python/log.md" lang="ko" >}})로 최적화하려는 메트릭을 기록합니다. 구성에 정의된 메트릭을 기록해야 합니다. 구성 사전 (이 예제에서는 `sweep_configuration`) 내에서 `val_acc` 값을 최대화하기 위해 스윕을 정의했습니다.
7. 54행: [`wandb.agent`]({{< relref path="/ref/python/agent.md" lang="ko" >}}) API 호출로 스윕을 시작합니다. 스윕 ID (19행), 스윕이 실행할 함수의 이름 (`function=main`)을 제공하고 시도할 최대 실행 횟수를 4개 (`count=4`)로 설정합니다. W&B Sweep을 시작하는 방법에 대한 자세한 내용은 [스윕 에이전트 시작]({{< relref path="./start-sweep-agents.md" lang="ko" >}})을 참조하십시오.


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

# 구성을 전달하여 스윕을 초기화합니다.
# (선택 사항) 프로젝트 이름을 제공합니다.
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")


# 하이퍼파라미터를 가져오는 트레이닝 함수를 정의합니다.
# `wandb.config`의 값을 가져와서 사용하여
# 모델을 트레이닝하고 메트릭을 반환합니다.
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
    # 하드 코딩된 값을 정의하는 대신
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

{{% /tab %}}
{{% tab header="CLI" %}}

W&B Sweep을 생성하려면 먼저 YAML 구성 파일을 만듭니다. 구성 파일에는 스윕이 탐색할 하이퍼파라미터가 포함되어 있습니다. 진행되는 예제에서 배치 크기 (`batch_size`), 에포크 (`epochs`) 및 학습률 (`lr`) 하이퍼파라미터는 각 스윕 중에 다양합니다.

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

W&B Sweep 구성을 만드는 방법에 대한 자세한 내용은 [스윕 구성 정의]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ko" >}})을 참조하십시오.

YAML 파일의 `program` 키에 Python 스크립트 이름을 제공해야 합니다.

다음으로 코드 예제에 다음을 추가합니다.

1. 1-2행: Wieghts & Biases Python SDK (`wandb`) 및 PyYAML (`yaml`)을 가져옵니다. PyYAML은 YAML 구성 파일을 읽어오는 데 사용됩니다.
2. 18행: 구성 파일을 읽어옵니다.
3. 21행: [`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ko" >}}) API를 사용하여 [W&B Run]({{< relref path="/ref/python/run.md" lang="ko" >}})으로 데이터를 동기화하고 기록하는 백그라운드 프로세스를 생성합니다. config 오브젝트를 config 파라미터에 전달합니다.
4. 25 - 27행: 하드 코딩된 값을 사용하는 대신 `wandb.config`에서 하이퍼파라미터 값을 정의합니다.
5. 33-39행: [`wandb.log`]({{< relref path="/ref/python/log.md" lang="ko" >}})로 최적화하려는 메트릭을 기록합니다. 구성에 정의된 메트릭을 기록해야 합니다. 구성 사전 (이 예제에서는 `sweep_configuration`) 내에서 `val_acc` 값을 최대화하기 위해 스윕을 정의했습니다.


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
    # 기본 하이퍼파라미터를 설정합니다.
    with open("./config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    run = wandb.init(config=config)

    # `wandb.config`에서 값을 정의합니다.
    # 하드 코딩된 값을 정의하는 대신
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

CLI로 이동합니다. CLI 내에서 스윕 에이전트가 시도해야 하는 최대 실행 횟수를 설정합니다. 이 단계는 선택 사항입니다. 다음 예제에서는 최대 횟수를 5로 설정합니다.

```bash
NUM=5
```

다음으로 [`wandb sweep`]({{< relref path="/ref/cli/wandb-sweep.md" lang="ko" >}}) 코맨드로 스윕을 초기화합니다. YAML 파일 이름을 제공합니다. 선택적으로 프로젝트 플래그 (`--project`)에 대한 프로젝트 이름을 제공합니다.

```bash
wandb sweep --project sweep-demo-cli config.yaml
```

그러면 스윕 ID가 반환됩니다. 스윕을 초기화하는 방법에 대한 자세한 내용은 [스윕 초기화]({{< relref path="./initialize-sweeps.md" lang="ko" >}})을 참조하십시오.

스윕 ID를 복사하고 진행 중인 코드 조각에서 `sweepID`를 바꾸어 [`wandb agent`]({{< relref path="/ref/cli/wandb-agent.md" lang="ko" >}}) 코맨드로 스윕 작업을 시작합니다.

```bash
wandb agent --count $NUM your-entity/sweep-demo-cli/sweepID
```

스윕 작업을 시작하는 방법에 대한 자세한 내용은 [스윕 작업 시작]({{< relref path="./start-sweep-agents.md" lang="ko" >}})을 참조하십시오.

{{% /tab %}}
{{< /tabpane >}}



## 메트릭 기록 시 고려 사항

스윕 구성에서 지정한 메트릭을 W&B에 명시적으로 기록해야 합니다. 하위 디렉토리 내에서 스윕에 대한 메트릭을 기록하지 마십시오.

예를 들어 진행 중인 의사 코드를 고려하십시오. 사용자는 검증 손실 (`"val_loss": loss`)을 기록하려고 합니다. 먼저 값을 사전에 전달합니다 (16행). 그러나 `wandb.log`에 전달된 사전은 사전의 키-값 쌍에 명시적으로 엑세스하지 않습니다.

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

대신 Python 사전 내에서 키-값 쌍에 명시적으로 엑세스합니다. 예를 들어 진행 중인 코드 (사전을 만든 후 `wandb.log` 메소드에 사전을 전달할 때 키-값 쌍을 지정)

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
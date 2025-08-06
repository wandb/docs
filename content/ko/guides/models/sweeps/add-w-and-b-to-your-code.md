---
title: 코드에 W&B (wandb)를 추가하세요
description: Python 코드 스크립트나 Jupyter Notebook에 W&B를 추가하세요.
menu:
  default:
    identifier: ko-guides-models-sweeps-add-w-and-b-to-your-code
    parent: sweeps
weight: 2
---

W&B Python SDK를 스크립트나 노트북에 추가하는 방법에는 여러 가지가 있습니다. 이 섹션에서는 W&B Python SDK를 자신의 코드에 자연스럽게 통합하는 "베스트 프랙티스" 예시를 제공합니다.

### 기존 트레이닝 스크립트

다음과 같은 Python 스크립트가 있다고 가정해봅시다. 여기서는 일반적인 트레이닝 루프를 모방한 `main` 함수를 정의합니다. 각 에포크마다 트레이닝 및 검증 데이터셋에서 정확도와 손실값을 계산합니다. 이 예제에서는 값을 임의로 생성합니다.

하이퍼파라미터 값을 저장하는 `config`라는 사전을 정의했습니다. 마지막에 `main` 함수를 호출하여 모의 트레이닝 코드를 실행합니다.

```python
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

# 하이퍼파라미터 값을 담은 config 변수
config = {"lr": 0.0001, "bs": 16, "epochs": 5}

def main():
    # 하드코딩 값 대신 `wandb.Run.config` 값을 정의하는 방식과 비교
    lr = config["lr"]
    bs = config["bs"]
    epochs = config["epochs"]

    for epoch in np.arange(1, epochs):
        train_acc, train_loss = train_one_epoch(epoch, lr, bs)
        val_acc, val_loss = evaluate_one_epoch(epoch)

        print("epoch: ", epoch)
        print("training accuracy:", train_acc, "training loss:", train_loss)
        print("validation accuracy:", val_acc, "training loss:", val_loss)        
```

### W&B Python SDK가 적용된 트레이닝 스크립트

다음 코드는 기존 코드에 W&B Python SDK를 어떻게 추가하는지 보여줍니다. W&B Sweep job을 CLI에서 시작할 경우, CLI 탭을 참고하세요. Jupyter 노트북이나 Python 스크립트 내에서 시작할 경우 Python SDK 탭을 참고하세요.

{{< tabpane text=true >}} {{% tab header="Python script or notebook" %}}
W&B Sweep 생성을 위해 아래 주요 변경이 적용되었습니다.

1. W&B Python SDK를 임포트합니다.
2. 스윕 구성을 정의하는 키-값 쌍으로 구성된 사전 오브젝트를 만듭니다. 아래 예시에서는 배치 크기(`batch_size`), 에포크(`epochs`), 러닝레이트(`lr`) 하이퍼파라미터가 스윕에서 다양하게 변화합니다. 자세한 내용은 [스윕 구성 정의]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ko" >}})를 참고하세요.
3. 스윕 구성 사전을 [`wandb.sweep()`]({{< relref path="/ref/python/sdk/functions/sweep.md" lang="ko" >}})에 전달하여 스윕을 초기화합니다. 이때 반환되는 sweep ID(`sweep_id`)가 생성됩니다. 자세한 내용은 [스윕 초기화]({{< relref path="./initialize-sweeps.md" lang="ko" >}})를 참고하세요.
4. [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ko" >}}) API를 사용하여 백그라운드 프로세스를 생성, [W&B Run]({{< relref path="/ref/python/sdk/classes/run.md" lang="ko" >}})으로 데이터 동기화 및 로깅을 진행합니다.
5. (선택) 하드코딩 값 대신 `wandb.config`에서 값을 받아오도록 구성합니다.
6. [`wandb.Run.log()`]({{< relref path="/ref/python/sdk/classes/run.md/#method-runlog" lang="ko" >}})를 사용해 최적화하려는 메트릭을 기록합니다. 반드시 스윕 구성에서 정의한 메트릭을 로깅해야 합니다. 아래 예시에서는 `val_acc` 값이 스윕의 최적화 대상입니다.
7. [`wandb.agent`]({{< relref path="/ref/python/sdk/functions/agent.md" lang="ko" >}}) API를 사용하여 스윕을 시작합니다. sweep ID, 실행할 함수명(`function=main`), 시도할 run 개수(`count=4`)를 지정합니다. 자세한 내용은 [스윕 에이전트 시작]({{< relref path="./start-sweep-agents.md" lang="ko" >}})를 참고하세요.

```python
import wandb
import numpy as np
import random

# 하이퍼파라미터 값을 받아 모델을 트레이닝하고,
# 메트릭을 반환하는 트레이닝 함수
def train_one_epoch(epoch, lr, bs):
    acc = 0.25 + ((epoch / 30) + (random.random() / 10))
    loss = 0.2 + (1 - ((epoch - 1) / 10 + random.random() / 5))
    return acc, loss

def evaluate_one_epoch(epoch):
    acc = 0.1 + ((epoch / 20) + (random.random() / 10))
    loss = 0.25 + (1 - ((epoch - 1) / 10 + random.random() / 6))
    return acc, loss

# 스윕 구성용 사전 정의
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

# (선택) 프로젝트 이름 지정
project = "my-first-sweep"

def main():
    # with 문을 사용해 run을 자동 종료합니다. 
    # 이는 각 run 마지막에 run.finish()를 호출하는 것과 동일합니다.
    with wandb.init(project=project) as run:

        # 하이퍼파라미터 값을 명시적으로 지정하는 대신
        # wandb.Run.config에서 값을 가져옵니다
        lr = run.config["lr"]
        bs = run.config["batch_size"]
        epochs = run.config["epochs"]

        # 트레이닝 루프를 실행하며 W&B로 성능 값을 로깅합니다
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
    # 스윕을 설정 사전으로 초기화
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project)

    # 스윕 job 시작하기
    wandb.agent(sweep_id, function=main, count=4)

```

{{% alert %}}
위 코드조각은 [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ko" >}}) API를 `with` 컨텍스트 매니저 안에서 호출해 백그라운드로 동기화 및 로깅 프로세스를 생성하는 예시입니다. 이는 값 업로드 후 run이 올바르게 종료되도록 보장합니다. 대안으로, 트레이닝 스크립트 시작과 끝에 각각 `wandb.init()`과 `wandb.Run.finish()`를 호출하는 방식도 사용할 수 있습니다.
{{% /alert %}}

{{% /tab %}} {{% tab header="CLI" %}}

W&B Sweep을 만들기 위해 먼저 YAML 구성 파일을 만듭니다. 이 파일에는 스윕이 탐색할 하이퍼파라미터가 포함됩니다. 아래 예시에서는 배치 크기(`batch_size`), 에포크(`epochs`), 러닝레이트(`lr`) 하이퍼파라미터가 각 스윕마다 변하도록 설정되어 있습니다.

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
    values: [16, 32, 64]
  lr:
    min: 0.0001
    max: 0.1
  epochs:
    values: [5, 10, 15]
```

W&B Sweep 구성을 만드는 자세한 방법은 [스윕 구성 정의]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ko" >}})를 참고하세요.

YAML 파일의 `program` 키에는 Python 스크립트명을 입력해야 합니다.

다음으로, 코드 예시에 아래 내용을 추가합니다.

1. W&B Python SDK(`wandb`)와 PyYAML(`yaml`)을 import합니다. PyYAML은 YAML 구성 파일을 읽는 데 사용됩니다.
2. 구성 파일을 읽어옵니다.
3. [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ko" >}}) API를 사용해 [W&B Run]({{< relref path="/ref/python/sdk/classes/run.md" lang="ko" >}}) 프로세스를 생성, config 객체를 파라미터로 전달합니다.
4. 하드코딩된 값 대신 `wandb.Run.config`에서 하이퍼파라미터 값을 정의합니다.
5. 스윕 구성에서 정의한 메트릭(`val_acc`)을 [`wandb.Run.log()`]({{< relref path="/ref/python/sdk/classes/run.md/#method-runlog" lang="ko" >}})로 기록합니다.

```python
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
    # 기본 하이퍼파라미터를 설정합니다
    with open("./config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    with wandb.init(config=config) as run:
        for epoch in np.arange(1, run.config['epochs']):
            train_acc, train_loss = train_one_epoch(epoch, run.config['lr'], run.config['batch_size'])
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

# main 함수 호출
main()
```

CLI에서는 스윕 에이전트가 시도할 최대 run 수를 지정할 수 있습니다(선택 사항). 아래 예시에서는 최대 5개로 설정합니다.

```bash
NUM=5
```

다음으로, [`wandb sweep`]({{< relref path="/ref/cli/wandb-sweep.md" lang="ko" >}}) 커맨드로 스윕을 초기화합니다. YAML 파일 이름을 지정하고, 프로젝트 이름을 `--project` 플래그로 추가할 수 있습니다(선택).

```bash
wandb sweep --project sweep-demo-cli config.yaml
```

이 명령은 sweep ID를 반환합니다. 스윕 초기화 방법은 [스윕 초기화]({{< relref path="./initialize-sweeps.md" lang="ko" >}})를 참고하세요.

sweep ID를 복사하여 아래 코드조각 중 `sweepID` 부분에 입력하고, [`wandb agent`]({{< relref path="/ref/cli/wandb-agent.md" lang="ko" >}}) 커맨드로 스윕 작업을 시작하세요.

```bash
wandb agent --count $NUM your-entity/sweep-demo-cli/sweepID
```

자세한 내용은 [스윕 작업 시작하기]({{< relref path="./start-sweep-agents.md" lang="ko" >}})를 참고하세요.

{{% /tab %}} {{< /tabpane >}}

## 메트릭 로깅 시 유의사항

스윕의 메트릭을 반드시 W&B에 명시적으로 기록해야 합니다. 스윕의 메트릭을 서브디렉토리 내에서 기록하는 것은 금지됩니다.

예를 들어, 아래와 같은 의사코드를 보세요. 사용자가 검증 손실(`"val_loss": loss`)을 로깅하려고 합니다. 먼저 값을 사전에 저장합니다. 하지만 `wandb.Run.log()`에 전달되는 사전이 명시적으로 해당 키-값 쌍에 접근하지 않습니다:

```python
# W&B Python 라이브러리 임포트 및 로그인
import wandb
import random

def train():
    # 트레이닝 및 검증 메트릭을 시뮬레이션
    offset = random.random() / 5
    epoch = 5  # 에포크 값 시뮬레이션
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset
    return loss, acc

def main():
    with wandb.init(entity="<entity>", project="my-first-sweep") as run:
        val_loss, val_acc = train()
        # 잘못된 예: 사전에서 명확히 키-값 쌍에 접근하지 않음
        # 올바른 로깅 예는 다음 코드 블록 참고
        run.log({"val_loss": val_loss, "val_acc": val_acc})

sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}

# 스윕 구성 사전으로 스윕 초기화
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")

# 스윕 job 시작
wandb.agent(sweep_id, function=main, count=10)
```

올바른 방법은 Python 사전 내부에서 키-값 쌍에 명확하게 접근하도록 하는 것입니다. 아래 예시에서는 `wandb.Run.log()`에 사전을 전달할 때 각 메트릭의 키-값을 직접 지정합니다:

```python title="train.py"
# W&B Python 라이브러리 임포트 및 로그인
import wandb
import random

def train():
    offset = random.random() / 5
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset

    return loss, acc

def main():
    with wandb.init(entity="<entity>", project="my-first-sweep") as run:
        # 올바른 예: 메트릭 로깅 시 사전에서 키-값 쌍 명확히 지정
        val_loss, val_acc = train()
        run.log({"val_loss": val_loss, "val_acc": val_acc})

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
---
description: Resume a paused or exited W&B Run
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 실행 재개

<head>
  <title>W&B 실행 재개</title>
</head>

W&B를 통해 `wandb.init()`에 `resume=True`를 전달함으로써 자동으로 실행을 재개할 수 있습니다. 프로세스가 성공적으로 종료되지 않은 경우, 다음 번에 실행할 때 W&B는 마지막 단계부터 로깅을 시작할 것입니다.

<Tabs
  defaultValue="keras"
  values={[
    {label: 'Keras', value: 'keras'},
    {label: 'PyTorch', value: 'pytorch'},
  ]}>
  <TabItem value="keras">

```python
import keras
import numpy as np
import wandb
from wandb.keras import WandbCallback

wandb.init(project="preemptible", resume=True)

if wandb.run.resumed:
    # 최고의 모델 복원
    model = keras.models.load_model(wandb.restore("model-best.h5").name)
else:
    a = keras.layers.Input(shape=(32,))
    b = keras.layers.Dense(10)(a)
    model = keras.models.Model(input=a, output=b)

model.compile("adam", loss="mse")
model.fit(
    np.random.rand(100, 32),
    np.random.rand(100, 10),
    # 재개된 에포크 설정
    initial_epoch=wandb.run.step,
    epochs=300,
    # 매 에포크마다 개선된 경우 최고의 모델 저장
    callbacks=[WandbCallback(save_model=True, monitor="loss")],
)
```

  </TabItem>
  <TabItem value="pytorch">


```python
import wandb
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_NAME = "pytorch-resume-run"
CHECKPOINT_PATH = "./checkpoint.tar"
N_EPOCHS = 100

# 더미 데이터
X = torch.randn(64, 8, requires_grad=True)
Y = torch.empty(64, 1).random_(2)
model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())

metric = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
epoch = 0
run = wandb.init(project=PROJECT_NAME, resume=True)
if wandb.run.resumed:
    checkpoint = torch.load(wandb.restore(CHECKPOINT_PATH))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

model.train()
while epoch < N_EPOCHS:
    optimizer.zero_grad()
    output = model(X)
    loss = metric(output, Y)
    wandb.log({"loss": loss.item()}, step=epoch)
    loss.backward()
    optimizer.step()

    # 체크포인트 위치 저장
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        CHECKPOINT_PATH,
    )
    wandb.save(CHECKPOINT_PATH)  # 체크포인트를 wandb에 저장
    epoch += 1
```


  </TabItem>
</Tabs>

### 재개 지침

아래에 설명된 대로 W&B를 사용하여 실행을 재개하는 방법은 여러 가지가 있습니다:

1.  [`resume`](./resuming.md)

    W&B와 함께 실행을 재개하는 권장 방법입니다.

    1. 위에서 설명한 것처럼, `wandb.init()`에 `resume=True`를 전달하여 실행을 재개할 수 있습니다. 이는 중단된 실행을 자동으로 이어받는 것으로 생각할 수 있습니다. 프로세스가 성공적으로 종료되지 않은 경우, 다음 번에 실행할 때 W&B는 마지막 단계부터 로깅을 시작할 것입니다.
       * 주의: 이는 실패한 디렉터리와 동일한 디렉터리에서 스크립트를 실행하는 경우에만 작동합니다. 파일은 `wandb/wandb-resume.json`에 저장됩니다.
    2. 다른 형태의 재개는 실제 실행 ID를 제공해야 합니다: `wandb.init(id=run_id)` 그리고 재개할 때(실제로 재개되는지 확인하려면) `wandb.init(id=run_id, resume="must")`를 사용합니다.
       * `run_id`를 관리함으로써 재개를 완전히 제어할 수도 있습니다. 우리는 `run_id`를 생성하는 유틸리티를 제공합니다: `wandb.util.generate_id()`. 각각의 고유 실행에 대해 이러한 고유 ID 중 하나를 설정하는 한, `resume="allow"`라고 할 수 있으며 W&B는 자동으로 해당 ID로 실행을 재개할 것입니다.

    자동 재개와 제어된 재개에 대한 더 많은 맥락은 [이 섹션](resuming.md#resume-runs)에서 찾을 수 있습니다.
2. [`wandb.restore`](../track/save-restore.md#examples-of-wandb.restore)
   * 이를 통해 중단된 지점부터 시작하여 실행에 새로운 역사적 메트릭 값을 로깅할 수 있지만, 코드의 상태를 재설정하는 것은 처리하지 않으므로, 로드할 수 있는 체크포인트를 작성했는지 확인해야 합니다!
   * [`wandb.save`](../track/save-restore.md#examples-of-wandbsave)를 사용하여 체크포인트 파일을 통해 실행의 상태를 기록할 수 있습니다. `wandb.save()`를 통해 체크포인트 파일을 생성한 다음, `wandb.init(resume=<run-id>)`을 통해 사용할 수 있습니다. [이 리포트](https://wandb.ai/lavanyashukla/save\_and\_restore/reports/Saving-and-Restoring-Models-with-W-B--Vmlldzo3MDQ3Mw)는 W&B와 함께 모델을 저장하고 복원하는 방법을 설명합니다.

#### 자동 및 제어된 재개

자동 재개는 실패한 프로세스와 동일한 파일 시스템 위에서 프로세스가 다시 시작될 때만 작동합니다. 파일 시스템을 공유할 수 없는 경우, `WANDB_RUN_ID`를 설정할 수 있습니다: 단일 스크립트 실행에 해당하는 프로젝트당 전역적으로 고유한 문자열입니다. 64자를 초과할 수 없습니다. 모든 비단어 문자는 대시로 변환됩니다.

```python
# 나중에 재개할 때 이 ID를 저장합니다
id = wandb.util.generate_id()
wandb.init(id=id, resume="allow")
# 또는 환경 변수를 통해
os.environ["WANDB_RESUME"] = "allow"
os.environ["WANDB_RUN_ID"] = wandb.util.generate_id()
wandb.init()
```

`WANDB_RESUME`을 `"allow"`로 설정한 경우, 항상 `WANDB_RUN_ID`를 고유한 문자열로 설정하고 프로세스의 재시작을 자동으로 처리할 수 있습니다. `WANDB_RESUME`을 `"must"`로 설정한 경우, 재개할 실행이 아직 존재하지 않으면 W&B는 새 실행을 자동 생성하는 대신 오류를 발생시킬 것입니다.

:::caution
여러 프로세스가 동시에 같은 `run_id`를 사용하는 경우 예상치 못한 결과가 기록되고 속도 제한이 발생할 수 있습니다.
:::

:::info
`wandb.init()`에서 `notes`를 지정한 경우 실행을 재개하면 UI에서 추가한 모든 노트가 덮어쓰여집니다.
:::

:::info
스윕의 일부로 실행된 실행을 재개하는 것은 지원되지 않습니다.
:::

### 선점 가능한 스윕

선점 가능한 큐에 있는 SLURM 작업, EC2 스팟 인스턴스 또는 Google Cloud 선점 가능한 VM과 같이 선점이 가능한 컴퓨팅 환경에서 스윕 에이전트를 실행하는 경우, 중단된 스윕 실행을 자동으로 다시 큐에 넣어 완료될 때까지 재시도할 수 있습니다.

현재 실행이 곧 선점될 것이라는 것을 알게 되면,

```
wandb.mark_preempting()
```

를 호출하여 W&B 백엔드에 실행이 선점될 것이라고 즉시 신호합니다. 선점 표시가 있는 실행이 상태 코드 0으로 종료되면, W&B는 실행이 성공적으로 종료되었다고 간주하고 재큐에 넣지 않을 것입니다. 선점 표시가 있는 실행이 0이 아닌 상태로 종료되면, W&B는 실행이 선점되었다고 간주하고, 실행을 스윕과 연결된 실행 큐에 자동으로 추가할 것입니다. 실행의 마지막 심장 박동 후 5분이 지나면 W&B는 실행 상태 없이 종료된 실행을 선점된 것으로 표시한 다음, 스윕 실행 큐에 추가할 것입니다. 스윕 에이전트는 큐가 소진될 때까지 큐에서 실행을 소비한 다음, 표준 스윕 검색 알고리즘을 기반으로 새로운 실행 생성을 재개할 것입니다.

기본적으로, 재큐에 넣은 실행은 초기 단계에서 로깅을 시작합니다. 중단된 단계에서 로깅을 재개하도록 실행을 지시하려면 재개된 실행을 `wandb.init(resume=True)`로 초기화합니다.
---
description: Initialize a W&B Sweep
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 스윕 초기화

<head>
  <title>W&B 스윕 시작하기</title>
</head>

W&B는 클라우드(표준) 또는 로컬(로컬)에서 하나 이상의 컴퓨터에서 스윕을 관리하기 위해 _스윕 컨트롤러_를 사용합니다. 실행이 완료된 후, 스윕 컨트롤러는 실행할 새로운 실행 세트에 대한 지침을 발행합니다. 이러한 지침은 실행을 실제로 수행하는 _에이전트_에 의해 수집됩니다. 일반적인 W&B 스윕에서 컨트롤러는 W&B 서버에 있습니다. 에이전트는 _당신의_ 컴퓨터(들)에 있습니다.

다음 코드 조각은 CLI와 Jupyter Notebook 또는 Python 스크립트 내에서 스윕을 초기화하는 방법을 보여줍니다.

:::caution
1. 스윕을 초기화하기 전에, YAML 파일이나 스크립트의 중첩된 Python 사전 개체에 스윕 구성이 정의되어 있는지 확인하세요. 자세한 정보는 [스윕 구성 정의](../../guides/sweeps/define-sweep-configuration.md)를 참조하세요.
2. W&B 스윕과 W&B 실행은 동일한 프로젝트에 있어야 합니다. 따라서, W&B를 초기화할 때 제공하는 이름([`wandb.init`](../../ref/python/init.md))은 W&B 스윕을 초기화할 때 제공하는 프로젝트 이름([`wandb.sweep`](../../ref/python/sweep.md))과 일치해야 합니다.
:::

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python 스크립트 또는 Jupyter Notebook', value: 'python'},
    {label: 'CLI', value: 'cli'},
  ]}>
  <TabItem value="python">

W&B SDK를 사용하여 스윕을 초기화하세요. 스윕 구성 사전을 `sweep` 파라미터에 전달하세요. 선택적으로 W&B 실행의 출력을 저장하고자 하는 프로젝트의 이름(`project`)을 프로젝트 파라미터에 제공하세요. 프로젝트가 지정되지 않은 경우, 실행은 "Uncategorized" 프로젝트에 저장됩니다.

```python
import wandb

# 예시 스윕 구성
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

sweep_id = wandb.sweep(sweep=sweep_configuration, project="project-name")
```

[`wandb.sweep`](../../ref/python/sweep) 함수는 스윕 ID를 반환합니다. 스윕 ID에는 엔티티 이름과 프로젝트 이름이 포함됩니다. 스윕 ID를 메모하세요.
  </TabItem>
  <TabItem value="cli">

W&B CLI를 사용하여 스윕을 초기화하세요. 구성 파일의 이름을 제공하세요. 선택적으로 `project` 플래그에 프로젝트 이름을 제공하세요. 프로젝트가 지정되지 않은 경우, W&B 실행은 "Uncategorized" 프로젝트에 저장됩니다.

스윕을 초기화하기 위해 [`wandb sweep`](../../ref/cli/wandb-sweep) 명령을 사용하세요. 다음 코드 예제는 `sweeps_demo` 프로젝트에 대한 스윕을 초기화하고 구성에 `config.yaml` 파일을 사용합니다.

```bash
wandb sweep --project sweeps_demo config.yaml
```

이 명령은 스윕 ID를 출력합니다. 스윕 ID에는 엔티티 이름과 프로젝트 이름이 포함됩니다. 스윕 ID를 메모하세요.
  </TabItem>
</Tabs>
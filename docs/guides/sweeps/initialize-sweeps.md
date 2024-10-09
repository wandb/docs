---
title: Initialize a sweep
description: W&B 스윕 초기화하기
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

W&B는 클라우드(기본), 로컬(로컬)에서 여러 대의 머신에 걸쳐 스윕을 관리하기 위해 _Sweep Controller_ 를 사용합니다. run이 완료되면 스윕 컨트롤러는 새로운 run을 실행하는 데 필요한 새로운 지침 세트를 발행합니다. 이러한 지침들은 실제로 run을 수행하는 _에이전트_ 에 의해 수집됩니다. 일반적인 W&B Sweep에서 컨트롤러는 W&B 서버에 위치하고, 에이전트는 _여러분의_ 머신(들)에 위치합니다.

다음 코드조각들은 CLI 및 Jupyter 노트북 또는 Python 스크립트 내에서 스윕을 초기화하는 방법을 설명합니다.

:::caution
1. 스윕을 초기화하기 전에 스윕 구성이 YAML 파일이나 스크립트 내의 중첩된 Python 사전 오브젝트로 정의되어 있는지 확인하세요. 자세한 정보는 [스윕 구성 정의](../../guides/sweeps/define-sweep-configuration.md)를 참조하세요.
2. W&B Sweep 및 W&B Run은 반드시 동일한 프로젝트에 있어야 합니다. 따라서 W&B를 초기화할 때 제공하는 이름([`wandb.init`](../../ref/python/init.md))은 W&B Sweep을 초기화할 때 제공하는 프로젝트의 이름([`wandb.sweep`](../../ref/python/sweep.md))과 일치해야 합니다.
:::

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python script or Jupyter Notebook', value: 'python'},
    {label: 'CLI', value: 'cli'},
  ]}>
  <TabItem value="python">

W&B SDK를 사용하여 스윕을 초기화하세요. 스윕 설정 사전 오브젝트를 `sweep` 파라미터에 전달하세요. 선택적으로 W&B Run의 출력이 저장될 _프로젝트_ 의 이름을 `project` 파라미터에 제공합니다. 만약 프로젝트가 지정되지 않으면, run은 "비분류" 프로젝트에 저장됩니다.

```python
import wandb

# 스윕 구성 예제
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

[`wandb.sweep`](../../ref/python/sweep) 함수는 스윕 ID를 반환합니다. 스윕 ID에는 entity 이름과 project 이름이 포함됩니다. 스윕 ID를 기록해 두세요.
  </TabItem>
  <TabItem value="cli">

W&B CLI를 사용하여 스윕을 초기화하세요. 구성 파일의 이름을 제공하세요. 선택적으로 `project` 플래그에 프로젝트 이름을 제공합니다. 만약 프로젝트가 지정되지 않으면, W&B Run은 "비분류" 프로젝트에 저장됩니다.

[`wandb sweep`](../../ref/cli/wandb-sweep) 코맨드를 사용하여 스윕을 초기화하세요. 아래의 코드 예제는 `sweeps_demo` 프로젝트에 대한 스윕을 초기화하고 `config.yaml` 파일을 구성에 사용합니다.

```bash
wandb sweep --project sweeps_demo config.yaml
```

이 코맨드는 스윕 ID를 출력합니다. 스윕 ID에는 entity 이름과 project 이름이 포함됩니다. 스윕 ID를 기록해 두세요.
  </TabItem>
</Tabs>
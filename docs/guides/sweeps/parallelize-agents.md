---
title: Parallelize agents
description: 멀티 코어 또는 멀티 GPU 머신에서 W&B 스윕 에이전트를 병렬로 실행하세요.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

W&B Sweep 에이전트를 다중 코어 또는 다중 GPU 머신에서 병렬화하세요. 시작하기 전에 W&B Sweep을 초기화했는지 확인하십시오. W&B Sweep을 초기화하는 방법에 대한 자세한 내용은 [Initialize sweeps](./initialize-sweeps.md)를 참조하십시오.

### 다중 CPU 머신에서 병렬화하기

귀하의 유스 케이스에 따라 CLI를 사용하거나 Jupyter 노트북 내에서 W&B Sweep 에이전트를 병렬화하는 방법을 배우려면 다음 탭을 탐색하세요.

<Tabs
  defaultValue="cli_text"
  values={[
    {label: 'CLI', value: 'cli_text'},
    {label: 'Jupyter Notebook', value: 'jupyter'},
  ]}>

  <TabItem value="cli_text">

터미널에서 여러 CPU에 걸쳐 W&B Sweep 에이전트를 병렬화하기 위해 [`wandb agent`](../../ref/cli/wandb-agent.md) 커맨드를 사용하십시오. 스윕을 [초기화했을 때](./initialize-sweeps.md) 반환된 스윕 ID를 제공하십시오.

1. 로컬 머신에서 두 개 이상의 터미널 창을 열십시오.
2. 아래 코드조각을 복사하여 붙여넣고 `sweep_id`를 스윕 ID로 바꿉니다:

```bash
wandb agent sweep_id
```

  </TabItem>

  <TabItem value="jupyter">

Jupyter 노트북 내에서 여러 CPU에 걸쳐 W&B Sweep 에이전트를 병렬화하기 위해 W&B Python SDK 라이브러리를 사용하십시오. 스윕을 [초기화했을 때](./initialize-sweeps.md) 반환된 스윕 ID를 확인하십시오. 또한, 스윕이 실행할 함수의 이름을 `function` 파라미터로 제공하십시오.

1. 여러 Jupyter 노트북을 엽니다.
2. 여러 Jupyter 노트북에 W&B Sweep ID를 복사하여 붙여넣어 W&B Sweep을 병렬화 합니다. 예를 들어, 스윕 ID가 `sweep_id`라는 변수에 저장되어 있고 함수 이름이 `function_name`인 경우, 다음 코드조각을 여러 Jupyter 노트북에 붙여넣어 스윕을 병렬화할 수 있습니다:

```python
wandb.agent(sweep_id=sweep_id, function=function_name)
```

  </TabItem>
</Tabs>

### 다중 GPU 머신에서 병렬화하기

CUDA Toolkit을 사용하여 여러 GPU에 걸쳐 W&B Sweep 에이전트를 병렬화하는 절차를 따르십시오:

1. 로컬 머신에서 두 개 이상의 터미널 창을 엽니다.
2. W&B Sweep 작업을 시작할 때 (`wandb agent`), 사용할 GPU 인스턴스는 `CUDA_VISIBLE_DEVICES`로 지정하십시오. 사용할 GPU 인스턴스에 해당하는 정수 값을 `CUDA_VISIBLE_DEVICES`에 할당하십시오.

예를 들어, 로컬 머신에 두 개의 NVIDIA GPU가 있다고 가정합니다. 터미널 창을 열고 `CUDA_VISIBLE_DEVICES`를 `0`으로 설정합니다 (`CUDA_VISIBLE_DEVICES=0`). 스윕을 초기화할 때 반환된 W&B Sweep ID로 다음 예에서 `sweep_ID`를 바꿉니다:

터미널 1

```bash
CUDA_VISIBLE_DEVICES=0 wandb agent sweep_ID
```

두 번째 터미널 창을 엽니다. `CUDA_VISIBLE_DEVICES`를 `1`로 설정합니다 (`CUDA_VISIBLE_DEVICES=1`). 이전 코드조각에 언급된 `sweep_ID`에 동일한 W&B Sweep ID를 붙여넣습니다:

터미널 2

```bash
CUDA_VISIBLE_DEVICES=1 wandb agent sweep_ID
```
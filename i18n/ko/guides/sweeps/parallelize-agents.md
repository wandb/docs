---
description: Parallelize W&B Sweep agents on multi-core or multi-GPU machine.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 에이전트 병렬 처리하기

<head>
  <title>에이전트 병렬 처리하기</title>
</head>

멀티 코어 또는 멀티 GPU 기계에서 W&B 스윕 에이전트를 병렬로 처리하세요. 시작하기 전에, W&B 스윕 초기화가 되었는지 확인하세요. W&B 스윕을 초기화하는 방법에 대한 자세한 정보는 [스윕 초기화하기](./initialize-sweeps.md)를 참조하세요.

### 멀티 CPU 기기에서 병렬 처리하기

사용 사례에 따라, CLI 또는 Jupyter Notebook 내에서 W&B 스윕 에이전트를 병렬로 처리하는 방법을 알아보려면 다음 탭을 탐색하세요.


<Tabs
  defaultValue="cli_text"
  values={[
    {label: 'CLI', value: 'cli_text'},
    {label: 'Jupyter Notebook', value: 'jupyter'},
  ]}>
  <TabItem value="cli_text">

터미널에서 여러 CPU에 걸쳐 W&B 스윕 에이전트를 병렬로 처리하려면 [`wandb agent`](../../ref/cli/wandb-agent.md) 명령을 사용하세요. [스윕 초기화하기](./initialize-sweeps.md)에서 반환된 스윕 ID를 제공하세요.

1. 로컬 기기에서 여러 터미널 창을 엽니다.
2. 아래 코드 조각을 복사하여 붙여넣고 `sweep_id`를 귀하의 스윕 ID로 교체하세요:


```bash
wandb agent sweep_id
```


  </TabItem>
  <TabItem value="jupyter">

Jupyter Notebook 내에서 여러 CPU에 걸쳐 W&B 스윕 에이전트를 병렬로 처리하려면 W&B Python SDK 라이브러리를 사용하세요. [스윕 초기화하기](./initialize-sweeps.md)에서 반환된 스윕 ID를 확인하세요. 또한, 스윕이 실행할 함수의 이름을 `function` 파라미터에 제공하세요:

1. 여러 Jupyter Notebook을 엽니다.
2. 여러 Jupyter Notebook에 W&B 스윕 ID를 복사하여 붙여넣어 W&B 스윕을 병렬로 처리하세요. 예를 들어, `sweep_id`라는 변수에 스윕 ID가 저장되어 있고 함수 이름이 `function_name`이라면 아래 코드 조각을 여러 jupyter notebooks에 붙여넣어 스윕을 병렬로 처리할 수 있습니다:


```python
wandb.agent(sweep_id=sweep_id, function=function_name)
```

  </TabItem>
</Tabs>

### 멀티 GPU 기기에서 병렬 처리하기

CUDA Toolkit을 사용하여 터미널을 통해 여러 GPU에 걸쳐 W&B 스윕 에이전트를 병렬로 처리하려면 다음 절차를 따르세요:

1. 로컬 기기에서 여러 터미널 창을 엽니다.
2. W&B 스윕 작업을 시작할 때 `CUDA_VISIBLE_DEVICES`를 사용하여 사용할 GPU 인스턴스를 지정하세요 ([`wandb agent`](../../ref/cli/wandb-agent.md)). 사용할 GPU 인스턴스에 해당하는 정수 값을 `CUDA_VISIBLE_DEVICES`에 할당하세요.

예를 들어, 로컬 기기에 NVIDIA GPU가 두 개 있다고 가정합니다. 터미널 창을 열고 `CUDA_VISIBLE_DEVICES`를 `0`(`CUDA_VISIBLE_DEVICES=0`)으로 설정하세요. 다음 예제에서 스윕을 초기화할 때 반환된 W&B 스윕 ID로 `sweep_ID`를 교체하세요:

터미널 1

```bash
CUDA_VISIBLE_DEVICES=0 wandb agent sweep_ID
```

두 번째 터미널 창을 엽니다. `CUDA_VISIBLE_DEVICES`를 `1`(`CUDA_VISIBLE_DEVICES=1`)로 설정하세요. 앞서 언급한 코드 조각에 동일한 W&B 스윕 ID를 `sweep_ID`로 붙여넣으세요:

터미널 2

```bash
CUDA_VISIBLE_DEVICES=1 wandb agent sweep_ID
```
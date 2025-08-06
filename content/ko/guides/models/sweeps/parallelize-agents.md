---
title: 에이전트 병렬화
description: 멀티 코어 또는 멀티 GPU 머신에서 W&B 스윕 에이전트를 병렬로 실행하세요.
menu:
  default:
    identifier: ko-guides-models-sweeps-parallelize-agents
    parent: sweeps
weight: 6
---

여러 개의 코어 또는 여러 개의 GPU가 장착된 머신에서 W&B Sweep 에이전트를 병렬로 실행할 수 있습니다. 시작하기 전에 W&B Sweep을 초기화했는지 확인하세요. W&B Sweep 초기화 방법에 대한 자세한 내용은 [스윕 초기화]({{< relref path="./initialize-sweeps.md" lang="ko" >}})를 참고하세요.

### 다중 CPU 머신에서 병렬화하기

유스 케이스에 따라, CLI 또는 Jupyter Notebook 내에서 W&B Sweep 에이전트를 병렬 처리하는 방법을 아래 탭에서 확인할 수 있습니다.


{{< tabpane text=true >}}
  {{% tab header="CLI" %}}
[`wandb agent`]({{< relref path="/ref/cli/wandb-agent.md" lang="ko" >}}) 코맨드를 사용해서 여러 CPU에서 스윕 에이전트를 터미널로 병렬 처리할 수 있습니다. [스윕을 초기화할 때]({{< relref path="./initialize-sweeps.md" lang="ko" >}}) 반환된 sweep ID를 입력하세요. 

1. 로컬 머신에서 하나 이상의 터미널 창을 엽니다.
2. 아래 코드조각을 복사해서 붙여넣고 `sweep_id`를 본인의 sweep ID로 바꿔 사용하세요:

```bash
wandb agent sweep_id
```  
  {{% /tab %}}
  {{% tab header="Jupyter Notebook" %}}
W&B Python SDK 라이브러리를 활용해 Jupyter Notebook 내에서 여러 CPU에 W&B Sweep 에이전트를 병렬로 실행할 수 있습니다. 또한, [스윕을 초기화할 때]({{< relref path="./initialize-sweeps.md" lang="ko" >}}) 반환된 sweep ID가 있는지 확인하세요. 그리고 스윕이 실행할 함수의 이름을 `function` 파라미터로 입력해야 합니다.

1. 여러 개의 Jupyter Notebook을 엽니다.
2. 여러 Jupyter Notebook에 동일한 W&B Sweep ID를 복사하여 붙여넣으면 Sweep을 병렬로 실행할 수 있습니다. 예를 들어, sweep ID가 `sweep_id` 변수에 저장되어 있고 함수 이름이 `function_name`일 때, 아래 코드조각을 각 노트북에 붙여넣어 스윕을 병렬화할 수 있습니다: 

```python
wandb.agent(sweep_id=sweep_id, function=function_name)
```  
  {{% /tab %}}
{{< /tabpane >}}



### 다중 GPU 머신에서 병렬화하기

CUDA Toolkit을 사용하여 터미널에서 여러 GPU에 W&B Sweep 에이전트를 병렬로 실행하려면 다음 절차를 따라 주세요:

1. 로컬 머신에서 하나 이상의 터미널 창을 엽니다.
2. W&B Sweep 작업([`wandb agent`]({{< relref path="/ref/cli/wandb-agent.md" lang="ko" >}}))을 시작할 때 `CUDA_VISIBLE_DEVICES`를 이용해 사용할 GPU 인스턴스를 지정하세요. `CUDA_VISIBLE_DEVICES`에 사용할 GPU 인스턴스에 해당하는 정수 값(예: 0, 1 등)을 할당합니다.

예를 들어, 로컬 머신에 NVIDIA GPU가 두 개 있다면 첫 번째 터미널 창을 열고 `CUDA_VISIBLE_DEVICES`를 `0`으로 설정합니다 (`CUDA_VISIBLE_DEVICES=0`). 아래 코드 예시에서 `sweep_ID`는 W&B Sweep을 초기화했을 때 반환된 Sweep ID로 교체하세요:

터미널 1

```bash
CUDA_VISIBLE_DEVICES=0 wandb agent sweep_ID
```

두 번째 터미널 창을 열고, `CUDA_VISIBLE_DEVICES`를 `1`로 설정하세요 (`CUDA_VISIBLE_DEVICES=1`). 앞선 코드조각에서 사용한 동일한 W&B Sweep ID(즉, `sweep_ID`)를 입력해 실행하면 됩니다:

터미널 2

```bash
CUDA_VISIBLE_DEVICES=1 wandb agent sweep_ID
```
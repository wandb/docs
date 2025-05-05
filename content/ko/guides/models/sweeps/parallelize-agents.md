---
title: Parallelize agents
description: 멀티 코어 또는 멀티 GPU 머신에서 W&B 스윕 에이전트를 병렬화하세요.
menu:
  default:
    identifier: ko-guides-models-sweeps-parallelize-agents
    parent: sweeps
weight: 6
---

멀티 코어 또는 멀티 GPU 머신에서 W&B 스윕 에이전트를 병렬화하세요. 시작하기 전에 W&B 스윕을 초기화했는지 확인하세요. W&B 스윕을 초기화하는 방법에 대한 자세한 내용은 [스윕 초기화]({{< relref path="./initialize-sweeps.md" lang="ko" >}})를 참조하세요.

### 멀티 CPU 머신에서 병렬화

유스 케이스에 따라 다음 탭을 살펴보고 CLI 또는 Jupyter Notebook 내에서 W&B 스윕 에이전트를 병렬화하는 방법을 알아보세요.

{{< tabpane text=true >}}
  {{% tab header="CLI" %}}
[`wandb agent`]({{< relref path="/ref/cli/wandb-agent.md" lang="ko" >}}) 코맨드를 사용하여 터미널에서 여러 CPU에 걸쳐 W&B 스윕 에이전트를 병렬화하세요. [스윕을 초기화]({{< relref path="./initialize-sweeps.md" lang="ko" >}})할 때 반환된 스윕 ID를 제공하세요.

1. 로컬 머신에서 둘 이상의 터미널 창을 여세요.
2. 아래 코드 조각을 복사하여 붙여넣고 `sweep_id`를 스윕 ID로 바꾸세요.

```bash
wandb agent sweep_id
```
  {{% /tab %}}
  {{% tab header="Jupyter Notebook" %}}
W&B Python SDK 라이브러리를 사용하여 Jupyter Notebook 내에서 여러 CPU에 걸쳐 W&B 스윕 에이전트를 병렬화하세요. [스윕을 초기화]({{< relref path="./initialize-sweeps.md" lang="ko" >}})할 때 반환된 스윕 ID가 있는지 확인하세요. 또한 스윕이 실행할 함수의 이름을 `function` 파라미터에 제공하세요.

1. 둘 이상의 Jupyter Notebook을 여세요.
2. 여러 Jupyter Notebook에 W&B 스윕 ID를 복사하여 붙여넣어 W&B 스윕을 병렬화하세요. 예를 들어, 스윕 ID가 `sweep_id`라는 변수에 저장되어 있고 함수의 이름이 `function_name`인 경우 다음 코드 조각을 여러 Jupyter Notebook에 붙여넣어 스윕을 병렬화할 수 있습니다.

```python
wandb.agent(sweep_id=sweep_id, function=function_name)
```
  {{% /tab %}}
{{< /tabpane >}}

### 멀티 GPU 머신에서 병렬화

CUDA 툴킷을 사용하여 터미널에서 여러 GPU에 걸쳐 W&B 스윕 에이전트를 병렬화하려면 다음 절차를 따르세요.

1. 로컬 머신에서 둘 이상의 터미널 창을 여세요.
2. W&B 스윕 작업을 시작할 때 `CUDA_VISIBLE_DEVICES`를 사용하여 사용할 GPU 인스턴스를 지정하세요([`wandb agent`]({{< relref path="/ref/cli/wandb-agent.md" lang="ko" >}})). 사용할 GPU 인스턴스에 해당하는 정수 값을 `CUDA_VISIBLE_DEVICES`에 할당하세요.

예를 들어, 로컬 머신에 두 개의 NVIDIA GPU가 있다고 가정해 보겠습니다. 터미널 창을 열고 `CUDA_VISIBLE_DEVICES`를 `0`으로 설정하세요(`CUDA_VISIBLE_DEVICES=0`). 다음 예제에서 `sweep_ID`를 W&B 스윕을 초기화할 때 반환되는 W&B 스윕 ID로 바꾸세요.

터미널 1

```bash
CUDA_VISIBLE_DEVICES=0 wandb agent sweep_ID
```

두 번째 터미널 창을 여세요. `CUDA_VISIBLE_DEVICES`를 `1`로 설정하세요(`CUDA_VISIBLE_DEVICES=1`). 이전 코드 조각에 언급된 `sweep_ID`에 대해 동일한 W&B 스윕 ID를 붙여넣으세요.

터미널 2

```bash
CUDA_VISIBLE_DEVICES=1 wandb agent sweep_ID
```
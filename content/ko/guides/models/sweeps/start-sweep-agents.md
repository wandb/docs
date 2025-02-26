---
title: Start or stop a sweep agent
description: 하나 이상의 시스템에서 W&B 스윕 에이전트 를 시작하거나 중지합니다.
menu:
  default:
    identifier: ko-guides-models-sweeps-start-sweep-agents
    parent: sweeps
weight: 5
---

하나 이상의 머신에서 하나 이상의 에이전트로 W&B 스윕을 시작하세요. W&B 스윕 에이전트는 하이퍼파라미터에 대해 W&B 스윕( `wandb sweep`)을 초기화할 때 시작한 W&B 서버를 쿼리하고 이를 사용하여 모델 트레이닝을 실행합니다.

W&B 스윕 에이전트를 시작하려면 W&B 스윕을 초기화할 때 반환된 W&B 스윕 ID를 제공하세요. W&B 스윕 ID는 다음과 같은 형식을 갖습니다.

```bash
entity/project/sweep_ID
```

여기서:

* entity: W&B 사용자 이름 또는 팀 이름입니다.
* project: W&B Run의 출력을 저장할 프로젝트의 이름입니다. 프로젝트를 지정하지 않으면 run이 "Uncategorized" 프로젝트에 배치됩니다.
* sweep_ID: W&B에서 생성한 의사 난수 고유 ID입니다.

Jupyter Notebook 또는 Python 스크립트 내에서 W&B 스윕 에이전트를 시작하는 경우 W&B 스윕이 실행할 함수 이름을 제공하세요.

다음 코드 조각은 W&B로 에이전트를 시작하는 방법을 보여줍니다. 구성 파일이 이미 있고 W&B 스윕을 이미 초기화했다고 가정합니다. 구성 파일을 정의하는 방법에 대한 자세한 내용은 [스윕 구성 정의]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ko" >}})을 참조하세요.

{{< tabpane text=true >}}
{{% tab header="CLI" %}}
`wandb agent` 코맨드를 사용하여 스윕을 시작하세요. 스윕을 초기화할 때 반환된 스윕 ID를 제공하세요. 아래 코드 조각을 복사하여 붙여넣고 `sweep_id`를 스윕 ID로 바꾸세요.

```bash
wandb agent sweep_id
```
{{% /tab %}}
{{% tab header="Python script or notebook" %}}
W&B Python SDK 라이브러리를 사용하여 스윕을 시작하세요. 스윕을 초기화할 때 반환된 스윕 ID를 제공하세요. 또한 스윕이 실행할 함수의 이름을 제공하세요.

```python
wandb.agent(sweep_id=sweep_id, function=function_name)
```
{{% /tab %}}
{{< /tabpane >}}

### W&B 에이전트 중지

{{% alert color="secondary" %}}
랜덤 및 베이지안 탐색은 영원히 실행됩니다. 코맨드 라인, Python 스크립트 또는 [Sweeps UI]({{< relref path="./visualize-sweep-results.md" lang="ko" >}}) 내에서 프로세스를 중지해야 합니다.
{{% /alert %}}

선택적으로 스윕 에이전트가 시도해야 하는 W&B Runs의 수를 지정하세요. 다음 코드 조각은 CLI 및 Jupyter Notebook, Python 스크립트 내에서 최대 [W&B Runs]({{< relref path="/ref/python/run.md" lang="ko" >}}) 수를 설정하는 방법을 보여줍니다.

{{< tabpane text=true >}}
  {{% tab header="Python script or notebook" %}}
먼저 스윕을 초기화합니다. 자세한 내용은 [스윕 초기화]({{< relref path="./initialize-sweeps.md" lang="ko" >}})을 참조하세요.

```
sweep_id = wandb.sweep(sweep_config)
```

다음으로 스윕 작업을 시작합니다. 스윕 시작에서 생성된 스윕 ID를 제공합니다. 시도할 최대 run 수를 설정하려면 count 파라미터에 정수 값을 전달하세요.

```python
sweep_id, count = "dtzl1o7u", 10
wandb.agent(sweep_id, count=count)
```

{{% alert color="secondary" %}}
스윕 에이전트가 완료된 후 동일한 스크립트 또는 노트북 내에서 새 run을 시작하는 경우 새 run을 시작하기 전에 `wandb.teardown()`을 호출해야 합니다.
{{% /alert %}}
  {{% /tab %}}
  {{% tab header="CLI" %}}
먼저 [`wandb sweep`]({{< relref path="/ref/cli/wandb-sweep.md" lang="ko" >}}) 코맨드로 스윕을 초기화합니다. 자세한 내용은 [스윕 초기화]({{< relref path="./initialize-sweeps.md" lang="ko" >}})을 참조하세요.

```
wandb sweep config.yaml
```

시도할 최대 run 수를 설정하려면 count 플래그에 정수 값을 전달하세요.

```python
NUM=10
SWEEPID="dtzl1o7u"
wandb agent --count $NUM $SWEEPID
```
  {{% /tab %}}
{{< /tabpane >}}

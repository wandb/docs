---
title: 스윕 에이전트 시작 또는 중지
description: 하나 이상의 머신에서 W&B 스윕 에이전트를 시작하거나 중지하세요.
menu:
  default:
    identifier: ko-guides-models-sweeps-start-sweep-agents
    parent: sweeps
weight: 5
---

하나 이상의 머신에서 하나 이상의 W&B Sweep 에이전트를 실행하세요. W&B Sweep 에이전트는 W&B Sweep(`wandb sweep`)을 초기화할 때 시작된 W&B 서버에 쿼리하여 하이퍼파라미터를 받아 모델 트레이닝을 진행합니다.

W&B Sweep 에이전트를 시작하려면, W&B Sweep을 초기화할 때 반환된 W&B Sweep ID를 입력해야 합니다. W&B Sweep ID의 형식은 아래와 같습니다:

```bash
entity/project/sweep_ID
```

여기서:

* entity: W&B 사용자명 또는 팀명입니다.
* project: W&B Run의 결과가 저장될 프로젝트 이름입니다. 프로젝트를 지정하지 않으면, run이 "Uncategorized" 프로젝트에 저장됩니다.
* sweep_ID: W&B에서 생성한 고유한 난수 ID입니다.

Jupyter Notebook이나 Python 스크립트 내에서 W&B Sweep 에이전트를 실행한다면, W&B Sweep이 수행할 함수 이름도 지정해야 합니다.

아래 코드조각들은 W&B 에이전트 실행 방법을 보여줍니다. 이미 설정 파일(sweep configuration file)을 보유하고 있고, W&B Sweep이 초기화된 상태라고 가정합니다. 설정 파일 정의 방법에 관한 자세한 내용은 [스윕 구성 정의]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ko" >}})를 참고하세요.

{{< tabpane text=true >}}
{{% tab header="CLI" %}}
`sweep`을 시작하려면 `wandb agent` 명령어를 사용하세요. sweep을 초기화할 때 반환된 sweep ID를 입력합니다. 아래 코드조각을 복사하여 `sweep_id`를 실제 sweep ID로 바꿔 사용하세요:

```bash
wandb agent sweep_id
```
{{% /tab %}}
{{% tab header="Python script or notebook" %}}
W&B Python SDK 라이브러리를 이용해 sweep을 시작할 수 있습니다. sweep을 초기화할 때 반환된 sweep ID를 입력하고, 실행할 함수 이름도 함께 지정해야 합니다.

```python
wandb.agent(sweep_id=sweep_id, function=function_name)
```
{{% /tab %}}
{{< /tabpane >}}



### W&B agent 중지하기

{{% alert color="secondary" %}}
랜덤 또는 베이지안 탐색은 별도로 종료하지 않는 한 계속 실행됩니다. 커맨드라인, 파이썬 스크립트, 또는 [Sweeps UI]({{< relref path="./visualize-sweep-results.md" lang="ko" >}})에서 직접 프로세스를 종료해야 합니다.
{{% /alert %}}

선택적으로 Sweep agent가 시도할 W&B Run의 개수를 지정할 수 있습니다. 아래 코드조각들은 CLI 및 Jupyter Notebook, Python 스크립트에서 [W&B Run]({{< relref path="/ref/python/sdk/classes/run.md" lang="ko" >}})의 최대 개수를 설정하는 방법을 보여줍니다.

{{< tabpane text=true >}}
  {{% tab header="Python script or notebook" %}}
먼저 sweep을 초기화하세요. 자세한 내용은 [sweeps 초기화]({{< relref path="./initialize-sweeps.md" lang="ko" >}})를 참고하세요.

```
sweep_id = wandb.sweep(sweep_config)
```

다음으로 sweep job을 시작합니다. sweep 초기화 시 생성된 sweep ID를 이용하세요. 실행할 run의 최대 개수를 정하고 싶다면, count 파라미터에 정수 값을 전달하세요.

```python
sweep_id, count = "dtzl1o7u", 10
wandb.agent(sweep_id, count=count)
```

{{% alert color="secondary" %}}
같은 스크립트나 노트북에서 sweep agent가 완료된 뒤에 새로운 run을 시작하려면, 새 run을 시작하기 전에 `wandb.teardown()`을 호출해야 합니다.
{{% /alert %}}  
  {{% /tab %}}
  {{% tab header="CLI" %}}
먼저 [`wandb sweep`]({{< relref path="/ref/cli/wandb-sweep.md" lang="ko" >}}) 명령어로 sweep을 초기화하세요. 자세한 내용은 [sweeps 초기화]({{< relref path="./initialize-sweeps.md" lang="ko" >}})를 참고하세요.

```
wandb sweep config.yaml
```

실행할 run의 최대 개수를 지정하려면 count 플래그에 정수 값을 전달하세요.

```python
NUM=10
SWEEPID="dtzl1o7u"
wandb agent --count $NUM $SWEEPID
```  
  {{% /tab %}}
{{< /tabpane >}}
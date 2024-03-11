---
description: Start or stop a W&B Sweep Agent on one or more machines.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 스윕 에이전트 시작하기

<head>
  <title>W&B 스윕 시작 또는 중지하기</title>
</head>

하나 이상의 머신에서 하나 이상의 에이전트로 W&B 스윕을 시작합니다. W&B 스윕 에이전트는 W&B 스윕을 초기화할 때 실행한 W&B 서버(`wandb sweep)`에서 하이퍼파라미터를 조회하고 이를 사용하여 모델 트레이닝을 실행합니다.

W&B 스윕 에이전트를 시작하려면, W&B 스윕을 초기화할 때 반환된 W&B 스윕 ID를 제공해야 합니다. W&B 스윕 ID는 다음과 같은 형태입니다:

```bash
entity/project/sweep_ID
```

여기서:

* entity: W&B 사용자 이름 또는 팀 이름입니다.
* project: W&B Run의 출력이 저장될 프로젝트의 이름입니다. 프로젝트가 지정되지 않은 경우, run은 "Uncategorized" 프로젝트에 저장됩니다.
* sweep\_ID: W&B에 의해 생성된 유사 랜덤, 고유 ID입니다.

Jupyter Notebook 또는 Python 스크립트 내에서 W&B 스윕 에이전트를 시작할 경우 실행할 함수의 이름을 제공합니다.

다음 코드 조각은 W&B와 함께 에이전트를 시작하는 방법을 보여줍니다. 이미 설정 파일을 가지고 있고 W&B 스윕을 초기화했다고 가정합니다. 설정 파일을 정의하는 방법에 대한 자세한 정보는 [스윕 구성 정의하기](./define-sweep-configuration.md)를 참조하세요.

<Tabs
  defaultValue="cli"
  values={[
    {label: 'CLI', value: 'cli'},
    {label: 'Python 스크립트 또는 Jupyter Notebook', value: 'python'},
  ]}>
  <TabItem value="cli">

`wandb agent` 코맨드를 사용하여 스윕을 시작합니다. 스윕을 초기화할 때 반환된 스윕 ID를 제공하세요. 아래의 코드 조각을 복사하여 붙여넣고 `sweep_id`를 스윕 ID로 교체하세요:

```bash
wandb agent sweep_id
```
  </TabItem>
  <TabItem value="python">

W&B Python SDK 라이브러리를 사용하여 스윕을 시작합니다. 스윕을 초기화할 때 반환된 스윕 ID를 제공하세요. 추가로, 스윕이 실행할 함수의 이름을 제공하세요.

```python
wandb.agent(sweep_id=sweep_id, function=function_name)
```
  </TabItem>
</Tabs>

### W&B 에이전트 중지하기

:::caution
랜덤 및 베이지안 탐색은 영원히 실행됩니다. 커맨드라인, 파이썬 스크립트 내부 또는 [Sweeps UI](./visualize-sweep-results.md)에서 프로세스를 중지해야 합니다.
:::

스윕 에이전트가 시도해야 할 W&B Runs의 수를 선택적으로 지정하세요. 다음 코드 조각은 CLI와 Jupyter Notebook, Python 스크립트 내에서 최대 [W&B Runs](../../ref/python/run.md) 수를 설정하는 방법을 보여줍니다.

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python 스크립트 또는 Jupyter Notebook', value: 'python'},
    {label: 'CLI', value: 'cli'},
  ]}>
  <TabItem value="python">

먼저, 스윕을 초기화하세요. 자세한 정보는 [스윕 초기화하기](./initialize-sweeps.md)를 참조하세요.

```
sweep_id = wandb.sweep(sweep_config)
```

다음으로, 스윕 작업을 시작하세요. 스윕 시작 시 생성된 스윕 ID를 제공하세요. count 매개변수에 정수 값을 전달하여 시도할 최대 실행 횟수를 설정하세요.

```python
sweep_id, count = "dtzl1o7u", 10
wandb.agent(sweep_id, count=count)
```

:::caution
같은 스크립트 또는 노트북에서 스윕 에이전트가 완료된 후 새로운 run을 시작하는 경우, 새 run을 시작하기 전에 `wandb.teardown()`을 호출해야 합니다.
:::


  </TabItem>

  <TabItem value="cli">

먼저, [`wandb sweep`](../../ref/cli/wandb-sweep.md) 명령어로 스윕을 초기화하세요. 자세한 정보는 [스윕 초기화하기](./initialize-sweeps.md)를 참조하세요.

```
wandb sweep config.yaml
```

count 플래그에 정수 값을 전달하여 시도할 최대 실행 횟수를 설정하세요.

```
NUM=10
SWEEPID="dtzl1o7u"
wandb agent --count $NUM $SWEEPID
```
  </TabItem>
</Tabs>
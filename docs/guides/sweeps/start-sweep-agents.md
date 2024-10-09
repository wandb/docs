---
title: Start or stop a sweep agent
description: 하나 이상의 머신에서 W&B 스윕 에이전트를 시작하거나 중지합니다.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

W&B Sweep을 하나 이상의 머신에서 하나 이상의 에이전트로 시작하세요. W&B Sweep 에이전트는 하이퍼파라미터를 위해 W&B Sweep(`wandb sweep`)을 초기화할 때 시작한 W&B 서버를 쿼리하고, 모델 트레이닝을 수행합니다.

W&B Sweep 에이전트를 시작하려면, W&B Sweep을 초기화할 때 반환된 W&B Sweep ID를 제공해야 합니다. W&B Sweep ID는 다음과 같은 형식입니다:

```bash
entity/project/sweep_ID
```

다음은 다음과 같습니다:

* entity: 사용자의 W&B 사용자 이름 또는 팀 이름.
* project: W&B Run의 출력을 저장할 프로젝트 이름입니다. 프로젝트를 지정하지 않으면, "미분류" 프로젝트에 저장됩니다.
* sweep_ID: W&B에서 생성한 임의의 고유 ID입니다.

Jupyter Notebook 또는 Python 스크립트 내에서 W&B Sweep 에이전트를 시작할 경우 실행할 함수의 이름을 제공하십시오.

이후의 코드조각은 W&B로 에이전트를 시작하는 방법을 보여줍니다. 이미 설정 파일이 있고 W&B Sweep을 초기화했다고 가정합니다. 설정 파일을 정의하는 방법에 대한 자세한 정보는 [Define sweep configuration](./define-sweep-configuration.md)에서 확인하세요.

<Tabs
  defaultValue="cli"
  values={[
    {label: 'CLI', value: 'cli'},
    {label: 'Python script or Jupyter Notebook', value: 'python'},
  ]}>
  <TabItem value="cli">

`sweep`를 시작하려면 `wandb agent` 코맨드를 사용하세요. sweep을 초기화할 때 반환된 sweep ID를 제공하십시오. 아래 코드조각을 복사하여 붙여넣고 `sweep_id`를 귀하의 sweep ID로 대체하세요:

```bash
wandb agent sweep_id
```
  </TabItem>
  <TabItem value="python">

W&B Python SDK 라이브러리를 사용하여 `sweep`을 시작하세요. sweep을 초기화할 때 반환된 sweep ID 및 sweep이 실행할 함수의 이름을 제공하십시오.

```python
wandb.agent(sweep_id=sweep_id, function=function_name)
```
  </TabItem>
</Tabs>

### W&B 에이전트 중지

:::caution
랜덤 및 베이지안 탐색은 계속 실행됩니다. 커맨드라인, Python 스크립트 내, 또는 [Sweeps UI](./visualize-sweep-results.md)에서 프로세스를 중지해야 합니다.
:::

선택적으로, Sweep 에이전트가 시도해야 할 W&B Run의 수를 지정할 수 있습니다. 다음 코드조각은 Jupyter Notebook, Python 스크립트 내 및 CLI에서 최대 [W&B Run](../../ref/python/run.md)의 수를 설정하는 방법을 보여줍니다.

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python script or Jupyter Notebook', value: 'python'},
    {label: 'CLI', value: 'cli'},
  ]}>
  <TabItem value="python">

먼저, sweep을 초기화하세요. 자세한 정보는 [Initialize sweeps](./initialize-sweeps.md)를 참조하세요.

```
sweep_id = wandb.sweep(sweep_config)
```

다음으로, 수위 작업을 시작합니다. 수위 초기화에서 생성된 수위 ID를 제공하십시오. 실행할 모든 run의 최대 수를 설정하기 위해 count 파라미터에 정수 값을 전달하십시오.

```python
sweep_id, count = "dtzl1o7u", 10
wandb.agent(sweep_id, count=count)
```

:::caution
같은 스크립트 또는 노트북 내에서 sweep 에이전트가 완료된 후 새로운 run을 시작하는 경우, 새로운 run을 시작하기 전에 `wandb.teardown()`을 호출해야 합니다.
:::

  </TabItem>

  <TabItem value="cli">

먼저, [`wandb sweep`](../../ref/cli/wandb-sweep.md) 명령어로 sweep을 초기화합니다. 자세한 정보는 [Initialize sweeps](./initialize-sweeps.md)를 참조하세요.

```
wandb sweep config.yaml
```

시도할 최대 run 수를 설정하기 위해 count 플래그에 정수 값을 전달합니다.

```
NUM=10
SWEEPID="dtzl1o7u"
wandb agent --count $NUM $SWEEPID
```
  </TabItem>
</Tabs>
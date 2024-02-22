---
description: Start or stop a W&B Sweep Agent on one or more machines.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 스윕 에이전트 시작하기

<head>
  <title>W&B 스윕 시작 또는 중지</title>
</head>

하나 이상의 기기에서 하나 이상의 에이전트에서 W&B 스윕을 시작합니다. W&B 스윕 에이전트는 W&B 스윕을 초기화할 때 시작한 W&B 서버(`wandb sweep`을 통해)에 쿼리하여 하이퍼파라미터를 가져와 모델 학습을 실행합니다.

W&B 스윕 에이전트를 시작하려면, W&B 스윕을 초기화할 때 반환된 W&B 스윕 ID를 제공합니다. W&B 스윕 ID는 다음과 같은 형태입니다:

```bash
entity/project/sweep_ID
```

여기서:

* entity: 귀하의 W&B 사용자 이름 또는 팀 이름입니다.
* project: W&B 실행의 결과가 저장될 프로젝트의 이름입니다. 프로젝트가 지정되지 않은 경우, 실행은 "Uncategorized" 프로젝트에 저장됩니다.
* sweep\_ID: W&B에 의해 생성된 유사 무작위, 고유 ID입니다.

Jupyter 노트북이나 Python 스크립트 내에서 W&B 스윕 에이전트를 시작할 경우 실행할 함수의 이름을 제공합니다.

다음 코드 조각은 W&B에서 에이전트를 시작하는 방법을 보여줍니다. 이미 구성 파일을 가지고 있고 W&B 스윕을 초기화했다고 가정합니다. 구성 파일을 정의하는 방법에 대한 자세한 내용은 [스윕 구성 정의하기](./define-sweep-configuration.md)를 참조하세요.

<Tabs
  defaultValue="cli"
  values={[
    {label: 'CLI', value: 'cli'},
    {label: 'Python 스크립트 또는 Jupyter 노트북', value: 'python'},
  ]}>
  <TabItem value="cli">

`wandb agent` 명령을 사용하여 스윕을 시작합니다. 스윕을 초기화할 때 반환된 스윕 ID를 제공합니다. 아래 코드 조각을 복사하여 붙여넣고 `sweep_id`를 귀하의 스윕 ID로 교체하세요:

```bash
wandb agent sweep_id
```
  </TabItem>
  <TabItem value="python">

W&B Python SDK 라이브러리를 사용하여 스윕을 시작합니다. 스윕을 초기화할 때 반환된 스윕 ID를 제공합니다. 또한, 스윕이 실행할 함수의 이름을 제공합니다.

```python
wandb.agent(sweep_id=sweep_id, function=function_name)
```
  </TabItem>
</Tabs>

### W&B 에이전트 중지

:::caution
랜덤 및 베이지안 탐색은 영원히 실행됩니다. 명령줄, 파이썬 스크립트 내부 또는 [스윕 UI](./visualize-sweep-results.md)에서 프로세스를 중지해야 합니다.
:::

스윕 에이전트가 시도할 W&B 실행의 최대 수를 선택적으로 지정합니다. 다음 코드 조각은 CLI 및 Jupyter 노트북, Python 스크립트 내에서 최대 실행 수를 설정하는 방법을 보여줍니다.

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python 스크립트 또는 Jupyter 노트북', value: 'python'},
    {label: 'CLI', value: 'cli'},
  ]}>
  <TabItem value="python">

먼저, 스윕을 초기화하세요. 자세한 내용은 [스윕 초기화하기](./initialize-sweeps.md)를 참조하세요.

```
sweep_id = wandb.sweep(sweep_config)
```

다음으로, 스윕 작업을 시작하세요. 스윕 초기화에서 생성된 스윕 ID를 제공합니다. count 매개변수에 정수 값을 전달하여 시도할 최대 실행 수를 설정합니다.

```python
sweep_id, count = "dtzl1o7u", 10
wandb.agent(sweep_id, count=count)
```

:::caution
스윕 에이전트가 종료된 후 같은 스크립트나 노트북 내에서 새로운 실행을 시작하려면, 새로운 실행을 시작하기 전에 `wandb.teardown()`을 호출해야 합니다.
:::


  </TabItem>

  <TabItem value="cli">

먼저, [`wandb sweep`](../../ref/cli/wandb-sweep.md) 명령을 사용하여 스윕을 초기화하세요. 자세한 내용은 [스윕 초기화하기](./initialize-sweeps.md)를 참조하세요.

```
wandb sweep config.yaml
```

count 플래그에 정수 값을 전달하여 시도할 최대 실행 수를 설정합니다.

```
NUM=10
SWEEPID="dtzl1o7u"
wandb agent --count $NUM $SWEEPID
```
  </TabItem>
</Tabs>
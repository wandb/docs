---
title: Resume a run
description: W&B Run 일시 중지 또는 종료 후 재개
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

run이 중지되거나 충돌할 경우 어떻게 행동해야 하는지를 지정합니다. 실행을 재개하거나 자동으로 재개할 수 있도록 하려면 해당 run과 관련된 고유한 run ID를 `id` 파라미터에 지정해야 합니다:

```python
run = wandb.init(entity="<entity>", \ 
        project="<project>", id="<run ID>", resume="<resume>")
```

:::tip
W&B는 run을 저장하려는 W&B 프로젝트의 이름을 제공할 것을 권장합니다.
:::

W&B의 응답 방식을 결정하기 위해 `resume` 파라미터에 다음 인수 중 하나를 전달합니다. 각 경우마다 W&B는 먼저 run ID가 이미 존재하는지 확인합니다.

|Argument | Description | Run ID exists| Run ID does not exist | Use case |
| --- | --- | -- | --| -- |
| `"must"` | W&B는 지정된 run ID로 run을 재개해야 합니다. | W&B는 동일한 run ID로 run을 재개합니다. | W&B는 오류를 발생시킵니다. | 동일한 run ID를 사용해야 하는 run을 재개합니다. |
| `"allow"`| run ID가 존재하는 경우 W&B가 run을 재개하도록 허용합니다. | W&B는 동일한 run ID로 run을 재개합니다. | W&B는 지정된 run ID로 새 run을 초기화합니다. | 기존의 run을 덮어쓰지 않고 run을 재개합니다. |
| `"never"`| W&B가 run ID로 지정된 run을 절대 재개할 수 없습니다. | W&B는 오류를 발생시킵니다. | W&B는 지정된 run ID로 새 run을 초기화합니다. | |

또한 `resume="auto"`를 지정하여 W&B가 자동으로 run을 다시 시작하도록 할 수 있습니다. 그러나 동일한 디렉토리에서 run을 재시작해야 함을 확인해야 합니다. 자세한 내용은 [자동 재개를 활성화](#enable-runs-to-automatically-resume) 섹션을 참조하세요.

아래 모든 예에서, `<>`로 둘러싸인 값을 자신의 것으로 변경하세요.

## 동일한 run ID를 사용해야 하는 run 재개
run이 중지되거나 충돌, 실패할 경우 동일한 run ID로 run을 재개합니다. 이를 위해 다음을 지정하여 run을 초기화합니다:

* `resume` 파라미터를 `"must"`로 설정 (`resume="must"`)
* 중지되거나 충돌한 run의 run ID를 제공

다음 코드조각은 W&B Python SDK로 이를 수행하는 방법을 보여줍니다:

```python
run = wandb.init(entity="<entity>", \ 
        project="<project>", id="<run ID>", resume="must")
```

:::caution
동일한 `id`를 여러 프로세스가 동시에 사용할 경우, 예기치 않은 결과가 발생할 수 있습니다.

여러 프로세스를 관리하는 방법에 대한 자세한 내용은 [분산 트레이닝 실험 로그](../track/log/distributed-training.md)를 참조하십시오.
:::

## 기존 run을 덮어쓰지 않고 run 재개
중지되거나 충돌한 run을 기존 run을 덮어쓰지 않고 재개합니다. 이는 프로세스가 성공적으로 종료되지 않을 때 특히 유용합니다. 다음에 W&B를 시작하면 W&B는 마지막 단계부터 로그를 시작합니다.

run을 W&B로 초기화할 때 `resume` 파라미터를 `"allow"`로 설정합니다 (`resume="allow"`). 중지되거나 충돌한 run의 run ID를 제공합니다. 다음 코드조각은 W&B Python SDK로 이를 수행하는 방법을 보여줍니다:

```python
import wandb

run = wandb.init(entity="<entity>", \ 
        project="<project>", id="<run ID>", resume="allow")
```

## run의 자동 재개 활성화
다음 코드조각은 Python SDK 또는 환경 변수로 run이 자동으로 재개되도록 설정하는 방법을 보여줍니다.

<Tabs
  defaultValue="python"
  values={[
    {label: 'W&B Python SDK', value: 'python'},
    {label: 'Shell script', value: 'bash'},
  ]}>
  <TabItem value="python">

다음 코드조각은 Python SDK로 W&B run ID를 지정하는 방법을 보여줍니다.

`<>`로 둘러싸인 값을 자신의 것으로 변경하세요:

```python
run = wandb.init(entity="<entity>", \ 
        project="<project>", id="<run ID>", resume="<resume>")
```

  </TabItem>
  <TabItem value="bash">

다음 예시는 bash 스크립트에서 W&B `WANDB_RUN_ID` 변수를 지정하는 방법을 보여줍니다:

```bash title="run_experiment.sh"
RUN_ID="$1"

WANDB_RESUME=allow WANDB_RUN_ID="$RUN_ID" python eval.py
```
터미널 내에서 W&B run ID와 함께 셸 스크립트를 실행할 수 있습니다. 다음 코드조각은 run ID `akj172`를 전달합니다:

```bash
sh run_experiment.sh akj172 
```

  </TabItem>
</Tabs>

:::important
자동 재개는 프로세스가 실패한 프로세스와 동일한 파일 시스템 상에서 다시 시작되는 경우에만 작동합니다.
:::

예를 들어, `Users/AwesomeEmployee/Desktop/ImageClassify/training/`이라는 `directory` 안에서 `train.py`라는 python 스크립트를 실행한다고 가정하겠습니다. `train.py` 내에서 run은 자동 재개가 가능하도록 설정됩니다. 다음으로 트레이닝 스크립트가 중단되었다고 가정합시다. 이 run을 재개하려면, `train.py` 스크립트를 `Users/AwesomeEmployee/Desktop/ImageClassify/training/` 내에서 다시 시작해야 합니다.

:::tip
파일 시스템을 공유할 수 없는 경우, `WANDB_RUN_ID` 환경 변수를 지정하거나 W&B Python SDK로 run ID를 전달하십시오. run ID에 대한 자세한 내용은 "What are runs?" 페이지의 [Create a run](./intro.md#create-a-run) 섹션을 참조하세요.
:::

## 중단 가능한 Sweeps run 재개
중단된 [sweep](../sweeps/intro.md) run을 자동으로 다시 대기열에 추가합니다. 이것은 특히 SLURM 작업의 중단 가능한 대기열, EC2 스팟 인스턴스, 또는 Google 클라우드 중단 가능한 VM과 같은 중단 가능성이 있는 컴퓨팅 환경에서 sweep 에이전트를 실행할 때 유용합니다.

W&B가 중단된 sweep run을 자동으로 다시 대기열에 추가할 수 있도록 [`mark_preempting`](../../ref/python/run/#mark_preempting) 함수를 사용합니다. 예를 들어, 다음 코드조각

```python
run = wandb.init()  # Run을 초기화합니다
run.mark_preempting()
```
다음 표는 sweep run의 종료 상태에 따라 W&B가 run을 처리하는 방식을 설명합니다.

|Status| Behavior |
|------| ---------|
|Status code 0| Run이 성공적으로 종료된 것으로 간주되어 다시 대기열에 추가되지 않습니다.|
|Nonzero status| W&B는 자동으로 run을 sweep과 연결된 run 대기열에 추가합니다.|
|No status| Run은 sweep run 대기열에 추가됩니다. Sweep 에이전트는 대기열이 비워질 때까지 run을 소비합니다. 대기열이 비워지면 sweep 대기열은 sweep 검색 알고리즘에 기반하여 새로운 run을 생성하기 시작합니다.|`

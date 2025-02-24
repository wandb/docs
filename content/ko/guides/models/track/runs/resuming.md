---
title: Resume a run
description: 일시 중단되거나 종료된 W&B Run 다시 시작하기
menu:
  default:
    identifier: ko-guides-models-track-runs-resuming
    parent: what-are-runs
---

run이 중지되거나 충돌할 경우 run이 어떻게 작동해야 하는지 지정합니다. run을 재개하거나 자동으로 재개하도록 설정하려면 `id` 파라미터에 대해 해당 run과 연결된 고유한 run ID를 지정해야 합니다.

```python
run = wandb.init(entity="<entity>", \ 
        project="<project>", id="<run ID>", resume="<resume>")
```

{{% alert %}}
W&B는 run을 저장할 W&B **Project** 이름을 제공할 것을 권장합니다.
{{% /alert %}}

W&B가 어떻게 응답해야 하는지 결정하기 위해 다음 인수 중 하나를 `resume` 파라미터에 전달합니다. 각 경우에 W&B는 먼저 run ID가 이미 존재하는지 확인합니다.

|인수 | 설명 | Run ID 존재| Run ID 존재하지 않음 | 유스 케이스 |
| --- | --- | -- | --| -- |
| `"must"` | W&B는 run ID로 지정된 run을 반드시 재개해야 합니다. | W&B는 동일한 run ID로 run을 재개합니다. | W&B에서 오류를 발생시킵니다. | 동일한 run ID를 사용해야 하는 run을 재개합니다.  |
| `"allow"`| W&B가 run ID가 존재하는 경우 run을 재개하도록 허용합니다. | W&B는 동일한 run ID로 run을 재개합니다. | W&B는 지정된 run ID로 새 run을 초기화합니다. | 기존 run을 덮어쓰지 않고 run을 재개합니다. |
| `"never"`| W&B가 run ID로 지정된 run을 재개하도록 허용하지 않습니다. | W&B에서 오류를 발생시킵니다. | W&B는 지정된 run ID로 새 run을 초기화합니다. | |

`resume="auto"`를 지정하여 W&B가 자동으로 run을 다시 시작하도록 할 수도 있습니다. 그러나 동일한 디렉토리에서 run을 다시 시작해야 합니다. 자세한 내용은 [run이 자동으로 재개되도록 설정]({{< relref path="#enable-runs-to-automatically-resume" lang="ko" >}}) 섹션을 참조하십시오.

아래의 모든 예제에서 `<>`로 묶인 값을 사용자 고유의 값으로 바꾸십시오.

## 동일한 run ID를 사용해야 하는 run 재개
run이 중지되거나, 충돌하거나, 실패하는 경우 동일한 run ID를 사용하여 재개할 수 있습니다. 이렇게 하려면 run을 초기화하고 다음을 지정합니다.

* `resume` 파라미터를 `"must"`(`resume="must"`)로 설정합니다.
* 중지되거나 충돌한 run의 run ID를 제공합니다.

다음 코드 조각은 W&B Python SDK로 이를 수행하는 방법을 보여줍니다.

```python
run = wandb.init(entity="<entity>", \ 
        project="<project>", id="<run ID>", resume="must")
```

{{% alert color="secondary" %}}
여러 **process**가 동시에 동일한 `id`를 사용하는 경우 예기치 않은 결과가 발생합니다.

여러 **process**를 관리하는 방법에 대한 자세한 내용은 [분산 트레이닝 **Experiments** 로깅]({{< relref path="/guides/models/track/log/distributed-training.md" lang="ko" >}})을 참조하십시오.
{{% /alert %}}

## 기존 run을 덮어쓰지 않고 run 재개
기존 run을 덮어쓰지 않고 중지되거나 충돌한 run을 재개합니다. 이는 **process**가 성공적으로 종료되지 않은 경우 특히 유용합니다. 다음에 W&B를 시작하면 W&B는 마지막 단계부터 로깅을 시작합니다.

W&B로 run을 초기화할 때 `resume` 파라미터를 `"allow"`(`resume="allow"`)로 설정합니다. 중지되거나 충돌한 run의 run ID를 제공합니다. 다음 코드 조각은 W&B Python SDK로 이를 수행하는 방법을 보여줍니다.

```python
import wandb

run = wandb.init(entity="<entity>", \ 
        project="<project>", id="<run ID>", resume="allow")
```

## run이 자동으로 재개되도록 설정
다음 코드 조각은 Python SDK 또는 환경 변수를 사용하여 run이 자동으로 재개되도록 설정하는 방법을 보여줍니다.

{{< tabpane text=true >}}
  {{% tab header="W&B Python SDK" %}}
다음 코드 조각은 Python SDK로 W&B run ID를 지정하는 방법을 보여줍니다.

`<>`로 묶인 값을 사용자 고유의 값으로 바꿉니다.

```python
run = wandb.init(entity="<entity>", \ 
        project="<project>", id="<run ID>", resume="<resume>")
```  
  {{% /tab %}}
  {{% tab header="Shell script" %}}

다음 예제는 bash 스크립트에서 W&B `WANDB_RUN_ID` 변수를 지정하는 방법을 보여줍니다.

```bash title="run_experiment.sh"
RUN_ID="$1"

WANDB_RESUME=allow WANDB_RUN_ID="$RUN_ID" python eval.py
```
**터미널** 내에서 W&B run ID와 함께 셸 스크립트를 실행할 수 있습니다. 다음 코드 조각은 run ID `akj172`를 전달합니다.

```bash
sh run_experiment.sh akj172 
```

{{% /tab %}}
{{< /tabpane >}}

{{% alert color="secondary" %}}
자동 재개는 **process**가 실패한 **process**와 동일한 파일 시스템 위에서 다시 시작되는 경우에만 작동합니다.
{{% /alert %}}

예를 들어 `Users/AwesomeEmployee/Desktop/ImageClassify/training/` 디렉토리에서 `train.py`라는 Python 스크립트를 실행한다고 가정합니다. `train.py` 내에서 스크립트는 자동 재개를 활성화하는 run을 만듭니다. 다음으로 트레이닝 스크립트가 중지되었다고 가정합니다. 이 run을 재개하려면 `Users/AwesomeEmployee/Desktop/ImageClassify/training/` 내에서 `train.py` 스크립트를 다시 시작해야 합니다.

{{% alert %}}
파일 시스템을 공유할 수 없는 경우 `WANDB_RUN_ID` 환경 변수를 지정하거나 W&B Python SDK로 run ID를 전달합니다. run ID에 대한 자세한 내용은 "run이란 무엇입니까?" 페이지의 [사용자 지정 run ID]({{< relref path="./#custom-run-ids" lang="ko" >}}) 섹션을 참조하십시오.
{{% /alert %}}

## 선점형 Sweeps run 재개
중단된 [**sweep**]({{< relref path="/guides/models/sweeps/" lang="ko" >}}) run을 자동으로 다시 대기열에 넣습니다. 이는 선점형 대기열의 SLURM 작업, EC2 스팟 인스턴스 또는 Google Cloud 선점형 VM과 같이 선점이 적용되는 컴퓨팅 환경에서 스윕 에이전트를 실행하는 경우 특히 유용합니다.

[`mark_preempting`]({{< relref path="/ref/python/run.md#mark_preempting" lang="ko" >}}) 함수를 사용하여 W&B가 중단된 **sweep** run을 자동으로 다시 대기열에 넣도록 설정합니다. 예를 들어 다음 코드 조각은 다음과 같습니다.

```python
run = wandb.init()  # run 초기화
run.mark_preempting()
```
다음 표는 W&B가 **sweep** run의 종료 상태를 기반으로 run을 처리하는 방법을 간략하게 설명합니다.

|상태| 행동 |
|------| ---------|
|상태 코드 0| run이 성공적으로 종료된 것으로 간주되며 다시 대기열에 넣지 않습니다.  |
|0이 아닌 상태| W&B는 run을 **sweep**과 연결된 run 대기열에 자동으로 추가합니다.|
|상태 없음| run이 **sweep** run 대기열에 추가됩니다. **Sweep agent**는 대기열이 비워질 때까지 run 대기열에서 run을 소비합니다. 대기열이 비워지면 **sweep** 대기열은 **sweep** 검색 알고리즘을 기반으로 새 run 생성을 재개합니다.|

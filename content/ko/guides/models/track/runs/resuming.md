---
title: run 재개하기
description: 일시 중지되거나 종료된 W&B run 다시 시작하기
menu:
  default:
    identifier: ko-guides-models-track-runs-resuming
    parent: what-are-runs
---

run이 중지되거나 크래시가 발생할 때 어떻게 동작할지 지정할 수 있습니다. run을 재개하거나 자동으로 재개하도록 하려면 해당 run에 연결된 고유한 run ID를 `id` 파라미터에 지정해야 합니다:

```python
run = wandb.init(entity="<entity>", \ 
        project="<project>", id="<run ID>", resume="<resume>")
```

{{% alert %}}
W&B는 run을 저장하려는 W&B Project 이름을 지정할 것을 권장합니다.
{{% /alert %}}

`resume` 파라미터에 다음 인수 중 하나를 전달하여 W&B의 동작 방식을 결정할 수 있습니다. 각 경우에 W&B는 먼저 run ID가 이미 존재하는지 확인합니다.

|인수 | 설명 | run ID 존재 시| run ID 미존재 시 | 유스 케이스 |
| --- | --- | -- | --| -- |
| `"must"` | 지정된 run ID로 반드시 run을 재개해야 합니다. | 동일한 run ID로 run을 재개합니다. | 에러를 발생시킵니다. | 반드시 같은 run ID를 사용해 run을 재개해야 할 때 사용합니다. |
| `"allow"`| run ID가 존재하면 run을 재개합니다. | 동일한 run ID로 run을 재개합니다. | 지정한 run ID로 새로운 run을 초기화합니다. | 기존 run을 덮어쓰지 않고 run을 재개할 때 사용합니다. |
| `"never"`| run ID로 지정된 run을 절대 재개하지 않습니다. | 에러를 발생시킵니다. | 지정한 run ID로 새로운 run을 초기화합니다. | |

또한 `resume="auto"`로 지정하면 W&B가 자동으로 run을 재시작하려고 시도합니다. 다만, 동일한 디렉토리에서 run을 재시작해야 합니다. 자세한 내용은 [run을 자동으로 재개 활성화]({{< relref path="#enable-runs-to-automatically-resume" lang="ko" >}}) 섹션을 참고하세요.

아래 예시 전체에서 `<>`로 감싸진 값을 본인의 값으로 변경하세요.

## 반드시 같은 run ID로 run 재개하기
run이 중지되거나 크래시 혹은 실패한 경우, 동일한 run ID로 재개할 수 있습니다. 이를 위해서는 run을 초기화할 때 다음과 같이 설정하세요:

* `resume` 파라미터를 `"must"` (`resume="must"`)로 설정
* 중지되었거나 크래시가 발생한 run의 run ID를 입력

아래 코드조각은 W&B Python SDK에서 이를 구현하는 방법을 보여줍니다:

```python
run = wandb.init(entity="<entity>", \ 
        project="<project>", id="<run ID>", resume="must")
```

{{% alert color="secondary" %}}
여러 프로세스가 동시에 동일한 `id`를 사용하면 예기치 않은 결과가 발생할 수 있습니다.

여러 프로세스 관리를 위한 자세한 내용은 [분산 트레이닝 실험 로깅]({{< relref path="/guides/models/track/log/distributed-training.md" lang="ko" >}})을 참고하세요.
{{% /alert %}}

## 기존 run을 덮어쓰지 않고 run 재개하기
중지되었거나 크래시가 발생한 run을 기존 run을 덮어쓰지 않고 재개할 수 있습니다. 프로세스가 정상적으로 종료되지 않는 경우 특히 유용합니다. 다음 번 W&B를 시작할 때, W&B는 마지막 step부터 로그를 다시 작성합니다.

run을 초기화할 때 `resume` 파라미터를 `"allow"` (`resume="allow"`)로 설정하세요. 그리고 중지되거나 크래시된 run의 run ID를 입력하세요. 아래 코드조각은 W&B Python SDK에서 이를 구현하는 방법입니다:

```python
import wandb

run = wandb.init(entity="<entity>", \ 
        project="<project>", id="<run ID>", resume="allow")
```

## run을 자동으로 재개하도록 설정하기
아래 코드조각은 Python SDK 또는 환경 변수로 run의 자동 재개를 설정하는 방법을 보여줍니다.

{{< tabpane text=true >}}
  {{% tab header="W&B Python SDK" %}}
아래 코드조각은 Python SDK에서 W&B run ID를 지정하는 방법입니다.

`<>`로 감싸진 값을 본인의 값으로 교체하세요:

```python
run = wandb.init(entity="<entity>", \ 
        project="<project>", id="<run ID>", resume="<resume>")
```
  {{% /tab %}}
  {{% tab header="Shell script" %}}

아래 예시는 bash 스크립트에서 W&B `WANDB_RUN_ID` 변수를 지정하는 방법을 보여줍니다:

```bash title="run_experiment.sh"
RUN_ID="$1"

WANDB_RESUME=allow WANDB_RUN_ID="$RUN_ID" python eval.py
```
터미널에서, 스크립트와 함께 W&B run ID를 넘겨 실행할 수 있습니다. 다음 코드조각은 run ID `akj172`를 전달하는 예입니다:

```bash
sh run_experiment.sh akj172 
```

{{% /tab %}}
{{< /tabpane >}}

{{% alert color="secondary" %}}
자동 재개 기능은 실패한 프로세스와 동일한 파일시스템에서 프로세스를 재시작할 때만 동작합니다.
{{% /alert %}}

예를 들어, `Users/AwesomeEmployee/Desktop/ImageClassify/training/` 디렉토리에서 `train.py`라는 python 스크립트를 실행한다고 가정해보겠습니다. `train.py` 내에서 run을 생성하며 자동 재개를 활성화합니다. 이후 트레이닝 스크립트가 중지되었다면, run을 재개하려면 반드시 `Users/AwesomeEmployee/Desktop/ImageClassify/training/` 디렉토리 내에서 다시 `train.py`를 실행해야 합니다.

{{% alert %}}
파일시스템을 공유할 수 없다면, `WANDB_RUN_ID` 환경 변수를 지정하거나 W&B Python SDK로 run ID를 넘겨주세요. run ID에 대한 자세한 내용은 "What are runs?" 페이지 내 [Custom run IDs]({{< relref path="./#custom-run-ids" lang="ko" >}}) 섹션을 확인하세요.
{{% /alert %}}

## Preemptible Sweeps run 재개하기
중단된 [sweep]({{< relref path="/guides/models/sweeps/" lang="ko" >}}) run을 자동으로 재큐잉 할 수 있습니다. 이는 SLURM 프리엠티브 큐의 작업, EC2 spot 인스턴스, Google Cloud preemptible VM 등 중단될 수 있는 컴퓨팅 환경에서 sweep agent를 실행할 때 특히 유용합니다.

[`mark_preempting`]({{< relref path="/ref/python/sdk/classes/run#mark_preempting" lang="ko" >}}) 함수를 사용하여 중단된 sweep run을 자동으로 큐에 다시 등록할 수 있습니다. 예시:

```python
run = wandb.init()  # run 초기화
run.mark_preempting()
```
아래 표는 sweep run의 종료 상태에 따라 W&B가 run을 어떻게 처리하는지 정리한 것입니다.

|상태| 동작 |
|------| ---------|
|상태 코드 0| run이 정상적으로 종료된 것으로 간주하고 재큐잉 하지 않습니다.  |
|0이 아닌 상태| W&B가 해당 run을 sweep에 연결된 run 큐에 자동으로 추가합니다.|
|상태 없음| run이 sweep run 큐에 추가됩니다. sweep 에이전트는 큐가 빌 때까지 run 큐에서 run을 소모합니다. 큐가 비면 sweep 큐에서 sweep search 알고리즘에 따라 새로운 run을 계속 생성합니다.|
---
title: run이란 무엇인가요?
description: W&B의 기본 구성 요소인 Run에 대해 알아보세요.
cascade:
- url: guides/runs/:filename
menu:
  default:
    identifier: ko-guides-models-track-runs-_index
    parent: experiments
url: guides/runs
weight: 5
---

*run*은 W&B에서 로그된 하나의 연산 단위입니다. W&B Run은 전체 프로젝트의 원자적인 요소로 생각할 수 있습니다. 즉, 각 run은 모델 트레이닝 후 결과를 기록하거나, 하이퍼파라미터 스윕, 기타 연산 등 특정 연산 과정을 기록한 하나의 레코드입니다.

run을 시작하는 일반적인 패턴은 다음과 같습니다(이외에도 다양함):

* 모델을 트레이닝할 때
* 하이퍼파라미터를 변경해 새로운 experiment를 수행할 때
* 다른 모델로 새로운 기계학습 experiment를 진행할 때
* 데이터를 로그하거나 모델을 [W&B Artifact]({{< relref path="/guides/core/artifacts/" lang="ko" >}})로 저장할 때
* [W&B Artifact 다운로드]({{< relref path="/guides/core/artifacts/download-and-use-an-artifact.md" lang="ko" >}}) 시

W&B는 생성한 run들을 [*projects*]({{< relref path="/guides/models/track/project-page.md" lang="ko" >}})에 저장합니다. W&B App의 프로젝트 워크스페이스에서 run과 그 속성을 확인할 수 있습니다. 또는 [`wandb.Api.Run`]({{< relref path="/ref/python/sdk/classes/run.md" lang="ko" >}}) 객체를 사용해 run 속성에 프로그래밍적으로 엑세스할 수도 있습니다.

`wandb.Run.log()`로 기록한 모든 내용은 해당 run에 저장됩니다.

```python
import wandb

entity = "nico"  # 본인의 W&B entity로 대체하세요
project = "awesome-project"

with wandb.init(entity=entity, project=project) as run:
    run.log({"accuracy": 0.9, "loss": 0.1})
```

첫 번째 줄은 W&B Python SDK를 임포트합니다. 두 번째 줄은 `awesome-project`라는 프로젝트에서 `nico`라는 entity로 run을 초기화합니다. 세 번째 줄은 모델의 accuracy와 loss를 해당 run에 기록합니다.

터미널에서 다음과 같이 출력됩니다:

```bash
wandb: Syncing run earnest-sunset-1
wandb: ⭐️ View project at https://wandb.ai/nico/awesome-project
wandb: 🚀 View run at https://wandb.ai/nico/awesome-project/runs/1jx1ud12
wandb:                                                                                
wandb: 
wandb: Run history:
wandb: accuracy ▁
wandb:     loss ▁
wandb: 
wandb: Run summary:
wandb: accuracy 0.9
wandb:     loss 0.5
wandb: 
wandb: 🚀 View run earnest-sunset-1 at: https://wandb.ai/nico/awesome-project/runs/1jx1ud12
wandb: ⭐️ View project at: https://wandb.ai/nico/awesome-project
wandb: Synced 6 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20241105_111006-1jx1ud12/logs
```

터미널에 출력된 URL은 W&B App UI에서 해당 run의 워크스페이스로 이동합니다. 워크스페이스에 생성된 패널은 이 단일 포인트에 해당됩니다.

{{< img src="/images/runs/single-run-call.png" alt="Single run workspace" >}}

메트릭을 단 한 번만 기록하는 것은 그리 유용하지 않을 수 있습니다. 트레이닝 과정을 시뮬레이션하는 좀 더 현실적인 예시는 메트릭을 일정 간격으로 기록하는 경우입니다. 다음 코드를 참고하세요:

```python
import wandb
import random

config = {
    "epochs": 10,
    "learning_rate": 0.01,
}

with wandb.init(project="awesome-project", config=config) as run:
    print(f"lr: {config['learning_rate']}")
      
    # 트레이닝 run 시뮬레이션
    for epoch in range(config['epochs']):
      offset = random.random() / 5
      acc = 1 - 2**-epoch - random.random() / (epoch + 1) - offset
      loss = 2**-epoch + random.random() / (epoch + 1) + offset
      print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
      run.log({"accuracy": acc, "loss": loss})
```

실행 결과는 다음과 같습니다:

```bash
wandb: Syncing run jolly-haze-4
wandb: ⭐️ View project at https://wandb.ai/nico/awesome-project
wandb: 🚀 View run at https://wandb.ai/nico/awesome-project/runs/pdo5110r
lr: 0.01
epoch=0, accuracy=-0.10070974957523078, loss=1.985328507123956
epoch=1, accuracy=0.2884687745057535, loss=0.7374362314407752
epoch=2, accuracy=0.7347387967382066, loss=0.4402409835486663
epoch=3, accuracy=0.7667969248039795, loss=0.26176963846423457
epoch=4, accuracy=0.7446848791003173, loss=0.24808611724405083
epoch=5, accuracy=0.8035095836268268, loss=0.16169791827329466
epoch=6, accuracy=0.861349032371624, loss=0.03432578493587426
epoch=7, accuracy=0.8794926436276016, loss=0.10331872172219471
epoch=8, accuracy=0.9424839917077272, loss=0.07767793473500445
epoch=9, accuracy=0.9584880427028566, loss=0.10531971149250456
wandb: 🚀 View run jolly-haze-4 at: https://wandb.ai/nico/awesome-project/runs/pdo5110r
wandb: Find logs at: wandb/run-20241105_111816-pdo5110r/logs
```

트레이닝 스크립트는 `wandb.Run.log()`를 10회 호출합니다. 매번 해당 에포크의 accuracy와 loss가 W&B에 기록됩니다. 앞선 출력 값에서 W&B가 보여주는 URL을 클릭하면, W&B App UI의 해당 run 워크스페이스로 이동합니다.

W&B는 이 시뮬레이션 트레이닝 루프 전체를 `jolly-haze-4`라는 하나의 run에 기록합니다. 이는 스크립트에서 `wandb.init()` 메소드를 단 한 번만 호출했기 때문입니다.

{{< img src="/images/runs/run_log_example_2.png" alt="Training run with logged metrics" >}}

또 다른 예시로, [sweep]({{< relref path="/guides/models/sweeps/" lang="ko" >}}) 도중에는, W&B가 지정한 하이퍼파라미터 탐색 공간을 여러 조합으로 실험합니다. sweep이 생성하는 각 새로운 하이퍼파라미터 조합은 고유한 run으로 실행됩니다.


## W&B Run 초기화

[`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init" lang="ko" >}})를 사용해 W&B Run을 초기화하세요. 아래 샘플 코드에서는 W&B Python SDK를 임포트하고 run을 초기화하는 방법을 보여줍니다.

꺾쇠 괄호(`< >`)로 표시된 값을 본인 값으로 바꿔주세요:

```python
import wandb

with wandb.init(entity="<entity>", project="<project>") as run:
    # 여기에 본인 코드를 작성하세요
```

run을 초기화할 때, W&B는 해당 run을 지정한 프로젝트 필드(즉, `wandb.init(project="<project>"`)의 프로젝트에 기록합니다. 지정한 프로젝트가 없다면 새 프로젝트를 만듭니다. 이미 존재하는 프로젝트명이라면 해당 프로젝트에 run을 저장합니다.

{{% alert %}}
프로젝트명을 지정하지 않으면 W&B는 run을 `Uncategorized`라는 프로젝트에 저장합니다.
{{% /alert %}}

W&B의 각 run은 [*run ID*로 알려진 고유 식별자]({{< relref path="#unique-run-identifiers" lang="ko" >}})를 가집니다. [직접 고유 ID를 지정할 수도 있고]({{< relref path="#unique-run-identifiers" lang="ko" >}}), [W&B가 무작위로 생성하도록 할 수도 있습니다]({{< relref path="#autogenerated-run-ids" lang="ko" >}}).

각 run은 또한 사람이 읽기 편한 비고유 [run 이름]({{< relref path="#name-your-run" lang="ko" >}})도 가집니다. run 이름을 직접 지정하거나, 랜덤으로 생성되도록 둘 수도 있습니다. 초기화 이후에도 run 이름은 변경할 수 있습니다.

예를 들면 아래와 같습니다:

```python title="basic.py"
import wandb

run = wandb.init(entity="wandbee", project="awesome-project")
```
이 코드를 실행하면 다음과 같은 결과가 나옵니다:

```bash
🚀 View run exalted-darkness-6 at: 
https://wandb.ai/nico/awesome-project/runs/pgbn9y21
Find logs at: wandb/run-20241106_090747-pgbn9y21/logs
```

위 코드에서 id 파라미터를 지정하지 않았기 때문에 W&B가 고유 run ID를 생성합니다. 여기서 `nico`는 run을 기록한 entity, `awesome-project`는 run이 기록된 프로젝트 이름, `exalted-darkness-6`은 run 이름, 그리고 `pgbn9y21`은 run ID입니다.

{{% alert title="Notebook users" %}}
run의 마지막에 `run.finish()`를 호출하여 run이 종료되었음을 명시하세요. 이를 통해 run이 정상적으로 프로젝트에 기록되고, 백그라운드에 남아있지 않도록 할 수 있습니다.

```python title="notebook.ipynb"
import wandb

run = wandb.init(entity="<entity>", project="<project>")
# 트레이닝 코드, 로그 기록 등 
run.finish()
```
{{% /alert %}}

[run을 그룹화]({{< relref path="grouping.md" lang="ko" >}})하여 experiments로 묶을 경우, 특정 run을 그룹 내/외로 이동시킬 수 있고, 그룹 간 이동도 가능합니다.

각 run에는 현재 상태(state)가 있습니다. 전체 가능한 상태는 [Run states]({{< relref path="#run-states" lang="ko" >}})에서 확인하세요.

## Run states
다음 표는 run이 가질 수 있는 상태를 설명합니다:

| 상태 | 설명 |
| ----- | ----- |
| `Crashed` | run의 내부 프로세스에서 heartbeat 신호가 끊긴 것. 머신이 충돌했을 때 발생할 수 있습니다. | 
| `Failed` | run이 비정상 종료(exit status가 0이 아님)로 끝났을 때. | 
| `Finished`| run이 종료되었고 모든 데이터가 동기화되었거나, `wandb.Run.finish()`가 호출됨. |
| `Killed` | run이 강제로 중단됨. |
| `Running` | run이 아직 실행 중이며 최근 heartbeat 신호를 정상적으로 전송함.  |


## 고유 run 식별자

Run ID는 각 run을 고유하게 식별하는 값입니다. 기본적으로, 새 run을 초기화할 때 [W&B가 임의이자 고유한 run ID를 생성]({{< relref path="#autogenerated-run-ids" lang="ko" >}})합니다. 직접 [고유 run ID를 지정할 수도 있습니다]({{< relref path="#custom-run-ids" lang="ko" >}}).

### 자동 생성 run ID

run을 초기화할 때 ID를 직접 지정하지 않으면, W&B가 무작위로 run ID를 생성합니다. W&B App에서 해당 run의 고유 ID를 확인할 수 있습니다.

1. [W&B App](https://wandb.ai/home)으로 이동합니다.
2. 해당 run을 초기화할 때 지정한 프로젝트로 이동합니다.
3. 프로젝트 워크스페이스의 **Runs** 탭을 선택합니다.
4. **Overview** 탭을 선택합니다.

W&B는 **Run path** 필드에 run의 고유 ID를 표시합니다. Run path는 팀 이름, 프로젝트명, run ID로 구성되며, 고유 ID는 run path의 맨 마지막 부분입니다.

아래 이미지에서 고유 run ID는 `9mxi1arc`입니다:

{{< img src="/images/runs/unique-run-id.png" alt="Run ID location" >}}


### 커스텀 run ID 사용
[`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init" lang="ko" >}}) 메소드에 `id` 파라미터를 전달해서 직접 run ID를 지정할 수 있습니다. 

```python 
import wandb

run = wandb.init(entity="<project>", project="<project>", id="<run-id>")
```

run의 고유 ID를 사용하여 W&B App에서 해당 run의 개요 페이지로 바로 이동할 수 있습니다. URL 구조는 다음과 같습니다:

```text title="특정 run을 위한 W&B App URL"
https://wandb.ai/<entity>/<project>/<run-id>
```

꺽쇠 괄호(`< >`)로 표시된 값들은 실제 entity, project, run ID 값으로 대체하세요.

## run 이름 지정하기
run의 이름은 사람이 읽기 쉬운, 유일하지 않은 식별자입니다. 

기본적으로 W&B는 run을 새로 초기화할 때 임의의 이름을 생성합니다. run의 이름은 프로젝트 워크스페이스와 [run overview 페이지]({{< relref path="#overview-tab" lang="ko" >}}) 상단에 나타납니다.

{{% alert %}}
run 이름은 프로젝트 워크스페이스에서 run을 빠르게 식별하는 데 유용하게 활용할 수 있습니다.
{{% /alert %}}

run 이름을 직접 지정하려면 [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init" lang="ko" >}}) 메소드의 `name` 파라미터에 전달하세요.

```python 
import wandb

with wandb.init(entity="<project>", project="<project>", name="<run-name>") as run:
    # 여기에 본인 코드를 작성하세요
```

### run 이름 변경하기

run을 초기화한 후에도 워크스페이스나 **Runs** 페이지에서 이름을 변경할 수 있습니다.

1. W&B 프로젝트 페이지로 이동합니다.
1. 프로젝트 사이드바에서 **Workspace** 또는 **Runs** 탭을 선택합니다.
1. 이름을 바꿀 run을 검색하거나 스크롤해서 찾습니다.

    run 이름 위에 마우스를 올리고 세로 점 3개(⋮) 아이콘을 클릭한 뒤, 다음 중 원하는 범위를 선택합니다:
    - **프로젝트에 대해 run 이름 변경**: 프로젝트 전체에서 run 이름이 변경됩니다.
    - **해당 워크스페이스에서만 run 이름 변경**: 해당 워크스페이스에서만 이름이 바뀝니다.
1. 새 run 이름을 입력합니다. 랜덤 이름을 다시 생성하려면 입력 칸을 비워두세요.
1. 폼을 제출하면, 변경된 run 이름이 표시됩니다. 워크스페이스 내에서 커스텀 이름을 부여한 run 옆엔 정보 아이콘이 뜨며, 마우스를 올리면 상세 정보가 표시됩니다.

[report]({{< relref path="/guides/core/reports/edit-a-report.md" lang="ko" >}})의 run set에서도 run 이름을 바꿀 수 있습니다:

1. 리포트에서 연필 아이콘을 클릭해 리포트 에디터를 엽니다.
1. run set에서 이름을 변경할 run을 찾고, 해당 이름에 마우스를 올린 뒤 세로 점 3개(⋮)를 클릭한 후 아래 중 하나를 선택하세요:

  - **프로젝트에 대해 run 이름 변경**: 전체 프로젝트에서 run 이름이 변경됩니다. 랜덤 이름을 새로 생성하려면 입력 칸을 비워두세요.
  - **해당 패널 그리드에서만 run 이름 변경**: 리포트 내에서만 이름이 바뀌며, 다른 곳에서는 기존 이름이 유지됩니다. 이 경우 랜덤 이름으로 변경은 지원되지 않습니다.

  폼을 제출하세요.
1. **Publish report**를 클릭하세요.

## run에 노트 작성하기
특정 run에 달아둔 노트는 run 페이지의 **Overview** 탭과 프로젝트 페이지의 run 테이블에 표시됩니다.

1. W&B 프로젝트로 이동
2. 프로젝트 사이드바에서 **Workspace** 탭 선택
3. run 셀렉터에서 노트를 추가할 run을 선택
4. **Overview** 탭 선택
5. **Description** 필드 옆 연필 아이콘을 눌러 노트 추가

## run 중지하기
run은 W&B App이나 프로그래밍적으로 중지할 수 있습니다.

{{< tabpane text=true >}}
  {{% tab header="프로그래밍 방식" %}}
1. run을 초기화한 터미널이나 코드 에디터로 이동합니다.
2. `Ctrl+D`를 눌러 run을 중지합니다.

예를 들어, 위 설명대로 하면 터미널은 아래와 유사하게 보입니다:

```bash
KeyboardInterrupt
wandb: 🚀 View run legendary-meadow-2 at: https://wandb.ai/nico/history-blaster-4/runs/o8sdbztv
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 1 other file(s)
wandb: Find logs at: ./wandb/run-20241106_095857-o8sdbztv/logs
```

run이 더 이상 활성화 상태가 아닌지 W&B App에서 확인하려면:

1. 해당 run이 기록된 프로젝트로 이동합니다.
2. 중지한 run의 이름을 선택  
  {{% alert %}}
  중지한 run의 이름은 터미널이나 코드 에디터 출력에서 확인할 수 있습니다. 예: 위의 예시에서는 run 이름이 `legendary-meadow-2`입니다.
  {{% /alert %}}
3. 프로젝트 사이드바에서 **Overview** 탭 선택

**State** 필드 옆 run의 상태가 `running`에서 `Killed`로 바뀐 것을 확인하세요.

{{< img src="/images/runs/stop-run-terminal.png" alt="Run stopped via terminal" >}}  
  {{% /tab %}}
  {{% tab header="W&B App" %}}

1. 해당 run이 기록되는 프로젝트로 이동합니다.
2. run 셀렉터에서 중지할 run을 선택하세요.
3. 프로젝트 사이드바에서 **Overview** 탭 선택
4. **State** 필드 옆의 상단 버튼을 클릭합니다.
{{< img src="/images/runs/stop-run-manual.png" alt="Manual run stop button" >}}

**State** 필드 옆 run 상태가 `running`에서 `Killed`로 변경됩니다.

{{< img src="/images/runs/stop-run-manual-status.png" alt="Run status after manual stop" >}}  
  {{% /tab %}}
{{< /tabpane >}}

가능한 run 상태의 전체 목록은 [State fields]({{< relref path="#run-states" lang="ko" >}})를 참고하세요.

## 로그된 run 보기

각 run에 대한 정보(상태, 연관된 artifacts, 실행 중 생성된 로그 파일 등)를 확인할 수 있습니다. 

{{< img src="/images/runs/demo-project.gif" alt="Project navigation demo" >}}

특정 run을 확인하려면:

1. [W&B App](https://wandb.ai/home)으로 이동하세요.
2. run을 초기화할 때 지정한 W&B 프로젝트로 이동하세요.
3. 프로젝트 사이드바에서 **Workspace** 탭을 선택하세요.
4. run 셀렉터에서 원하는 run을 클릭하거나, 이름 일부를 입력해 관련 run을 필터링하세요.

참고로 특정 run의 URL 경로는 아래와 같습니다:

```text
https://wandb.ai/<team-name>/<project-name>/runs/<run-id>
```

꺾쇠 괄호(`< >`)로 표시된 값들은 팀명, 프로젝트명, run ID의 실제 값으로 바꿔주면 됩니다.

### run 표시 방식 커스터마이즈하기
이 섹션에서는 프로젝트 **Workspace** 및 **Runs** 탭에서 run 표시 방식을 커스터마이즈하는 방법을 설명합니다. 두 탭은 같은 표시 설정을 공유합니다.

{{% alert %}}
워크스페이스는 설정과 무관하게 최대 1000개의 run만 표시할 수 있습니다.
{{% /alert %}}

열 표시 여부 커스터마이즈:
1. 프로젝트 사이드바에서 **Runs** 탭을 누릅니다.
1. run 목록 위쪽의 **Columns** 클릭
1. 숨겨진 컬럼 이름을 클릭하면 표시, 보이는 컬럼 이름을 클릭하면 숨김 처리됩니다.
  
    컬럼명은 퍼지 검색, 완전 일치, 정규식 등으로 검색 가능하며 드래그로 순서도 조정할 수 있습니다.
1. **Done**을 눌러 컬럼 브라우저 종료

보이는 컬럼을 기준으로 run 목록 정렬하기:

1. 컬럼명 위에 마우스를 올려 action `...` 메뉴 클릭
1. **Sort ascending** 또는 **Sort descending** 클릭

고정된(pinned) 컬럼은 우측, 고정 해제된 컬럼은 좌측(Workspace 탭에는 표시 안 됨)에 표시됩니다.

컬럼 고정 방법:
1. 프로젝트 사이드바에서 **Runs** 탭 클릭
1. **Pin column** 클릭

컬럼 고정 해제 방법:
1. 프로젝트 사이드바에서 **Workspace** 또는 **Runs** 탭 클릭
1. 컬럼명 위에 마우스를 올리고 action `...` 메뉴 클릭
1. **Unpin column** 클릭

기본적으로, 긴 run 이름은 가독성을 위해 중간이 생략됩니다. 이름 잘림(truncation) 방식을 변경하려면:

1. run 목록 상단의 action `...` 메뉴 클릭
1. **Run name cropping**에서 끝/중간/앞부분 중 잘림 방식을 선택

자세한 내용은 [**Runs** 탭]({{< relref path="/guides/models/track/project-page.md#runs-tab" lang="ko" >}})을 참고하세요.

### Overview 탭
**Overview** 탭에서는 프로젝트 내 특정 run의 정보를 확인할 수 있습니다. 예를 들면:

* **Author**: 해당 run을 생성한 W&B entity
* **Command**: run을 실행한 커맨드
* **Description**: 생성 시 입력한 run 설명. 입력하지 않으면 비어 있음. App UI 또는 Python SDK로 추가 가능
* **Tracked Hours**: run이 실제로 연산 또는 데이터 로깅에 사용된 시간. 중지/대기 시간 제외
* **Runtime**: run의 전체 진행 시간(중지, 대기 포함)
* **Git repository**: run과 연결된 git 저장소. [Git 활성화]({{< relref path="/guides/models/app/settings-page/user-settings.md#personal-github-integration" lang="ko" >}}) 필요
* **Host name**: run이 실행된 컴퓨터의 이름(로컬 실행 시 내 컴퓨터명 표시)
* **Name**: run의 이름
* **OS**: run을 시작한 운영체제
* **Python executable**: run 시작에 사용된 파이썬 실행 파일
* **Python version**: run을 생성한 파이썬 버전
* **Run path**: `entity/project/run-ID` 형태의 고유 run 식별자
* **Start time**: run 초기화 시각
* **State**: [run 상태]({{< relref path="#run-states" lang="ko" >}})
* **System hardware**: run에 사용된 하드웨어
* **Tags**: 문자열 리스트. 관련 run을 묶거나 임시 라벨(`baseline`, `production` 등) 용도로 활용
* **W&B CLI version**: 해당 run을 실행한 머신에 설치된 W&B CLI 버전
* **Git state**: run을 초기화할 때의 마지막 git 커밋 SHA. git 미사용 또는 정보 부족 시 비어 있음

이 외에 다음 정보가 overview 섹션 아래에 표시됩니다:

* **Artifact Outputs**: 해당 run이 생성한 Artifact 목록
* **Config**: [`wandb.Run.config`]({{< relref path="/guides/models/track/config.md" lang="ko" >}})로 저장된 설정 파라미터 리스트
* **Summary**: [`wandb.Run.log()`]({{< relref path="/guides/models/track/log/" lang="ko" >}})로 저장된 총괄(summary) 파라미터 목록. 기본적으로 마지막 기록값이 사용

{{< img src="/images/app_ui/wandb_run_overview_page.png" alt="W&B Dashboard run overview tab" >}}

프로젝트 개요 예시는 [여기](https://wandb.ai/stacey/deep-drive/overview)에서 확인할 수 있습니다.

### Workspace 탭
Workspace 탭에서는 자동 생성/커스텀 플롯, 시스템 메트릭 등 다양한 시각화 결과를 조회, 검색, 그룹화, 정렬할 수 있습니다.

{{< img src="/images/app_ui/wandb-run-page-workspace-tab.png" alt="Run workspace tab" >}}

예시 워크스페이스 보기: [여기](https://wandb.ai/stacey/deep-drive/workspace?nw=nwuserstacey)

### Runs 탭

Runs 탭에서는 run을 필터링, 그룹화, 정렬할 수 있습니다.

{{< img src="/images/runs/run-table-example.png" alt="Runs table" >}}

다음 탭에서는 Runs 탭에서 자주 사용하는 기능을 보여줍니다.

{{< tabpane text=true >}}
   {{% tab header="컬럼 커스터마이즈" %}}
Runs 탭에는 프로젝트 내 run의 많은 정보가 컬럼으로 표시됩니다.

- 모든 보이는 컬럼을 확인하려면 가로 스크롤을 사용하세요.
- 컬럼 순서를 변경하려면, 원하는 컬럼을 다른 위치로 드래그하세요.
- 컬럼을 고정하려면, 컬럼명 위에 마우스를 올리고 action 메뉴 `...` 클릭 후 **Pin column** 선택. 고정 컬럼은 **Name** 바로 다음에 나타나며, 해제하려면 **Unpin column** 선택
- 컬럼을 숨기려면, 컬럼명 위에 마우스를 올리고 action 메뉴 `...` 클릭 후 **Hide column** 선택. 현재 숨김 처리된 컬럼은 **Columns** 클릭 시 확인 가능
- 여러 컬럼을 한 번에 숨기거나(보이기/숨기기/고정/해제), **Columns**를 클릭하세요
  - 숨김 컬럼 클릭 시 보이기
  - 보이는 컬럼 클릭 시 숨기기
  - 컬럼명 옆 고정 아이콘 클릭 시 고정

Runs 탭의 커스터마이즈는 [Workspace 탭]({{< relref path="#workspace-tab" lang="ko" >}})의 Runs 셀렉터에도 반영됩니다.

   {{% /tab %}}

   {{% tab header="정렬" %}}
테이블의 특정 컬럼 값을 기준으로 모든 row를 정렬할 수 있습니다. 

1. 컬럼명 위에 마우스를 올려 '케밥 메뉴'(세로 점 3개)가 표시됩니다.
2. 해당 메뉴 클릭
3. **Sort Asc** 또는 **Sort Desc**를 선택해 오름차순/내림차순 정렬

{{< img src="/images/data_vis/data_vis_sort_kebob.png" alt="Confident predictions" >}}

위 이미지는 `val_acc` 컬럼의 정렬 옵션을 보여줍니다.   
   {{% /tab %}}
   {{% tab header="필터" %}}
**Filter** 버튼을 통해 원하는 표현식으로 row 전체를 필터링할 수 있습니다.

{{< img src="/images/data_vis/filter.png" alt="Incorrect predictions filter" >}}

**Add filter** 선택 시 하나 이상의 필터 조건을 추가할 수 있습니다. 드롭다운 3개가 나타나며, 왼쪽부터 순서대로: Column name, Operator, Value

|                   | Column name | Binary relation    | Value       |
| -----------       | ----------- | ----------- | ----------- |
| 지원 값   | String       |  &equals;, &ne;, &le;, &ge;, IN, NOT IN,  | Integer, float, string, timestamp, null |


식 에디터는 컬럼명 자동완성, 논리 연산 구조 안내 등 옵션을 보여줍니다. "and" 또는 "or" (때론 괄호도 사용)로 여러 논리 조건을 조합할 수 있습니다.

{{< img src="/images/data_vis/filter_example.png" alt="Run filtering example" >}}
위 이미지는 `val_loss` 컬럼을 기준으로 필터를 걸어 검증 손실이 1보다 작거나 같은 run만 표시하는 예시입니다.   
   {{% /tab %}}
   {{% tab header="그룹화" %}}
**Group by** 버튼을 사용해 특정 컬럼 기준으로 row 전체를 그룹화할 수 있습니다. 

{{< img src="/images/data_vis/group.png" alt="Error distribution analysis" >}}

기본적으로 다른 수치형 컬럼들은 해당 그룹별로 값 분포를 보여주는 히스토그램으로 전환됩니다. 그룹화는 데이터의 상위 패턴을 이해하는 데 도움이 됩니다.

{{% alert %}}
**Group by** 기능은 [run의 run group]({{< relref path="grouping.md" lang="ko" >}})과는 다릅니다. run group을 기준으로도 그룹화를 할 수 있습니다. run을 다른 group으로 이동하려면 [Assign a group or job type to a run]({{< relref path="#assign-a-group-or-job-type-to-a-run" lang="ko" >}}) 항목을 참고하세요.
{{% /alert %}}

   {{% /tab %}}
{{< /tabpane >}}

### Logs 탭
**Log 탭**에서는 커맨드라인에 출력된 결과(표준 출력 `stdout`, 표준 에러 `stderr` 등)를 보여줍니다.

우측 상단 **Download** 버튼을 눌러 로그 파일을 내려받을 수 있습니다.

{{< img src="/images/app_ui/wandb_run_page_log_tab.png" alt="Run logs tab" >}}

Log 예시는 [여기에서 볼 수 있습니다](https://app.wandb.ai/stacey/deep-drive/runs/pr0os44x/logs).

### Files 탭
**Files 탭**에서는 특정 run과 연관된 파일(예: 모델 체크포인트, 검증 세트 예시 등)을 볼 수 있습니다.

{{< img src="/images/app_ui/wandb_run_page_files_tab.png" alt="Run files tab" >}}

Files 탭 예시는 [여기](https://app.wandb.ai/stacey/deep-drive/runs/pr0os44x/files/media/images)에서 확인하세요.

### Artifacts 탭
**Artifacts** 탭에는 해당 run에서 입력/출력된 [artifacts]({{< relref path="/guides/core/artifacts/" lang="ko" >}}) 목록이 표시됩니다.

{{< img src="/images/app_ui/artifacts_tab.png" alt="Run artifacts tab" >}}

[artifact 그래프 예시 보기]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph.md" lang="ko" >}}).

## run 삭제

W&B App에서 하나 이상의 run을 프로젝트에서 삭제할 수 있습니다.

1. 삭제하려는 run이 포함된 프로젝트로 이동합니다.
2. 프로젝트 사이드바에서 **Runs** 탭을 선택합니다.
3. 삭제할 run 옆 체크박스를 선택합니다.
4. 테이블 상단의 **Delete** 버튼(휴지통 아이콘) 클릭
5. 표시되는 확인 창에서 **Delete** 선택

{{% alert %}}
특정 ID의 run이 삭제된 이후 해당 ID는 다시 사용할 수 없습니다. 삭제된 ID로 run을 새로 생성하려고 할 경우 에러가 발생하며, 실행이 차단됩니다.
{{% /alert %}}

{{% alert %}}
run이 매우 많은 프로젝트에서는 검색창에서 Regex 등으로 삭제할 run을 필터링하거나, filter 버튼으로 상태/태그/기타 속성을 기준으로 원하는 run만 필터링해 삭제할 수 있습니다.
{{% /alert %}}

## run 정리하기

이 섹션에서는 그룹 및 job type을 이용해 run을 정리하는 방법을 안내합니다. run을 (예: 실험명) 그룹으로 묶고, (전처리, 트레이닝, 평가, 디버깅 등) job type을 지정하면 워크플로우를 더 효율적으로 관리할 수 있으며, 모델 비교도 쉬워집니다.

### run에 그룹 또는 job type 지정하기

W&B에서 각 run은 **group** 및 **job type**으로 분류할 수 있습니다:

- **Group**: 실험 전체를 포괄하는 카테고리로, run 정렬/필터링 용도로 활용
- **Job type**: run의 역할(예: `preprocessing`, `training`, `evaluation` 등)

아래 [예시 워크스페이스](https://wandb.ai/stacey/model_iterz?workspace=user-stacey)는, Fashion-MNIST 데이터셋으로 데이터를 점진적으로 늘리며 베이스라인 모델을 학습하는 경우입니다. 워크스페이스에서는 데이터 양에 따라 다른 색상을 사용합니다:

- **노란색~진한 초록**: 베이스라인 모델의 데이터 양 증가
- **연파랑~보라~핑크**: 더 복잡한 "double" 모델(파라미터 추가)의 데이터양 변화를 의미

W&B의 필터 및 검색바를 활용하면, 아래와 같은 조건으로 run 비교가 가능합니다:
- 동일한 데이터셋으로 트레이닝한 경우
- 동일한 테스트 세트로 평가한 경우

필터를 적용하면 **Table** 뷰가 자동으로 업데이트되어, 모델의 성능 차이를 신속하게 파악할 수 있습니다. 예를 들어, 어떤 모델이 특정 클래스에서 어려움을 겪는지 쉽게 확인할 수 있습니다.
---
title: Runs
description: W&B의 기본 빌딩 블록인 Runs에 대해 알아보세요.
slug: /guides/runs
displayed_sidebar: default
---

W&B에 의해 기록된 계산의 단일 단위는 *run*이라고 합니다. W&B run은 전체 프로젝트의 원자적 요소로 생각할 수 있습니다. 다음과 같은 경우 새로운 run을 시작해야 합니다:

* 모델을 훈련할 때
* 하이퍼파라미터를 변경할 때
* 다른 모델을 사용할 때
* 데이터를 로그하거나 모델을 [W&B Artifact](../artifacts/intro.md)로 활용할 때
* [W&B Artifact 다운로드](../artifacts/download-and-use-an-artifact.md)를 할 때

예를 들어, [sweep](../sweeps/intro.md) 중에, W&B는 사용자가 지정한 하이퍼파라미터 검색 공간을 탐색합니다. 스윕에 의해 생성된 각 새로운 하이퍼파라미터 조합은 고유한 run으로 구현되고 기록됩니다.

:::tip
run을 생성 및 관리할 때 고려해야 할 몇 가지 핵심 사항:
* `wandb.log`로 로그한 모든 것이 해당 run에 기록됩니다. W&B에서 오브젝트를 로그하는 방법에 대한 자세한 정보는 [Log Media and Objects](../track/log/intro.md)를 참조하세요.
* 각 run은 특정 W&B 프로젝트와 연결됩니다.
* W&B App UI의 run 프로젝트 워크스페이스 내에서 run과 그 속성을 볼 수 있습니다.
* 어떤 프로세스에서도 하나의 활성 [`wandb.Run`](../../ref/python/run.md)만 있을 수 있으며, 이는 `wandb.run`으로 엑세스 가능합니다.
:::

## Create a run

[`wandb.init()`](../../ref/python/init.md)를 사용하여 W&B run을 생성하세요:

```python
import wandb

run = wandb.init()
```

새로운 run을 생성할 때 프로젝트 이름과 W&B 엔티티를 지정하는 것이 좋습니다. W&B는 제공된 W&B 엔티티 내에 프로젝트가 존재하지 않는 경우 새 프로젝트를 생성합니다. 프로젝트가 이미 존재하는 경우, W&B는 run을 해당 프로젝트에 저장합니다.

예를 들어, 다음 코드 조각은 `wandbee` 엔티티 내에 범위가 설정된 `model_registry_example`이라는 프로젝트에 저장된 run을 초기화합니다:

```python
import wandb

run = wandb.init(entity="wandbee", \
        project="model_registry_example")
```

W&B는 생성된 run의 이름과 해당 특정 run에 대한 자세한 정보를 찾을 수 있는 URL 경로를 출력합니다.

예를 들어, 위 코드 조각은 다음 출력을 생성합니다:
![](/images/runs/run_example.png)

## Organize runs with run names and run IDs

기본적으로, W&B는 새로운 run을 초기화할 때 무작위 이름과 run ID를 생성합니다.

앞의 예시에서, W&B는 `likely-lion-9`이라는 run 이름과 `xlm66ixq`라는 run ID를 생성합니다. `likely-lion-9` run은 `model_registry_example`이라는 프로젝트에 저장됩니다.

:::note
W&B에 의해 생성된 run 이름은 고유성을 보장하지 않습니다.
:::

run을 초기화할 때 `id` 파라미터를 사용하여 고유한 run ID 식별자를 제공하고 `name` 파라미터를 사용하여 run의 이름을 제공할 수 있습니다. 예를 들어,

```python 
import wandb

run = wandb.init(
    entity="<project>", 
    project="<project>", 
    name="<run-name>", 
    id="<run-id>"
)
```

run 이름과 run IDs를 사용하여 W&B App UI 내의 프로젝트에서 실험을 빠르게 찾으세요. 특정 run에 대한 자세한 정보를 URL에서 찾을 수 있습니다:

```text title="W&B App URL for a specific run"
https://wandb.ai/entity/project-name/run-id
```

여기서:
* `entity`: run을 초기화한 W&B 엔티티입니다.
* `project`: run이 저장된 프로젝트입니다.
* `run-id`: 해당 run의 run ID입니다.

:::tip
W&B는 run을 초기화할 때 프로젝트 이름을 지정할 것을 권장합니다. 프로젝트가 지정되지 않은 경우, W&B는 run을 "Uncategorized"라는 프로젝트에 저장합니다.
:::

[`wandb.init`](../../ref/python/init.md) 참조 문서에서 사용할 수 있는 모든 파라미터의 전체 목록을 확인하세요.

## View a run

run이 기록된 프로젝트 내에서 특정 run을 조회하세요:

1. [https://wandb.ai/home](https://wandb.ai/home)에서 W&B App UI로 이동합니다.
2. run을 초기화할 때 지정한 W&B 프로젝트로 이동합니다.
3. 프로젝트의 워크스페이스 내에서 **Runs**라는 테이블을 볼 수 있습니다. 이 테이블은 프로젝트에 포함된 모든 run 목록을 표시합니다. 표시된 run 목록 중 조회하고 싶은 run을 선택하세요!
  ![Example project workspace called 'sweep-demo'](/images/app_ui/workspace_tab_example.png)
4. 다음으로 **Overview Tab** 아이콘을 선택합니다.

다음 이미지는 **sparkling-glade-2**라는 Run에 대한 정보를 보여줍니다:

![W&B Dashboard run overview tab](/images/app_ui/wandb_run_overview_page.png)

**Overview Tab**은 선택한 run에 대한 다음 정보를 보여줍니다:

* **Run name**: run의 이름입니다.
* **Description**: run을 설명한 내용입니다. run을 생성할 때 지정되지 않은 경우 이 필드는 초기적으로 비어 있습니다. W&B App UI나 프로그래밍적으로 run에 대한 설명을 제공할 수 있습니다.
* **Privacy**: run의 프라이버시 설정입니다. **Private** 또는 **Public**으로 설정할 수 있습니다.
    * **Private**: (기본값) 사용자만 볼 수 있고 기여할 수 있습니다.
    * **Public**: 누구나 볼 수 있습니다.
* **Tags**: (목록, 선택 사항) 문자열 목록입니다. 태그는 run을 함께 조직하거나, "baseline" 또는 "production"과 같은 임시 레이블을 적용하는 데 유용합니다.
* **Author**: run을 생성한 W&B 사용자 이름입니다.
* **Run state**: run의 상태입니다:
  * **finished**: 스크립트가 종료되고 데이터가 완전히 동기화되었거나 `wandb.finish()`를 호출함
  * **failed**: 스크립트가 0이 아닌 종료 상태로 종료됨
  * **crashed**: 내부 프로세스에서 하트비트를 보내지 않게 되었고, 이는 머신이 크래시할 때 발생할 수 있음
  * **running**: 스크립트가 여전히 실행 중이며 최근에 하트비트를 보냈음
* **Start time**: run이 시작된 타임스탬프입니다.
* **Duration**: run이 **finish**, **fail** 또는 **crash**하는 데 소요된 시간(초)입니다.
* **Run path**: 고유한 run 식별자입니다. 형태는 `entity/project/run-ID`입니다.
* **Host name**: run이 실행된 위치입니다. 로컬 머신에서 run을 실행한 경우 머신 이름이 표시됩니다.
* **Operating system**: run에 사용된 운영 체제입니다.
* **Python version**: run에 사용된 Python 버전입니다.
* **Python executable**: run을 시작한 코맨드입니다.
* **System Hardware**: run을 생성한 하드웨어입니다.
* **W&B CLI version**: run command를 호스팅한 머신에 설치된 W&B ClI 버전입니다.
* **Job Type**:

개요 섹션 아래에는 추가적으로 다음 정보가 있습니다:

* **Artifact Outputs**: run에 의해 생성된 Artifact 출력입니다.
* **Config**: [`wandb.config`](../../guides/track/config.md)와 함께 저장된 설정 파라미터 목록입니다.
* **Summary**: [`wandb.log()`](../../guides/track/log/intro.md)와 함께 저장된 요약 파라미터 목록입니다. 기본적으로 이 값은 마지막으로 로그된 값으로 설정됩니다.

프로젝트 내 여러 Run을 조직하는 방법에 대한 자세한 정보는 [Runs Table](../app/features/runs-table.md) 문서를 참조하세요.

프로젝트 워크스페이스의 라이브 예제를 보려면, [이 예제 프로젝트를](https://app.wandb.ai/example-team/sweep-demo) 참조하세요.

## End a run
W&B는 run을 자동으로 종료하고 해당 run의 데이터를 W&B 프로젝트에 로그합니다. [`run.finish`](../../ref/python/run.md#finish) 코맨드를 사용하여 run을 수동으로 종료할 수 있습니다. 예를 들어:

```python
import wandb

run = wandb.init()
run.finish()
```

:::info
`[`wandb.init`](../../ref/python/init.md)`를 자식 프로세스에서 호출한 경우, 자식 프로세스의 끝에서 [`wandb.finish`](../../ref/python/finish.md) 메소드를 사용할 것을 W&B는 권장합니다.
:::

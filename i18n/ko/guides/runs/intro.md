---
description: Learn about the basic building block of W&B, Runs.
slug: /guides/runs
displayed_sidebar: default
---

# Runs

W&B에서 로그된 단일 계산 단위를 *Run*이라고 합니다.

W&B Run을 전체 프로젝트의 원자적 요소로 생각하세요. 하이퍼파라미터를 변경하거나, 다른 모델을 사용하거나, [W&B 아티팩트](../artifacts/intro.md)를 생성하는 등의 작업을 할 때마다 새로운 Run을 생성하고 시작해야 합니다.

예를 들어, [W&B 스윕](../sweeps/intro.md)에서는 W&B가 하이퍼파라미터 탐색을 수행하고 가능한 모델의 공간을 탐색합니다. 각각의 새로운 하이퍼파라미터 조합은 W&B Run으로 구현됩니다.

다음과 같은 작업에 W&B Runs를 사용하세요:

* 모델을 훈련할 때마다.
* 데이터나 모델을 [W&B 아티팩트](../artifacts/intro.md)로 로그합니다.
* [W&B 아티팩트 다운로드](../artifacts/download-and-use-an-artifact.md).

`wandb.log`로 로그하는 모든 것은 해당 Run에 기록됩니다. W&B에서 오브젝트를 로그하는 방법에 대한 자세한 정보는 [미디어 및 오브젝트 로그](../track/log/intro.md)를 참조하세요.

프로젝트 내의 Runs를 프로젝트의 [워크스페이스](#view-runs)에서 확인하세요.

## Run 생성

[`wandb.init()`](../../ref/python/init.md)으로 W&B Run을 생성하세요:

```python
import wandb

run = wandb.init(project="my-project-name")
```

선택적으로 `project` 필드에 프로젝트 이름을 제공할 수 있습니다. Run 오브젝트를 생성할 때 프로젝트 이름을 지정하는 것이 좋습니다. 제공한 이름으로 프로젝트가 이미 존재하지 않으면 W&B는 새 프로젝트를 생성합니다. 프로젝트는 실험, runs, 아티팩트 등을 한 곳에 편리하게 정리하는 데 도움이 되며, *프로젝트 워크스페이스*라고 하는 개인적인 모래상자를 제공합니다. 프로젝트의 워크스페이스는 runs를 비교할 수 있는 개인적인 공간을 제공합니다.

:::info
프로젝트가 지정되지 않은 경우, W&B Run은 "Uncategorized"라는 프로젝트에 저장됩니다.
:::

어떤 프로세스에서든 활성 [`wandb.Run`](../../ref/python/run.md)은 하나만 존재하며, `wandb.run`으로 접근할 수 있습니다:

```python
import wandb

assert wandb.run is None

wandb.init()

assert wandb.run is not None
```

동일한 노트북이나 스크립트에서 하나 이상의 Runs를 시작하기 위해서는 완료되지 않은 Run을 마쳐야 합니다.

## Run 종료
W&B는 [`wandb.finish`](../../ref/python/finish.md)를 자동으로 호출하여 run을 최종화하고 정리합니다. 그러나 자식 프로세스에서 [`wandb.init`](../../ref/python/init.md)을 호출하는 경우, 자식 프로세스의 끝에서 `wandb.finish`를 명시적으로 호출해야 합니다.

:::note
스크립트가 종료될 때 wandb.finish API가 자동으로 호출됩니다.
:::

[`wandb.finish`](../../ref/python/finish.md) API를 사용하거나 `with` 문을 사용하여 Run을 수동으로 종료할 수 있습니다. 다음 코드 예제는 `with` 파이썬 문을 사용하여 Run을 종료하는 방법을 보여줍니다:

```python
import wandb

wandb.init()
wandb.finish()

assert wandb.run is None

with wandb.init() as run:
    pass  # 여기에 데이터 로그

assert wandb.run is None
```

## 프로젝트의 모든 Runs 보기
W&B App UI를 사용하여 프로젝트와 관련된 Runs를 봅니다. W&B App으로 이동하여 프로젝트 이름을 검색하세요.

다음 예에서는 "my-first-run"이라는 프로젝트를 검색합니다:

![](/images/runs/search_run_name_landing_page.png)

프로젝트를 선택하세요. 이 작업은 해당 프로젝트의 워크스페이스로 리디렉션됩니다. 프로젝트의 워크스페이스는 runs를 비교할 수 있는 개인적인 모래상자를 제공합니다. 프로젝트를 사용하여 비교할 수 있는 모델을 정리하고, 다른 아키텍처, 하이퍼파라미터, 데이터셋, 전처리 등으로 동일한 문제를 해결하는 작업을 조직하세요.

프로젝트의 워크스페이스 내에서, **Runs**이라고 표시된 테이블을 볼 수 있습니다. 이 테이블은 프로젝트에 있는 모든 Runs를 나열합니다. 즉, 이 runs는 생성될 때 `project` 인수를 받았습니다.

다음 이미지는 "sweep-demo"라는 프로젝트 워크스페이스를 보여줍니다:

![Example project workspace called 'sweep-demo'](/images/app_ui/workspace_tab_example.png)

**Runs 사이드바**는 프로젝트의 모든 runs를 나열합니다. 단일 Run 위로 마우스를 가져가면 다음을 수정하거나 보기 위해 선택할 수 있습니다:

* **케밥 메뉴**: Run 이름을 변경하거나 Run을 삭제하거나 활성 Run을 중지하는 데 이 케밥 메뉴를 사용하세요.
* **가시성 아이콘**: 특정 run을 숨기려면 눈 아이콘을 선택하세요.
* **색상**: run 색상을 우리의 프리셋 중 하나 또는 사용자 정의 색상으로 변경하세요.
* **검색**: 이름으로 runs를 검색하세요. 이것은 또한 플롯에서 볼 수 있는 runs를 필터링합니다.
* **필터**: 사이드바 필터를 사용하여 볼 수 있는 runs의 집합을 좁히세요.
* **그룹**: runs를 동적으로 그룹화하기 위해 config 열을 선택하세요. 예를 들어 아키텍처별로. 그룹화는 평균 값에 따라 선이 나타나고 그래프의 분산에 대한 음영 처리된 영역이 있는 플롯을 만듭니다.
* **정렬**: 예를 들어 가장 낮은 손실 또는 가장 높은 정확도를 가진 runs에 따라 runs를 정렬하세요. 정렬은 그래프에 나타나는 runs에 영향을 미칩니다.
* **확장 버튼**: 사이드바를 전체 테이블로 확장하세요
* **Run 수**: 괄호 안의 숫자는 프로젝트의 총 run 수입니다. 숫자 (N 시각화)는 눈이 켜져 있고 각 플롯에서 시각화할 수 있는 runs의 수입니다. 아래 예에서는 183개의 runs 중 처음 10개만 그래프에 표시됩니다. 그래프를 편집하여 보이는 최대 runs 수를 늘리세요.

프로젝트에서 여러 Runs를 조직하는 방법에 대한 자세한 내용은 [Runs 테이블](../app/features/runs-table.md) 문서를 참조하세요.

프로젝트의 워크스페이스에 대한 실제 예를 보려면 [이 예제 프로젝트](https://app.wandb.ai/example-team/sweep-demo)를 참조하세요.

## 프로젝트에서 특정 Run 조사

Run 페이지를 사용하여 특정 Run에 대한 자세한 정보를 탐색하세요.

1. 프로젝트로 이동하여 **Runs 사이드바**에서 특정 Run을 선택하세요.
2. 다음으로, **Overview 탭** 아이콘을 선택하세요.

다음 이미지는 "sparkling-glade-2"라는 Run에 대한 정보를 보여줍니다:

![W&B 대시보드 run overview 탭](/images/app_ui/wandb_run_overview_page.png)

**Overview 탭**은 선택한 Run에 대한 다음 정보를 보여줍니다:

* Run 이름: Run의 이름입니다.
* 설명: 제공한 Run의 설명입니다. Run을 생성할 때 설명이 지정되지 않았다면 이 필드는 처음에 비어 있습니다. W&B App UI 또는 프로그래매틱하게 Run에 대한 설명을 선택적으로 제공할 수 있습니다.
* 개인 정보 보호: Run의 개인 정보 보호 설정입니다. **Private** 또는 **Public**으로 설정할 수 있습니다.
    * **Private**: (기본값) 본인만 볼 수 있고 기여할 수 있습니다.
    * **Public**: 누구나 볼 수 있습니다.
* 태그: (리스트, 선택 사항) 문자열의 리스트입니다. 태그는 runs를 함께 조직하거나 "베이스라인" 또는 "프로덕션"과 같은 임시 라벨을 적용하는 데 유용합니다.
* 저자: Run을 생성한 W&B 사용자 이름입니다.
* Run 상태: Run의 상태입니다:
  * **finished**: 스크립트가 종료되고 데이터가 완전히 동기화되었거나 `wandb.finish()`를 호출했습니다
  * **failed**: 스크립트가 0이 아닌 종료 상태로 종료되었습니다
  * **crashed**: 스크립트가 내부 프로세스에서 심장 박동을 보내지 않아 중지되었습니다. 이는 기계가 충돌하는 경우 발생할 수 있습니다
  * **running**: 스크립트가 여전히 실행 중이며 최근에 심장 박동을 보냈습니다
* 시작 시간: Run이 시작된 타임스탬프입니다.
* 지속 시간: Run이 **finish**, **fail**, 또는 **crash**하는 데 걸린 시간(초)입니다.
* 호스트 이름: Run이 시작된 위치입니다. 로컬 기계에서 Run을 시작한 경우 기계의 이름이 표시됩니다.
* 운영 체제: Run에 사용된 운영 체제입니다.
* Python 버전: Run에 사용된 Python 버전입니다.
* Python 실행 파일: Run을 시작한 코맨드입니다.
* 시스템 하드웨어: Run을 생성하는 데 사용된 하드웨어입니다.
* W&B CLI 버전: Run 코맨드를 호스팅한 기계에 설치된 W&B CLI 버전입니다.

개요 섹션 아래에서, 다음 정보를 추가로 찾을 수 있습니다:

* **아티팩트 출력**: Run에서 생성된 아티팩트 출력입니다.
* **Config**: [`wandb.config`](../../guides/track/config.md)로 저장된 config 파라미터 목록입니다.
* **요약**: [`wandb.log()`](../../guides/track/log/intro.md)로 저장된 요약 파라미터 목록입니다. 기본적으로, 이 값은 로그된 마지막 값으로 설정됩니다.
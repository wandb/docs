---
title: Automation 이벤트 및 범위
menu:
  default:
    identifier: ko-guides-core-automations-automation-events
    parent: automations
weight: 2
---

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

Automation은 특정 프로젝트 또는 Registry 내에서 이벤트가 발생할 때 시작될 수 있습니다. 이 페이지에서는 각 범위에서 automation를 트리거할 수 있는 이벤트들을 설명합니다. Automation에 대해 더 알고 싶다면 [Automation 개요]({{< relref path="/guides/core/automations/" lang="ko" >}}) 또는 [Automation 생성하기]({{< relref path="create-automations/" lang="ko" >}})를 참고하세요.

## Registry
이 섹션에서는 [Registry]({{< relref path="/guides/core/registry/" lang="ko" >}})에서 automation를 위한 범위와 이벤트에 대해 설명합니다.

1. https://wandb.ai/registry/ 에서 **Registry** 앱으로 이동하세요.
1. Registry의 이름을 클릭한 후, **Automations** 탭에서 automation를 확인하거나 생성할 수 있습니다.

![Registry Automations 탭에서 automation가 보이는 스크린샷](/images/automations/registry_automations_tab.png)

[Automation 생성하기]({{< relref path="create-automations/" lang="ko" >}})에 대해 더 알아보세요.

### 범위
Registry automation는 다음과 같은 범위에서 생성할 수 있습니다:
- [Registry]({{< relref path="/guides/core/registry/" lang="ko" >}}) 레벨: 해당 registry 내의 모든 collection에서 발생하는 이벤트를 감시합니다. 향후 추가되는 collection도 포함됩니다.
- collection 레벨: 특정 registry 내 한 개의 collection에 대해 적용됩니다.

### 이벤트
Registry automation는 다음과 같은 이벤트를 감지할 수 있습니다:
- **새 버전이 collection에 연결됨**: Registry에 새로운 모델이나 데이터셋이 추가될 때 테스트 및 검증 작업을 수행할 수 있습니다.
- **artifact 에일리어스가 추가됨**: 새 artifact 버전에 특정 에일리어스가 적용될 때 워크플로우의 특정 단계를 트리거할 수 있습니다. 예를 들어, `production` 에일리어스가 적용되면 모델을 배포하도록 할 수 있습니다.

## Project
이 섹션에서는 [project]({{< relref path="/guides/models/track/project-page.md" lang="ko" >}})에서 automation를 위한 범위와 이벤트에 대해 설명합니다.

1. W&B 앱에서 `https://wandb.ai/<team>/<project-name>` 경로로 본인의 W&B Project로 이동하세요.
1. **Automations** 탭에서 automation를 조회 및 생성할 수 있습니다.

![Project Automations 탭에서 automation가 보이는 스크린샷](/images/automations/project_automations_tab.png)

[Automation 생성하기]({{< relref path="create-automations/" lang="ko" >}})에 대해 더 알아보세요.

### 범위
Project automation는 다음과 같은 범위에서 생성할 수 있습니다:
- Project 레벨: project 내의 모든 collection에서 일어나는 이벤트를 감시합니다.
- collection 레벨: 사용자가 지정한 필터에 맞는 project 내 모든 collection을 대상으로 합니다.

### Artifact 이벤트
이 부분에서는 artifact와 관련된 automation 트리거 이벤트를 설명합니다.

- **artifact에 새 버전이 추가됨**: artifact의 각 버전에 반복적으로 동작을 적용할 수 있습니다. 예를 들어, 새로운 데이터셋 artifact 버전이 생성될 때 트레이닝 작업을 시작할 수 있습니다.
- **artifact 에일리어스가 추가됨**: project 또는 collection 내의 artifact 새 버전에 특정 에일리어스가 적용될 때 워크플로우의 특정 단계를 트리거할 수 있습니다. 예를 들어, artifact에 `test-set-quality-check` 에일리어스가 붙으면 후처리 단계를 실행하거나, 새로운 artifact 버전마다 `latest` 에일리어스가 붙으면 워크플로우를 실행할 수 있습니다. 한 시점에는 하나의 artifact 버전만이 동일한 에일리어스를 가질 수 있습니다.
- **artifact tag가 추가됨**: project 또는 collection 내 artifact 버전에 특정 tag가 붙을 때 워크플로우의 특정 단계를 트리거할 수 있습니다. 예를 들어, artifact 버전에 "europe" tag가 추가되면 지역별 워크플로우를 실행할 수 있습니다. artifact tag는 그룹화 및 필터링 용도로 사용되며, 하나의 tag로 여러 artifact 버전에 동시 적용될 수 있습니다.

### Run 이벤트
Automation은 [run의 상태]({{< relref path="/guides/models/track/runs/#run-states" lang="ko" >}}) 변화나 [메트릭 값]({{< relref path="/guides/models/track/log/#what-data-is-logged-with-specific-wb-api-calls" lang="ko" >}}) 변화에 따라 트리거될 수 있습니다.

#### Run 상태 변화
{{% alert %}}
- 현재는 [W&B Multi-tenant Cloud]({{< relref path="/guides/hosting/#wb-multi-tenant-cloud" lang="ko" >}}) 에서만 지원됩니다.
- **Killed** 상태의 run은 automation를 트리거할 수 없습니다. 이 상태는 관리자 사용자에 의해 run이 강제로 중지된 경우를 의미합니다.
{{% /alert %}}

run의 [상태]({{< relref path="/guides/models/track/runs/_index.md#run-states" lang="ko" >}})가 **Running**, **Finished**, 또는 **Failed**로 바뀔 때 워크플로우를 트리거합니다. 원한다면 run을 시작한 사용자나 run의 이름으로 automation를 트리거할 run을 더 세부적으로 필터링할 수 있습니다.

![run 상태 변화 automation 예시 스크린샷](/images/automations/run_status_change.png)

run 상태는 전체 run의 속성이므로, run 상태 automation는 **Automations** 페이지에서만 만들 수 있고 workspace에서는 만들 수 없습니다.

#### Run 메트릭 변화
{{% alert %}}
현재는 [W&B Multi-tenant Cloud]({{< relref path="/guides/hosting/#wb-multi-tenant-cloud" lang="ko" >}}) 에서만 지원됩니다.
{{% /alert %}}

로그된 메트릭의 값(예: run의 history 상의 메트릭 또는 `cpu`와 같이 CPU 사용률을 보여주는 [시스템 메트릭]({{< relref path="/guides/models/app/settings-page/system-metrics.md" lang="ko" >}}))을 기반으로 워크플로우를 트리거할 수 있습니다. W&B는 시스템 메트릭을 15초마다 자동으로 기록합니다.

run 메트릭 automation는 project의 **Automations** 탭이나 워크스페이스의 라인 플롯 패널에서 바로 만들 수 있습니다.

run 메트릭 automation를 설정할 때는, 지정한 임계값과 메트릭의 값을 어떻게 비교할지 선택합니다. 선택지는 이벤트 종류나 적용한 필터에 따라 달라집니다.

필요하다면 run을 시작한 사용자나 run의 이름으로 automation를 트리거할 run을 더 세부적으로 필터링할 수 있습니다.

##### 임계값 (Threshold)
**Run 메트릭 임계값 충족** 이벤트의 경우, 다음을 설정합니다:
1. 최근에 기록된 값 중 몇 개를 고려할지 (기본값 5)
1. 해당 구간(window) 내에서 **Average**(평균), **Min**(최소), **Max**(최대) 중 무엇을 평가할지
1. 비교 방식 선택:
      - 초과
      - 이상
      - 미만
      - 이하
      - 같지 않음
      - 동일

예를 들어 평균 `accuracy`가 `.6`을 초과할 때 automation를 실행할 수 있습니다.

![run 메트릭 임계값 automation 예시 스크린샷](/images/automations/run_metrics_threshold_automation.png)

##### 변화 임계값 (Change threshold)
**Run 메트릭 변화 임계값 충족** 이벤트의 경우, automation는 두 개의 "구간"의 값을 비교하여 시작 여부를 결정합니다:

- _현재 구간_: 최근에 기록된 값들 (기본값 10)
- _이전 구간_: 그 전의 값들 (기본값 50)

현재 구간과 이전 구간은 연속적이며 서로 겹치지 않습니다.

Automation을 만들 때는 다음과 같이 설정합니다:
1. 현재 구간의 값 개수 (기본값 10)
1. 이전 구간의 값 개수 (기본값 50)
1. 상대값(**Relative**, 기본값) 또는 절대값(Absolute) 중 평가 방식
1. 비교 방식 선택:
      - 최소한 만큼 증가
      - 최소한 만큼 감소
      - 증가 또는 감소

예를 들어 평균 `loss`가 최소 `.25` 이상 감소하면 automation를 실행할 수 있습니다.

![run 메트릭 변화 임계값 automation 예시 스크린샷](/images/automations/run_metrics_change_threshold_automation.png)

#### Run 필터
이 섹션에서는 automation가 어떤 run을 평가 대상으로 삼는지 설명합니다.

- 기본적으로, project에 있는 모든 run이 이벤트 발생 시 automation를 트리거할 수 있습니다. 특정 run만 대상으로 하고 싶다면 run 필터를 지정하세요.
- 각 run이 개별적으로 평가되며, 각각 automation를 트리거할 수 있습니다.
- 각 run의 값들은 별도의 구간(window)에 포함되어 임계값과 따로 비교됩니다.
- 24시간 동안 각 run별로 해당 automation는 최대 한 번만 실행될 수 있습니다.

## 다음 단계
- [Slack automation 생성하기]({{< relref path="create-automations/slack.md" lang="ko" >}})
- [웹훅 automation 생성하기]({{< relref path="create-automations/webhook.md" lang="ko" >}})
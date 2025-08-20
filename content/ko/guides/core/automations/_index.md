---
title: Automation
menu:
  default:
    identifier: ko-guides-core-automations-_index
    parent: core
weight: 4
---

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

이 페이지는 W&B의 _automations_ 에 대해 설명합니다. [automation 생성]({{< relref path="create-automations/" lang="ko" >}})을 통해, Artifacts의 버전이 새로 생성되거나 [run metric]({{< relref path="/guides/models/track/runs.md" lang="ko" >}})이 기준값을 달성하거나 변경되는 등 W&B 내의 이벤트에 따라 automation된 모델 테스트, 배포 등 워크플로우 단계를 트리거할 수 있습니다.

예를 들어, 새로운 버전이 생성될 때 Slack 채널에 알림을 보내거나, `production` 에일리어스가 artifact에 추가되면 automation된 테스트 webhook을 트리거하거나, 특정 run의 `loss`가 허용 범위 내에 있는 경우에만 검증 작업을 시작할 수 있습니다.

## 개요
automation은 registry나 project 내에서 특정 [이벤트]({{< relref path="automation-events.md" lang="ko" >}})가 발생하면 시작됩니다.

[Registry]({{< relref path="/guides/core/registry/" lang="ko" >}})에서는 다음과 같은 경우 automation을 실행할 수 있습니다:
- 새로운 artifact 버전이 collection에 연결될 때: 예를 들어, 후보 모델에 대한 테스트 및 검증 워크플로우를 트리거합니다.
- artifact 버전에 에일리어스가 추가될 때: 예를 들어, 모델 버전에 에일리어스가 추가되면 배포 워크플로우를 트리거합니다.

[project]({{< relref path="/guides/models/track/project-page.md" lang="ko" >}})에서는 다음과 같은 경우 automation을 실행할 수 있습니다:
- artifact에 새로운 버전이 추가될 때: 예를 들어, 데이터셋 artifact의 새 버전이 collection에 추가되면 트레이닝 작업을 시작합니다.
- artifact 버전에 에일리어스가 추가될 때: 예를 들어, 데이터셋 artifact에 "redaction" 에일리어스가 추가되면 PII 비식별화 워크플로우를 실행합니다.
- artifact 버전에 tag가 추가될 때: 예를 들어, "europe" tag가 artifact 버전에 추가되면 지역별 워크플로우를 트리거합니다.
- run의 metric이 설정된 기준값을 달성하거나 초과할 때
- run의 metric이 설정된 기준만큼 변경될 때
- run의 상태가 **Running**, **Failed**, **Finished** 중 하나로 변경될 때

원하는 경우 user 또는 run 이름으로 run을 필터링할 수 있습니다.

자세한 내용은 [Automation events and scopes]({{< relref path="automation-events.md" lang="ko" >}})를 참고하세요.

[automation을 생성하려면]({{< relref path="create-automations/" lang="ko" >}}), 다음 단계를 수행합니다:

1. 필요한 경우, automation이 필요로 하는 민감한 문자열(예: 엑세스 토큰, 비밀번호, 민감한 설정 값 등)을 위한 [secrets]({{< relref path="/guides/core/secrets.md" lang="ko" >}})을 설정합니다. secrets 는 **Team Settings**에서 정의합니다. secrets 는 주로 webhook automation에서 자격 증명이나 토큰을 안전하게 외부 서비스로 전달하려고 평문이나 코드 내에 노출하지 않기 위해 사용합니다.
1. W&B가 Slack에 알림을 게시하거나 webhook을 실행하도록 승인하려면 webhook 또는 Slack 알림을 설정하세요. 하나의 automation action(webhook 또는 Slack 알림)은 여러 automation에서 사용할 수 있습니다. 이 action들은 **Team Settings**에 정의합니다.
1. project 또는 registry에서 automation을 생성합니다:
    1. 감지할 [event]({{< relref path="#automation-events" lang="ko" >}})를 정의합니다. 예를 들어, 새로운 artifact 버전이 추가될 때 등.
    1. 이벤트 발생 시 실행할 action을 정의합니다(Slack 채널에 게시 또는 webhook 실행). webhook의 경우, 필요하다면 엑세스 토큰용 secret 또는 payload에 함께 보낼 secret을 지정합니다.

## 제한사항
[Run metric automations]({{< relref path="automation-events.md#run-metrics-events" lang="ko" >}})는 현재 [W&B Multi-tenant Cloud]({{< relref path="/guides/hosting/#wb-multi-tenant-cloud" lang="ko" >}})에서만 지원됩니다.

## 다음 단계
- [automation 생성]({{< relref path="create-automations/" lang="ko" >}})
- [Automation events and scopes]({{< relref path="automation-events.md" lang="ko" >}}) 자세히 알아보기
- [secret 생성]({{< relref path="/guides/core/secrets.md" lang="ko" >}})
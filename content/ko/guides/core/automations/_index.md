---
title: Automations
menu:
  default:
    identifier: ko-guides-core-automations-_index
    parent: core
weight: 4
---

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

이 페이지는 W&B의 _자동화_ 에 대해 설명합니다. [자동화 생성]({{< relref path="create-automations/" lang="ko" >}})을 통해, [artifact]({{< relref path="/guides/core/artifacts" lang="ko" >}}) 아티팩트 버전이 생성될 때와 같이 W&B의 이벤트에 따라 자동 모델 테스트 및 배포와 같은 워크플로우 단계를 트리거할 수 있습니다.

예를 들어, 새로운 버전이 생성될 때 Slack 채널에 게시하거나, `production` 에일리어스가 아티팩트에 추가될 때 자동 테스트를 트리거하는 훅을 실행하는 자동화를 설정할 수 있습니다.

## 개요
자동화는 특정 [이벤트]({{< relref path="automation-events.md" lang="ko" >}})가 레지스트리 또는 프로젝트에서 발생할 때 실행될 수 있습니다.

[Registry]({{< relref path="/guides/core/registry/" lang="ko" >}})의 아티팩트의 경우, 다음을 실행하도록 자동화를 구성할 수 있습니다.
- 새로운 아티팩트 버전이 컬렉션에 연결될 때. 예를 들어, 새로운 후보 모델에 대한 테스팅 및 유효성 검사 워크플로우를 트리거합니다.
- 에일리어스가 아티팩트 버전에 추가될 때. 예를 들어, 에일리어스가 모델 버전에 추가될 때 배포 워크플로우를 트리거합니다.

[project]({{< relref path="/guides/models/track/project-page.md" lang="ko" >}})의 아티팩트의 경우, 다음을 실행하도록 자동화를 구성할 수 있습니다.
- 새로운 버전이 아티팩트에 추가될 때. 예를 들어, 새로운 버전의 데이터셋 아티팩트가 주어진 컬렉션에 추가될 때 트레이닝 작업을 시작합니다.
- 에일리어스가 아티팩트 버전에 추가될 때. 예를 들어, "redaction" 에일리어스가 데이터셋 아티팩트에 추가될 때 PII 삭제 워크플로우를 트리거합니다.

자세한 내용은 [자동화 이벤트 및 범위]({{< relref path="automation-events.md" lang="ko" >}})를 참조하세요.

[자동화 생성]({{< relref path="create-automations/" lang="ko" >}}) 방법:

1. 필요한 경우, 엑세스 토큰, 비밀번호 또는 민감한 구성 세부 정보와 같이 자동화에 필요한 민감한 문자열에 대한 [secrets]({{< relref path="/guides/core/secrets.md" lang="ko" >}})를 구성합니다. Secrets는 **Team Settings**에서 정의됩니다. Secrets는 일반적으로 훅 자동화에서 자격 증명 또는 토큰을 일반 텍스트로 노출하거나 훅의 페이로드에 하드 코딩하지 않고 훅의 외부 서비스에 안전하게 전달하는 데 사용됩니다.
1. W&B가 Slack에 게시하거나 사용자를 대신하여 훅을 실행할 수 있도록 훅 또는 Slack 알림을 구성합니다. 단일 자동화 작업(훅 또는 Slack 알림)은 여러 자동화에서 사용할 수 있습니다. 이러한 작업은 **Team Settings**에서 정의됩니다.
1. 프로젝트 또는 레지스트리에서 자동화를 생성합니다.
    1. 새로운 아티팩트 버전이 추가될 때와 같이 감시할 [이벤트]({{< relref path="#automation-events" lang="ko" >}})를 정의합니다.
    1. 이벤트가 발생할 때 수행할 작업(Slack 채널에 게시 또는 훅 실행)을 정의합니다. 훅의 경우, 필요한 경우 엑세스 토큰에 사용할 secret 및/또는 페이로드와 함께 보낼 secret을 지정합니다.

## 다음 단계
- [자동화 생성]({{< relref path="create-automations/" lang="ko" >}}).
- [자동화 이벤트 및 범위]({{< relref path="automation-events.md" lang="ko" >}})에 대해 알아봅니다.
- [secret 생성]({{< relref path="/guides/core/secrets.md" lang="ko" >}}).

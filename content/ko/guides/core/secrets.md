---
title: Secrets
description: W&B secrets에 대한 개요, 작동 방식, 사용 시작 방법에 대해 설명합니다.
menu:
  default:
    identifier: ko-guides-core-secrets
    parent: core
url: guides/secrets
weight: 1
---

W&B Secret Manager를 사용하면 엑세스 토큰, bearer 토큰, API 키 또는 비밀번호와 같은 중요한 문자열인 _secrets_를 안전하고 중앙 집중식으로 저장, 관리 및 삽입할 수 있습니다. W&B Secret Manager는 중요한 문자열을 코드에 직접 추가하거나 웹 훅의 헤더 또는 [페이로드]({{< relref path="/guides/core/automations/" lang="ko" >}})를 구성할 때 불필요하게 만듭니다.

Secrets는 각 팀의 Secret Manager의 [팀 설정]({{< relref path="/guides/models/app/settings-page/team-settings/" lang="ko" >}})의 **Team secrets** 섹션에 저장되고 관리됩니다.

{{% alert %}}
* W&B 관리자만 secret을 생성, 편집 또는 삭제할 수 있습니다.
* Secrets는 Azure, GCP 또는 AWS에서 호스팅하는 [W&B Server 배포]({{< relref path="/guides/hosting/" lang="ko" >}})를 포함하여 W&B의 핵심 부분으로 포함됩니다. 다른 배포 유형을 사용하는 경우 W&B 계정 팀에 문의하여 W&B에서 secrets를 사용하는 방법에 대해 논의하십시오.
* W&B Server에서는 보안 요구 사항을 충족하는 보안 조치를 구성해야 합니다.

  - W&B는 고급 보안 기능으로 구성된 AWS, GCP 또는 Azure에서 제공하는 클라우드 제공업체의 secrets manager의 W&B 인스턴스에 secrets를 저장하는 것이 좋습니다.

  - 클라우드 secrets manager(AWS, GCP 또는 Azure)의 W&B 인스턴스를 사용할 수 없고 클러스터를 사용하는 경우 발생할 수 있는 보안 취약점을 방지하는 방법을 이해하지 못하는 경우 Kubernetes 클러스터를 secrets 저장소의 백엔드로 사용하지 않는 것이 좋습니다.
{{% /alert %}}

## secret 추가
secret을 추가하려면:

1. 수신 서비스가 들어오는 웹 훅을 인증하는 데 필요한 경우 필요한 토큰 또는 API 키를 생성합니다. 필요한 경우 비밀번호 관리자와 같이 중요한 문자열을 안전하게 저장합니다.
2. W&B에 로그인하여 팀의 **Settings** 페이지로 이동합니다.
3. **Team Secrets** 섹션에서 **New secret**을 클릭합니다.
4. 문자, 숫자 및 밑줄(`_`)을 사용하여 secret의 이름을 지정합니다.
5. 중요한 문자열을 **Secret** 필드에 붙여넣습니다.
6. **Add secret**을 클릭합니다.

웹 훅을 구성할 때 웹 훅 자동화에 사용할 secrets를 지정합니다. 자세한 내용은 [웹 훅 구성]({{< relref path="#configure-a-webhook" lang="ko" >}}) 섹션을 참조하십시오.

{{% alert %}}
secret을 생성하면 `${SECRET_NAME}` 형식을 사용하여 [웹 훅 자동화의 페이로드]({{< relref path="/guides/core/automations/create-automations/webhook.md" lang="ko" >}})에서 해당 secret에 엑세스할 수 있습니다.
{{% /alert %}}

## secret 교체
secret을 교체하고 값을 업데이트하려면:
1. secret의 행에서 연필 아이콘을 클릭하여 secret의 세부 정보를 엽니다.
2. **Secret**을 새 값으로 설정합니다. 선택적으로 **Reveal secret**을 클릭하여 새 값을 확인합니다.
3. **Add secret**을 클릭합니다. secret의 값이 업데이트되고 더 이상 이전 값으로 확인되지 않습니다.

{{% alert %}}
secret을 생성하거나 업데이트한 후에는 더 이상 현재 값을 표시할 수 없습니다. 대신 secret을 새 값으로 교체합니다.
{{% /alert %}}

## secret 삭제
secret을 삭제하려면:
1. secret의 행에서 휴지통 아이콘을 클릭합니다.
2. 확인 대화 상자를 읽은 다음 **Delete**를 클릭합니다. secret이 즉시 영구적으로 삭제됩니다.

## secrets에 대한 엑세스 관리
팀의 자동화는 팀의 secrets를 사용할 수 있습니다. secret을 제거하기 전에 secret을 사용하는 자동화가 작동을 멈추지 않도록 업데이트하거나 제거하십시오.
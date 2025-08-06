---
title: 시크릿
description: W&B 시크릿의 개요, 작동 방식, 그리고 시작 방법 안내.
menu:
  default:
    identifier: ko-guides-core-secrets
    parent: core
url: guides/secrets
weight: 1
---

W&B Secret Manager 를 사용하면 엑세스 토큰, 베어러 토큰, API 키, 비밀번호와 같은 민감한 문자열인 _시크릿_ 을 안전하고 중앙에서 저장, 관리, 주입할 수 있습니다. W&B Secret Manager 를 통해 코드를 작성하거나 웹훅의 헤더 또는 [payload]({{< relref path="/guides/core/automations/" lang="ko" >}})를 설정할 때 민감한 문자열을 직접 입력할 필요가 없습니다.

시크릿은 각 팀의 Secret Manager 에서 관리되며, [team settings]({{< relref path="/guides/models/app/settings-page/team-settings/" lang="ko" >}})의 **Team secrets** 섹션에서 확인할 수 있습니다.

{{% alert %}}
* 오직 W&B Admin 만 시크릿을 생성, 수정, 삭제할 수 있습니다.
* 시크릿은 [W&B Server deployments]({{< relref path="/guides/hosting/" lang="ko" >}})를 포함하여 W&B의 핵심 기능으로 제공되며, 여러분이 Azure, GCP, AWS 등에서 직접 호스팅할 때도 사용할 수 있습니다. 만약 다른 배포 환경을 사용하고 있다면, W&B 계정 팀에 연락하여 시크릿을 어떻게 사용할 수 있는지 상담해 주세요.
* W&B Server 환경에서는 여러분의 보안 요구사항을 충족하는 보안 설정을 직접 구성해야 합니다.

  - W&B는 AWS, GCP, Azure가 제공하는 클라우드 제공사의 secrets manager 내 W&B 인스턴스에 시크릿을 저장하는 것을 강력히 권장합니다. 해당 시스템들은 고급 보안 기능을 제공합니다.

  - 만약 클라우드 secrets manager (AWS, GCP, Azure) 환경의 W&B 인스턴스를 사용할 수 없고, 클러스터 사용에서 발생할 수 있는 보안 취약점을 충분히 이해할 수 있는 경우를 제외하고, 시크릿 저장소의 백엔드로 Kubernetes 클러스터를 사용하는 것은 추천하지 않습니다.
{{% /alert %}}

## 시크릿 추가하기
시크릿을 추가하려면:

1. 만약 받는 서비스가 인증을 위해 웹훅에 토큰이나 API 키를 요구하는 경우, 필요한 토큰 또는 API 키를 생성합니다. 필요하다면 비밀번호 관리자 등에 해당 민감한 문자열을 안전하게 저장하세요.
1. W&B에 로그인하여 팀의 **Settings** 페이지로 이동합니다.
1. **Team Secrets** 섹션에서 **New secret**을 클릭합니다.
1. 영문, 숫자, 밑줄(`_`)을 사용해 시크릿 이름을 입력합니다.
1. **Secret** 필드에 민감한 문자열을 붙여넣습니다.
1. **Add secret**을 클릭합니다.

웹훅 자동화를 구성할 때 사용할 시크릿을 지정할 수 있습니다. 자세한 내용은 [Configure a webhook]({{< relref path="#configure-a-webhook" lang="ko" >}}) 섹션을 참고하세요.

{{% alert %}}
시크릿을 생성하면, [webhook automation의 payload]({{< relref path="/guides/core/automations/create-automations/webhook.md" lang="ko" >}})에서 `${SECRET_NAME}` 형식으로 해당 시크릿에 엑세스할 수 있습니다.
{{% /alert %}}

## 시크릿 교체하기 (Rotate a secret)
시크릿의 값을 교체하려면:
1. 시크릿 행의 연필 아이콘을 클릭하여 시크릿 세부 정보를 엽니다.
1. **Secret**에 새로운 값을 입력합니다. 필요하다면 **Reveal secret**을 클릭해 새로운 값을 확인할 수 있습니다.
1. **Add secret**을 클릭합니다. 시크릿 값이 업데이트되며, 이전 값은 더 이상 사용할 수 없습니다.

{{% alert %}}
시크릿을 생성하거나 업데이트한 후에는 기존 값을 다시 볼 수 없습니다. 새로운 값으로 시크릿을 교체(rotate)해 주세요.
{{% /alert %}}

## 시크릿 삭제하기
시크릿을 삭제하려면:
1. 시크릿 행의 휴지통 아이콘을 클릭합니다.
1. 확인 대화 상자를 읽고 **Delete**를 클릭하세요. 해당 시크릿은 즉시, 영구적으로 삭제됩니다.

## 시크릿 엑세스 권한 관리
한 팀의 automations 는 해당 팀의 시크릿을 사용할 수 있습니다. 시크릿을 삭제하기 전, 해당 시크릿을 사용하는 automation 을 업데이트하거나 제거하여 중단 없이 동작하도록 하세요.
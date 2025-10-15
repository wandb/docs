---
title: Use service accounts to automate workflows
description: org 및 팀 범위의 서비스 계정을 사용하여 자동화된 워크플로우 또는 비대화형 워크플로우를 관리하세요.
displayed_sidebar: default
menu:
  default:
    identifier: ko-guides-hosting-iam-authentication-service-accounts
---

서비스 계정은 팀 내 또는 팀 간의 프로젝트에서 일반적인 작업을 자동으로 수행할 수 있는 비인간 또는 머신 사용자를 나타냅니다.

- 조직 관리자는 조직 범위에서 서비스 계정을 만들 수 있습니다.
- 팀 관리자는 해당 팀 범위에서 서비스 계정을 만들 수 있습니다.

서비스 계정의 API 키를 통해 호출자는 서비스 계정 범위 내에서 프로젝트를 읽거나 쓸 수 있습니다.

서비스 계정을 사용하면 여러 사용자 또는 팀에서 워크플로우를 중앙 집중식으로 관리하고, W&B Models에 대한 실험 트래킹을 자동화하거나, W&B Weave에 대한 추적을 기록할 수 있습니다. [환경 변수]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}}) `WANDB_USERNAME` 또는 `WANDB_USER_EMAIL`을 사용하여 서비스 계정에서 관리하는 워크플로우와 인간 사용자의 ID를 연결할 수 있습니다.

{{% alert %}}
서비스 계정은 [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}), 엔터프라이즈 라이선스가 있는 [자체 관리 인스턴스]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) 및 [SaaS 클라우드]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})의 엔터프라이즈 계정에서 사용할 수 있습니다.
{{% /alert %}}

## 조직 범위 서비스 계정

조직 범위의 서비스 계정은 [제한된 프로젝트]({{< relref path="../access-management/restricted-projects.md#visibility-scopes" lang="ko" >}})를 제외하고 팀에 관계없이 조직의 모든 프로젝트에서 읽고 쓸 수 있는 권한을 갖습니다. 조직 범위의 서비스 계정이 제한된 프로젝트에 엑세스하려면 해당 프로젝트의 관리자가 서비스 계정을 프로젝트에 명시적으로 추가해야 합니다.

조직 관리자는 조직 또는 계정 대시보드의 **Service Accounts** 탭에서 조직 범위의 서비스 계정에 대한 API 키를 얻을 수 있습니다.

새로운 조직 범위의 서비스 계정을 만들려면:

* 조직 대시보드의 **Service Accounts** 탭에서 **New service account** 버튼을 클릭합니다.
* **Name**을 입력합니다.
* 서비스 계정의 기본 팀을 선택합니다.
* **Create**를 클릭합니다.
* 새로 생성된 서비스 계정 옆에 있는 **Copy API key**를 클릭합니다.
* 복사한 API 키를 비밀 관리자 또는 안전하지만 엑세스 가능한 다른 위치에 저장합니다.

{{% alert %}}
조직 범위의 서비스 계정은 조직 내의 모든 팀이 소유한 제한되지 않은 프로젝트에 엑세스할 수 있더라도 기본 팀이 필요합니다. 이는 모델 트레이닝 또는 생성적 AI 앱의 환경에서 `WANDB_ENTITY` 변수가 설정되지 않은 경우 워크로드가 실패하는 것을 방지하는 데 도움이 됩니다. 다른 팀의 프로젝트에 조직 범위의 서비스 계정을 사용하려면 `WANDB_ENTITY` 환경 변수를 해당 팀으로 설정해야 합니다.
{{% /alert %}}

## 팀 범위 서비스 계정

팀 범위의 서비스 계정은 해당 팀의 [제한된 프로젝트]({{< relref path="../access-management/restricted-projects.md#visibility-scopes" lang="ko" >}})를 제외하고 팀 내의 모든 프로젝트에서 읽고 쓸 수 있습니다. 팀 범위의 서비스 계정이 제한된 프로젝트에 엑세스하려면 해당 프로젝트의 관리자가 서비스 계정을 프로젝트에 명시적으로 추가해야 합니다.

팀 관리자는 팀의 `<WANDB_HOST_URL>/<your-team-name>/service-accounts`에서 팀 범위의 서비스 계정에 대한 API 키를 얻을 수 있습니다. 또는 팀의 **Team settings**로 이동한 다음 **Service Accounts** 탭을 참조할 수 있습니다.

팀의 새로운 팀 범위 서비스 계정을 만들려면:

* 팀의 **Service Accounts** 탭에서 **New service account** 버튼을 클릭합니다.
* **Name**을 입력합니다.
* 인증 방법으로 **Generate API key (Built-in)**을 선택합니다.
* **Create**를 클릭합니다.
* 새로 생성된 서비스 계정 옆에 있는 **Copy API key**를 클릭합니다.
* 복사한 API 키를 비밀 관리자 또는 안전하지만 엑세스 가능한 다른 위치에 저장합니다.

팀 범위의 서비스 계정을 사용하는 모델 트레이닝 또는 생성적 AI 앱 환경에서 팀을 구성하지 않으면 모델 run 또는 weave 추적이 서비스 계정의 상위 팀 내에서 명명된 프로젝트에 기록됩니다. 이러한 시나리오에서 `WANDB_USERNAME` 또는 `WANDB_USER_EMAIL` 변수를 사용한 사용자 속성은 참조된 사용자가 서비스 계정의 상위 팀에 속하지 않는 한 _작동하지 않습니다_.

{{% alert color="warning" %}}
팀 범위의 서비스 계정은 상위 팀과 다른 팀의 [팀 또는 제한 범위 프로젝트]({{< relref path="../access-management/restricted-projects.md#visibility-scopes" lang="ko" >}})에 run을 기록할 수 없지만 다른 팀 내에서 공개 가시성 프로젝트에 run을 기록할 수 있습니다.
{{% /alert %}}

### 외부 서비스 계정

**Built-in** 서비스 계정 외에도 W&B는 JSON 웹 토큰(JWT)을 발급할 수 있는 ID 공급자(IdP)와 함께 [Identity federation]({{< relref path="./identity_federation.md#external-service-accounts" lang="ko" >}})를 사용하여 W&B SDK 및 CLI를 통해 팀 범위의 **External service accounts**를 지원합니다.

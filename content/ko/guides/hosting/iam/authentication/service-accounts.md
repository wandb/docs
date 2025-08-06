---
title: 서비스 계정을 사용하여 워크플로우를 자동화하세요
description: 조직 및 팀 범위의 서비스 계정을 사용하여 자동화된 또는 비대화형 워크플로우를 관리하세요.
displayed_sidebar: default
menu:
  default:
    identifier: ko-guides-hosting-iam-authentication-service-accounts
---

서비스 계정은 프로젝트 내 또는 여러 팀에 걸쳐 일반적인 작업을 자동으로 수행할 수 있는 비인간/머신 사용자입니다.

- 조직 관리자는 조직 범위에서 서비스 계정을 생성할 수 있습니다.
- 팀 관리자는 해당 팀 범위에서 서비스 계정을 생성할 수 있습니다.

서비스 계정의 API 키를 사용하면 호출자가 서비스 계정의 범위 내에 있는 프로젝트에서 읽기 및 쓰기를 할 수 있습니다.

서비스 계정을 사용하면 여러 사용자 또는 팀이 워크플로우를 중앙에서 관리할 수 있으며, W&B Models 실험 추적을 자동화하거나 W&B Weave의 트레이스를 로그하는 것도 가능합니다. 서비스 계정으로 관리되는 워크플로우에 인간 사용자의 신원을 연결하려면 [환경 변수]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}}) `WANDB_USERNAME` 또는 `WANDB_USER_EMAIL` 중 하나를 사용할 수 있습니다.

{{% alert %}}
서비스 계정은 [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}), 엔터프라이즈 라이선스가 있는 [셀프 관리 인스턴스]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}), 그리고 [SaaS 클라우드]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})의 엔터프라이즈 계정에서 사용할 수 있습니다.
{{% /alert %}}

## 조직 범위 서비스 계정

조직 범위의 서비스 계정은 팀과 상관없이 조직 내 모든 프로젝트에서 읽기와 쓰기가 가능합니다. 단, [제한된 프로젝트]({{< relref path="../access-management/restricted-projects.md#visibility-scopes" lang="ko" >}})는 제외됩니다. 조직 범위 서비스 계정이 제한된 프로젝트에 엑세스하려면 그 프로젝트의 관리자가 서비스 계정을 해당 프로젝트에 명시적으로 추가해야 합니다.

조직 관리자는 조직 또는 계정 대시보드의 **Service Accounts** 탭에서 조직 범위 서비스 계정의 API 키를 얻을 수 있습니다.

새 조직 범위 서비스 계정을 생성하려면:

* 조직 대시보드의 **Service Accounts** 탭에서 **New service account** 버튼을 클릭하세요.
* **Name**을 입력하세요.
* 서비스 계정의 기본 팀을 선택하세요.
* **Create**를 클릭하세요.
* 새로 만들어진 서비스 계정 옆에서 **Copy API key**를 클릭하세요.
* 복사된 API 키는 시크릿 매니저 또는 안전하면서도 엑세스 가능한 장소에 저장하세요.

{{% alert %}}
조직 범위 서비스 계정은 조직 내 모든 팀의 비제한 프로젝트에 엑세스할 수 있지만, 기본 팀이 필요합니다. 이는 모델 트레이닝이나 생성형 AI 앱 환경에서 `WANDB_ENTITY` 변수가 설정되지 않은 경우 workload가 실패하지 않도록 도와줍니다. 다른 팀의 프로젝트에 조직 범위 서비스 계정을 사용하려면, 해당 팀으로 `WANDB_ENTITY` 환경 변수를 설정해야 합니다.
{{% /alert %}}

## 팀 범위 서비스 계정

팀 범위의 서비스 계정은 자신의 팀 내 모든 프로젝트에서 읽기와 쓰기가 가능하지만, 그 팀 내의 [제한된 프로젝트]({{< relref path="../access-management/restricted-projects.md#visibility-scopes" lang="ko" >}})에는 엑세스할 수 없습니다. 팀 범위 서비스 계정이 제한된 프로젝트에 엑세스하려면 해당 프로젝트의 관리자가 서비스 계정을 프로젝트에 명시적으로 추가해야 합니다.

팀 관리자는 `<WANDB_HOST_URL>/<your-team-name>/service-accounts`에서 팀 범위 서비스 계정의 API 키를 얻을 수 있습니다. 또는 **Team settings**로 이동한 뒤 **Service Accounts** 탭을 참조해도 됩니다.

팀을 위한 새 팀 범위 서비스 계정을 생성하려면:

* 팀의 **Service Accounts** 탭에서 **New service account** 버튼을 클릭하세요.
* **Name**을 입력하세요.
* 인증 메서드로 **Generate API key (Built-in)** 을 선택하세요.
* **Create**를 클릭하세요.
* 새로 만들어진 서비스 계정 옆에서 **Copy API key**를 클릭하세요.
* 복사된 API 키는 시크릿 매니저 또는 안전하면서도 엑세스 가능한 장소에 저장하세요.

팀 범위 서비스 계정을 사용하는 모델 트레이닝 또는 생성형 AI 앱 환경에서 팀을 별도로 지정하지 않으면, 해당 서비스 계정의 부모 팀 내에 있는 지정된 프로젝트에 모델 run이나 weave trace가 로그됩니다. 이 경우 `WANDB_USERNAME` 또는 `WANDB_USER_EMAIL` 변수를 이용한 사용자 attribution은 _작동하지 않습니다_ (참조된 사용자가 서비스 계정의 부모 팀 소속이 아닌 한).

{{% alert color="warning" %}}
팀 범위 서비스 계정은 부모 팀과 다른 팀의 [팀 혹은 제한 범위 프로젝트]({{< relref path="../access-management/restricted-projects.md#visibility-scopes" lang="ko" >}})에는 run을 로그할 수 없지만, 다른 팀의 오픈 상태 프로젝트에는 run을 로그할 수 있습니다.
{{% /alert %}}

### 외부 서비스 계정

**Built-in** 서비스 계정 외에도, W&B는 [Identity federation]({{< relref path="./identity_federation.md#external-service-accounts" lang="ko" >}})을 사용하는 W&B SDK 및 CLI와 함께 팀 범위 **External service accounts**도 지원합니다. 이때 IdP(Identity Provider)가 JWT(JSON Web Token)를 발행할 수 있어야 합니다.
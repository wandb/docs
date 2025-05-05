---
title: What is a service account, and why is it useful?
menu:
  support:
    identifier: ko-support-kb-articles-service_account_useful
support:
- administrator
toc_hide: true
type: docs
url: /ko/support/:filename
---

서비스 계정(Enterprise 전용 기능)은 특정 사용자에게 국한되지 않은 팀 및 프로젝트 전반에서 일반적인 작업을 자동화할 수 있는 비인간 또는 머신 사용자를 나타냅니다. 팀 내에서 서비스 계정을 생성하고 해당 API 키를 사용하여 해당 팀 내의 프로젝트에서 읽고 쓸 수 있습니다.

무엇보다도 서비스 계정은 주기적인 재 트레이닝, 야간 빌드 등과 같이 wandb 에 기록된 자동화된 작업을 추적하는 데 유용합니다. 원하는 경우 [환경 변수]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}}) `WANDB_USERNAME` 또는 `WANDB_USER_EMAIL`을 사용하여 이러한 머신에서 실행된 run 과 사용자 이름을 연결할 수 있습니다.

자세한 내용은 [팀 서비스 계정 행동]({{< relref path="/guides/models/app/settings-page/teams.md#team-service-account-behavior" lang="ko" >}})을 참조하십시오.

팀의 서비스 계정에 대한 API 키는 `<WANDB_HOST_URL>/<your-team-name>/service-accounts`에서 얻을 수 있습니다. 또는 팀의 **Team settings** 으로 이동한 다음 **Service Accounts** 탭을 참조할 수 있습니다.

팀의 새 서비스 계정을 만들려면 다음을 수행하십시오.
* 팀의 **Service Accounts** 탭에서 **+ New service account** 버튼을 누릅니다.
* **Name** 필드에 이름을 입력합니다.
* 인증 방법으로 **Generate API key (Built-in)** 를 선택합니다.
* **Create** 버튼을 누릅니다.
* 새로 생성된 서비스 계정에 대해 **Copy API key** 버튼을 클릭하고 비밀 관리자 또는 다른 안전하지만 액세스 가능한 위치에 저장합니다.

{{% alert %}}
**Built-in** 서비스 계정 외에도 W&B 는 [SDK 및 CLI 에 대한 ID 페더레이션]({{< relref path="/guides/hosting/iam/authentication/identity_federation.md#external-service-accounts" lang="ko" >}})을 사용하는 **External service accounts** 도 지원합니다. JSON Web Tokens (JWT)를 발행할 수 있는 ID 공급자에서 관리되는 서비스 ID 를 사용하여 W&B 작업을 자동화하려는 경우 외부 서비스 계정을 사용하십시오.
{{% /alert %}}

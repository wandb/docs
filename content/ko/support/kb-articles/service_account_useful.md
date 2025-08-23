---
title: 서비스 계정이란 무엇이며, 왜 필요한가요?
menu:
  support:
    identifier: ko-support-kb-articles-service_account_useful
support:
- 관리자
toc_hide: true
type: docs
url: /support/:filename
---

서비스 계정(Enterprise 전용 기능)은 인간이 아닌, 또는 머신 사용자로서 팀과 Projects 전체에 걸쳐 자주 수행하는 작업이나 특정 인간 사용자에 국한되지 않은 작업을 자동화할 수 있습니다. 팀 내에서 서비스 계정을 생성하고, 해당 계정의 API 키를 사용하여 해당 팀의 Projects 에서 읽기 및 쓰기를 수행할 수 있습니다.

서비스 계정은 주기적인 재트레이닝, 야간 빌드 등과 같이 자동화된 작업을 wandb 에 로그할 때 유용합니다. 필요하다면, 이러한 머신이 실행한 Runs 에 [환경 변수]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}}) `WANDB_USERNAME` 또는 `WANDB_USER_EMAIL`을 지정해서 사용자 이름을 연결할 수 있습니다.

자세한 내용은 [Team Service Account 행동]({{< relref path="/guides/models/app/settings-page/teams.md#team-service-account-behavior" lang="ko" >}})을 참고하세요.

팀 내 서비스 계정의 API 키는 `<WANDB_HOST_URL>/<your-team-name>/service-accounts` 에서 확인할 수 있습니다. 또는, 팀의 **Team settings** 에서 **Service Accounts** 탭을 참고할 수도 있습니다.

팀에 새로운 서비스 계정을 생성하려면:
* 팀의 **Service Accounts** 탭에서 **+ New service account** 버튼을 클릭하세요.
* **Name** 필드에 이름을 지정하세요.
* 인증 방식으로 **Generate API key (Built-in)** 을 선택하세요.
* **Create** 버튼을 누르세요.
* 생성된 서비스 계정의 **Copy API key** 버튼을 눌러 API 키를 복사한 후, 시크릿 매니저 등 안전하면서도 엑세스 가능한 곳에 보관하세요.

{{% alert %}}
**Built-in** 서비스 계정 이외에, W&B 는 [SDK 및 CLI용 아이덴티티 페더레이션]({{< relref path="/guides/hosting/iam/authentication/identity_federation.md#external-service-accounts" lang="ko" >}})을 통한 **External service accounts** 도 지원합니다. 아이덴티티 공급자에서 관리하는 서비스 아이덴티티를 통해 JSON Web Token (JWT) 발급이 가능하다면, External service accounts 를 활용해 W&B 작업을 자동화하는 것이 가능합니다.
{{% /alert %}}
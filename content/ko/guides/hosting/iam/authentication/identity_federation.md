---
title: Use federated identities with SDK
menu:
  default:
    identifier: ko-guides-hosting-iam-authentication-identity_federation
    parent: authentication
---

W&B SDK를 통해 조직의 자격 증명을 사용하여 로그인하려면 아이덴티티 연동을 사용하세요. W&B 조직 관리자가 조직에 대해 SSO를 구성한 경우, 이미 조직의 자격 증명을 사용하여 W&B 앱 UI에 로그인하고 있을 것입니다. 이러한 의미에서 아이덴티티 연동은 W&B SDK에 대한 SSO와 유사하지만, JSON Web Tokens (JWTs)를 직접 사용한다는 차이점이 있습니다. 아이덴티티 연동은 API 키의 대안으로 사용할 수 있습니다.

[RFC 7523](https://datatracker.ietf.org/doc/html/rfc7523)은 SDK와의 아이덴티티 연동의 기본 토대를 형성합니다.

{{% alert %}}
아이덴티티 연동은 모든 플랫폼 유형 (SaaS Cloud, Dedicated Cloud, 자체 관리 인스턴스)의 `Enterprise` 플랜에서 `Preview`로 사용할 수 있습니다. 궁금한 점이 있으면 W&B 팀에 문의하십시오.
{{% /alert %}}

{{% alert %}}
이 문서의 목적을 위해 `아이덴티티 공급자`와 `JWT 발급자`라는 용어는 같은 의미로 사용됩니다. 둘 다 이 기능의 맥락에서 동일한 대상을 지칭합니다.
{{% /alert %}}

## JWT 발급자 설정

첫 번째 단계로, 조직 관리자는 W&B 조직과 공개적으로 엑세스 가능한 JWT 발급자 간의 연동을 설정해야 합니다.

* 조직 대시보드의 **설정** 탭으로 이동합니다.
* **인증** 옵션에서 `JWT 발급자 설정`을 누릅니다.
* 텍스트 상자에 JWT 발급자 URL을 추가하고 `생성`을 누릅니다.

W&B는 자동으로 `${ISSUER_URL}/.well-known/oidc-configuration` 경로에서 OIDC 검색 문서를 찾고, 검색 문서의 관련 URL에서 JSON Web Key Set (JWKS)를 찾으려고 시도합니다. JWKS는 JWT가 관련 아이덴티티 공급자에 의해 발급되었는지 확인하기 위해 JWT의 실시간 유효성 검사에 사용됩니다.

## JWT를 사용하여 W&B에 엑세스하기

W&B 조직에 대해 JWT 발급자가 설정되면, 사용자는 해당 아이덴티티 공급자가 발급한 JWT를 사용하여 관련 W&B 프로젝트에 엑세스할 수 있습니다. JWT 사용 메커니즘은 다음과 같습니다.

* 조직에서 사용할 수 있는 메커니즘 중 하나를 사용하여 아이덴티티 공급자에 로그인해야 합니다. 일부 공급자는 API 또는 SDK를 사용하여 자동화된 방식으로 엑세스할 수 있는 반면, 일부는 관련 UI를 사용해야만 엑세스할 수 있습니다. 자세한 내용은 W&B 조직 관리자 또는 JWT 발급자 소유자에게 문의하십시오.
* 아이덴티티 공급자에 로그인한 후 JWT를 검색했으면, 안전한 위치의 파일에 저장하고 환경 변수 `WANDB_IDENTITY_TOKEN_FILE`에 절대 파일 경로를 구성합니다.
* W&B SDK 또는 CLI를 사용하여 W&B 프로젝트에 엑세스합니다. SDK 또는 CLI는 JWT를 자동으로 감지하고 JWT가 성공적으로 유효성 검사된 후 W&B 엑세스 토큰으로 교환해야 합니다. W&B 엑세스 토큰은 AI 워크플로우를 활성화하기 위한 관련 API에 엑세스하는 데 사용됩니다. 즉, Runs, 메트릭, Artifacts 등을 기록하는 데 사용됩니다. 엑세스 토큰은 기본적으로 `~/.config/wandb/credentials.json` 경로에 저장됩니다. 환경 변수 `WANDB_CREDENTIALS_FILE`을 지정하여 해당 경로를 변경할 수 있습니다.

{{% alert %}}
JWT는 API 키, 비밀번호 등과 같은 오래 지속되는 자격 증명의 단점을 해결하기 위한 짧은 수명의 자격 증명입니다. 아이덴티티 공급자에 구성된 JWT 만료 시간에 따라 JWT를 계속 갱신하고 환경 변수 `WANDB_IDENTITY_TOKEN_FILE`에서 참조하는 파일에 저장해야 합니다.

W&B 엑세스 토큰에도 기본 만료 기간이 있으며, 그 후 SDK 또는 CLI는 JWT를 사용하여 자동으로 갱신하려고 시도합니다. 사용자 JWT도 그때까지 만료되어 갱신되지 않으면 인증 실패가 발생할 수 있습니다. 가능하다면, JWT 검색 및 만료 후 갱신 메커니즘은 W&B SDK 또는 CLI를 사용하는 AI 워크로드의 일부로 구현되어야 합니다.
{{% /alert %}}

### JWT 유효성 검사

JWT를 W&B 엑세스 토큰으로 교환한 다음 프로젝트에 엑세스하는 워크플로우의 일부로, JWT는 다음 유효성 검사를 거칩니다.

* JWT 서명은 W&B 조직 수준에서 JWKS를 사용하여 확인됩니다. 이는 첫 번째 방어선이며, 실패하면 JWKS 또는 JWT 서명 방식에 문제가 있음을 의미합니다.
* JWT의 `iss` 클레임은 조직 수준에서 구성된 발급자 URL과 같아야 합니다.
* JWT의 `sub` 클레임은 W&B 조직에 구성된 사용자의 이메일 주소와 같아야 합니다.
* JWT의 `aud` 클레임은 AI 워크플로우의 일부로 엑세스하는 프로젝트를 수용하는 W&B 조직의 이름과 같아야 합니다. [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 또는 [자체 관리]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) 인스턴스의 경우, 인스턴스 수준 환경 변수 `SKIP_AUDIENCE_VALIDATION`을 `true`로 구성하여 대상 클레임의 유효성 검사를 건너뛰거나 `wandb`를 대상으로 사용할 수 있습니다.
* JWT의 `exp` 클레임은 토큰이 유효한지 또는 만료되었는지 확인하고 갱신해야 하는지 확인하기 위해 검사됩니다.

## 외부 서비스 계정

W&B는 오랫동안 오래 지속되는 API 키가 있는 기본 제공 서비스 계정을 지원해 왔습니다. SDK 및 CLI에 대한 아이덴티티 연동 기능을 사용하면 JWT를 인증에 사용할 수 있는 외부 서비스 계정을 가져올 수도 있습니다. 단, 해당 계정이 조직 수준에서 구성된 동일한 발급자가 발급한 것이어야 합니다. 팀 관리자는 기본 제공 서비스 계정과 마찬가지로 팀 범위 내에서 외부 서비스 계정을 구성할 수 있습니다.

외부 서비스 계정을 구성하려면:

* 팀의 **서비스 계정** 탭으로 이동합니다.
* `새 서비스 계정`을 누릅니다.
* 서비스 계정 이름을 제공하고 `인증 방법`으로 `Federated Identity`를 선택하고 `Subject`를 제공하고 `생성`을 누릅니다.

외부 서비스 계정의 `sub` 클레임은 팀 관리자가 팀 수준 **서비스 계정** 탭에서 해당 계정의 주제로 구성한 것과 동일해야 합니다. 해당 클레임은 [JWT 유효성 검사]({{< relref path="#jwt-validation" lang="ko" >}})의 일부로 확인됩니다. `aud` 클레임 요구 사항은 사람 사용자 JWT와 유사합니다.

[외부 서비스 계정의 JWT를 사용하여 W&B에 엑세스하는 경우]({{< relref path="#using-the-jwt-to-access-wb" lang="ko" >}}), 초기 JWT를 생성하고 지속적으로 갱신하는 워크플로우를 자동화하는 것이 일반적으로 더 쉽습니다. 외부 서비스 계정을 사용하여 기록된 Runs를 사람 사용자에게 귀속시키려면, 기본 제공 서비스 계정에서 수행하는 방식과 유사하게 AI 워크플로우에 대해 환경 변수 `WANDB_USERNAME` 또는 `WANDB_USER_EMAIL`을 구성할 수 있습니다.

{{% alert %}}
W&B는 유연성과 단순성 간의 균형을 유지하기 위해 다양한 수준의 데이터 민감도를 가진 AI 워크로드에서 기본 제공 및 외부 서비스 계정을 혼합하여 사용하는 것이 좋습니다.
{{% /alert %}}

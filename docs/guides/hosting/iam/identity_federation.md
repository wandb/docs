---
title: Use federated identities with SDK
displayed_sidebar: default
---

W&B SDK를 통해 조직 자격 증명을 사용하여 ID 연합으로 로그인하세요. 만약 조직의 W&B 관리자에 의해 SSO가 설정되어 있다면, 이미 W&B 앱 UI에 로그인할 때 조직 자격 증명을 사용하게 됩니다. 이러한 의미에서 ID 연합은 W&B SDK에서 사용되는 SSO와 비슷하지만 JSON 웹 토큰(JWT)을 직접 사용합니다. ID 연합은 API 키의 대안으로 사용할 수 있습니다.

[RFC 7523](https://datatracker.ietf.org/doc/html/rfc7523)은 SDK와의 ID 연합을 위한 기본 기반을 형성합니다.

:::info
ID 연합은 모든 플랫폼 유형의 `Enterprise` 플랜에서 `Preview`로 사용 가능합니다 - SaaS Cloud, Dedicated Cloud, 그리고 Self-managed 인스턴스. 문의 사항이 있으면 W&B 팀에 문의하세요.
:::

:::note
이 문서의 목적을 위해, `identity provider`와 `JWT issuer`라는 용어는 서로 교환 가능하게 사용됩니다. 이 두 용어는 이 기능의 컨텍스트에서 같은 것을 의미합니다.
:::

## JWT 발급자 설정

첫 번째 단계로, 조직 관리자는 W&B 조직과 공개적으로 액세스 가능한 JWT 발급자 간의 연합을 설정해야 합니다.

* 조직 대시보드에서 **Settings** 탭으로 이동하십시오.
* **Authentication** 옵션에서 `Set up JWT Issuer`를 누르세요.
* JWT 발급자 URL을 텍스트 상자에 추가하고 `Create`를 누르세요.

W&B는 경로 `${ISSUER_URL}/.well-known/oidc-configuration`에서 OIDC 발견 문서를 자동으로 찾고, 발견 문서의 관련 URL에서 JSON 웹 키 세트(JWKS)를 찾으려고 합니다. JWKS는 적절한 ID 제공자가 발급했는지를 확인하기 위해 JWT를 실시간으로 검증하는 데 사용됩니다.

## JWT를 사용하여 W&B 엑세스하기

한 번 W&B 조직에 대한 JWT 발급자가 설정되면, 사용자는 해당 ID 제공자가 발행한 JWT를 사용하여 관련 W&B Projects에 접근할 수 있습니다. JWT를 사용하는 메커니즘은 다음과 같습니다.

* 조직에서 이용 가능한 메커니즘 중 하나를 사용하여 ID 제공자에 로그인해야 합니다. 일부 제공자는 API나 SDK를 통해 자동으로 접근할 수 있으며, 일부는 관련 UI를 통해서만 접근할 수 있습니다. 자세한 사항은 W&B 조직 관리자나 JWT 발급자 소유자에게 문의하세요.
* ID 제공자에 로그인한 후 JWT를 검색하면 보안된 위치에 파일로 저장하고 환경 변수 `WANDB_IDENTITY_TOKEN_FILE`에 절대 파일 경로를 설정하십시오.
* W&B SDK나 CLI를 사용하여 W&B Project에 엑세스하세요. SDK나 CLI는 JWT를 자동으로 감지하고 JWT가 성공적으로 검증된 후 W&B 엑세스 토큰으로 교환해야 합니다. W&B 엑세스 토큰은 기본적으로 경로 `~/.config/wandb/credentials.json`에 저장됩니다. 환경 변수 `WANDB_CREDENTIALS_FILE`을 지정하여 해당 경로를 변경할 수 있습니다.

:::info
JWT는 API 키, 비밀번호 등과 같은 장기간 자격 증명의 단점을 해결하기 위한 단기 자격 증명으로 설계되었습니다. ID 제공자에 설정된 JWT 만료 시간에 따라, JWT를 지속적으로 갱신하고, 환경 변수 `WANDB_IDENTITY_TOKEN_FILE`에 참조된 파일에 저장되도록 해야 합니다.

W&B 액세스 토큰도 기본 만료 기간이 있으며, 그 이후 SDK나 CLI는 자동으로 JWT를 사용하여 갱신을 시도합니다. 그 시간까지 만약 사용자 JWT 또한 만료되고 갱신되지 않았다면 인증 실패를 초래할 수 있습니다. 가능하다면 JWT 검색 및 만료 후 갱신 메커니즘은 W&B SDK나 CLI를 사용하는 AI 작업의 일환으로 구현되어야 합니다.
:::

### JWT 검증

JWT를 W&B 엑세스 토큰으로 교환하고 프로젝트에 엑세스하기 위한 워크플로우의 일환으로, JWT는 다음과 같은 검증을 거칩니다.

* W&B 조직 수준에서 JWKS를 사용하여 JWT 서명이 검증됩니다. 이는 첫 번째 방어선이며, 실패할 경우 JWKS에 문제가 있거나 JWT 서명 방식에 문제 있다는 것을 의미합니다.
* JWT의 `iss` 클레임은 조직 수준에서 설정된 발급자 URL과 같아야 합니다.
* JWT의 `sub` 클레임은 W&B 조직에 설정된 사용자의 이메일 주소와 같아야 합니다.
* JWT의 `aud` 클레임은 당신이 AI 워크플로우의 일환으로 접근하고 있는 프로젝트를 포함한 W&B 조직의 이름과 같아야 합니다. [Dedicated Cloud](../hosting-options/dedicated_cloud.md) 또는 [Self-managed](../hosting-options/self-managed.md) 인스턴스의 경우, 인스턴스 수준의 환경 변수 `SKIP_AUDIENCE_VALIDATION`를 `true`로 설정하여 대상 클레임 검증을 건너뛰거나 `wandb`를 대상자로 사용할 수 있습니다.
* JWT의 `exp` 클레임을 확인하여 토큰이 유효한지 또는 만료되어 갱신이 필요한지 확인합니다.

## 외부 서비스 계정

W&B는 긴 수명의 API 키를 갖춘 내장 서비스 계정을 오랫동안 지원해 왔습니다. SDK 및 CLI에 대한 ID 연합 기능을 통해 조직 수준에서 설정된 같은 발급자가 발행한 경우 JWT를 인증에 사용할 수 있는 외부 서비스 계정도 가져올 수 있습니다. 팀 관리자는 내장 서비스 계정과 같이 팀의 범위 내에서 외부 서비스 계정을 설정할 수 있습니다.

외부 서비스 계정을 설정하려면:

* 팀의 **Service Accounts** 탭으로 이동
* `New service account`를 누르세요.
* 서비스 계정의 이름을 제공하고, `Authentication Method`로 `Federated Identity`를 선택하고, `Subject`을 제공하고 `Create`를 누르세요.

외부 서비스 계정의 JWT 내 `sub` 클레임은 팀 관리자가 팀 수준 **Service Accounts** 탭에서 주제로 설정한 것과 동일해야 합니다. 이 클레임은 [JWT 검증](#jwt-validation) 과정의 일부로 확인됩니다. `aud` 클레임 요구 사항은 인적 사용자 JWT와 비슷합니다.

[외부 서비스 계정의 JWT를 사용하여 W&B에 접근](#using-the-jwt-to-access-wb)할 때, 초기 JWT를 생성하고 지속적으로 갱신하는 워크플로우를 자동화하는 것이 일반적으로 더 쉽습니다. 외부 서비스 계정을 사용하여 기록된 run을 인적 사용자에 귀속시키고 싶다면, 내장 서비스 계정의 경우와 유사하게 AI 워크플로우에 대한 환경 변수 `WANDB_USERNAME` 또는 `WANDB_USER_EMAIL`을 설정할 수 있습니다.

:::note
W&B는 유연성과 단순성의 균형을 맞추기 위해, 다양한 수준의 데이터 민감도를 가진 AI 워크로드에서 내장 및 외부 서비스 계정을 혼합하여 사용할 것을 권장합니다.
:::

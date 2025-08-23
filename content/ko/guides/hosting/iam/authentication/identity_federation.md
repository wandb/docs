---
title: SDK에서 연합 ID 사용하기
menu:
  default:
    identifier: ko-guides-hosting-iam-authentication-identity_federation
    parent: authentication
---

W&B SDK를 통해 조직의 자격 증명을 사용하여 ID 연동(Identity Federation)으로 로그인할 수 있습니다. 만약 W&B 조직의 관리자가 조직을 위한 SSO를 구성했다면, 이미 W&B 앱 UI에서 조직 자격 증명을 사용해 로그인하고 계실 것입니다. 그런 의미에서 ID 연동은 W&B SDK를 위한 SSO와 비슷하지만, JSON Web Token(JWT)을 직접 사용한다는 점이 다릅니다. ID 연동은 API 키의 대안으로 사용할 수 있습니다.

[RFC 7523](https://datatracker.ietf.org/doc/html/rfc7523)는 SDK와의 ID 연동에 대한 기반을 형성합니다.

{{% alert %}}
ID 연동은 모든 플랫폼 유형( SaaS Cloud, 전용 클라우드, 자체 관리형 인스턴스 )의 `Enterprise` 플랜에서 `Preview`로 제공됩니다. 문의 사항이 있으시면 W&B 팀에 문의하십시오.
{{% /alert %}}

{{% alert %}}
이 문서에서는 `identity provider(신원 제공자)`와 `JWT issuer(JWT 발급자)` 용어를 동의어로 사용합니다. 둘 다 이 기능의 맥락에서는 같은 것을 의미합니다.
{{% /alert %}}

## JWT 발급자 설정

첫 번째 단계로, 조직 관리자가 W&B 조직과 공개적으로 접근 가능한 JWT 발급자 간에 연동(Federation)을 설정해야 합니다.

* 조직 대시보드의 **Settings** 탭으로 이동합니다.
* **Authentication** 옵션에서 `Set up JWT Issuer` 버튼을 누릅니다.
* 텍스트 박스에 JWT 발급자 URL을 입력한 뒤 `Create`를 클릭합니다.

W&B는 자동으로 `${ISSUER_URL}/.well-known/oidc-configuration` 경로에서 OIDC 검색 문서를 찾고, 해당 문서 내에서 JSON Web Key Set(JWKS)이 위치한 관련 URL을 확인합니다. JWKS는 JWT가 해당 신원 제공자에서 발급되었는지 실시간으로 검증하는 데 사용됩니다.

## JWT를 이용한 W&B 엑세스

JWT 발급자가 W&B 조직에 대해 설정되면, 사용자는 해당 신원 제공자에서 발급받은 JWT로 관련 W&B 프로젝트에 엑세스할 수 있습니다. JWT 사용 방식은 다음과 같습니다.

* 조직에서 사용할 수 있는 메커니즘 중 하나로 신원 제공자에 로그인해야 합니다. 일부 제공자는 API 또는 SDK를 통해 자동화된 방식으로 접근할 수 있고, 일부는 관련 UI를 통해서만 접근할 수 있습니다. 자세한 내용은 W&B 조직의 관리자나 JWT 발급 담당자에게 문의하세요.
* 신원 제공자에 로그인한 뒤 JWT를 발급받으면, 해당 토큰을 안전한 위치의 파일에 저장하고, 해당 파일의 절대 경로를 환경 변수 `WANDB_IDENTITY_TOKEN_FILE`에 설정합니다.
* W&B SDK나 CLI를 이용해 W&B 프로젝트에 엑세스하세요. SDK 또는 CLI는 JWT를 자동으로 감지하고, 유효성 검증이 성공하면 해당 JWT를 W&B 엑세스 토큰으로 교환합니다. 이 W&B 엑세스 토큰은 AI 워크플로우 활성화를 위한 API 엑세스( Ex: run, 메트릭, 아티팩트 등 )에 사용됩니다. 엑세스 토큰은 기본적으로 `~/.config/wandb/credentials.json` 경로에 저장되며, 환경 변수 `WANDB_CREDENTIALS_FILE`을 지정해 경로를 변경할 수 있습니다.

{{% alert %}}
JWT는 API 키, 비밀번호 등과 같이 장기간 사용되는 자격 증명의 단점을 보완하려는 단기(Short-lived) 자격 증명입니다. 신원 제공자에서 설정한 JWT 만료 시간에 따라, JWT를 지속적으로 갱신하고 환경 변수 `WANDB_IDENTITY_TOKEN_FILE`이 참조하는 파일에 항상 최신 JWT가 저장되도록 해야 합니다.

W&B 엑세스 토큰 또한 기본 만료 시간이 존재하며, 만료 시 SDK 혹은 CLI가 자동으로 JWT를 이용해 재발급을 시도합니다. 이때 사용자의 JWT도 만료되어 갱신되지 않은 상태라면 인증이 실패할 수 있습니다. 가능하다면, JWT의 발급과 만료 후 갱신 프로세스를 W&B SDK나 CLI를 사용하는 AI 워크로드 내에 포함시키는 것이 좋습니다.
{{% /alert %}}

### JWT 검증

JWT를 W&B 엑세스 토큰으로 교환하고, 프로젝트에 엑세스하는 워크플로우의 일부로 JWT에는 아래 검증이 수행됩니다.

* JWT 서명은 조직 레벨의 JWKS를 이용해 검증됩니다. 이 단계가 실패하면, JWKS 또는 JWT 서명 방식에 문제가 있다는 의미입니다.
* JWT의 `iss` 클레임은 조직 레벨에 구성된 발급자 URL과 같아야 합니다.
* JWT의 `sub` 클레임은 W&B 조직에 등록된 사용자의 이메일 주소와 같아야 합니다.
* JWT의 `aud` 클레임은 AI 워크플로우로 접근하려는 프로젝트를 보유한 W&B 조직명의 값이어야 합니다. [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 또는 [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) 인스턴스의 경우, 인스턴스 레벨 환경 변수 `SKIP_AUDIENCE_VALIDATION`을 `true`로 설정하여 `audience` 클레임 검증을 생략하거나, `wandb` 값을 사용할 수 있습니다.
* JWT의 `exp` 클레임을 확인하여 토큰의 유효 기간이 남아 있는지, 만료되어 갱신이 필요한지 확인합니다.

## 외부 서비스 계정

W&B는 오랫동안 장기 API 키를 사용하는 내장 서비스 계정을 지원해왔습니다. SDK 및 CLI에서도 ID 연동 기능이 지원됨에 따라, 동일한 발급자에서 발급된 JWT만 사용한다는 조건 하에, 외부 서비스 계정을 도입해 JWT로 인증할 수 있습니다. 팀 관리자는 해당 팀의 범위 내에서 내장 서비스 계정처럼 외부 서비스 계정을 구성할 수 있습니다.

외부 서비스 계정 구성 방법:

* 팀의 **Service Accounts** 탭으로 이동
* `New service account` 클릭
* 서비스 계정 이름 입력 후, `Authentication Method`로 `Federated Identity`를 선택, `Subject` 입력 후 `Create` 클릭

외부 서비스 계정 JWT의 `sub` 클레임 값은 팀 관리자가 팀 레벨 **Service Accounts** 탭에서 주체(Subject)로 구성한 것과 동일해야 합니다. 이 클레임은 [JWT 검증]({{< relref path="#jwt-validation" lang="ko" >}}) 과정에서 검증됩니다. `aud` 클레임은 실제 사용자 JWT와 같은 요구 사항을 갖습니다.

[외부 서비스 계정 JWT로 W&B 엑세스]({{< relref path="#using-the-jwt-to-access-wb" lang="ko" >}})를 할 때는, 초기 JWT를 생성하고 주기적으로 갱신하는 워크플로우를 자동화하기가 일반적으로 더 쉽습니다. 만약 외부 서비스 계정을 이용해 기록된 run을 실제 사용자에게 귀속시키고 싶다면, 내장 서비스 계정과 마찬가지로 워크플로우에 환경 변수 `WANDB_USERNAME` 또는 `WANDB_USER_EMAIL`을 설정할 수 있습니다.

{{% alert %}}
W&B는 여러 수준의 데이터 민감도가 요구되는 AI 워크로드에 대해 내장 서비스 계정과 외부 서비스 계정을 혼합해 사용하는 것을 권장합니다. 이를 통해 유연성과 단순성 간의 균형을 이룰 수 있습니다.
{{% /alert %}}
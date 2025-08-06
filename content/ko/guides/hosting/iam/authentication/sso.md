---
title: OIDC로 SSO 구성하기
menu:
  default:
    identifier: ko-guides-hosting-iam-authentication-sso
    parent: authentication
---

W&B Server는 OpenID Connect (OIDC) 호환 아이덴티티 공급자에 대한 지원을 통해 Okta, Keycloak, Auth0, Google, Entra와 같은 외부 아이덴티티 공급자를 통한 사용자 아이덴티티와 그룹 멤버십 관리를 할 수 있도록 해줍니다.

## OpenID Connect (OIDC)

W&B Server는 외부 Identity Provider (IdP)와 연동하기 위해 아래 OIDC 인증 플로우를 지원합니다.
1. Form Post가 포함된 Implicit Flow
2. Proof Key for Code Exchange (PKCE)가 적용된 Authorization Code Flow

이러한 인증 플로우는 사용자를 인증하고, W&B Server에 접근 제어에 필요한 ID 토큰 형태로 아이덴티티 정보를 제공합니다.

ID 토큰은 사용자의 이름, 사용자 이름, 이메일, 그룹 등의 정보를 포함한 JWT입니다. W&B Server는 이 토큰을 활용하여 사용자를 인증하고, 시스템 내에서 해당 역할 또는 그룹에 매핑합니다.

W&B Server에서는 access 토큰이 사용자를 대신해 API 요청을 허용하지만, 핵심적으로 사용자 인증과 아이덴티티 확보를 위한 것이 목적이므로 ID 토큰만이 필요합니다.

[환경 변수]({{< relref path="../advanced_env_vars.md" lang="ko" >}})를 사용하여 [Dedicated cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 또는 [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) 인스턴스에 IAM 옵션을 설정할 수 있습니다.

[Dedicated cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 또는 [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) W&B Server 설치에서 Identity Provider 설정을 돕기 위해, 다양한 IdP별 가이드라인을 따라 설정하실 수 있습니다. 만약 W&B SaaS 버전을 사용하신다면, 조직 내 Auth0 테넌트 구성 지원을 위해 [support@wandb.com](mailto:support@wandb.com)으로 문의해 주세요.

{{< tabpane text=true >}}
{{% tab header="Cognito" value="cognito" %}}
AWS Cognito를 인증용으로 설정하려면 아래 절차를 따르세요:

1. 먼저 AWS 계정에 로그인 후 [AWS Cognito](https://aws.amazon.com/cognito/) 앱으로 이동합니다.

    {{< img src="/images/hosting/setup_aws_cognito.png" alt="AWS Cognito setup" >}}

2. IdP에서 애플리케이션 설정 시 허용되는 콜백 URL을 입력합니다:
     * `http(s)://YOUR-W&B-HOST/oidc/callback`을 콜백 URL로 추가하세요. `YOUR-W&B-HOST`는 실제 W&B 호스트 주소로 교체해야 합니다.

3. IdP가 universal logout을 지원하는 경우, Logout URL은 `http(s)://YOUR-W&B-HOST`로 설정하세요. `YOUR-W&B-HOST`는 실제 W&B 호스트 주소로 교체해야 합니다.

    예를 들어, 애플리케이션이 `https://wandb.mycompany.com`에서 동작한다면, `YOUR-W&B-HOST`는 `wandb.mycompany.com`으로 대체해야 합니다.

    아래 이미지는 AWS Cognito에 콜백 및 로그아웃 URL을 지정하는 방법을 보여줍니다.

    {{< img src="/images/hosting/setup_aws_cognito_ui_settings.png" alt="Host configuration" >}}

    _wandb/local_은 기본적으로 [`implicit` grant 및 `form_post` response type](https://auth0.com/docs/get-started/authentication-and-authorization-flow/implicit-flow-with-form-post)을 사용합니다. 

    _wandb/local_에서 [PKCE Code Exchange](https://www.oauth.com/oauth2-servers/pkce/) 흐름을 사용하는 `authorization_code` grant 방식으로 구성할 수도 있습니다.

4. AWS Cognito가 토큰을 앱에 전달하는 방법을 결정하기 위해 하나 이상의 OAuth grant 유형을 선택하세요.
5. W&B 사용을 위해 필요한 OIDC scope를 AWS Cognito App에서 선택합니다:
    * "openid" 
    * "profile"
    * "email"

    예시로, AWS Cognito App UI는 아래 이미지와 유사해야 합니다:

    {{< img src="/images/hosting/setup_aws_required_fields.png" alt="Required fields" >}}

    설정 페이지에서 **Auth Method**를 선택하거나 환경 변수 OIDC_AUTH_METHOD를 설정하여 _wandb/local_이 사용할 grant 방식을 지정해야 합니다.

    Auth Method는 반드시 `pkce`로 지정해야 합니다.

6. Client ID와 OIDC issuer의 URL이 필요합니다. OpenID discovery 문서는 반드시 `$OIDC_ISSUER/.well-known/openid-configuration` 위치에 존재해야 합니다.

    예시로, **App Integration** 탭 내 **User Pools** 섹션에서 User Pool ID를 Cognito IdP URL 뒤에 붙여 issuer URL을 생성할 수 있습니다:

    {{< img src="/images/hosting/setup_aws_cognito_issuer_url.png" alt="AWS Cognito issuer URL" >}}

    "Cognito domain"을 IDP URL로 사용하지 마세요. Cognito의 discovery 문서는 `https://cognito-idp.$REGION.amazonaws.com/$USER_POOL_ID` 형태로 제공됩니다.

{{% /tab %}}

{{% tab header="Okta" value="okta"%}}
Okta를 인증용으로 설정하려면 다음 단계를 따르세요:

1. [Okta Portal](https://login.okta.com/)에 로그인합니다.

2. 왼쪽에서 **Applications**를 선택한 뒤, 다시 **Applications**를 클릭하세요.
    {{< img src="/images/hosting/okta_select_applications.png" alt="Okta Applications menu" >}}

3. "Create App integration" 버튼을 클릭하세요.
    {{< img src="/images/hosting/okta_create_new_app_integration.png" alt="Create App integration button" >}}

4. "Create a new app integration" 화면에서 **OIDC - OpenID Connect**와 **Single-Page Application**을 선택 후 "Next"를 클릭하세요.
    {{< img src="/images/hosting/okta_create_a_new_app_integration.png" alt="OIDC Single-Page Application selection" >}}

5. "New Single-Page App Integration" 화면에서 아래 값을 입력 후 **Save**를 클릭하세요:
    - App integration 이름, 예시: "W&B"
    - Grant type: **Authorization Code**와 **Implicit (hybrid)** 모두 선택
    - Sign-in redirect URIs: https://YOUR_W_AND_B_URL/oidc/callback
    - Sign-out redirect URIs: https://YOUR_W_AND_B_URL/logout
    - Assignments: **Skip group assignment for now** 선택
    {{< img src="/images/hosting/okta_new_single_page_app_integration.png" alt="Single-Page App configuration" >}}

6. 방금 생성한 Okta 애플리케이션의 overview 화면에서 **General** 탭의 **Client Credentials** 아래 **Client ID**를 확인하세요.
    {{< img src="/images/hosting/okta_make_note_of_client_id.png" alt="Okta Client ID location" >}}

7. Okta OIDC Issuer URL을 찾으려면 왼쪽에서 **Settings** > **Account**로 이동하세요. Okta UI에서 **Organization Contact** 아래에 회사명이 표시됩니다.
    {{< img src="/images/hosting/okta_identify_oidc_issuer_url.png" alt="Okta organization settings" >}}

OIDC issuer URL은 아래와 같은 형식입니다: `https://COMPANY.okta.com`. COMPANY 부분을 실제 값으로 바꿔서 기록해두세요.
{{% /tab %}}

{{% tab header="Entra" value="entra"%}}
1. [Azure Portal](https://portal.azure.com/)에 로그인하세요.

2. "Microsoft Entra ID" 서비스를 선택합니다.
    {{< img src="/images/hosting/entra_select_entra_service.png" alt="Microsoft Entra ID service" >}}

3. 왼쪽에서 "App registrations"를 선택하세요.
    {{< img src="/images/hosting/entra_app_registrations.png" alt="App registrations menu" >}}

4. 상단의 "New registration"을 클릭하세요.
    {{< img src="/images/hosting/entra_new_app_registration.png" alt="New registration button" >}}

    "Register an application" 화면에서 아래 내용을 입력하세요:
    {{< img src="/images/hosting/entra_register_an_application.png" alt="Application registration form" >}}

    - 이름 지정 (예: "Weights and Biases application")
    - 기본 계정 유형은: "Accounts in this organizational directory only (Default Directory only - Single tenant)"입니다. 필요에 따라 수정하세요.
    - Redirect URI는 **Web** 타입으로 지정하고 값은 `https://YOUR_W_AND_B_URL/oidc/callback`으로 입력하세요.
    - "Register"를 클릭하세요.

    - "Application (client) ID"와 "Directory (tenant) ID"를 기록해 두세요.

      {{< img src="/images/hosting/entra_app_overview_make_note.png" alt="Application and Directory IDs" >}}

5. 왼쪽 메뉴에서 **Authentication**을 클릭하세요.
    {{< img src="/images/hosting/entra_select_authentication.png" alt="Authentication menu" >}}

    - **Front-channel logout URL** 아래에: `https://YOUR_W_AND_B_URL/logout`을 입력하세요.
    - "Save"를 클릭하세요.

      {{< img src="/images/hosting/entra_logout_url.png" alt="Front-channel logout URL" >}}

6. 왼쪽에서 "Certificates & secrets"를 클릭하세요.
    {{< img src="/images/hosting/entra_select_certificates_secrets.png" alt="Certificates & secrets menu" >}}

    - "Client secrets"를 클릭 후 "New client secret"을 클릭하세요.
      {{< img src="/images/hosting/entra_new_secret.png" alt="New client secret button" >}}

      "Add a client secret" 화면에서 아래처럼 입력하세요:
      {{< img src="/images/hosting/entra_add_new_client_secret.png" alt="Client secret configuration" >}}

      - 설명 입력 (예: "wandb")
      - "Expires"는 기본값을 사용하거나 필요에 따라 변경하세요.
      - "Add"를 클릭하세요.

    - Secret의 "Value"를 기록해 두세요. "Secret ID"는 필요하지 않습니다.
    {{< img src="/images/hosting/entra_make_note_of_secret_value.png" alt="Client secret value" >}}

이제 세 가지 값을 기록해 두셔야 합니다:
- OIDC Client ID
- OIDC Client Secret
- OIDC Issuer URL에 필요한 Tenant ID

OIDC issuer URL의 형식은 다음과 같습니다: `https://login.microsoftonline.com/${TenantID}/v2.0`
{{% /tab %}}
{{< /tabpane >}}

## W&B Server에서 SSO 설정하기

SSO를 설정하려면 관리자 권한과 아래 정보가 필요합니다:
- OIDC Client ID
- OIDC Auth method (`implicit` 또는 `pkce`)
- OIDC Issuer URL
- OIDC Client Secret(선택사항; IdP 설정 방식에 따라 필요함)

IdP에서 OIDC Client Secret을 요구하는 경우, [환경 변수]({{< relref path="/guides/hosting/env-vars.md" lang="ko" >}}) `OIDC_CLIENT_SECRET`로 지정하세요.
- UI에서는 **System Console** > **Settings** > **Advanced** > **User Spec**으로 이동한 후, 아래와 같이 `extraENV` 섹션에 `OIDC_CLIENT_SECRET`을 추가하세요.
- Helm에서는 아래와 같이 `values.global.extraEnv`를 구성하세요.

```yaml
values:
  global:
    extraEnv:
      OIDC_CLIENT_SECRET="<your_secret>"
```

{{% alert %}}
SSO 설정 후 인스턴스에 로그인할 수 없는 경우, 환경 변수 `LOCAL_RESTORE=true`를 설정해 인스턴스를 재시작하세요. 이 경우 컨테이너 로그에 임시 비밀번호가 출력되며 SSO가 비활성화됩니다. 문제를 해결한 후에는 해당 환경 변수를 제거해 SSO를 다시 활성화해야 합니다.
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab header="System Console" value="console" %}}
System Console은 System Settings 페이지의 차세대 버전입니다. [W&B Kubernetes Operator]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/" lang="ko" >}}) 기반 배포에서 사용할 수 있습니다.

1. [W&B Management Console 엑세스 가이드]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/#access-the-wb-management-console" lang="ko" >}})를 참고하세요.

2. **Settings** > **Authentication**로 이동합니다. **Type** 드롭다운에서 **OIDC**를 선택하세요.
    {{< img src="/images/hosting/sso_configure_via_console.png" alt="System Console OIDC configuration" >}}

3. 정보를 입력하세요.

4. **Save**를 클릭하세요.

5. 로그아웃 후, 이번에는 IdP 로그인 화면을 통해 다시 로그인하세요.
{{% /tab %}}
{{% tab header="System settings" value="settings" %}}
1. Weights&Biases 인스턴스에 로그인하세요. 
2. W&B App으로 이동합니다.

    {{< img src="/images/hosting/system_settings.png" alt="W&B App navigation" >}}

3. 드롭다운에서 **System Settings**를 선택하세요:

    {{< img src="/images/hosting/system_settings_select_settings.png" alt="System Settings dropdown" >}}

4. Issuer, Client ID 그리고 Authentication Method를 입력합니다.
5. **Update settings**를 선택하세요.

{{< img src="/images/hosting/system_settings_select_update.png" alt="Update settings button" >}}
{{% /tab %}}
{{< /tabpane >}}

{{% alert %}}
SSO 설정 후 인스턴스에 로그인할 수 없으면, `LOCAL_RESTORE=true` 환경 변수로 인스턴스를 재시작하세요. 임시 비밀번호가 컨테이너 로그에 출력되고 SSO가 꺼집니다. 문제를 해결한 후에는 해당 환경 변수를 반드시 제거해 SSO 사용을 다시 활성화해야 합니다.
{{% /alert %}}

## Security Assertion Markup Language (SAML)
W&B Server는 SAML을 지원하지 않습니다.

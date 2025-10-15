---
title: Configure SSO with OIDC
menu:
  default:
    identifier: ko-guides-hosting-iam-authentication-sso
    parent: authentication
---

Weights & Biases 서버는 OpenID Connect (OIDC) 호환 ID 공급자를 지원하므로 Okta, Keycloak, Auth0, Google, Entra와 같은 외부 ID 공급자를 통해 사용자 ID 및 그룹 멤버십을 관리할 수 있습니다.

## OpenID Connect (OIDC)

W&B 서버는 외부 IdP(Identity Provider)와 통합하기 위해 다음과 같은 OIDC 인증 흐름을 지원합니다.
1. Form Post를 사용하는 암시적 흐름
2. PKCE(Proof Key for Code Exchange)를 사용하는 Authorization Code 흐름

이러한 흐름은 사용자를 인증하고 엑세스 제어를 관리하는 데 필요한 ID 정보(ID 토큰 형태)를 W&B 서버에 제공합니다.

ID 토큰은 이름, 사용자 이름, 이메일, 그룹 멤버십과 같은 사용자 ID 정보가 포함된 JWT입니다. W&B 서버는 이 토큰을 사용하여 사용자를 인증하고 시스템 내 적절한 역할 또는 그룹에 매핑합니다.

W&B 서버의 맥락에서 엑세스 토큰은 사용자를 대신하여 API에 대한 요청을 승인하지만 W&B 서버의 주요 관심사는 사용자 인증 및 ID이므로 ID 토큰만 필요합니다.

환경 변수를 사용하여 [IAM 옵션 구성]({{< relref path="../advanced_env_vars.md" lang="ko" >}})을 [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 또는 [자체 관리형]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) 인스턴스에 설정할 수 있습니다.

[전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 또는 [자체 관리형]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) W&B 서버 설치를 위해 ID 공급자를 구성하는 데 도움이 되도록 다양한 IdP에 대한 다음 가이드라인을 따르세요. SaaS 버전의 W&B를 사용하는 경우 조직의 Auth0 테넌트 구성에 대한 지원은 [support@wandb.com](mailto:support@wandb.com)으로 문의하세요.

{{< tabpane text=true >}}
{{% tab header="Cognito" value="cognito" %}}
인증을 위해 AWS Cognito를 설정하려면 아래 절차를 따르세요.

1. 먼저 AWS 계정에 로그인하고 [AWS Cognito](https://aws.amazon.com/cognito/) 앱으로 이동합니다.

    {{< img src="/images/hosting/setup_aws_cognito.png" alt="인증이 아닌 권한 부여에 OIDC를 사용하는 경우 퍼블릭 클라이언트는 설정을 간소화합니다." >}}

2. 허용된 콜백 URL을 제공하여 IdP에서 애플리케이션을 구성합니다.
     * `http(s)://YOUR-W&B-HOST/oidc/callback`을 콜백 URL로 추가합니다. `YOUR-W&B-HOST`를 W&B 호스트 경로로 바꿉니다.

3. IdP가 유니버설 로그아웃을 지원하는 경우 로그아웃 URL을 `http(s)://YOUR-W&B-HOST`로 설정합니다. `YOUR-W&B-HOST`를 W&B 호스트 경로로 바꿉니다.

    예를 들어 애플리케이션이 `https://wandb.mycompany.com`에서 실행 중인 경우 `YOUR-W&B-HOST`를 `wandb.mycompany.com`으로 바꿉니다.

    아래 이미지는 AWS Cognito에서 허용된 콜백 및 로그아웃 URL을 제공하는 방법을 보여줍니다.

    {{< img src="/images/hosting/setup_aws_cognito_ui_settings.png" alt="인스턴스에 여러 호스트에서 엑세스할 수 있는 경우 여기에 모두 포함해야 합니다." >}}


    _wandb/local_은 기본적으로 [`form_post` 응답 유형과 함께 `암시적` 권한을 사용합니다](https://auth0.com/docs/get-started/authentication-and-authorization-flow/implicit-flow-with-form-post).

    [PKCE Code Exchange](https://www.oauth.com/oauth2-servers/pkce/) 흐름을 사용하는 `authorization_code` 권한을 수행하도록 _wandb/local_을 구성할 수도 있습니다.

4. AWS Cognito가 앱에 토큰을 전달하는 방법을 구성하려면 하나 이상의 OAuth 권한 유형을 선택합니다.
5. W&B에는 특정 OpenID Connect (OIDC) 범위가 필요합니다. AWS Cognito 앱에서 다음을 선택합니다.
    * "openid"
    * "profile"
    * "email"

    예를 들어 AWS Cognito 앱 UI는 다음 이미지와 유사해야 합니다.

    {{< img src="/images/hosting/setup_aws_required_fields.png" alt="필수 필드" >}}

    설정 페이지에서 **인증 방법**을 선택하거나 OIDC_AUTH_METHOD 환경 변수를 설정하여 _wandb/local_에 사용할 권한을 알립니다.

    인증 방법을 `pkce`로 설정해야 합니다.

6. 클라이언트 ID와 OIDC 발급자 URL이 필요합니다. OpenID 검색 문서는 `$OIDC_ISSUER/.well-known/openid-configuration`에서 사용할 수 있어야 합니다.

    예를 들어 **사용자 풀** 섹션 내의 **앱 통합** 탭에서 사용자 풀 ID를 Cognito IdP URL에 추가하여 발급자 URL을 생성할 수 있습니다.

    {{< img src="/images/hosting/setup_aws_cognito_issuer_url.png" alt="AWS Cognito의 발급자 URL 스크린샷" >}}

    IDP URL에 "Cognito 도메인"을 사용하지 마세요. Cognito는 `https://cognito-idp.$REGION.amazonaws.com/$USER_POOL_ID`에서 검색 문서를 제공합니다.

{{% /tab %}}

{{% tab header="Okta" value="okta"%}}
인증을 위해 Okta를 설정하려면 아래 절차를 따르세요.

1. [https://login.okta.com/](https://login.okta.com/) 에서 Okta 포털에 로그인합니다.

2. 왼쪽에서 **애플리케이션**을 선택한 다음 **애플리케이션**을 다시 선택합니다.
    {{< img src="/images/hosting/okta_select_applications.png" alt="" >}}

3. "앱 통합 생성"을 클릭합니다.
    {{< img src="/images/hosting/okta_create_new_app_integration.png" alt="" >}}

4. "새 앱 통합 만들기"라는 화면에서 **OIDC - OpenID Connect** 및 **단일 페이지 애플리케이션**을 선택합니다. 그런 다음 "다음"을 클릭합니다.
    {{< img src="/images/hosting/okta_create_a_new_app_integration.png" alt="" >}}

5. "새 단일 페이지 앱 통합"이라는 화면에서 다음과 같이 값을 채우고 **저장**을 클릭합니다.
    - 앱 통합 이름(예: "Weights & Biases")
    - 권한 유형: **Authorization Code** 및 **Implicit (hybrid)** 모두 선택
    - 로그인 리디렉션 URI: https://YOUR_W_AND_B_URL/oidc/callback
    - 로그아웃 리디렉션 URI: https://YOUR_W_AND_B_URL/logout
    - 할당: **지금 그룹 할당 건너뛰기** 선택
    {{< img src="/images/hosting/okta_new_single_page_app_integration.png" alt="" >}}

6. 방금 생성한 Okta 애플리케이션의 개요 화면에서 **일반** 탭 아래의 **클라이언트 자격 증명** 아래에 있는 **클라이언트 ID**를 기록해 둡니다.
    {{< img src="/images/hosting/okta_make_note_of_client_id.png" alt="" >}}

7. Okta OIDC 발급자 URL을 식별하려면 왼쪽에서 **설정**을 선택한 다음 **계정**을 선택합니다.
    Okta UI는 **조직 연락처** 아래에 회사 이름을 표시합니다.
    {{< img src="/images/hosting/okta_identify_oidc_issuer_url.png" alt="" >}}

OIDC 발급자 URL의 형식은 `https://COMPANY.okta.com`입니다. COMPANY를 해당 값으로 바꿉니다. 기록해 둡니다.
{{% /tab %}}

{{% tab header="Entra" value="entra"%}}
1. [https://portal.azure.com/](https://portal.azure.com/) 에서 Azure Portal에 로그인합니다.

2. "Microsoft Entra ID" 서비스를 선택합니다.
    {{< img src="/images/hosting/entra_select_entra_service.png" alt="" >}}

3. 왼쪽에서 "앱 등록"을 선택합니다.
    {{< img src="/images/hosting/entra_app_registrations.png" alt="" >}}

4. 상단에서 "새 등록"을 클릭합니다.
    {{< img src="/images/hosting/entra_new_app_registration.png" alt="" >}}

    "애플리케이션 등록"이라는 화면에서 다음과 같이 값을 채웁니다.
    {{< img src="/images/hosting/entra_register_an_application.png" alt="" >}}

    - 이름(예: "Weights and Biases application")을 지정합니다.
    - 기본적으로 선택된 계정 유형은 "이 조직 디렉터리의 계정만 해당(기본 디렉터리만 해당 - 단일 테넌트)"입니다. 필요한 경우 수정합니다.
    - **웹** 유형으로 리디렉션 URI를 `https://YOUR_W_AND_B_URL/oidc/callback` 값으로 구성합니다.
    - "등록"을 클릭합니다.

    - "애플리케이션(클라이언트) ID" 및 "디렉터리(테넌트) ID"를 기록해 둡니다.

      {{< img src="/images/hosting/entra_app_overview_make_note.png" alt="" >}}


5. 왼쪽에서 **인증**을 클릭합니다.
    {{< img src="/images/hosting/entra_select_authentication.png" alt="" >}}

    - **프런트 채널 로그아웃 URL** 아래에 `https://YOUR_W_AND_B_URL/logout`을 지정합니다.
    - "저장"을 클릭합니다.

      {{< img src="/images/hosting/entra_logout_url.png" alt="" >}}


6. 왼쪽에서 "인증서 및 암호"를 클릭합니다.
    {{< img src="/images/hosting/entra_select_certificates_secrets.png" alt="" >}}

    - "클라이언트 암호"를 클릭한 다음 "새 클라이언트 암호"를 클릭합니다.
      {{< img src="/images/hosting/entra_new_secret.png" alt="" >}}

      "클라이언트 암호 추가"라는 화면에서 다음과 같이 값을 채웁니다.
      {{< img src="/images/hosting/entra_add_new_client_secret.png" alt="" >}}

      - 설명(예: "wandb")을 입력합니다.
      - "만료"는 그대로 두거나 필요한 경우 변경합니다.
      - "추가"를 클릭합니다.


    - 암호의 "값"을 기록해 둡니다. "암호 ID"는 필요하지 않습니다.
    {{< img src="/images/hosting/entra_make_note_of_secret_value.png" alt="" >}}

이제 세 가지 값을 기록해 두어야 합니다.
- OIDC 클라이언트 ID
- OIDC 클라이언트 암호
- OIDC 발급자 URL에 테넌트 ID가 필요합니다.

OIDC 발급자 URL의 형식은 `https://login.microsoftonline.com/${TenantID}/v2.0`입니다.
{{% /tab %}}
{{< /tabpane >}}

## W&B 서버에서 SSO 설정

SSO를 설정하려면 관리자 권한과 다음 정보가 필요합니다.
- OIDC 클라이언트 ID
- OIDC 인증 방법(`implicit` 또는 `pkce`)
- OIDC 발급자 URL
- OIDC 클라이언트 암호(선택 사항, IdP 설정 방법에 따라 다름)

{{% alert %}}
IdP에 OIDC 클라이언트 암호가 필요한 경우 `OIDC_CLIENT_SECRET` 환경 변수를 사용하여 지정합니다.
{{% /alert %}}

W&B 서버 UI를 사용하거나 [환경 변수]({{< relref path="/guides/hosting/env-vars.md" lang="ko" >}})를 `wandb/local` 포드로 전달하여 SSO를 구성할 수 있습니다. 환경 변수가 UI보다 우선합니다.

{{% alert %}}
SSO를 구성한 후 인스턴스에 로그인할 수 없는 경우 `LOCAL_RESTORE=true` 환경 변수를 설정하여 인스턴스를 다시 시작할 수 있습니다. 그러면 컨테이너 로그에 임시 암호가 출력되고 SSO가 비활성화됩니다. SSO 문제를 해결한 후에는 SSO를 다시 활성화하려면 해당 환경 변수를 제거해야 합니다.
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab header="System Console" value="console" %}}
시스템 콘솔은 시스템 설정 페이지의 후속 버전입니다. [W&B Kubernetes Operator]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/" lang="ko" >}}) 기반 배포에서 사용할 수 있습니다.

1. [W&B 관리 콘솔 엑세스]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/#access-the-wb-management-console" lang="ko" >}})를 참조하세요.

2. **설정**으로 이동한 다음 **인증**으로 이동합니다. **유형** 드롭다운에서 **OIDC**를 선택합니다.
    {{< img src="/images/hosting/sso_configure_via_console.png" alt="" >}}

3. 값을 입력합니다.

4. **저장**을 클릭합니다.

5. 로그아웃한 다음 다시 로그인합니다. 이번에는 IdP 로그인 화면을 사용합니다.
{{% /tab %}}
{{% tab header="System settings" value="settings" %}}
1. Weights & Biases 인스턴스에 로그인합니다.
2. W&B 앱으로 이동합니다.

    {{< img src="/images/hosting/system_settings.png" alt="" >}}

3. 드롭다운에서 **시스템 설정**을 선택합니다.

    {{< img src="/images/hosting/system_settings_select_settings.png" alt="" >}}

4. 발급자, 클라이언트 ID 및 인증 방법을 입력합니다.
5. **설정 업데이트**를 선택합니다.

{{< img src="/images/hosting/system_settings_select_update.png" alt="" >}}
{{% /tab %}}
{{< /tabpane >}}

{{% alert %}}
SSO를 구성한 후 인스턴스에 로그인할 수 없는 경우 `LOCAL_RESTORE=true` 환경 변수를 설정하여 인스턴스를 다시 시작할 수 있습니다. 그러면 컨테이너 로그에 임시 암호가 출력되고 SSO가 꺼집니다. SSO 문제를 해결한 후에는 SSO를 다시 활성화하려면 해당 환경 변수를 제거해야 합니다.
{{% /alert %}}

## SAML(Security Assertion Markup Language)
W&B 서버는 SAML을 지원하지 않습니다.

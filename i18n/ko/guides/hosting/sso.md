---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 싱글 사인온 (SSO) 설정

[contact@wandb.com](mailto:contact@wandb.com)으로 이메일을 보내 W&B에서 지원하는 신원 제공자(예: SAML, Ping Federate, Active Directory 등)로 [Auth0](https://auth0.com) 테넌트를 구성하도록 요청하세요.

이미 Auth0을 사용하거나 Open ID Connect 호환 서버가 있는 경우, 아래 지침에 따라 Open ID를 사용한 인증을 설정하세요.

:::info
W&B 서버는 기본적으로 수동 사용자 관리로 작동합니다. _wandb/local_의 라이선스 버전은 SSO를 활성화합니다.
:::

## Open ID Connect

_wandb/local_은 인증을 위해 Open ID Connect (OIDC)를 사용합니다. 유스 케이스에 따라 탭 중 하나를 선택하여 AWS Cognito 또는 Okta를 사용하여 W&B 서버를 Open ID Connect로 인증하는 방법을 알아보세요.

:::tip
신원 제공자(IdP)에서 단일 페이지 또는 공개 클라이언트 애플리케이션을 선택하세요.
:::

<Tabs
  defaultValue="aws"
  values={[
    {label: 'AWS', value: 'aws'},
    {label: 'Okta', value: 'okta'},
  ]}>
  <TabItem value="aws">

아래 절차에 따라 AWS Cognito를 인증을 위해 설정하세요:

1. 먼저 AWS 계정에 로그인하고 [AWS Cognito](https://aws.amazon.com/cognito/) 앱으로 이동하세요.

![OIDC를 인증에만 사용하므로 공개 클라이언트는 설정을 단순화합니다](/images/hosting/setup_aws_cognito.png)

2. IdP에서 애플리케이션을 구성하기 위해 허용된 콜백 URL을 제공하세요:
     * 콜백 URL로 `http(s)://YOUR-W&B-HOST/oidc/callback`을 추가하세요. `YOUR-W&B-HOST`를 귀하의 W&B 호스트 경로로 대체하세요.

3. IdP가 범용 로그아웃을 지원하는 경우, 로그아웃 URL을 `http(s)://YOUR-W&B-HOST`로 설정하세요. `YOUR-W&B-HOST`를 귀하의 W&B 호스트 경로로 대체하세요.

예를 들어, 애플리케이션이 `https://wandb.mycompany.com`에서 실행되는 경우, `YOUR-W&B-HOST`를 `wandb.mycompany.com`으로 대체하세요.

아래 이미지는 AWS Cognito에서 허용된 콜백 및 로그아웃 URL을 제공하는 방법을 보여줍니다.

![인스턴스가 여러 호스트에서 접근 가능한 경우 여기에 모두 포함시키세요.](/images/hosting/setup_aws_cognito_ui_settings.png)

_wandb/local_은 기본적으로 ["implicit" 그랜트와 "form\_post" 응답 유형](https://auth0.com/docs/get-started/authentication-and-authorization-flow/implicit-flow-with-form-post)을 사용합니다.

또한, _wandb/local_을 [PKCE 코드 교환](https://www.oauth.com/oauth2-servers/pkce/) 흐름을 사용하는 "authorization\_code" 그랜트로 구성할 수 있습니다.

4. AWS Cognito가 앱에 토큰을 전달하는 방식을 구성하려면 하나 이상의 OAuth 그랜트 유형을 선택하세요.
5. W&B는 특정 OpenID Connect (OIDC) 범위를 요구합니다. AWS Cognito 앱에서 다음을 선택하세요:
    * "openid"
    * "profile"
    * "email"

예를 들어, 귀하의 AWS Cognito 앱 UI는 아래 이미지와 유사해야 합니다:

![openid, profile, 및 email이 필요합니다](/images/hosting/setup_aws_required_fields.png)

설정 페이지에서 **인증 메소드**를 선택하거나 OIDC\_AUTH\_METHOD 환경 변수를 설정하여 _wandb/local_이 어떤 그랜트를 사용할지 알려주세요.

:::info
AWS Cognito 제공자의 경우 인증 메소드를 "pkce"로 설정해야 합니다.
:::

6. 클라이언트 ID와 OIDC 발급자의 URL이 필요합니다. OpenID 발견 문서는 `$OIDC_ISSUER/.well-known/openid-configuration`에서 사용 가능해야 합니다.

예를 들어, AWS Cognito를 사용하는 경우, **앱 통합** 탭 내의 **사용자 풀** 섹션에서 사용자 풀 ID를 Cognito IdP URL에 추가하여 발급자 URL을 생성할 수 있습니다:

![발급자 URL은 https://cognito-idp.us-east-1.amazonaws.com/us-east-1\_uiIFNdacd가 됩니다](/images/hosting/setup_aws_cognito_issuer_url.png)

:::info
IDP url에 "Cognito 도메인"을 사용하지 마세요. Cognito는 `https://cognito-idp.$REGION.amazonaws.com/$USER_POOL_ID`에서 발견 문서를 제공합니다.
:::

  </TabItem>
  <TabItem value="okta">

1. 새 애플리케이션을 설정하세요. Okta의 앱 UI로 이동하여 **앱 추가**를 선택하세요:

![](/images/hosting/okta.png)

2. **앱 통합 이름** 필드에 앱 이름을 제공하세요(예: Weights and Biases).
3. 그랜트 유형 `implicit (hybrid)`을 선택하세요.

W&B는 PKCE가 있는 Authorization Code 그랜트 유형도 지원합니다.

![](/images/hosting/pkce.png)

4. 허용된 콜백 url을 제공하세요:
    * 다음 허용 콜백 URL을 추가하세요 `http(s)://YOUR-W&B-HOST/oidc/callback`.

5. IdP가 범용 로그아웃을 지원하는 경우, **로그아웃 URL**을 `http(s)://YOUR-W&B-HOST`로 설정하세요.

![](/images/hosting/redirect_uri.png)
예를 들어, 귀하의 애플리케이션이 로컬 호스트의 8080 포트(`https://localhost:8080`)에서 실행되는 경우,
리디렉션 URI는 다음과 같아야 합니다: `https://localhost:8080/oidc/callback`.

6. **로그아웃 리디렉션 URI** 필드에 `http(s)://YOUR-W&B-HOST/logout`으로 로그아웃 리디렉션을 설정하세요:

![](/images/hosting/signout_redirect.png)

7. OIDC 발급자, 클라이언트 ID, 인증 방법을 https://deploy.wandb.ai/system-admin에서 wandb/local에 제공하거나 환경 변수로 설정하세요.

  </TabItem>
</Tabs>

## W&B 앱에서 SSO 구성

모든 설정을 완료하면 Issuer, 클라이언트 ID, 인증 방법을 W&B 앱의 `wandb/local`에 제공하거나 환경 변수로 설정하여 SSO를 구성할 수 있습니다. W&B 앱 UI에서 SSO를 구성하는 단계를 안내하는 절차는 다음과 같습니다:

1. Weights and Biases 서버에 로그인하세요.
2. W&B 앱으로 이동하세요.

![](/images/hosting/system_settings.png)

3. 드롭다운에서 **시스템 설정**을 선택하세요:

![](/images/hosting/system_settings_select_settings.png)

4. 발급자, 클라이언트 ID, 인증 방법을 입력하세요.
5. **설정 업데이트**를 선택하세요.

![](/images/hosting/system_settings_select_update.png)

:::info
SSO를 구성한 후 인스턴스에 로그인할 수 없는 경우, `LOCAL_RESTORE=true` 환경 변수를 설정하여 인스턴스를 다시 시작할 수 있습니다. 이렇게 하면 임시 비밀번호가 컨테이너 로그에 출력되고 SSO가 비활성화됩니다. SSO와 관련된 문제를 해결한 후에는 SSO를 다시 활성화하기 위해 해당 환경 변수를 제거해야 합니다.
:::
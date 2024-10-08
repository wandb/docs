---
title: Configure SSO with OIDC
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

이메일 [contact@wandb.com](mailto:contact@wandb.com)로 문의하시면 W&B에서 지원하는 아이덴티티 제공자(예: SAML, Ping Federate, Active Directory 등)와 함께 [Auth0](https://auth0.com) 테넌트를 구성해 드립니다.

이미 Auth0를 사용 중이거나 Open ID Connect 호환 서버를 보유한 경우, Open ID로 권한 설정을 위해 아래의 지침을 따라주세요.

:::info
W&B 서버는 기본적으로 수동 사용자 관리를 수행합니다. _wandb/local_의 라이선스 버전에서는 SSO도 활성화됩니다.
:::

## Open ID Connect

_wandb/local_은 Open ID Connect (OIDC)를 사용하여 인증을 수행합니다. 귀하의 유스 케이스에 따라 아래의 탭 중 하나를 선택하여 AWS Cognito 또는 Okta를 통해 W&B 서버를 Open ID Connect로 인증하는 방법을 알아보세요.

:::tip
정체성 제공자(IdP)에서 싱글 페이지 또는 퍼블릭 클라이언트 애플리케이션을 선택하십시오.
:::

<Tabs
  defaultValue="aws"
  values={[
    {label: 'AWS', value: 'aws'},
    {label: 'Okta', value: 'okta'},
  ]}>
  <TabItem value="aws">

AWS Cognito를 설정하려면 아래 절차를 따르세요:

1. 먼저 AWS 계정에 로그인하고 [AWS Cognito](https://aws.amazon.com/cognito/) 앱으로 이동합니다.

![OIDC를 인증 목적으로만 사용하고 권한 부여에는 사용하지 않으므로, 퍼블릭 클라이언트가 설정을 간소화합니다.](/images/hosting/setup_aws_cognito.png)

2. IdP 애플리케이션을 설정하기 위해 허용된 콜백 URL을 제공합니다:
     * `http(s)://YOUR-W&B-HOST/oidc/callback`을 콜백 URL로 추가합니다. `YOUR-W&B-HOST`를 W&B 호스트 경로로 대체하십시오.

3. IdP가 유니버설 로그아웃을 지원하는 경우, 로그아웃 URL을 `http(s)://YOUR-W&B-HOST`로 설정합니다. `YOUR-W&B-HOST`를 W&B 호스트 경로로 대체하세요.

예를 들어, 애플리케이션이 `https://wandb.mycompany.com`에서 실행되고 있는 경우, `YOUR-W&B-HOST`를 `wandb.mycompany.com`으로 대체합니다.

아래 이미지는 AWS Cognito에서 허용된 콜백 및 로그아웃 URL을 제공하는 방법을 보여줍니다.

![여러 호스트에서 인스턴스에 엑세스할 수 있는 경우, 여기 모두 포함되어야 합니다.](/images/hosting/setup_aws_cognito_ui_settings.png)

_wandb/local_은 기본적으로 ["implicit" grant with the "form_post" response type](https://auth0.com/docs/get-started/authentication-and-authorization-flow/implicit-flow-with-form-post)을 사용합니다.

또한 _wandb/local_을 사용하여 [PKCE Code Exchange](https://www.oauth.com/oauth2-servers/pkce/) 흐름을 이용한 "authorization_code" grant를 수행하도록 설정할 수 있습니다.

4. AWS Cognito가 앱에 토큰을 전달하는 방법을 구성하기 위해 하나 이상의 OAuth grant 유형을 선택합니다.
5. W&B는 특정 OpenID Connect (OIDC) 스코프를 필요로 합니다. AWS Cognito App에서 다음을 선택하세요:
    * "openid"
    * "profile"
    * "email"

예를 들어, AWS Cognito 앱 UI는 다음 이미지와 유사해야 합니다:

![openid, profile, 및 email이 필요합니다.](/images/hosting/setup_aws_required_fields.png)

설정 페이지에서 **Auth Method**를 선택하거나 OIDC_AUTH_METHOD 환경 변수를 설정하여 _wandb/local_에 어떤 grant를 사용할지 지정합니다.

:::info
AWS Cognito 제공자 용으로는 Auth Method를 "pkce"로 설정해야 합니다.
:::

6. OIDC 발행자의 URL과 Client ID가 필요합니다. OpenID 디스커버리 문서는 `$OIDC_ISSUER/.well-known/openid-configuration`에 제공되어야 합니다.

예를 들어, AWS Cognito를 사용하여 발행자 URL을 생성하려면 **User Pools** 섹션 내의 **App Integration** 탭에서 Cognito IdP URL에 사용자 풀 ID를 추가하십시오:

![발행자 URL은 https://cognito-idp.us-east-1.amazonaws.com/us-east-1_uiIFNdacd입니다.](/images/hosting/setup_aws_cognito_issuer_url.png)

:::info
IDP url로 "Cognito 도메인"을 사용하지 마십시오. Cognito는 `https://cognito-idp.$REGION.amazonaws.com/$USER_POOL_ID`에서 디스커버리 문서를 제공합니다.
:::

  </TabItem>
  <TabItem value="okta">

1. 먼저 새로운 애플리케이션을 설정합니다. Okta의 앱 UI로 이동하여 **Add apps**를 선택합니다:

![](/images/hosting/okta.png)

2. **App Integration name** 필드에 앱 이름을 제공합니다 (예: Weights and Biases).
3. `implicit (hybrid)` grant 유형을 선택합니다.

W&B는 PKCE를 사용한 Authorization Code grant 유형도 지원합니다.

![](/images/hosting/pkce.png)

4. 허용된 콜백 URL을 제공합니다:
    * 다음의 허용된 콜백 URL을 추가합니다. `http(s)://YOUR-W&B-HOST/oidc/callback`.

5. IdP가 유니버설 로그아웃을 지원하는 경우, **로그아웃 URL**을 `http(s)://YOUR-W&B-HOST`로 설정합니다.

![](/images/hosting/redirect_uri.png)
예를 들어, 애플리케이션이 포트 8080 (`https://localhost:8080`)에서 로컬 호스트로 실행되는 경우, 리다이렉트 URI는 다음과 같습니다: `https://localhost:8080/oidc/callback`.

6. **Sign-out redirects URIs** 필드에서 `http(s)://YOUR-W&B-HOST/logout`으로 서명 아웃 리다이렉트를 설정합니다.

![](/images/hosting/signout_redirect.png)

7. OIDC 발급자, Client ID 및 Auth 메소드를 https://deploy.wandb.ai/system-admin에서 wandb/local에 제공하거나 환경 변수로 설정합니다.

  </TabItem>
</Tabs>

## W&B 앱에서 SSO 구성

모든 설정이 완료되면 발급자, Client ID 및 Auth 메소드를 W&B 앱에서 `wandb/local`에 제공하거나 환경 변수로 설정할 수 있습니다. 다음 절차는 W&B 앱 UI를 사용하여 SSO를 구성하는 단계를 안내합니다:

1. Weights and Biases 서버에 로그인합니다.
2. W&B 앱으로 이동합니다.

![](/images/hosting/system_settings.png)

3. 드롭다운에서 **System Settings**를 선택합니다:

![](/images/hosting/system_settings_select_settings.png)

4. 발행자, Client ID 및 인증 방법을 입력합니다.
5. **Update settings**를 선택합니다.

![](/images/hosting/system_settings_select_update.png)

:::info
SSO를 설정한 후 인스턴스에 로그인할 수 없는 경우 `LOCAL_RESTORE=true` 환경 변수를 설정하여 인스턴스를 재시작할 수 있습니다. 이렇게 하면 컨테이너 로그에 임시 비밀번호가 출력되고 SSO가 비활성화됩니다. SSO의 문제를 해결한 후에는 SSO를 다시 활성화하기 위해 해당 환경 변수를 제거해야 합니다.
:::
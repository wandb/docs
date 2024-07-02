---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# OIDC を使用した SSO

W&B サポートの ID プロバイダー (SAML、Ping Federate、Active Directory など) の Auth0 テナントを設定するには、[contact@wandb.com](mailto:contact@wandb.com) にメールしてください。

既に Auth0 を使用している場合や、Open ID Connect 互換のサーバーを持っている場合は、以下の手順に従って Open ID を使用した認証を設定してください。

:::info
W&B サーバーはデフォルトで手動ユーザ管理を行います。ライセンスバージョンの _wandb/local_ も SSO が可能です。
:::

## Open ID Connect

_wandb/local_ は認証に Open ID Connect (OIDC) を使用します。あなたのユースケースに基づいて、AWS Cognito または Okta で W&B サーバーを Open ID Connect で認証する方法を学ぶために、以下のタブのいずれかを選択してください。

:::tip
ID プロバイダー (IdP) でシングルページまたはパブリッククライアントアプリケーションを選択してください。
:::



<Tabs
  defaultValue="aws"
  values={[
    {label: 'AWS', value: 'aws'},
    {label: 'Okta', value: 'okta'},
  ]}>
  <TabItem value="aws">

以下の手順に従って AWS Cognito を認証用に設定してください:

1. まず、AWS アカウントにサインインし、[AWS Cognito](https://aws.amazon.com/cognito/) アプリに移動します。

![認証のみで認可を行わないため、パブリッククライアントが設定を簡素化します。](/images/hosting/setup_aws_cognito.png)

2. 許可されたコールバック URL を指定して、IdP にアプリケーションを設定します。
     * `http(s)://YOUR-W&B-HOST/oidc/callback` をコールバック URL として追加します。 `YOUR-W&B-HOST` を W&B ホストパスに置き換えます。

3. IdP がユニバーサルログアウトをサポートしている場合、ログアウト URL を `http(s)://YOUR-W&B-HOST` に設定します。 `YOUR-W&B-HOST` を W&B ホストパスに置き換えます。

例として、アプリケーションが `https://wandb.mycompany.com` で実行されている場合、`YOUR-W&B-HOST` を `wandb.mycompany.com` に置き換えます。

以下の画像は、AWS Cognito に許可されたコールバックおよびサインアウト URL を提供する方法を示しています。

![インスタンスが複数のホストからアクセス可能な場合は、ここにすべて記載することを忘れないでください。](/images/hosting/setup_aws_cognito_ui_settings.png)

_wandb/local_ はデフォルトで ["フォームポスト" レスポンスタイプを使用した "暗黙的" 承認](https://auth0.com/docs/get-started/authentication-and-authorization-flow/implicit-flow-with-form-post) を使用します。

また、_wandb/local_ を [PKCE コードエクスチェンジ](https://www.oauth.com/oauth2-servers/pkce/) フローを使用した「承認コード」グラントを実行するように設定することもできます。

4. 1 つ以上の OAuth グラントタイプを選択して、AWS Cognito がトークンをアプリに配信する方法を設定します。
5. W&B は特定の OpenID Connect (OIDC) スコープを必要とします。以下を AWS Cognito アプリから選択します。
    * "openid" 
    * "profile"
    * "email"

例として、あなたの AWS Cognito アプリの UI は以下の画像のように見えるはずです。

![openid、profile、および email は必須です。](/images/hosting/setup_aws_required_fields.png)

設定ページで **Auth Method** を選択するか、環境変数 OIDC\_AUTH\_METHOD を設定してどのグラントを _wandb/local_ に使います。

:::info
AWS Cognito プロバイダーの場合、Auth Method を「pkce」に設定する必要があります
:::

6. クライアント ID と OIDC 発行者の URL が必要です。OpenID 発見ドキュメントは `$OIDC_ISSUER/.well-known/openid-configuration` にある必要があります。

例として、AWS Cognito を使用する場合、ユーザプール ID を Cognito IdP URL に追加して発行者の URL を生成できます。これは **User Pools** セクション内の **App Integration** タブから行えます。

![発行者 URL は https://cognito-idp.us-east-1.amazonaws.com/us-east-1\_uiIFNdacd となります。](/images/hosting/setup_aws_cognito_issuer_url.png)

:::info
IDP の URL に「Cognito ドメイン」を使用しないでください。Cognito は `https://cognito-idp.$REGION.amazonaws.com/$USER_POOL_ID` に発見ドキュメントを提供します。
:::

  </TabItem>
  <TabItem value="okta">

1. まず、新しいアプリケーションを設定します。Okta のアプリ UI に進み、**Add apps** を選択します。

![](/images/hosting/okta.png)

2. **App Integration name** フィールドにアプリの名前を入力します。（例：Weights and Biases）
3. グラントタイプ `implicit (hybrid)` を選択します。

W&B は PKCE を使用した承認コードグラントタイプもサポートします

![](/images/hosting/pkce.png)

4. 許可されたコールバック URL を提供します。
    * 許可されたコールバック URL として `http(s)://YOUR-W&B-HOST/oidc/callback` を追加します。

5. IdP がユニバーサルログアウトをサポートしている場合、**Logout URL** を `http(s)://YOUR-W&B-HOST` に設定します。

![](/images/hosting/redirect_uri.png)
例として、アプリケーションがポート 8080 のローカルホストで実行されている場合（`https://localhost:8080`）、リダイレクト URI は `https://localhost:8080/oidc/callback` のようになります。

6. **Sign-out redirects URIs** フィールドに `http(s)://YOUR-W&B-HOST/logout` と設定します。

![](/images/hosting/signout_redirect.png)

7. OIDC 発行者、クライアント ID、そして Auth メソッドを https://deploy.wandb.ai/system-admin で _wandb/local_ に提供するか、環境変数として設定します。



  </TabItem>
</Tabs>

## W&B アプリで SSO 設定を行う

すべてが設定されたら、発行者、クライアント ID、そして Auth メソッドを W&B アプリ上の `wandb/local` に提供するか、環境変数として設定します。以下の手順では、W&B アプリの UI を使用して SSO を設定する手順を説明します。

1. Weights and Biases サーバーにサインインします。
2. W&B アプリに移動します。

![](/images/hosting/system_settings.png)

3. ドロップダウンから **System Settings** を選択します。

![](/images/hosting/system_settings_select_settings.png)

4. 発行者、クライアント ID、および認証メソッドを入力します。
5. **Update settings** を選択します。

![](/images/hosting/system_settings_select_update.png)


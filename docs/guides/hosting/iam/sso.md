---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# SSO の OIDC 使用法

W&B でサポートされているアイデンティティプロバイダー (SAML、Ping Federate、Active Directory など) を使用した [Auth0](https://auth0.com) テナントの設定については、[contact@wandb.com](mailto:contact@wandb.com) にメールでお問い合わせください。

すでに Auth0 を使用している場合や Open ID Connect 互換のサーバーをお持ちの場合は、以下の手順に従って Open ID 認証を設定してください。

:::info
W&B サーバーはデフォルトで手動のユーザー管理を行います。_wandb/local_ のライセンスバージョンでも SSO が有効になります。
:::

## Open ID Connect

_wandb/local_ は Open ID Connect (OIDC) を使用して認証します。ユースケースに基づいて、AWS Cognito または Okta を使用して W&B サーバーを Open ID Connect で認証する方法を学ぶために適切なタブを選択してください。

:::tip
アイデンティティプロバイダー (IdP) でシングルページまたはパブリッククライアントアプリケーションを選択してください。
:::



<Tabs
  defaultValue="aws"
  values={[
    {label: 'AWS', value: 'aws'},
    {label: 'Okta', value: 'okta'},
  ]}>
  <TabItem value="aws">

次の手順に従って AWS Cognito の認証を設定してください:

1. まず、AWS アカウントにサインインし、[AWS Cognito](https://aws.amazon.com/cognito/) アプリに移動します。

![認証のためだけに OIDC を使用し、認可を使用しないため、パブリッククライアントはセットアップを簡素化します](/images/hosting/setup_aws_cognito.png)

2. IdP でアプリケーションを構成するために許可されたコールバック URL を指定します:
     * `http(s)://YOUR-W&B-HOST/oidc/callback` をコールバック URL として追加します。`YOUR-W&B-HOST` を W&B ホストパスに置き換えてください。

3. あなたの IdP がユニバーサルログアウトをサポートしている場合は、ログアウト URL を `http(s)://YOUR-W&B-HOST` に設定します。`YOUR-W&B-HOST` を W&B ホストパスに置き換えてください。

例えば、アプリケーションが `https://wandb.mycompany.com` で動作している場合、`YOUR-W&B-HOST` を `wandb.mycompany.com` に置き換えます。

以下の画像では、AWS Cognito に許可されたコールバックおよびサインアウト URL を指定する方法を示しています。

![インスタンスが複数のホストからアクセス可能である場合は、ここにそれらすべてを含めることを確認してください。](/images/hosting/setup_aws_cognito_ui_settings.png)

_wandb/local_ はデフォルトで ["インプリシット" グラントを "form\_post" レスポンスタイプで使用](https://auth0.com/docs/get-started/authentication-and-authorization-flow/implicit-flow-with-form-post) します。

また、_wandb/local_ を設定して、[PKCEコード交換](https://www.oauth.com/oauth2-servers/pkce/) フローを使用する "authorization\_code" グラントを実行することもできます。

4. AWS Cognito があなたのアプリにトークンを提供する方法を設定するために、1つ以上の OAuth グラントタイプを選択します。
5. W&B は特定の OpenID Connect (OIDC) スコープを必要とします。AWS Cognito アプリから以下を選択してください:
    * "openid"
    * "profile"
    * "email"

例えば、あなたの AWS Cognito アプリ UI は次の画像のように見えるはずです:

![openid、profile、および email が必要です](/images/hosting/setup_aws_required_fields.png)

設定ページで **Auth Method** を選択するか、OIDC\_AUTH\_METHOD 環境変数を設定して、_wandb/local_ にどのグラントを使用するかを伝えます。

:::info
AWS Cognito プロバイダーの場合、認証方法を "pkce" に設定する必要があります。
:::

6. クライアント ID と OIDC 発行者の URL が必要です。OpenID ディスカバリドキュメントは `$OIDC_ISSUER/.well-known/openid-configuration` で利用可能でなければなりません。

例えば、AWS Cognito では、**User Pools** セクションの **App Integration** タブからユーザープール ID を Cognito IdP URL に追加して発行者 URL を生成できます:

![発行者 URL は https://cognito-idp.us-east-1.amazonaws.com/us-east-1\_uiIFNdacd のようになります](/images/hosting/setup_aws_cognito_issuer_url.png)

:::info
IDP URL に "Cognito ドメイン" を使用しないでください。Cognito はディスカバリドキュメントを `https://cognito-idp.$REGION.amazonaws.com/$USER_POOL_ID` で提供します。
:::

  </TabItem>
  <TabItem value="okta">

1. まず、新しいアプリケーションを設定します。Okta のアプリアイコンに移動して **Add apps** を選択します:

![](/images/hosting/okta.png)

2. **App Integration name** フィールドにアプリの名前 (例: Weights and Biases) を入力します。
3. グラントタイプ `implicit (hybrid)` を選択します。

W&B は PKCE による Authorization Code グラントタイプもサポートしています。

![](/images/hosting/pkce.png)

4. 許可されたコールバック URL を指定します:
    * 以下の許可されたコールバック URL を追加します `http(s)://YOUR-W&B-HOST/oidc/callback`。

5. あなたの IdP がユニバーサルログアウトをサポートしている場合は、**Logout URL** を `http(s)://YOUR-W&B-HOST` に設定します。

![](/images/hosting/redirect_uri.png)
例えば、あなたのアプリケーションがポート 8080 でローカルホストで動作している場合 (`https://localhost:8080`)、リダイレクト URI は次のようになります: `https://localhost:8080/oidc/callback`。

6. **Sign-out redirects URIs** フィールドに `http(s)://YOUR-W&B-HOST/logout` をサインアウトリダイレクトとして設定します:

![](/images/hosting/signout_redirect.png)

7. OIDC 発行者、クライアント ID、および認証方法を wandb/local に https://deploy.wandb.ai/system-admin で提供するか、環境変数として設定します。

  </TabItem>
</Tabs>

## W&B アプリで SSO を設定

すべてが設定されたら、Issuer、Client ID、および認証方法を W&B アプリおよび_環境変数として提供できます。以下の手順は、W&B アプリ UI で SSO を設定する方法を説明します:

1. Weights and Biases サーバーにサインインします。
2. W&B アプリに移動します。

![](/images/hosting/system_settings.png)

3. ドロップダウンから **System Settings** を選択します:

![](/images/hosting/system_settings_select_settings.png)

4. Issuer、Client ID、および認証方法を入力します。
5. **Update settings** を選択します。

![](/images/hosting/system_settings_select_update.png)


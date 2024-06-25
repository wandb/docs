---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# SSO using OIDC

Email [contact@wandb.com](mailto:contact@wandb.com) to configure an [Auth0](https://auth0.com) tenant for you with identity providers supported by W&B (such as SAML, Ping Federate, Active Directory, and more).

If you already use Auth0 or have an Open ID Connect compatible server, follow the instructions below to set up authorization with Open ID.

:::info
W&B Server はデフォルトで手動のユーザー管理で操作します。ライセンスバージョンの _wandb/local_ では SSO も利用可能です。
:::

## Open ID Connect

_wandb/local_ は Open ID Connect (OIDC) を認証に使用します。ユースケースに基づいて、AWS Cognito または Okta で W&B Server を Open ID Connect を使用して認証する方法について学ぶためのタブを選択してください。

:::tip
アイデンティティープロバイダー（IdP）でシングルページまたはパブリッククライアントアプリケーションを選択してください。
:::

<Tabs
  defaultValue="aws"
  values={[
    {label: 'AWS', value: 'aws'},
    {label: 'Okta', value: 'okta'},
  ]}>
  <TabItem value="aws">

以下の手順に従って AWS Cognito で認証を設定してください:

1. 最初に AWS アカウントにサインインし、[AWS Cognito](https://aws.amazon.com/cognito/) アプリに移動します。

![OIDCを認証にのみ使用し、認証に使用しないため、パブリッククライアントがセットアップを簡素化します](/images/hosting/setup_aws_cognito.png)

2. 許可されたコールバック URL を提供して IdP にアプリケーションを設定します:
   * `http(s)://YOUR-W&B-HOST/oidc/callback` をコールバック URL として追加します。`YOUR-W&B-HOST` を W&B ホストパスに置き換えます。

3. IdP がユニバーサルログアウトをサポートしている場合、ログアウト URL を `http(s)://YOUR-W&B-HOST` に設定します。`YOUR-W&B-HOST` を W&B ホストパスに置き換えます。

例えば、アプリケーションが `https://wandb.mycompany.com` で実行されている場合、`YOUR-W&B-HOST` を `wandb.mycompany.com` に置き換えます。

以下の画像は、AWS Cognito に許可されたコールバックおよびサインアウト URL を提供する方法を示しています。

![インスタンスが複数のホストからアクセス可能な場合、それらすべてをここに含めてください。](/images/hosting/setup_aws_cognito_ui_settings.png)


_wandb/local_ はデフォルトで [「暗黙的」グラントと「form\_post」レスポスタイプ](https://auth0.com/docs/get-started/authentication-and-authorization-flow/implicit-flow-with-form-post) を使用します。

また、_wandb/local_ を設定して [PKCE コード交換](https://www.oauth.com/oauth2-servers/pkce/) フローを使用して "authorization\_code" グラントを実行することもできます。

4. 1つ以上のOAuth グラントタイプを選択して、AWS Cognito がアプリへのトークンをどのように提供するかを設定します。
5. W&B には特定の OpenID Connect (OIDC) スコープが必要です。AWS Cognito アプリから以下を選択します:
    * "openid" 
    * "profile"
    * "email"

例えば、AWS Cognito アプリのUIは次の画像のようになります:

![openid、profile、およびemailは必須です](/images/hosting/setup_aws_required_fields.png)

設定ページで **Auth Method** を選択するか、OIDC\_AUTH\_METHOD 環境変数を設定して、_wandb/local_ にどのグラントを使用するかを指示します。

:::info
AWS Cognito プロバイダーの場合、Auth Method を "pkce" に設定する必要があります。
:::

6. クライアント ID と OIDC 発行者の URL が必要です。OpenID ディスカバリードキュメントは `$OIDC_ISSUER/.well-known/openid-configuration` で利用可能でなければなりません。

例えば、AWS Cognito では、**User Pools** セクションの **App Integration** タブから、ユーザープール ID を Cognito IdP URL に追加して発行者 URL を生成できます。

![発行者 URL は https://cognito-idp.us-east-1.amazonaws.com/us-east-1\_uiIFNdacd になります。](/images/hosting/setup_aws_cognito_issuer_url.png)

:::info
IDP URL に "Cognito ドメイン" を使用しないでください。Cognito は `https://cognito-idp.$REGION.amazonaws.com/$USER_POOL_ID` でディスカバリードキュメントを提供します。
:::

  </TabItem>
  <TabItem value="okta">

1. 最初に新しいアプリケーションを設定します。 Okta のアプリケーション UI に移動して、**Add apps** を選択します:

![](/images/hosting/okta.png)

2. **App Integration name** フィールドにアプリの名前を入力します（例: Weights and Biases）
3. グラント タイプ `implicit (hybrid)` を選択します

W&B は PKCE との Authorization Code グラント タイプもサポートしています

![](/images/hosting/pkce.png)

4. 許可されたコールバック URL を提供します:
    * 次の許可されたコールバック URL を追加します `http(s)://YOUR-W&B-HOST/oidc/callback`。

5. IdP がユニバーサルログアウトをサポートしている場合、**Logout URL** を `http(s)://YOUR-W&B-HOST` に設定します。

![](/images/hosting/redirect_uri.png)
例えば、アプリケーションがポート 8080 でローカルホストで実行されている場合（`https://localhost:8080`）、
リダイレクト URI は次のようになります: `https://localhost:8080/oidc/callback`。

6. サインアウトリダイレクト URI フィールドに `http(s)://YOUR-W&B-HOST/logout` を設定します:

![](/images/hosting/signout_redirect.png)

7. OIDC Issuer、クライアント ID、および Auth method を wandb/local に提供するか、環境変数として設定します。

  </TabItem>
</Tabs>

## W&B アプリで SSO を設定する

すべての設定が完了したら、Issuer、Client ID、および Auth method を `wandb/local` に W&B アプリ経由で提供するか、環境変数を設定します。以下の手順は、W&B アプリ UI を使用して SSO を設定する手順を説明しています:

1. Weights and Biases サーバーにサインインします。
2. W&B アプリに移動します。

![](/images/hosting/system_settings.png)

3. ドロップダウンから **System Settings** を選択します:

![](/images/hosting/system_settings_select_settings.png)

4. Issuer、Client ID、および Authentication Method を入力します。
5. **Update settings** を選択します。

![](/images/hosting/system_settings_select_update.png)

:::info
SSO の設定後、インスタンスにログインできない場合は、`LOCAL_RESTORE=true` 環境変数を設定してインスタンスを再起動することができます。これにより、コンテナログに一時的なパスワードが出力され、SSO が無効になります。SSO の問題を解決したら、その環境変数を削除して SSO を再度有効にする必要があります。
:::


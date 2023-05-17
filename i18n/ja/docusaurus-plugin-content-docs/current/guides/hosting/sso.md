---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# シングルサインオン（SSO)の設定

[contact@wandb.com](mailto:contact@wandb.com)にメールして、W&Bでサポートされているアイデンティティプロバイダ（SAML、Ping Federate、Active Directoryなど）を含む[Auth0](https://auth0.com)テナントを設定してください。

すでにAuth0を使用しているか、Open ID Connect互換のサーバーがある場合は、以下の手順に従ってOpen IDでの認証を設定してください。

:::info
W&Bサーバーは、デフォルトで手動ユーザー管理で運用されています。_wandb/local_ のライセンス版でも SSOが利用可能です。
:::

## Open ID Connect

_wandb/local_ は、Open ID Connect（OIDC）を認証に使用します。ユースケースに応じて、タブのいずれかを選択し、AWS CognitoまたはOktaを使用して、Open ID ConnectでW&Bサーバーを認証する方法を学びます。

:::tip
アイデンティティプロバイダ（IdP）で、シングルページアプリケーションまたはパブリッククライアントアプリケーションを選択してください。
:::



<Tabs
  defaultValue="aws"
  values={[
    {label: 'AWS', value: 'aws'},
    {label: 'Okta', value: 'okta'},
  ]}>
  <TabItem value="aws">
以下の手順に従って、AWS Cognitoを設定して認証を行います。

1. まず、AWSアカウントにサインインし、[AWS Cognito](https://aws.amazon.com/cognito/) アプリに移動します。

![Because we're only using OIDC for authentication and not authorization, public clients simplify setup](/images/hosting/setup_aws_cognito.png)

2. 許可されたコールバックURLを提供して、IdP内のアプリケーションを設定します。
     * コールバックURLとして `http(s)://YOUR-W&B-HOST/oidc/callback` を追加します。`YOUR-W&B-HOST` をあなたのW&Bホストパスに置き換えてください。

3. あなたのIdPがユニバーサルログアウトをサポートしている場合は、ログアウトURLを `http(s)://YOUR-W&B-HOST` に設定します。`YOUR-W&B-HOST` をあなたのW&Bホストパスに置き換えてください。

例えば、アプリケーションが `https://wandb.mycompany.com` で実行されている場合、`YOUR-W&B-HOST` を `wandb.mycompany.com` に置き換えます。

以下の画像は、AWS Cognitoで許可されたコールバックURLとサインアウトURLを提供する方法を示しています。

![If your instance is accessible from multiple hosts, be sure to include all of them here.](/images/hosting/setup_aws_cognito_ui_settings.png)

_wandb/local_ はデフォルトで ["implicit" grant with the "form\_post" response type](https://auth0.com/docs/get-started/authentication-and-authorization-flow/implicit-flow-with-form-post) を使用します。

また、_wandb/local_ を [PKCE Code Exchange](https://www.oauth.com/oauth2-servers/pkce/)  フローを使用した "authorization\_code" グラントで実行するように設定することもできます。

4. AWS Cognitoがアプリにトークンを配信する方法を設定するために、1つ以上のOAuthグラントタイプを選択します。
5. W&Bでは特定のOpenID Connect（OIDC）スコープが必要です。AWS Cognitoアプリから以下を選択してください：
    * "openid"
    * "profile"
    * "email"
たとえば、あなたのAWS CognitoアプリUIは、次の画像のようになるはずです：

![openid, profile, emailが必須](/images/hosting/setup_aws_required_fields.png)

設定ページで**Auth Method** を選択するか、OIDC_AUTH_METHOD環境変数を設定して、_wandb/local_ にどのグラントを伝えるかを指定します。

:::info
AWS Cognitoプロバイダの場合、Auth Methodを "pkce" に設定する必要があります
:::

6. クライアントIDとOIDC発行者（issuer）のURLが必要です。OpenIDディスカバリドキュメントは `$OIDC_ISSUER/.well-known/openid-configuration`で利用可能でなければなりません。

例えば、AWS Cognitoでは、**User Pools**セクションの**App Integration**タブからCognito IdP URLにユーザープールIDを追加することで、発行者のURLを生成できます：

![発行者のURLは https://cognito-idp.us-east-1.amazonaws.com/us-east-1\_uiIFNdacd になります](/images/hosting/setup_aws_cognito_issuer_url.png)

:::info
IDPのURLには "Cognitoドメイン" を使用しないでください。Cognitoはディスカバリドキュメントを `https://cognito-idp.$REGION.amazonaws.com/$USER_POOL_ID` で提供しています。
:::

<!-- 7. 最後に、OIDC発行者、クライアントID、およびAuth Methodを_wandb/local_ の `https://deploy.wandb.ai/system-admin` に提供するか、環境変数として設定します。

次の画像は、W&BアプリUI（`https://deploy.wandb.ai/system-admin`）で、SSOを有効にし、OIDC発行者、クライアントID、および認証方法を提供する方法を示しています： -->

<!-- すべての設定が完了したら、発行者、クライアントID、およびAuth Methodを `wandb/local` の`/system-admin` や環境変数に提供して、SSOが構成されます。

1. Weights and Biasesサーバーにサインインする
2. W&Bアプリに移動します。
![](/images/hosting/system_settings.png)

3. ドロップダウンから、**システム設定** を選択してください：

![](/images/hosting/system_settings_select_settings.png)

4. 発行者、クライアントID、および認証方法を入力してください。
5. **設定を更新** を選択してください。

![](/images/hosting/system_settings_select_update.png)

![](/images/hosting/enable_sso.png) -->

  </TabItem>
  <TabItem value="okta">


1. 最初に新しいアプリケーションを設定します。OktaのApp UIに移動し、**アプリを追加** を選択してください：

![](/images/hosting/okta.png)

2. **アプリ統合名**フィールドにアプリの名前を提供してください（例：Weights and Biases）
3. グラントタイプ `implicit (hybrid)` を選択してください。

W&Bは、PKCE付きのAuthorization Codeグラントタイプもサポートしています

![](/images/hosting/pkce.png)

4. 許可されたコールバックURLを提供してください：
    * 次の許可されたコールバックURLを追加してください: `http(s)://YOUR-W&B-HOST/oidc/callback`.
5. あなたのIdPがユニバーサルログアウトをサポートしている場合、**ログアウトURL** を `http(s)://YOUR-W&B-HOST` に設定します。

![](/images/hosting/redirect_uri.png)
例えば、アプリケーションがローカルホストのポート8080（`https://localhost:8080`）で実行されている場合、リダイレクトURIは次のようになります：`https://localhost:8080/oidc/callback`.

6. **サインアウトリダイレクトURI**フィールドに、`http(s)://YOUR-W&B-HOST/logout` を設定します。

![](/images/hosting/signout_redirect.png)

7. OIDC Issuer、Client ID、およびAuth methodを https://deploy.wandb.ai/system-admin にあるwandb/localに提供するか、環境変数として設定します。


  </TabItem>
</Tabs>

## W&BアプリでSSOを設定する

すべての設定が完了したら、Issuer、Client ID、およびAuth methodをW&Bアプリの`wandb/local`に提供するか、環境変数を設定します。次の手順では、W&BアプリUIでSSOを設定する方法を説明します。

1. Weights and Biasesサーバーにサインインします。
2. W&Bアプリに移動します。

![](/images/hosting/system_settings.png)

3. ドロップダウンから**システム設定**を選択します。

![](/images/hosting/system_settings_select_settings.png)
4. 発行者、クライアントID、および認証方法を入力してください。

5. **設定を更新** を選択してください。



![](/images/hosting/system_settings_select_update.png)





:::info

SSOを設定した後でインスタンスにログインできない場合は、`LOCAL_RESTORE=true` 環境変数が設定された状態でインスタンスを再起動できます。これにより、一時的なパスワードがコンテナのログに出力され、SSOが無効になります。SSOの問題が解決されたら、再びSSOを有効にするためにその環境変数を削除する必要があります。

:::
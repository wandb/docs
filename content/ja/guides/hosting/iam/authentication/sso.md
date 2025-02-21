---
title: Configure SSO with OIDC
menu:
  default:
    identifier: ja-guides-hosting-iam-authentication-sso
    parent: authentication
---

W&B Server の OpenID Connect (OIDC) 互換のアイデンティティプロバイダーのサポートにより、Okta、Keycloak、Auth0、Google、Entra などの外部アイデンティティプロバイダーを介して、 ユーザー のアイデンティティとグループメンバーシップを管理できます。

## OpenID Connect (OIDC)

W&B Server は、外部 Identity Provider (IdP) との統合のために、以下の OIDC 認証フローをサポートしています。
1. Form Post を使用した Implicit Flow
2. Proof Key for Code Exchange (PKCE) を使用した Authorization Code Flow

これらのフローは ユーザー を認証し、アクセス制御を管理するために必要なアイデンティティ情報 (ID トークンの形式) を W&B Server に提供します。

ID トークンは、名前、 ユーザー 名、メール、グループメンバーシップなど、 ユーザー のアイデンティティ情報を含む JWT です。W&B Server はこのトークンを使用して ユーザー を認証し、システム内の適切なロールまたはグループにマッピングします。

W&B Server のコンテキストでは、アクセス トークンは ユーザー に代わって API へのリクエストを承認しますが、W&B Server の主な関心事は ユーザー 認証とアイデンティティであるため、ID トークンのみが必要です。

環境変数を使用して、[IAM オプションを設定]({{< relref path="../advanced_env_vars.md" lang="ja" >}}) して、[Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンスを設定できます。

[Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) W&B Server インストール用に Identity Provider を構成する際は、以下のガイドラインに従って、さまざまな IdP に対応してください。W&B の SaaS バージョンを使用している場合は、[support@wandb.com](mailto:support@wandb.com) に連絡して、組織の Auth0 テナントの構成を支援してもらってください。

{{< tabpane text=true >}}
{{% tab header="Cognito" value="cognito" %}}
認証に AWS Cognito を設定するには、以下の手順に従ってください。

1. まず、AWS アカウントにサインインし、[AWS Cognito](https://aws.amazon.com/cognito/) アプリに移動します。

    {{< img src="/images/hosting/setup_aws_cognito.png" alt="認証に OIDC を使用し、承認には使用しない場合、パブリッククライアントはセットアップを簡素化します" >}}

2. IdP で アプリケーション を構成するために、許可された コールバック URL を指定します。
     * `http(s)://YOUR-W&B-HOST/oidc/callback` を コールバック URL として追加します。`YOUR-W&B-HOST` を W&B ホストパスに置き換えます。

3. IdP がユニバーサルログアウトをサポートしている場合は、ログアウト URL を `http(s)://YOUR-W&B-HOST` に設定します。`YOUR-W&B-HOST` を W&B ホストパスに置き換えます。

    たとえば、 アプリケーション が `https://wandb.mycompany.com` で実行されている場合、`YOUR-W&B-HOST` を `wandb.mycompany.com` に置き換えます。

    以下の画像は、AWS Cognito で許可された コールバック とサインアウト URL を指定する方法を示しています。

    {{< img src="/images/hosting/setup_aws_cognito_ui_settings.png" alt="インスタンスに複数のホストからアクセスできる場合は、ここにすべて含めてください。" >}}


    _wandb/local_ は、デフォルトで [`form_post` レスポンスタイプの `implicit` グラント](https://auth0.com/docs/get-started/authentication-and-authorization-flow/implicit-flow-with-form-post) を使用します。

    [PKCE Code Exchange](https://www.oauth.com/oauth2-servers/pkce/) フローを使用する `authorization_code` グラントを実行するように _wandb/local_ を構成することもできます。

4. AWS Cognito がトークンを アプリ に配信する方法を構成するために、1 つ以上の OAuth グラントタイプを選択します。
5. W&B には、特定の OpenID Connect (OIDC) スコープが必要です。AWS Cognito アプリから以下を選択します。
    * "openid"
    * "profile"
    * "email"

    たとえば、AWS Cognito アプリ UI は以下の画像のようになります。

    {{< img src="/images/hosting/setup_aws_required_fields.png" alt="必須フィールド" >}}

    設定ページで **Auth Method** を選択するか、OIDC_AUTH_METHOD 環境変数を設定して、どのグラントを _wandb/local_ に指示するかを指定します。

    Auth Method を `pkce` に設定する必要があります。

6. Client ID と OIDC 発行者の URL が必要です。OpenID ディスカバリドキュメントは `$OIDC_ISSUER/.well-known/openid-configuration` で利用可能である必要があります。

    たとえば、**User Pools** セクション内の **App Integration** タブから Cognito IdP URL に ユーザー プール ID を追加して、発行者 URL を生成できます。

    {{< img src="/images/hosting/setup_aws_cognito_issuer_url.png" alt="AWS Cognito の発行者 URL のスクリーンショット" >}}

    IDP URL に "Cognito ドメイン" を使用しないでください。Cognito は、`https://cognito-idp.$REGION.amazonaws.com/$USER_POOL_ID` でディスカバリドキュメントを提供します。

{{% /tab %}}

{{% tab header="Okta" value="okta"%}}
認証に Okta を設定するには、以下の手順に従ってください。

1. https://login.okta.com/ で Okta ポータルにログインします。

2. 左側で、**Applications** を選択し、もう一度 **Applications** を選択します。
    {{< img src="/images/hosting/okta_select_applications.png" alt="" >}}

3. 「Create App integration」をクリックします。
    {{< img src="/images/hosting/okta_create_new_app_integration.png" alt="" >}}

4. 「Create a new app integration」という画面で、**OIDC - OpenID Connect** と **Single-Page Application** を選択します。次に「Next」をクリックします。
    {{< img src="/images/hosting/okta_create_a_new_app_integration.png" alt="" >}}

5. 「New Single-Page App Integration」という画面で、以下の値を入力して **Save** をクリックします。
    - App integration name (例: "Weights & Biases")
    - Grant type: **Authorization Code** と **Implicit (hybrid)** の両方を選択します。
    - Sign-in redirect URIs: https://YOUR_W_AND_B_URL/oidc/callback
    - Sign-out redirect URIs: https://YOUR_W_AND_B_URL/logout
    - Assignments: **Skip group assignment for now** を選択します。
    {{< img src="/images/hosting/okta_new_single_page_app_integration.png" alt="" >}}

6. 作成した Okta アプリケーション の概要画面で、**General** タブの **Client Credentials** の下にある **Client ID** をメモしておきます。
    {{< img src="/images/hosting/okta_make_note_of_client_id.png" alt="" >}}

7. Okta OIDC Issuer URL を特定するには、左側で **Settings**、次に **Account** を選択します。
    Okta UI には、**Organization Contact** の下に会社名が表示されます。
    {{< img src="/images/hosting/okta_identify_oidc_issuer_url.png" alt="" >}}

OIDC 発行者 URL の形式は `https://COMPANY.okta.com` です。COMPANY を対応する 値 に置き換えます。メモしておきます。
{{% /tab %}}

{{% tab header="Entra" value="entra"%}}
1. Azure ポータル (https://portal.azure.com/) にログインします。

2. 「Microsoft Entra ID」サービスを選択します。
    {{< img src="/images/hosting/entra_select_entra_service.png" alt="" >}}

3. 左側で、「App registrations」を選択します。
    {{< img src="/images/hosting/entra_app_registrations.png" alt="" >}}

4. 上部で、「New registration」をクリックします。
    {{< img src="/images/hosting/entra_new_app_registration.png" alt="" >}}

    「Register an application」という画面で、以下の値を入力します。
    {{< img src="/images/hosting/entra_register_an_application.png" alt="" >}}

    - 名前を指定します (例: "Weights and Biases application")
    - デフォルトでは、選択されているアカウントタイプは「Accounts in this organizational directory only (Default Directory only - Single tenant)」です。必要に応じて変更してください。
    - **Web** タイプで、値が `https://YOUR_W_AND_B_URL/oidc/callback` であるリダイレクト URI を構成します。
    - 「Register」をクリックします。

    - 「Application (client) ID」と「Directory (tenant) ID」をメモしておきます。

      {{< img src="/images/hosting/entra_app_overview_make_note.png" alt="" >}}


5. 左側で、**Authentication** をクリックします。
    {{< img src="/images/hosting/entra_select_authentication.png" alt="" >}}

    - **Front-channel logout URL** で、`https://YOUR_W_AND_B_URL/logout` を指定します。
    - 「Save」をクリックします。

      {{< img src="/images/hosting/entra_logout_url.png" alt="" >}}


6. 左側で、「Certificates & secrets」をクリックします。
    {{< img src="/images/hosting/entra_select_certificates_secrets.png" alt="" >}}

    - 「Client secrets」をクリックし、次に「New client secret」をクリックします。
      {{< img src="/images/hosting/entra_new_secret.png" alt="" >}}

      「Add a client secret」という画面で、以下の値を入力します。
      {{< img src="/images/hosting/entra_add_new_client_secret.png" alt="" >}}

      - 説明を入力します (例: "wandb")
      - 「Expires」はそのままにするか、必要に応じて変更します。
      - 「Add」をクリックします。


    - シークレットの「Value」をメモしておきます。「Secret ID」は必要ありません。
    {{< img src="/images/hosting/entra_make_note_of_secret_value.png" alt="" >}}

これで、次の 3 つの 値 をメモしておく必要があります。
- OIDC Client ID
- OIDC Client Secret
- OIDC Issuer URL にはテナント ID が必要です。

OIDC 発行者 URL の形式は `https://login.microsoftonline.com/${TenantID}/v2.0` です。
{{% /tab %}}
{{< /tabpane >}}

## W&B Server で SSO を設定する

SSO を設定するには、管理者権限と以下の情報が必要です。
- OIDC Client ID
- OIDC Auth メソッド (`implicit` または `pkce`)
- OIDC Issuer URL
- OIDC Client Secret (オプション; IdP の設定方法によって異なります)

{{% alert %}}
IdP が OIDC Client Secret を必要とする場合は、環境変数 `OIDC_CLIENT_SECRET` で指定してください。
{{% /alert %}}

W&B Server UI を使用するか、[環境変数]({{< relref path="/guides/hosting/env-vars.md" lang="ja" >}}) を `wandb/local` pod に渡すことによって、SSO を構成できます。環境変数は UI よりも優先されます。

{{% alert %}}
SSO の構成後にインスタンスにログインできない場合は、`LOCAL_RESTORE=true` 環境変数を設定してインスタンスを再起動できます。これにより、コンテナログに一時パスワードが出力され、SSO が無効になります。SSO の問題を解決したら、その環境変数を削除して SSO を再度有効にする必要があります。
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab header="System Console" value="console" %}}
System Console は、System Settings ページの後継です。[W&B Kubernetes Operator]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/" lang="ja" >}}) ベースの デプロイメント で利用できます。

1. [W&B Management Console へのアクセス]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/#access-the-wb-management-console" lang="ja" >}}) を参照してください。

2. **Settings** に移動し、次に **Authentication** に移動します。**Type** ドロップダウンで **OIDC** を選択します。
    {{< img src="/images/hosting/sso_configure_via_console.png" alt="" >}}

3. 値 を入力します。

4. **Save** をクリックします。

5. ログアウトしてから再度ログインします。今回は IdP ログイン画面を使用します。
{{% /tab %}}
{{% tab header="System settings" value="settings" %}}
1. Weights & Biases インスタンスにサインインします。
2. W&B アプリ に移動します。

    {{< img src="/images/hosting/system_settings.png" alt="" >}}

3. ドロップダウンから **System Settings** を選択します。

    {{< img src="/images/hosting/system_settings_select_settings.png" alt="" >}}

4. 発行者、Client ID、および認証方法を入力します。
5. **Update settings** を選択します。

{{< img src="/images/hosting/system_settings_select_update.png" alt="" >}}
{{% /tab %}}
{{< /tabpane >}}

{{% alert %}}
SSO の構成後にインスタンスにログインできない場合は、`LOCAL_RESTORE=true` 環境変数を設定してインスタンスを再起動できます。これにより、コンテナログに一時パスワードが出力され、SSO がオフになります。SSO の問題を解決したら、その環境変数を削除して SSO を再度有効にする必要があります。
{{% /alert %}}

## Security Assertion Markup Language (SAML)
W&B Server は SAML をサポートしていません。

---
title: SSO を OIDC で設定
menu:
  default:
    identifier: ja-guides-hosting-iam-authentication-sso
    parent: authentication
---

W&B サーバーは、OpenID Connect (OIDC) 互換のアイデンティティ プロバイダーをサポートしており、Okta、Keycloak、Auth0、Google、および Entra などの外部アイデンティティ プロバイダーを通じてユーザー アイデンティティとグループ メンバーシップを管理できます。

## OpenID Connect (OIDC)

W&B サーバーは、外部アイデンティティプロバイダー (IdP) とのインテグレーションのために、次の OIDC 認証フローをサポートします。
1. フォームポストを使用したインプリシットフロー
2. コードエクスチェンジのための証明キーを使用した認可コードフロー (PKCE)

これらのフローはユーザーを認証し、必要なアイデンティティ情報 (ID トークンの形式) を W&B サーバーに提供してアクレス制御を管理します。

ID トークンは、ユーザーの名前、ユーザー名、メール、およびグループメンバーシップなど、ユーザーのアイデンティティ情報を含む JWT です。W&B サーバーはこのトークンを使用してユーザーを認証し、システム内で適切なロールやグループにマッピングします。

W&B サーバーのコンテキストでは、アクセストークンはユーザーを代表して API へのリクエストを認可しますが、W&B サーバーの主な関心はユーザー認証とアイデンティティであるため、ID トークンのみが必要です。

環境変数を使用して、[Dedicated cloud](../advanced_env_vars.md) の [IAM オプションを設定]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}})するか、[Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンスを設定することができます。

[Dedicated cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) W&B サーバーインストールのためにアイデンティティ プロバイダーを設定する際は、さまざまな IdP に関するこれらのガイドラインに従ってください。SaaS バージョンの W&B を使用している場合は、組織の Auth0 テナントを設定するための支援を求めるには [support@wandb.com](mailto:support@wandb.com) に連絡してください。

{{< tabpane text=true >}}
{{% tab header="Cognito" value="cognito" %}}
AWS Cognito を認証に設定する手順は以下の通りです： 

1. 最初に AWS アカウントにサインインし、[AWS Cognito](https://aws.amazon.com/cognito/) アプリに移動します。

    {{< img src="/images/hosting/setup_aws_cognito.png" alt="認証に OIDC を使用し認可に使用しない場合、パブリッククライアントはセットアップを簡素化します" >}}

2. IdP にアプリケーションを設定するための許可されたコールバック URL を入力します：
     * `http(s)://YOUR-W&B-HOST/oidc/callback` をコールバック URL として追加します。 `YOUR-W&B-HOST` を W&B ホストパスに置き換えます。

3. IdP がユニバーサルログアウトをサポートしている場合は、ログアウト URL を `http(s)://YOUR-W&B-HOST` に設定します。 `YOUR-W&B-HOST` を W&B ホストパスに置き換えます。

    たとえば、アプリケーションが `https://wandb.mycompany.com` で実行されている場合、`YOUR-W&B-HOST` を `wandb.mycompany.com` に置き換えます。

    下の画像は、AWS Cognito で許可されたコールバックとサインアウト URL を提供する方法を示しています。

    {{< img src="/images/hosting/setup_aws_cognito_ui_settings.png" alt="インスタンスが複数のホストからアクセス可能な場合は、ここにすべてを含めてください。" >}}

    _wandb/local_ はデフォルトで [`implicit` grant with the `form_post` response type](https://auth0.com/docs/get-started/authentication-and-authorization-flow/implicit-flow-with-form-post) を使用します。

    また、_wandb/local_ を設定して、[PKCE Code Exchange](https://www.oauth.com/oauth2-servers/pkce/) フローを使用する `authorization_code` grant を実行することもできます。

4. アプリにトークンを届ける方法を AWS Cognito で設定するために、1 つ以上の OAuth グラントタイプを選択します。
5. W&B は特定の OpenID Connect (OIDC) スコープを要求します。AWS Cognito App から以下を選択してください：
    * "openid" 
    * "profile"
    * "email"

    たとえば、AWS Cognito アプリの UI は以下の画像のようになります：

    {{< img src="/images/hosting/setup_aws_required_fields.png" alt="必須フィールド" >}}

    設定ページで **Auth Method** を選択するか、`OIDC_AUTH_METHOD` 環境変数を設定して、どのグラントが _wandb/local_ に適しているかを指定します。

    Auth Method を `pkce` に設定する必要があります。

6. クライアント ID および OIDC 発行者の URL が必要です。OpenID ディスカバリドキュメントは `$OIDC_ISSUER/.well-known/openid-configuration` で利用可能でなければなりません。

    たとえば、ユーザープール ID を **User Pools** セクションの **App Integration** タブから、Cognito IdP URL に追加することで発行者 URL を生成できます：

    {{< img src="/images/hosting/setup_aws_cognito_issuer_url.png" alt="AWS Cognito での発行者 URL のスクリーンショット" >}}

    IDP URL には「Cognito ドメイン」を使用しないでください。Cognito は `https://cognito-idp.$REGION.amazonaws.com/$USER_POOL_ID` でそのディスカバリドキュメントを提供します。

{{% /tab %}}

{{% tab header="Okta" value="okta"%}}
Okta を認証に設定する手順は以下の通りです： 

1. https://login.okta.com/ で Okta ポータルにログインします。

2. 左側のサイドバーで **Applications**、そして再度 **Applications** を選択します。
    {{< img src="/images/hosting/okta_select_applications.png" alt="" >}}

3. "Create App integration" をクリックします。
    {{< img src="/images/hosting/okta_create_new_app_integration.png" alt="" >}}

4. "Create a new app integration" 画面で **OIDC - OpenID Connect** と **Single-Page Application** を選択し、次に「Next」をクリックします。
    {{< img src="/images/hosting/okta_create_a_new_app_integration.png" alt="" >}}

5. "New Single-Page App Integration" 画面で、次の内容を入力し「Save」をクリックします：
    - アプリ統合名、例として "Weights & Biases"
    - グラントタイプ: **Authorization Code** と **Implicit (hybrid)** の両方を選択
    - サインイン リダイレクト URI: https://YOUR_W_AND_B_URL/oidc/callback
    - サインアウト リダイレクト URI: https://YOUR_W_AND_B_URL/logout
    - 割り当て: **Skip group assignment for now** を選択
    {{< img src="/images/hosting/okta_new_single_page_app_integration.png" alt="" >}}

6. 作成したばかりの Okta アプリケーションの概要画面で、**Client ID** を **Client Credentials** の **General** タブの下に記録します：
    {{< img src="/images/hosting/okta_make_note_of_client_id.png" alt="" >}}

7. Okta OIDC 発行者 URL を特定するには、左側のメニューで **Settings** そして **Account** を選択します。
    Okta UI は **Organization Contact** の下に企業名を表示します。
    {{< img src="/images/hosting/okta_identify_oidc_issuer_url.png" alt="" >}}

OIDC 発行者 URL は `https://COMPANY.okta.com` の形式です。該当する値で COMPANY を置き換えて、注意してください。
{{% /tab %}}

{{% tab header="Entra" value="entra"%}}
1. https://portal.azure.com/ で Azure ポータルにログインします。

2. 「Microsoft Entra ID」サービスを選択します。
    {{< img src="/images/hosting/entra_select_entra_service.png" alt="" >}}

3. 左側のサイドバーで「App registrations」を選択します。
    {{< img src="/images/hosting/entra_app_registrations.png" alt="" >}}

4. 上部で「New registration」をクリックします。
    {{< img src="/images/hosting/entra_new_app_registration.png" alt="" >}}

    「アプリケーションの登録」画面で次の値を入力します：
    {{< img src="/images/hosting/entra_register_an_application.png" alt="" >}}

    - 名前を指定します。例として「Weights and Biases application」
    - デフォルトでは選択されたアカウントタイプは「この組織ディレクトリ内のアカウントのみ (デフォルトディレクトリのみ - シングルテナント)」です。必要に応じて修正してください。
    - リダイレクト URI を **Web** タイプで設定し、値は `https://YOUR_W_AND_B_URL/oidc/callback`
    - 「登録」をクリックします。

    - 「アプリケーション (client) ID」と「ディレクトリ (テナント) ID」をメモしておいてください。

      {{< img src="/images/hosting/entra_app_overview_make_note.png" alt="" >}}

5. 左側のサイドバーで、**Authentication** をクリックします。
    {{< img src="/images/hosting/entra_select_authentication.png" alt="" >}}

    - **Front-channel logout URL** の下に次を指定します: `https://YOUR_W_AND_B_URL/logout`
    - 「保存」をクリックします。

      {{< img src="/images/hosting/entra_logout_url.png" alt="" >}}

6. 左側のサイドバーで「Certificates & secrets」をクリックします。
    {{< img src="/images/hosting/entra_select_certificates_secrets.png" alt="" >}}

    - 「Client secrets」をクリックし、「New client secret」をクリックします。
      {{< img src="/images/hosting/entra_new_secret.png" alt="" >}}

    「クライアントシークレットの追加」画面で次の値を入力します：
      {{< img src="/images/hosting/entra_add_new_client_secret.png" alt="" >}}

      - 説明を入力します。例として「wandb」
      - 「有効期限」はそのままにしておくか、必要に応じて変更します。
      - 「追加」をクリックします。

    - シークレットの「値」をメモしておいてください。「シークレット ID」は不要です。
    {{< img src="/images/hosting/entra_make_note_of_secret_value.png" alt="" >}}

これで次の 3 つの値をメモしておいてください：
- OIDC クライアント ID
- OIDC クライアントシークレット
- OIDC 発行者 URL に必要なテナント ID

OIDC 発行者 URL は次の形式です：`https://login.microsoftonline.com/${TenantID}/v2.0`
{{% /tab %}}
{{< /tabpane >}}

## W&B サーバーでの SSO 設定

SSO を設定するには、管理者権限と次の情報が必要です：
- OIDC クライアント ID
- OIDC 認証方法（`implicit` または `pkce`）
- OIDC 発行者 URL
- OIDC クライアントシークレット (オプション; IdP の設定方法に依存します)

{{% alert %}}
IdP が OIDC クライアントシークレットを要求する場合、環境変数 `OIDC_CLIENT_SECRET` で指定してください。
{{% /alert %}}

SSO の設定は、W&B サーバー UI を使用するか、`wandb/local` pod に[環境変数]({{< relref path="/guides/hosting/env-vars.md" lang="ja" >}}) を渡して設定することができます。環境変数が UI よりも優先されます。

{{% alert %}}
SSO を設定した後でインスタンスにログインできない場合、`LOCAL_RESTORE=true` 環境変数を設定してインスタンスを再起動できます。これにより、一時的なパスワードがコンテナのログに出力され、SSO が無効になります。SSO の問題を解決したら、環境変数を削除して SSO を再度有効化する必要があります。
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab header="System Console" value="console" %}}
System Console は System Settings ページの後継です。これは [W&B Kubernetes Operator]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/" lang="ja" >}}) ベースのデプロイメントで利用可能です。

1. [Access the W&B Management Console]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/#access-the-wb-management-console" lang="ja" >}}) を参照してください。

2. **Settings** に移動し、次に **Authentication** を選択します。**Type** ドロップダウンで **OIDC** を選択します。
    {{< img src="/images/hosting/sso_configure_via_console.png" alt="" >}}

3. 値を入力します。

4. **Save** をクリックします。

5. ログアウトし、IdP ログイン画面を使用して再度ログインします。
{{% /tab %}}
{{% tab header="System settings" value="settings" %}}
1. Weights&Biases インスタンスにサインインします。 
2. W&B アプリに移動します。 

    {{< img src="/images/hosting/system_settings.png" alt="" >}}

3. ドロップダウンから **System Settings** を選択します:

    {{< img src="/images/hosting/system_settings_select_settings.png" alt="" >}}

4. 発行者、クライアント ID、および認証方法を入力します。
5. **Update settings** を選択します。

{{< img src="/images/hosting/system_settings_select_update.png" alt="" >}}
{{% /tab %}}
{{< /tabpane >}}

{{% alert %}}
SSO を設定した後でインスタンスにログインできない場合、`LOCAL_RESTORE=true` 環境変数を設定してインスタンスを再起動できます。これにより、一時的なパスワードがコンテナのログに出力され、SSO がオフになります。SSO の問題を解決したら、環境変数を削除して SSO を再度有効化する必要があります。
{{% /alert %}}

## セキュリティ・アサーション・マークアップ言語 (SAML)
W&B サーバーは SAML をサポートしていません。
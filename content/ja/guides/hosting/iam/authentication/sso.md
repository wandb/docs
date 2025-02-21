---
title: Configure SSO with OIDC
menu:
  default:
    identifier: ja-guides-hosting-iam-authentication-sso
    parent: authentication
---

W&B Server の OpenID Connect (OIDC) 互換のアイデンティティプロバイダのサポートにより、Okta、Keycloak、Auth0、Google、Entra などの外部アイデンティティプロバイダを通じてユーザーのアイデンティティとグループメンバーシップの管理が可能になります。

## OpenID Connect (OIDC)

W&B Server は、外部アイデンティティプロバイダ (IdP) との統合のために、次の OIDC 認証フローをサポートしています。
1. フォーム投稿を用いた暗黙的フロー
2. コード交換のための証明キー付きの承認コードフロー (PKCE)

これらのフローはユーザーを認証し、W&B Server にアクセス制御を管理するために必要なアイデンティティ情報（ID トークンの形で）を提供します。

ID トークンは、ユーザーの名前、ユーザー名、メール、グループメンバーシップなどのアイデンティティ情報を含む JWT（JSON Web トークン）です。W&B Server はこのトークンを使用してユーザーを認証し、システム内の適切な役割やグループにマッピングします。

W&B Server のコンテキストでは、アクセストークンはユーザーの代わりに API へのリクエストを承認しますが、W&B Server の主な関心事はユーザー認証とアイデンティティであるため、ID トークンのみが必要です。

[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [セルフマネージド]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンスの [IAM オプションを設定する]({{< relref path="../advanced_env_vars.md" lang="ja" >}}) には、環境変数を使用できます。

[Dedicated cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) W&B Server インストールのために Identity Providers を設定するためのガイドラインを以下に示します。W&B の SaaS バージョンを使用している場合は、組織のために Auth0 テナントを設定するサポートが必要な場合に [support@wandb.com](mailto:support@wandb.com) にお問い合わせください。

{{< tabpane text=true >}}
{{% tab header="Cognito" value="cognito" %}}
AWS Cognito の認証を設定するには、以下の手順に従います：

1. まず、AWS アカウントにサインインし、[AWS Cognito](https://aws.amazon.com/cognito/) アプリに移動します。

    {{< img src="/images/hosting/setup_aws_cognito.png" alt="認証のために OIDC を使用し、認可ではない場合、パブリッククライアントはセットアップを簡素化します" >}}

2. IdP にアプリケーションを設定するための許可されたコールバック URL を指定します:
     * `http(s)://YOUR-W&B-HOST/oidc/callback` をコールバック URL として追加します。`YOUR-W&B-HOST` を W&B ホストのパスに置き換えます。

3. IdP がユニバーサルログアウトをサポートしている場合、ログアウト URL を `http(s)://YOUR-W&B-HOST` に設定します。`YOUR-W&B-HOST` を W&B ホストのパスに置き換えます。

    たとえば、アプリケーションが `https://wandb.mycompany.com` で実行されている場合、`YOUR-W&B-HOST` を `wandb.mycompany.com` に置き換えます。

    下の画像は、AWS Cognito で許可されたコールバックとサインアウト URL を提供する方法を示しています。

    {{< img src="/images/hosting/setup_aws_cognito_ui_settings.png" alt="インスタンスが複数のホストからアクセス可能な場合、ここにすべてを含めるようにしてください。" >}}


    _wandb/local_ はデフォルトで [`フォーム投稿の応答タイプを使用した暗黙の承認`](https://auth0.com/docs/get-started/authentication-and-authorization-flow/implicit-flow-with-form-post) を使用しています。

    また、_wandb/local_ を PKCE コード交換フローを使用する `authorization_code` grant を実行するように設定することもできます。

4. AWS Cognito がトークンをアプリに配信する方法を設定するために、1 つまたは複数の OAuth グラントタイプを選択します。
5. W&B は特定の OpenID Connect (OIDC) スコープを必要とします。AWS Cognito アプリから次のものを選択します:
    * "openid" 
    * "profile"
    * "email"

    たとえば、AWS Cognito アプリ UI は次の画像に似ているはずです:

    {{< img src="/images/hosting/setup_aws_required_fields.png" alt="必要なフィールド" >}}

    設定ページで **Auth Method** を選択するか、環境変数 OIDC_AUTH_METHOD を設定して、wandb/local がどのグラントを使用するかを指定します。

    Auth Method を `pkce` に設定する必要があります。

6. クライアント ID と OIDC 発行者の URL が必要です。OpenID ディスカバリ ドキュメントは、`$OIDC_ISSUER/.well-known/openid-configuration` に存在する必要があります。

    たとえば、**User Pools** セクション内の **App Integration** タブから Cognito IdP URL に User Pool ID を追加することで、発行者 URL を生成できます:

    {{< img src="/images/hosting/setup_aws_cognito_issuer_url.png" alt="AWS Cognito の発行者 URL のスクリーンショット" >}}

    IdP URL に「Cognito ドメイン」を使用しないでください。Cognito は、`https://cognito-idp.$REGION.amazonaws.com/$USER_POOL_ID` にディスカバリ ドキュメントを提供します。

{{% /tab %}}

{{% tab header="Okta" value="okta"%}}
Okta の認証を設定するには、以下の手順に従います：

1. https://login.okta.com/ から Okta ポータルにログインします。

2. 左側で **Applications** を選択し、再度 **Applications** を選択します。
    {{< img src="/images/hosting/okta_select_applications.png" alt="" >}}

3. 「Create App integration」をクリックします。
    {{< img src="/images/hosting/okta_create_new_app_integration.png" alt="" >}}

4. 「Create a new app integration」画面で、**OIDC - OpenID Connect** と **Single-Page Application** を選択します。次に「Next」をクリックします。
    {{< img src="/images/hosting/okta_create_a_new_app_integration.png" alt="" >}}

5. 「New Single-Page App Integration」画面で、次の値を入力し、**Save** をクリックします：
    - アプリ統合名（例："Weights & Biases"）
    - グラントタイプ: **Authorization Code** と **Implicit (hybrid)** の両方を選択
    - サインインリダイレクト URI: https://YOUR_W_AND_B_URL/oidc/callback
    - サインアウトリダイレクト URI: https://YOUR_W_AND_B_URL/logout
    - 割り当て: 「グループ割り当てを今はスキップ」を選択
    {{< img src="/images/hosting/okta_new_single_page_app_integration.png" alt="" >}}

6. 作成した Okta アプリケーションの概要画面で、**General** タブの **Client Credentials** の下にある **Client ID** をメモします：
    {{< img src="/images/hosting/okta_make_note_of_client_id.png" alt="" >}}

7. Okta OIDC Issuer URL を特定するには、左側で **Settings** を選択し、次に **Account** を選択します。
    Okta UI は **Organization Contact** の下に会社名を表示します。
    {{< img src="/images/hosting/okta_identify_oidc_issuer_url.png" alt="" >}}

コンパニーを対応する値で置き換えてください。これをメモします。
{{% /tab %}}

{{% tab header="Entra" value="entra"%}}
1. https://portal.azure.com/ にアクセスし、Azure ポータルにログインします。

2. 「Microsoft Entra ID」サービスを選択します。
    {{< img src="/images/hosting/entra_select_entra_service.png" alt="" >}}

3. 左側で「アプリ登録」を選択します。
    {{< img src="/images/hosting/entra_app_registrations.png" alt="" >}}

4. 上部で「新しい登録」をクリックします。
    {{< img src="/images/hosting/entra_new_app_registration.png" alt="" >}}

    「アプリケーションの登録」画面で、以下の値を入力します：
    {{< img src="/images/hosting/entra_register_an_application.png" alt="" >}}

    - 名前を指定（例："Weights and Biases application"）
    - デフォルトで選択されているアカウントタイプは:「この組織ディレクトリのアカウントのみ (デフォルトディレクトリのみ - シングルテナント)」です。必要に応じて変更します。
    - リダイレクト URI を **Web** タイプとして次の値で設定: `https://YOUR_W_AND_B_URL/oidc/callback`
    - 「登録」をクリック

    - 「アプリケーション (クライアント) ID」と「ディレクトリ (テナント) ID」をメモします。

      {{< img src="/images/hosting/entra_app_overview_make_note.png" alt="" >}}


5. 左側で **Authentication** をクリックします。
    {{< img src="/images/hosting/entra_select_authentication.png" alt="" >}}

    - **フロントチャネルログアウト URL** の下に次を指定: `https://YOUR_W_AND_B_URL/logout`
    - 「保存」をクリック

      {{< img src="/images/hosting/entra_logout_url.png" alt="" >}}


6. 左側で「証明書とシークレット」をクリックします。
    {{< img src="/images/hosting/entra_select_certificates_secrets.png" alt="" >}}

    - 「クライアントシークレット」をクリックし、「新しいクライアントシークレット」をクリック
      {{< img src="/images/hosting/entra_new_secret.png" alt="" >}}

      「クライアントシークレットの追加」画面で、次の値を入力します：
      {{< img src="/images/hosting/entra_add_new_client_secret.png" alt="" >}}

      - 説明を入力（例："wandb"）
      - 「有効期限」はそのままか、必要に応じて変更
      - 「追加」をクリック


    - シークレットの「値」をメモします。「シークレット ID」は不要です。
    {{< img src="/images/hosting/entra_make_note_of_secret_value.png" alt="" >}}

これで次の 3 つの値をメモしたことになります：
- OIDC クライアント ID
- OIDC クライアントシークレット
- OIDC 発行者 URL に必要なテナント ID

OIDC 発行者 URL の形式は次の通りです: `https://login.microsoftonline.com/${TenantID}/v2.0`
{{% /tab %}}
{{< /tabpane >}}

## W&B サーバー に SSO を設定する

SSO を設定するには、管理者権限が必要で、以下の情報が必要です：
- OIDC クライアント ID
- OIDC 認証メソッド（`implicit` または `pkce`）
- OIDC 発行者 URL
- OIDC クライアントシークレット（オプション；IdP の設定方法による）

{{% alert %}}
IdP が OIDC クライアントシークレットを要求する場合は、環境変数 `OIDC_CLIENT_SECRET` でそれを指定してください。
{{% /alert %}}

SSO は W&B Server UI または `wandb/local` ポッドに [環境変数]({{< relref path="/guides/hosting/env-vars.md" lang="ja" >}}) を渡すことで設定できます。環境変数が UI よりも優先されます。

{{% alert %}}
SSO 設定後にインスタンスにログインできなくなった場合は、環境変数 `LOCAL_RESTORE=true` を設定してインスタンスを再起動できます。これにより、一時的なパスワードがコンテナのログに出力され、SSO が無効になります。SSO の問題を解決した後、この環境変数を削除して SSO を再度有効にする必要があります。
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab header="システムコンソール" value="console" %}}
システムコンソールは、システム設定ページの後継であり、[W&B Kubernetes Operator]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/" lang="ja" >}}) に基づくデプロイメントで利用可能です。

1. [Access the W&B Management Console]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/#access-the-wb-management-console" lang="ja" >}}) を参照してください。

2. **Settings** に移動し、その後 **Authentication** を選択します。**Type** ドロップダウンで **OIDC** を選択します。
    {{< img src="/images/hosting/sso_configure_via_console.png" alt="" >}}

3. 値を入力します。

4. **Save** をクリックします。

5. ログアウトしてから再度ログインし、今度は IdP ログイン画面を使用します。
{{% /tab %}}
{{% tab header="システム設定" value="settings" %}}
1. Weights & Biases インスタンスにサインインします。
2. W&B アプリに移動します。

    {{< img src="/images/hosting/system_settings.png" alt="" >}}

3. ドロップダウンから **System Settings** を選択します：

    {{< img src="/images/hosting/system_settings_select_settings.png" alt="" >}}

4. 発行者、クライアント ID、および認証メソッドを入力します。
5. **Update settings** を選択します。

{{< img src="/images/hosting/system_settings_select_update.png" alt="" >}}
{{% /tab %}}
{{< /tabpane >}}

{{% alert %}}
SSO 設定後にインスタンスにログインできなくなった場合は、環境変数 `LOCAL_RESTORE=true` を設定してインスタンスを再起動できます。これにより、一時的なパスワードがコンテナのログに出力され、SSO が無効になります。SSO の問題を解決した後、この環境変数を削除して SSO を再度有効にする必要があります。
{{% /alert %}}

## Security Assertion Markup Language (SAML)
W&B Server は SAML をサポートしていません。
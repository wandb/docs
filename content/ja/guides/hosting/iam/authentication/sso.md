---
title: OIDC で SSO を設定
menu:
  default:
    identifier: sso
    parent: authentication
---

W&B Server の OpenID Connect (OIDC) 対応により、Okta、Keycloak、Auth0、Google、Entra などの外部アイデンティティプロバイダーを使って ユーザーID やグループメンバーシップを管理できます。

## OpenID Connect (OIDC)

W&B Server は、外部のアイデンティティプロバイダー（IdP）と連携するための以下の OIDC 認証フローに対応しています。
1. フォームポスト付きインプリシットフロー 
2. 証明コード付き認可コードフロー（PKCE 利用）

これらのフローによりユーザーを認証し、W&B Server はユーザーの ID 情報（ID トークンの形式で）を取得してアクセス制御を行います。

ID トークンは JWT であり、ユーザーの氏名、ユーザー名、メール、グループメンバーシップ等の ID 情報が含まれます。W&B Server はこのトークンを利用してユーザー認証を行い、適切なロールやグループに紐付けます。

W&B Server においては、アクセストークンはユーザーの代わりに API へのリクエスト権限を与えますが、W&B Server の主な用途はユーザー認証・識別のため、ID トークンのみを必要とします。

[環境変数を使って IAM オプションを設定]({{< relref "../advanced_env_vars.md" >}})し、[専用クラウド]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}) や [セルフマネージド]({{< relref "/guides/hosting/hosting-options/self-managed.md" >}}) インスタンスの設定が可能です。

[専用クラウド]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}) や [セルフマネージド]({{< relref "/guides/hosting/hosting-options/self-managed.md" >}}) の W&B Server インストール時に IdP の設定をサポートするため、さまざまな IdP ごとのガイドラインに従ってください。SaaS バージョンの W&B をご利用の場合は、組織用の Auth0 テナント設定支援のため [support@wandb.com](mailto:support@wandb.com) までご連絡ください。

{{< tabpane text=true >}}
{{% tab header="Cognito" value="cognito" %}}
AWS Cognito で認可を設定するには、以下の手順に従ってください。

1. まず、AWS アカウントにサインインし、[AWS Cognito](https://aws.amazon.com/cognito/) アプリに移動します。

    {{< img src="/images/hosting/setup_aws_cognito.png" alt="AWS Cognito setup" >}}

2. IdP でアプリケーション設定を行うため、許可されたコールバック URL を指定します:
     * コールバック URL として `http(s)://YOUR-W&B-HOST/oidc/callback` を追加してください。`YOUR-W&B-HOST` はご自身の W&B ホストのパスに置き換えてください。

3. IdP がユニバーサルログアウトをサポートしている場合は、ログアウト URL を `http(s)://YOUR-W&B-HOST` に設定します。`YOUR-W&B-HOST` はご自身の W&B ホストパスに置き換えてください。

    例えば、アプリケーションが `https://wandb.mycompany.com` で稼働している場合、`YOUR-W&B-HOST` を `wandb.mycompany.com` に置き換えてください。

    下記画像は、AWS Cognito でコールバックおよびサインアウト URL を指定する画面例です。

    {{< img src="/images/hosting/setup_aws_cognito_ui_settings.png" alt="Host configuration" >}}


    _wandb/local_ はデフォルトで [`implicit` グラントと `form_post` レスポンスタイプ](https://auth0.com/docs/get-started/authentication-and-authorization-flow/implicit-flow-with-form-post) を使用します。

    _wandb/local_ で [PKCE Code Exchange](https://www.oauth.com/oauth2-servers/pkce/) フローを利用する `authorization_code` グラントも設定可能です。

4. 1つ以上の OAuth グラントタイプを選び、AWS Cognito からのトークン発行方法を設定します。
5. W&B で必要な OpenID Connect (OIDC) スコープを選択します。AWS Cognito アプリで以下を有効にしてください:
    * "openid"
    * "profile"
    * "email"

    例えば、AWS Cognito アプリの画面は下記のようになります。

    {{< img src="/images/hosting/setup_aws_required_fields.png" alt="Required fields" >}}

    設定ページで **Auth Method** を選択するか、OIDC_AUTH_METHOD 環境変数をセットし、_wandb/local_ でどのグラントを使うか指定します。

    Auth Method は `pkce` に設定してください。

6. Client ID および OIDC 発行者 URL（Issuer URL）が必要です。OpenID のディスカバリードキュメントは `$OIDC_ISSUER/.well-known/openid-configuration` で利用可能でなければなりません。

    例えば、Issuer URL は **App Integration** タブ内の **User Pools** セクションから User Pool ID を Cognito IdP URL に追加することで生成できます:

    {{< img src="/images/hosting/setup_aws_cognito_issuer_url.png" alt="AWS Cognito issuer URL" >}}

    IDP URL には「Cognito domain」を使用しないでください。Cognito のディスカバリードキュメントは `https://cognito-idp.$REGION.amazonaws.com/$USER_POOL_ID` で提供されています。

{{% /tab %}}

{{% tab header="Okta" value="okta"%}}
Okta で認可を設定するには、以下の手順に従ってください。

1. [Okta Portal](https://login.okta.com/) にログインします。

2. 左側のメニューから **Applications** を選択し、再度 **Applications** を選びます。
    {{< img src="/images/hosting/okta_select_applications.png" alt="Okta Applications menu" >}}

3. 「Create App integration」をクリックします。
    {{< img src="/images/hosting/okta_create_new_app_integration.png" alt="Create App integration button" >}}

4. 「Create a new app integration」画面で、**OIDC - OpenID Connect** と **Single-Page Application** を選び、「Next」をクリックします。
    {{< img src="/images/hosting/okta_create_a_new_app_integration.png" alt="OIDC Single-Page Application selection" >}}

5. 「New Single-Page App Integration」画面で以下の値を入力して **Save** をクリックします。
    - App integration name：例「W&B」
    - Grant type：**Authorization Code** および **Implicit (hybrid)** の両方を選択
    - Sign-in redirect URIs: https://YOUR_W_AND_B_URL/oidc/callback
    - Sign-out redirect URIs: https://YOUR_W_AND_B_URL/logout
    - Assignments: **Skip group assignment for now** を選択
    {{< img src="/images/hosting/okta_new_single_page_app_integration.png" alt="Single-Page App configuration" >}}

6. 作成した Okta アプリケーションの概要画面で、**General** タブ内 **Client Credentials** にある **Client ID** をメモします:
    {{< img src="/images/hosting/okta_make_note_of_client_id.png" alt="Okta Client ID location" >}}

7. Okta OIDC Issuer URL を特定するため、左側の **Settings** > **Account** をクリック。
    Okta UI の **Organization Contact** 下部に会社名が表示されます。
    {{< img src="/images/hosting/okta_identify_oidc_issuer_url.png" alt="Okta organization settings" >}}

OIDC Issuer URL は `https://COMPANY.okta.com` という形式です。COMPANY は該当の値に置き換えてください。これもメモしておきましょう。
{{% /tab %}}

{{% tab header="Entra" value="entra"%}}
1. [Azure Portal](https://portal.azure.com/) にログインします。

2. 「Microsoft Entra ID」サービスを選択します。
    {{< img src="/images/hosting/entra_select_entra_service.png" alt="Microsoft Entra ID service" >}}

3. 左側の「App registrations」を選択します。
    {{< img src="/images/hosting/entra_app_registrations.png" alt="App registrations menu" >}}

4. 上部の「New registration」をクリックします。
    {{< img src="/images/hosting/entra_new_app_registration.png" alt="New registration button" >}}

    「Register an application」画面で下記を入力します:
    {{< img src="/images/hosting/entra_register_an_application.png" alt="Application registration form" >}}

    - 名前を指定（例:「Weights and Biases application」）
    - デフォルトでは「この組織ディレクトリ内のアカウントのみ（Default Directory のみ－シングルテナント）」が選択されています。必要に応じて変更可。
    - Redirect URI のタイプは **Web** を選び、値は `https://YOUR_W_AND_B_URL/oidc/callback`
    - 「Register」をクリック

    - 「Application (client) ID」と「Directory (tenant) ID」をメモしておきます。

      {{< img src="/images/hosting/entra_app_overview_make_note.png" alt="Application and Directory IDs" >}}


5. 左側で **Authentication** をクリックします。
    {{< img src="/images/hosting/entra_select_authentication.png" alt="Authentication menu" >}}

    - **Front-channel logout URL** には `https://YOUR_W_AND_B_URL/logout` を指定
    - 「Save」をクリック

      {{< img src="/images/hosting/entra_logout_url.png" alt="Front-channel logout URL" >}}


6. 左側の「Certificates & secrets」をクリック。
    {{< img src="/images/hosting/entra_select_certificates_secrets.png" alt="Certificates & secrets menu" >}}

    - 「Client secrets」をクリックし、「New client secret」を押します。
      {{< img src="/images/hosting/entra_new_secret.png" alt="New client secret button" >}}

      「Add a client secret」画面で下記を入力します:
      {{< img src="/images/hosting/entra_add_new_client_secret.png" alt="Client secret configuration" >}}

      - 説明を入力（例:「wandb」）
      - 「Expires」はそのまま、または必要に応じて変更
      - 「Add」をクリック


    - 秘密の「値（Value）」を必ず控えます。「Secret ID」は不要です。
    {{< img src="/images/hosting/entra_make_note_of_secret_value.png" alt="Client secret value" >}}

この段階で3つの値が控えてあるはずです:
- OIDC Client ID
- OIDC Client Secret
- Tenant ID（OIDC Issuer URL 生成に必要）

OIDC Issuer URL 形式は `https://login.microsoftonline.com/${TenantID}/v2.0` です。
{{% /tab %}}
{{< /tabpane >}}

## W&B Server で SSO をセットアップする

SSO 設定のためには、管理者権限と下記情報が必要です。
- OIDC Client ID
- OIDC Auth method（`implicit` または `pkce`）
- OIDC Issuer URL
- OIDC Client Secret（オプション・IdP の設定方法により異なる）

IdP が OIDC Client Secret を必要とする場合は、[環境変数]({{< relref "/guides/hosting/env-vars.md" >}}) `OIDC_CLIENT_SECRET` を渡して指定してください。
- UI では **System Console** > **Settings** > **Advanced** > **User Spec** に移動し、下記のように `extraENV` セクションに `OIDC_CLIENT_SECRET` を追加します。
- Helm では `values.global.extraEnv` を下記のように設定します。

```yaml
values:
  global:
    extraEnv:
      OIDC_CLIENT_SECRET="<your_secret>"
```

{{% alert %}}
SSO 設定後にインスタンスへログインできなくなった場合、`LOCAL_RESTORE=true` の環境変数をセットしてインスタンスを再起動してください。これにより一時的なパスワードがコンテナのログに出力され、SSO が無効化されます。SSO の問題解決後はこの環境変数を必ず削除してください。
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab header="System Console" value="console" %}}
System Console は System Settings ページの後継です。[W&B Kubernetes Operator]({{< relref "/guides/hosting/hosting-options/self-managed/kubernetes-operator/" >}}) ベースのデプロイメントで利用可能です。

1. [W&B 管理コンソールへのアクセス方法]({{< relref "/guides/hosting/hosting-options/self-managed/kubernetes-operator/#access-the-wb-management-console" >}}) を参照してください。

2. **Settings** から **Authentication** に進み、**Type** で **OIDC** を選択します。
    {{< img src="/images/hosting/sso_configure_via_console.png" alt="System Console OIDC configuration" >}}

3. 各値を入力します。

4. **Save** をクリック。

5. 一度ログアウトし、新たに IdP ログイン画面経由で再度ログインしてください。
{{% /tab %}}
{{% tab header="System settings" value="settings" %}}
1. Weights&Biases インスタンスにサインインします。
2. W&B App にアクセスします。

    {{< img src="/images/hosting/system_settings.png" alt="W&B App navigation" >}}

3. ドロップダウンから **System Settings** を選択します。

    {{< img src="/images/hosting/system_settings_select_settings.png" alt="System Settings dropdown" >}}

4. Issuer、Client ID、Authentication Method を入力します。
5. **Update settings** を選択します。

{{< img src="/images/hosting/system_settings_select_update.png" alt="Update settings button" >}}
{{% /tab %}}
{{< /tabpane >}}

{{% alert %}}
SSO 設定後にインスタンスへログインできなくなった場合、`LOCAL_RESTORE=true` の環境変数をセットしてインスタンスを再起動してください。これにより一時的なパスワードがコンテナログに出力され、SSO が無効になります。SSO の問題解決後はこの環境変数を必ず削除してください。
{{% /alert %}}

## Security Assertion Markup Language (SAML)
W&B Server は SAML をサポートしていません。
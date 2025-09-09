---
title: OIDC を使用した SSO の設定
menu:
  default:
    identifier: ja-guides-hosting-iam-authentication-sso
    parent: authentication
---

W&B Server の OpenID Connect (OIDC) 互換アイデンティティプロバイダー対応により、Okta、Keycloak、Auth0、Google、Entra などの外部 IdP を通じてユーザーのアイデンティティとグループメンバーシップを管理できます。

## OpenID Connect (OIDC)

W&B Server は、外部 IdP（Identity Provider）とのインテグレーションのために、次の OIDC 認証フローをサポートします。
1. Implicit Flow（Form Post）
2. Authorization Code フロー（Proof Key for Code Exchange: PKCE）

これらのフローはユーザーを認証し、W&B Server がアクセス制御を行うために必要なアイデンティティ情報（ID トークンの形式）を提供します。

ID トークンは JWT で、ユーザーの氏名、ユーザー名、メール、グループメンバーシップなどのアイデンティティ情報が含まれます。W&B Server はこのトークンを用いてユーザーを認証し、システム内の適切なロールやグループにマッピングします。

W&B Server の文脈では、アクセストークンはユーザーに代わって API へのリクエストを許可しますが、W&B Server の主な関心はユーザー認証とアイデンティティであるため、必要なのは ID トークンのみです。

環境変数を使って、[IAM オプションを設定]({{< relref path="../advanced_env_vars.md" lang="ja" >}}) し、[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [セルフマネージド]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンスに適用できます。

[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [セルフマネージド]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) の W&B Server インストールで IdP を設定する際は、以下の IdP ごとのガイドラインに従ってください。SaaS 版の W&B をお使いの場合は、組織向けに Auth0 テナントを設定する支援について [support@wandb.com](mailto:support@wandb.com) までお問い合わせください。

{{< tabpane text=true >}}
{{% tab header="Cognito" value="cognito" %}}
以下の手順で AWS Cognito による認可を設定します。 

1. まず AWS アカウントにサインインし、[AWS Cognito](https://aws.amazon.com/cognito/) アプリに移動します。

    {{< img src="/images/hosting/setup_aws_cognito.png" alt="AWS Cognito のセットアップ" >}}

2. IdP 側のアプリケーション設定として、許可するコールバック URL を指定します。
     * コールバック URL として `http(s)://YOUR-W&B-HOST/oidc/callback` を追加します。`YOUR-W&B-HOST` はお使いの W&B ホスト名に置き換えてください。

3. IdP がユニバーサルログアウトをサポートしている場合、Logout URL を `http(s)://YOUR-W&B-HOST` に設定します。`YOUR-W&B-HOST` はお使いの W&B ホスト名に置き換えてください。

    たとえば、アプリケーションが `https://wandb.mycompany.com` で動作している場合、`YOUR-W&B-HOST` は `wandb.mycompany.com` に置き換えます。

    下の画像は、AWS Cognito で許可されたコールバック URL とサインアウト URL を指定する方法を示しています。

    {{< img src="/images/hosting/setup_aws_cognito_ui_settings.png" alt="ホストの設定" >}}


    _wandb/local_ はデフォルトで、[`implicit` グラント（`form_post` レスポンスタイプ）](https://auth0.com/docs/get-started/authentication-and-authorization-flow/implicit-flow-with-form-post) を使用します。 

    また、_wandb/local_ を [PKCE Code Exchange](https://www.oauth.com/oauth2-servers/pkce/) を用いる `authorization_code` グラントで動作するように設定することもできます。 

4. OAuth のグラントタイプを 1 つ以上選択し、AWS Cognito がアプリにトークンを配布する方法を設定します。
5. W&B には特定の OpenID Connect (OIDC) スコープが必要です。AWS Cognito アプリで次を選択します。
    * "openid" 
    * "profile"
    * "email"

    たとえば、AWS Cognito アプリの UI は次の画像のようになります。

    {{< img src="/images/hosting/setup_aws_required_fields.png" alt="必須フィールド" >}}

    設定ページで **Auth Method** を選択するか、環境変数 OIDC_AUTH_METHOD を設定して、_wandb/local_ に使用するグラントを指定します。

    Auth Method は `pkce` に設定してください。

6. Client ID と OIDC Issuer の URL が必要です。OpenID のディスカバリドキュメントは `$OIDC_ISSUER/.well-known/openid-configuration` で取得できる必要があります。 

    たとえば、**User Pools** セクションの **App Integration** タブにある Cognito IdP の URL に User Pool ID を付加して Issuer URL を生成できます。

    {{< img src="/images/hosting/setup_aws_cognito_issuer_url.png" alt="AWS Cognito の Issuer URL" >}}

    IdP の URL に「Cognito domain」を使用しないでください。Cognito のディスカバリドキュメントは `https://cognito-idp.$REGION.amazonaws.com/$USER_POOL_ID` にあります。

{{% /tab %}}

{{% tab header="Okta" value="okta"%}}
以下の手順で Okta による認可を設定します。 

1. [Okta Portal](https://login.okta.com/) にログインします。 

2. 左側で **Applications**、続いて **Applications** を選択します。
    {{< img src="/images/hosting/okta_select_applications.png" alt="Okta の Applications メニュー" >}}

3. "Create App integration" をクリックします。
    {{< img src="/images/hosting/okta_create_new_app_integration.png" alt="Create App integration ボタン" >}}

4. "Create a new app integration" 画面で **OIDC - OpenID Connect** と **Single-Page Application** を選択し、"Next" をクリックします。
    {{< img src="/images/hosting/okta_create_a_new_app_integration.png" alt="OIDC Single-Page Application の選択" >}}

5. "New Single-Page App Integration" 画面で次のように入力し、**Save** をクリックします。
    - App integration name（例: "W&B"）
    - Grant type: **Authorization Code** と **Implicit (hybrid)** の両方を選択
    - Sign-in redirect URIs: https://YOUR_W_AND_B_URL/oidc/callback
    - Sign-out redirect URIs: https://YOUR_W_AND_B_URL/logout
    - Assignments: **Skip group assignment for now** を選択
    {{< img src="/images/hosting/okta_new_single_page_app_integration.png" alt="Single-Page App の設定" >}}

6. 作成した Okta アプリの概要画面で、**General** タブの **Client Credentials** の下にある **Client ID** を控えます。
    {{< img src="/images/hosting/okta_make_note_of_client_id.png" alt="Okta の Client ID の場所" >}}

7. Okta の OIDC Issuer URL を確認するには、左側で **Settings**、続いて **Account** を選択します。
    Okta の UI では **Organization Contact** の下に会社名が表示されます。
    {{< img src="/images/hosting/okta_identify_oidc_issuer_url.png" alt="Okta の組織設定" >}}

OIDC Issuer URL は次の形式です: `https://COMPANY.okta.com`。COMPANY を該当する値に置き換え、控えておきます。
{{% /tab %}}

{{% tab header="Entra" value="entra"%}}
1. [Azure Portal](https://portal.azure.com/) にログインします。

2. "Microsoft Entra ID" サービスを選択します。
    {{< img src="/images/hosting/entra_select_entra_service.png" alt="Microsoft Entra ID サービス" >}}

3. 左側で "App registrations" を選択します。
    {{< img src="/images/hosting/entra_app_registrations.png" alt="App registrations メニュー" >}}

4. 上部の "New registration" をクリックします。
    {{< img src="/images/hosting/entra_new_app_registration.png" alt="New registration ボタン" >}}

    "Register an application" 画面で次のように入力します。
    {{< img src="/images/hosting/entra_register_an_application.png" alt="アプリケーション登録フォーム" >}}

    - 名前を指定します（例: "Weights and Biases application"）。
    - 既定のアカウントタイプは「Accounts in this organizational directory only (Default Directory only - Single tenant)」です。必要に応じて変更します。
    - Redirect URI は種類を **Web**、値を `https://YOUR_W_AND_B_URL/oidc/callback` に設定します。
    - "Register" をクリックします。

    - "Application (client) ID" と "Directory (tenant) ID" を控えます。 

      {{< img src="/images/hosting/entra_app_overview_make_note.png" alt="Application と Directory の ID" >}}


5. 左側で **Authentication** をクリックします。
    {{< img src="/images/hosting/entra_select_authentication.png" alt="Authentication メニュー" >}}

    - **Front-channel logout URL** に `https://YOUR_W_AND_B_URL/logout` を指定します。
    - "Save" をクリックします。

      {{< img src="/images/hosting/entra_logout_url.png" alt="Front-channel logout URL" >}}


6. 左側で "Certificates & secrets" をクリックします。
    {{< img src="/images/hosting/entra_select_certificates_secrets.png" alt="Certificates & secrets メニュー" >}}

    - "Client secrets" をクリックし、"New client secret" をクリックします。
      {{< img src="/images/hosting/entra_new_secret.png" alt="New client secret ボタン" >}}

      "Add a client secret" 画面で次のように入力します。
      {{< img src="/images/hosting/entra_add_new_client_secret.png" alt="Client secret の設定" >}}

      - 説明を入力します（例: "wandb"）。
      - "Expires" はそのまま、必要に応じて変更します。
      - "Add" をクリックします。


    - シークレットの "Value" を控えます。"Secret ID" は不要です。
    {{< img src="/images/hosting/entra_make_note_of_secret_value.png" alt="Client secret の値" >}}

控えておく値は次の 3 つです。
- OIDC Client ID
- OIDC Client Secret
- OIDC Issuer URL に必要な Tenant ID

OIDC Issuer URL は次の形式です: `https://login.microsoftonline.com/${TenantID}/v2.0`
{{% /tab %}}
{{< /tabpane >}}

## W&B Server で SSO を設定する

SSO を設定するには管理者権限と以下の情報が必要です。
- OIDC Client ID
- OIDC Auth method（`implicit` または `pkce`）
- OIDC Issuer URL
- OIDC Client Secret（任意。使用する IdP の設定に依存）

IdP が OIDC Client Secret を必要とする場合は、[環境変数]({{< relref path="/guides/hosting/env-vars.md" lang="ja" >}}) `OIDC_CLIENT_SECRET` を指定してください。
- UI では、**System Console** > **Settings** > **Advanced** > **User Spec** に移動し、下記のように `extraENV` セクションへ `OIDC_CLIENT_SECRET` を追加します。
- Helm の場合は、下記のように `values.global.extraEnv` を設定します。

```yaml
values:
  global:
    extraEnv:
      OIDC_CLIENT_SECRET="<your_secret>"
```

{{% alert %}}
SSO を設定した後にインスタンスへログインできなくなった場合は、`LOCAL_RESTORE=true` の環境変数を設定してインスタンスを再起動できます。これによりコンテナのログに一時パスワードが出力され、SSO が無効化されます。SSO の問題を解決したら、この環境変数を削除して SSO を再度有効化してください。
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab header="System Console" value="console" %}}
System Console は System Settings ページの後継であり、[W&B Kubernetes Operator]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/" lang="ja" >}}) ベースのデプロイメントで利用できます。

1. [W&B Management Console にアクセス]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/#access-the-wb-management-console" lang="ja" >}}) を参照します。

2. **Settings** に進み、**Authentication** を選択します。**Type** のドロップダウンで **OIDC** を選びます。
    {{< img src="/images/hosting/sso_configure_via_console.png" alt="System Console の OIDC 設定" >}}

3. 値を入力します。

4. **Save** をクリックします。

5. 一度ログアウトし、IdP のログイン画面を使用して再度ログインします。

## Customer Namespace を確認する

W&B 専用クラウドまたは セルフマネージド 環境で、CoreWeave ストレージを用いたチーム単位の BYOB を設定する前に、組織の **Customer Namespace** を取得する必要があります。**Authentication** タブの下部で確認・コピーできます。

Customer Namespace を使って CoreWeave ストレージを設定する手順の詳細は、[専用クラウド / セルフマネージド 向けの CoreWeave 要件]({{< relref path="/guides/hosting/data-security/secure-storage-connector#coreweave-customer-namespace" lang="ja" >}}) を参照してください。

{{% /tab %}}
{{% tab header="System settings" value="settings" %}}
1. Weights & Biases インスタンスにサインインします。 
2. W&B App に移動します。 

    {{< img src="/images/hosting/system_settings.png" alt="W&B App へのナビゲーション" >}}

3. ドロップダウンから **System Settings** を選択します。

    {{< img src="/images/hosting/system_settings_select_settings.png" alt="System Settings のドロップダウン" >}}

4. Issuer、Client ID、Authentication Method を入力します。 
5. **Update settings** を選択します。

{{< img src="/images/hosting/system_settings_select_update.png" alt="Update settings ボタン" >}}
{{% /tab %}}
{{< /tabpane >}}

{{% alert %}}
SSO を設定した後にインスタンスへログインできない場合は、`LOCAL_RESTORE=true` の環境変数を設定してインスタンスを再起動できます。これによりコンテナのログに一時パスワードが出力され、SSO がオフになります。SSO の問題を解決したら、この環境変数を削除して SSO を再度有効化してください。
{{% /alert %}}

## Security Assertion Markup Language (SAML)
W&B Server は SAML をサポートしていません。
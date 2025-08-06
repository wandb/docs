---
title: OIDC で SSO を設定
menu:
  default:
    identifier: ja-guides-hosting-iam-authentication-sso
    parent: authentication
---

W&B Server の OpenID Connect (OIDC) 互換のアイデンティティプロバイダー対応により、Okta、Keycloak、Auth0、Google、Entra などの外部アイデンティティプロバイダーを通じてユーザーの管理やグループメンバーシップの管理が可能です。

## OpenID Connect (OIDC)

W&B Server は、外部アイデンティティプロバイダー（IdP）との連携において、以下の OIDC 認証フローをサポートしています。
1. Form Post を利用したインプリシットフロー
2. 認可コードフロー（PKCE: 証明鍵付きコード交換 を利用）

これらのフローによりユーザー認証を行い、W&B Server は（ID トークンとして）必要なユーザー情報を取得し、アクセス制御を管理します。

ID トークンは、ユーザーの名前、ユーザー名、メールアドレス、グループ情報などを含む JWT です。W&B Server はこのトークンをもとにユーザーを認証し、システム内の適切なロールやグループとマッピングします。

W&B Server のコンテキストにおいては、アクセストークンはユーザーの代理として API へのリクエスト認可に使われますが、W&B Server の主な関心はユーザー認証とアイデンティティ認識のため、基本的に必要なのは ID トークンのみです。

[環境変数を使って IAM オプションを設定する方法]({{< relref path="../advanced_env_vars.md" lang="ja" >}})や、[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [自己管理型]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}})インスタンスでの設定についてご覧いただけます。

[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) や [自己管理型]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) W&B Server 導入時に IdP の設定を行う際は、各 IdP ごとに以下のガイドラインをご参照ください。SaaS バージョンの W&B をご利用の場合、組織向け Auth0 テナント設定のサポートは [support@wandb.com](mailto:support@wandb.com) までご連絡ください。

{{< tabpane text=true >}}
{{% tab header="Cognito" value="cognito" %}}
AWS Cognito で認可を設定する手順は以下の通りです。

1. AWS アカウントにサインインし、[AWS Cognito](https://aws.amazon.com/cognito/) アプリにアクセスします。

    {{< img src="/images/hosting/setup_aws_cognito.png" alt="AWS Cognito setup" >}}

2. IdP のアプリケーション設定で許可するコールバック URL を入力します:
     * `http(s)://YOUR-W&B-HOST/oidc/callback` をコールバック URL として追加します。`YOUR-W&B-HOST` にはご自身の W&B ホストパスを指定してください。

3. IdP がユニバーサルログアウトをサポートしている場合は、ログアウト URL を `http(s)://YOUR-W&B-HOST` に設定します。こちらも `YOUR-W&B-HOST` をご自身のパスに置き換えてください。

    例：アプリケーションが `https://wandb.mycompany.com` 上で稼働している場合は、`YOUR-W&B-HOST` を `wandb.mycompany.com` に置き換えます。

    下記イメージは AWS Cognito でのコールバック・サインアウト URL の設定例です。

    {{< img src="/images/hosting/setup_aws_cognito_ui_settings.png" alt="Host configuration" >}}

    _wandb/local_ はデフォルトで [`implicit` グラントの `form_post` レスポンスタイプ](https://auth0.com/docs/get-started/authentication-and-authorization-flow/implicit-flow-with-form-post) を使用します。

    また、_wandb/local_ を [PKCE Code Exchange](https://www.oauth.com/oauth2-servers/pkce/) フローを用いた `authorization_code` グラントにも設定できます。

4. AWS Cognito でトークンの受け渡し方法を決める OAuth グラントタイプを1つ以上選択します。
5. W&B で必要な OIDC スコープを選択します。AWS Cognito で次のスコープを有効にしてください。
    * "openid"
    * "profile"
    * "email"

    例：AWS Cognito アプリ UI は下記のようになります。

    {{< img src="/images/hosting/setup_aws_required_fields.png" alt="Required fields" >}}

    設定ページで **Auth Method** を選択、もしくは OIDC_AUTH_METHOD 環境変数により _wandb/local_ のグラントタイプを指定します。

    Auth Method を必ず `pkce` にしてください。

6. クライアント ID と OIDC イシュア URL が必要です。OpenID のディスカバリードキュメントが `$OIDC_ISSUER/.well-known/openid-configuration` で取得できる必要があります。

    例：**User Pools** セクションの **App Integration** タブから Cognito IdP URL にユーザープール ID を追加するとイシュア URL が生成できます。

    {{< img src="/images/hosting/setup_aws_cognito_issuer_url.png" alt="AWS Cognito issuer URL" >}}

    IDP URL で「Cognito ドメイン」は使用しないでください。ディスカバリードキュメントは `https://cognito-idp.$REGION.amazonaws.com/$USER_POOL_ID` で提供されます。

{{% /tab %}}

{{% tab header="Okta" value="okta"%}}
Okta で認可を設定する手順は以下の通りです。

1. [Okta ポータル](https://login.okta.com/) にログインしてください。

2. 左メニューから **Applications** を2回選択します。
    {{< img src="/images/hosting/okta_select_applications.png" alt="Okta Applications menu" >}}

3. 「Create App integration」をクリック。
    {{< img src="/images/hosting/okta_create_new_app_integration.png" alt="Create App integration button" >}}

4. 「Create a new app integration」画面で **OIDC - OpenID Connect** と **Single-Page Application** を選択し、「Next」を押します。
    {{< img src="/images/hosting/okta_create_a_new_app_integration.png" alt="OIDC Single-Page Application selection" >}}

5. 「New Single-Page App Integration」画面で下記の値を入力し、**Save** をクリックします。
    - App integration name 例: "W&B"
    - Grant type: **Authorization Code** と **Implicit (hybrid)** の両方を選択
    - Sign-in redirect URIs: https://YOUR_W_AND_B_URL/oidc/callback
    - Sign-out redirect URIs: https://YOUR_W_AND_B_URL/logout
    - Assignments: **Skip group assignment for now** を選択
    {{< img src="/images/hosting/okta_new_single_page_app_integration.png" alt="Single-Page App configuration" >}}

6. 作成した Okta アプリケーションの overview 画面で、**General** タブの **Client Credentials** にある **Client ID** をメモしてください。
    {{< img src="/images/hosting/okta_make_note_of_client_id.png" alt="Okta Client ID location" >}}

7. Okta OIDC イシュア URL を特定するには、左側で **Settings** → **Account** を選択してください。Okta UI で **Organization Contact** の会社名が表示されます。
    {{< img src="/images/hosting/okta_identify_oidc_issuer_url.png" alt="Okta organization settings" >}}

OIDC イシュア URL の形式は次の通りです：`https://COMPANY.okta.com`。COMPANY 部分はご自身の値に置き換え、控えておいてください。
{{% /tab %}}

{{% tab header="Entra" value="entra"%}}
1. [Azure Portal](https://portal.azure.com/) にログインします。

2. 「Microsoft Entra ID」サービスを選択してください。
    {{< img src="/images/hosting/entra_select_entra_service.png" alt="Microsoft Entra ID service" >}}

3. 左側メニューから「App registrations」を選択します。
    {{< img src="/images/hosting/entra_app_registrations.png" alt="App registrations menu" >}}

4. 上部の「New registration」をクリックします。
    {{< img src="/images/hosting/entra_new_app_registration.png" alt="New registration button" >}}

    「Register an application」画面で以下の項目を入力してください。
    {{< img src="/images/hosting/entra_register_an_application.png" alt="Application registration form" >}}

    - 名前を入力（例："Weights and Biases application"）
    - デフォルトで 「この組織ディレクトリ内のアカウントのみ (Default Directory only - Single tenant)」が選ばれています。必要に応じて変更可能です。
    - リダイレクト URI：**Web** タイプで `https://YOUR_W_AND_B_URL/oidc/callback` を指定
    - 「Register」をクリック

    - 「Application (client) ID」と「Directory (tenant) ID」を控えておいてください。

      {{< img src="/images/hosting/entra_app_overview_make_note.png" alt="Application and Directory IDs" >}}


5. 左メニューから **Authentication** を選択
    {{< img src="/images/hosting/entra_select_authentication.png" alt="Authentication menu" >}}

    - **Front-channel logout URL** に `https://YOUR_W_AND_B_URL/logout` を指定
    - 「Save」をクリック

      {{< img src="/images/hosting/entra_logout_url.png" alt="Front-channel logout URL" >}}


6. 左メニューから「Certificates & secrets」を選択
    {{< img src="/images/hosting/entra_select_certificates_secrets.png" alt="Certificates & secrets menu" >}}

    - 「Client secrets」をクリックし、「New client secret」を押してください
      {{< img src="/images/hosting/entra_new_secret.png" alt="New client secret button" >}}

      「Add a client secret」画面で以下を入力
      {{< img src="/images/hosting/entra_add_new_client_secret.png" alt="Client secret configuration" >}}

      - 説明例："wandb"
      - 「Expires」は必要なら変更
      - 「Add」をクリック

    - シークレットの「値」を控えてください（「Secret ID」は不要です）
    {{< img src="/images/hosting/entra_make_note_of_secret_value.png" alt="Client secret value" >}}

控えておくべき値は3つです：
- OIDC Client ID
- OIDC Client Secret
- Tenant ID（OIDC イシュア URL 作成時に必要）

OIDC イシュア URL の形式は：`https://login.microsoftonline.com/${TenantID}/v2.0`
{{% /tab %}}
{{< /tabpane >}}

## W&B Server で SSO を設定する

SSO をセットアップするには、管理者権限と以下の情報が必要です。
- OIDC Client ID
- OIDC Auth method（`implicit` または `pkce`）
- OIDC Issuer URL
- OIDC Client Secret（任意、IdP の設定に応じて）

IdP が OIDC Client Secret を必要とする場合は、[環境変数]({{< relref path="/guides/hosting/env-vars.md" lang="ja" >}}) `OIDC_CLIENT_SECRET` で指定します。
- UI の場合、**System Console** > **Settings** > **Advanced** > **User Spec** へ進み、下記例のように `extraENV` セクションに `OIDC_CLIENT_SECRET` を追加します。
- Helm の場合は、下記のように `values.global.extraEnv` を設定してください。

```yaml
values:
  global:
    extraEnv:
      OIDC_CLIENT_SECRET="<your_secret>"
```

{{% alert %}}
SSO 設定後にインスタンスへログインできない場合、`LOCAL_RESTORE=true` の環境変数をセットしてインスタンスを再起動してください。この操作で一時パスワードがコンテナのログに出力され、SSO が無効になります。SSO に関する問題が解決したら、変数を削除して SSO を再度有効にしてください。
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab header="System Console" value="console" %}}
System Console は System Settings ページの後継です。[W&B Kubernetes Operator]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/" lang="ja" >}}) ベースのデプロイメントで利用できます。

1. [W&B 管理コンソールへのアクセス方法]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/#access-the-wb-management-console" lang="ja" >}}) を参照してください。

2. **Settings** → **Authentication** へ移動し、**Type** ドロップダウンで **OIDC** を選択します。
    {{< img src="/images/hosting/sso_configure_via_console.png" alt="System Console OIDC configuration" >}}

3. 必要な値を入力します。

4. **Save** をクリックします。

5. 一度ログアウトし、今回は IdP のログイン画面で再度ログインしてください。
{{% /tab %}}
{{% tab header="System settings" value="settings" %}}
1. Weights & Biases インスタンスにサインインします。
2. W&B アプリへ移動してください。

    {{< img src="/images/hosting/system_settings.png" alt="W&B App navigation" >}}

3. ドロップダウンリストから **System Settings** を選択します。

    {{< img src="/images/hosting/system_settings_select_settings.png" alt="System Settings dropdown" >}}

4. Issuer、Client ID、および Authentication Method を入力します。
5. **Update settings** を選択します。

{{< img src="/images/hosting/system_settings_select_update.png" alt="Update settings button" >}}
{{% /tab %}}
{{< /tabpane >}}

{{% alert %}}
SSO 設定後にインスタンスにログインできなくなった場合、`LOCAL_RESTORE=true` の環境変数をセットしてインスタンスを再起動してください。これで一時的なパスワードがコンテナログに出力され、SSO が無効になります。問題が解決したら、必ずこの環境変数を削除し SSO を再有効化してください。
{{% /alert %}}

## Security Assertion Markup Language (SAML)
W&B Server は SAML をサポートしていません。
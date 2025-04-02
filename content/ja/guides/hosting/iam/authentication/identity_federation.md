---
title: Use federated identities with SDK
menu:
  default:
    identifier: ja-guides-hosting-iam-authentication-identity_federation
    parent: authentication
---

Identity federation を使用して、W&B SDK 経由で組織の認証情報を使用してサインインします。W&B organization の管理者が organization 向けに SSO を設定している場合、すでに組織の認証情報を使用して W&B アプリの UI にサインインしているはずです。その意味で、identity federation は W&B SDK の SSO のようなものですが、JSON Web Tokens (JWT) を直接使用します。identity federation は、APIキー の代替として使用できます。

[RFC 7523](https://datatracker.ietf.org/doc/html/rfc7523) は、SDK との identity federation の基礎を形成します。

{{% alert %}}
Identity federation は、すべてのプラットフォームタイプ（SaaS Cloud、Dedicated Cloud、および Self-managed インスタンス）の `Enterprise` プランで `Preview` として利用できます。ご不明な点がございましたら、W&B チームにお問い合わせください。
{{% /alert %}}

{{% alert %}}
このドキュメントでは、`identity provider` と `JWT issuer` という用語は同じ意味で使用されます。どちらも、この機能のコンテキストでは同じものを指します。
{{% /alert %}}

## JWT issuer の設定

最初の手順として、organization の管理者は、W&B organization と公開されている JWT issuer の間の federation を設定する必要があります。

* organization の ダッシュボード で **Settings** タブに移動します
* **Authentication** オプションで、`Set up JWT Issuer` を押します
* テキストボックスに JWT issuer の URL を追加し、`Create` を押します

W&B は、`${ISSUER_URL}/.well-known/oidc-configuration` のパスで OIDC discovery document を自動的に検索し、discovery document 内の関連する URL で JSON Web Key Set (JWKS) を見つけようとします。JWKS は、JWT が関連する identity provider によって発行されたことを確認するために、JWT のリアルタイム検証に使用されます。

## JWT を使用して W&B にアクセスする

JWT issuer が W&B organization 用に設定されると、ユーザーはその identity provider によって発行された JWT を使用して、関連する W&B プロジェクト へのアクセスを開始できます。JWT を使用するメカニズムは次のとおりです。

* organization で利用可能なメカニズムのいずれかを使用して、identity provider にサインインする必要があります。一部のプロバイダーには、API または SDK を使用して自動化された方法でアクセスできますが、関連する UI を使用してのみアクセスできるプロバイダーもあります。詳細については、W&B organization の管理者または JWT issuer の所有者にお問い合わせください。
* identity provider へのサインイン後に JWT を取得したら、安全な場所にファイルに保存し、環境変数 `WANDB_IDENTITY_TOKEN_FILE` に絶対ファイルパスを設定します。
* W&B SDK または CLI を使用して W&B project にアクセスします。SDK または CLI は、JWT を自動的に検出し、JWT が正常に検証された後、W&B access token と交換する必要があります。W&B access token は、AI ワークフローを有効にするための関連する API にアクセスするために使用されます。つまり、run、メトリクス、Artifacts などを ログ に記録します。access token は、デフォルトで `~/.config/wandb/credentials.json` のパスに保存されます。環境変数 `WANDB_CREDENTIALS_FILE` を指定することで、そのパスを変更できます。

{{% alert %}}
JWT は、APIキー、パスワードなどの有効期間の長い認証情報の欠点に対処するための有効期間の短い認証情報です。identity provider で設定された JWT の有効期限に応じて、JWT を継続的に更新し、環境変数 `WANDB_IDENTITY_TOKEN_FILE` で参照されるファイルに保存されていることを確認する必要があります。

W&B access token にもデフォルトの有効期限があり、その後、SDK または CLI は JWT を使用して自動的に更新を試みます。その時点でユーザー JWT も期限切れになり、更新されない場合、認証エラーが発生する可能性があります。可能であれば、JWT の取得と有効期限後の更新メカニズムは、W&B SDK または CLI を使用する AI ワークロードの一部として実装する必要があります。
{{% /alert %}}

### JWT の検証

JWT を W&B access token と交換し、project にアクセスする ワークフロー の一部として、JWT は次の検証を受けます。

* JWT 署名は、W&B organization レベルで JWKS を使用して検証されます。これは最初の防御線であり、これが失敗した場合、JWKS または JWT の署名方法に問題があることを意味します。
* JWT の `iss` クレームは、organization レベルで設定された issuer URL と同じである必要があります。
* JWT の `sub` クレームは、W&B organization で設定されているユーザーのメールアドレスと同じである必要があります。
* JWT の `aud` クレームは、AI ワークフローの一部としてアクセスしている project を収容する W&B organization の名前と同じである必要があります。[Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンスの場合、インスタンスレベルの環境変数 `SKIP_AUDIENCE_VALIDATION` を `true` に設定して、オーディエンスクレームの検証をスキップするか、オーディエンスとして `wandb` を使用できます。
* JWT の `exp` クレームは、トークンが有効かどうか、または期限切れで更新が必要かどうかを確認するためにチェックされます。

## 外部サービスアカウント

W&B は、有効期間の長い APIキー を持つ組み込みのサービスアカウントを長年サポートしてきました。SDK および CLI 向けの identity federation 機能を使用すると、organization レベルで設定されているのと同じ issuer によって発行されている限り、JWT を認証に使用できる外部サービスアカウントも導入できます。team 管理者は、組み込みのサービスアカウントと同様に、team のスコープ内で外部サービスアカウントを設定できます。

外部サービスアカウントを設定するには:

* team の **Service Accounts** タブに移動します
* `New service account` を押します
* サービスアカウントの名前を入力し、`Authentication Method` として `Federated Identity` を選択し、`Subject` を入力して、`Create` を押します

外部サービスアカウントの JWT の `sub` クレームは、team 管理者が team レベルの **Service Accounts** タブでサブジェクトとして設定したものと同じである必要があります。そのクレームは、[JWT の検証]({{< relref path="#jwt-validation" lang="ja" >}}) の一部として検証されます。`aud` クレームの要件は、ヒューマンユーザー JWT の要件と同様です。

[外部サービスアカウントの JWT を使用して W&B にアクセスする]({{< relref path="#using-the-jwt-to-access-wb" lang="ja" >}}) 場合、通常は、初期 JWT を生成し、継続的に更新する ワークフロー を自動化する方が簡単です。外部サービスアカウントを使用して ログ に記録された run をヒューマンユーザーに帰属させたい場合は、組み込みのサービスアカウントの場合と同様に、AI ワークフローの環境変数 `WANDB_USERNAME` または `WANDB_USER_EMAIL` を設定できます。

{{% alert %}}
W&B は、柔軟性と簡素さのバランスを取るために、データ感度のレベルが異なる AI ワークロード全体で、組み込みおよび外部サービスアカウントを組み合わせて使用することをお勧めします。
{{% /alert %}}

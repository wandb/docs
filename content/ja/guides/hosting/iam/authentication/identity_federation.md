---
title: SDK でフェデレーテッドアイデンティティを使用する
menu:
  default:
    identifier: identity_federation
    parent: authentication
---

W&B SDK を使って、組織の認証情報を用いたアイデンティティフェデレーションでサインインできます。あなたの W&B 組織の管理者が組織の SSO を設定していれば、すでに W&B アプリの UI には組織の認証情報でサインインしています。この意味で、アイデンティティフェデレーションは W&B SDK 用の SSO のようなものですが、JSON Web Token (JWT) を直接使います。APIキーの代わりにアイデンティティフェデレーションを利用できます。

SDK でのアイデンティティフェデレーションは [RFC 7523](https://datatracker.ietf.org/doc/html/rfc7523) に準拠しています。

{{% alert %}}
アイデンティティフェデレーションは、すべてのプラットフォームタイプ（SaaS クラウド、専用クラウド、セルフ管理インスタンス）の `Enterprise` プランで `プレビュー` として提供中です。ご不明な点は W&B の担当チームまでお問い合わせください。
{{% /alert %}}

{{% alert %}}
このドキュメントでは、`identity provider` と `JWT issuer` という用語は同じ意味で使われています。この機能に関しては両者は同一です。
{{% /alert %}}

## JWT issuer のセットアップ

まず最初に、組織の管理者が W&B 組織と公開アクセシブルな JWT issuer との間でフェデレーションを設定します。

* 組織のダッシュボードにある **Settings** タブへ移動します
* **Authentication** オプションで `Set up JWT Issuer` をクリック
* テキストボックスに JWT issuer の URL を入力して `Create` を押します

W&B は自動で `${ISSUER_URL}/.well-known/oidc-configuration` パスにある OIDC ディスカバリドキュメントを参照し、その中にある JSON Web Key Set (JWKS) を取得します。JWKS は JWT が正しい identity provider から発行されたかどうかをリアルタイムで検証するために使われます。

## JWT を使った W&B へのアクセス

一度 W&B 組織に JWT issuer が設定されると、その identity provider で発行された JWT を使ってユーザーは関連する W&B Projects へアクセスできます。JWT 利用の流れは次の通りです：

* 組織内で利用可能な認証方式のひとつを使って identity provider へサインインしてください。いくつかのプロバイダは API や SDK を使った自動取得が可能ですが、UI からしか取得できない場合もあります。詳細は W&B 組織管理者または JWT issuer の管理者へご確認ください。
* identity provider へのサインイン後に取得した JWT を、安全な場所のファイルに保存し、絶対パスを環境変数 `WANDB_IDENTITY_TOKEN_FILE` に設定してください。
* W&B SDK または CLI から W&B Project にアクセスします。SDK または CLI は JWT を自動検出し、検証が成功すれば JWT と引き換えに W&B アクセストークンを取得します。この W&B アクセストークンによって AI ワークフロー用の API へのアクセス（runのログ、メトリクス、Artifactsの操作等）が可能となります。アクセストークンはデフォルトで `~/.config/wandb/credentials.json` に保存されます。保存場所の変更は環境変数 `WANDB_CREDENTIALS_FILE` で設定可能です。

{{% alert %}}
JWT は長寿命の APIキーやパスワードなどの課題を解決するため、短命の認証情報として設計されています。identity provider 側で設定された JWT の有効期限に応じて、定期的に JWT をリフレッシュし、環境変数 `WANDB_IDENTITY_TOKEN_FILE` で参照しているファイルに保存してください。

W&B アクセストークンもデフォルトの有効期限があり、それを過ぎると SDK や CLI が自動的に JWT を使ってリフレッシュを試みます。その時ユーザーの JWT も有効期限切れの場合、自動リフレッシュできずに認証失敗となる可能性があります。可能であれば、JWT の取得・リフレッシュ処理を W&B SDK や CLI を利用している AI ワークロード内で自動化してください。
{{% /alert %}}

### JWT の検証

JWT を W&B アクセストークンへ交換し、さらにプロジェクトへアクセスするワークフローの一環として、JWT には以下の検証が実施されます：

* JWT の署名が W&B 組織レベルの JWKS で検証されます。これが最初の防御線であり、ここで失敗した場合は JWKS または JWT の署名方法に問題があります。
* JWT の `iss` クレームが組織レベルで設定された issuer URL と一致する必要があります。
* JWT の `sub` クレームが W&B 組織で登録されているユーザーのメールアドレスと一致する必要があります。
* JWT の `aud` クレームが、AI ワークフローを通じてアクセスする W&B 組織名と一致する必要があります。[Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}) や [Self-managed]({{< relref "/guides/hosting/hosting-options/self-managed.md" >}}) インスタンスの場合、インスタンスレベルの環境変数 `SKIP_AUDIENCE_VALIDATION` を `true` に設定するか、audience に `wandb` を指定してこの検証をスキップできます。
* JWT の `exp` クレームがチェックされ、トークンの有効性が確認されます。有効期限切れの場合はリフレッシュが必要です。

## 外部サービスアカウント

W&B では以前から長寿命 APIキーを利用する組み込みのサービスアカウントが利用可能でした。SDK や CLI におけるアイデンティティフェデレーション機能により、JWT を使った認証が可能な外部サービスアカウントも利用できるようになりました（ただし、組織レベルで設定した同じ issuer から発行される必要があります）。チーム管理者は、組み込みサービスアカウントと同様に、チームのスコープで外部サービスアカウントを設定できます。

外部サービスアカウントの設定手順：

* チームの **Service Accounts** タブに移動します
* `New service account` をクリック
* サービスアカウント名を入力し、`Authentication Method` として `Federated Identity` を選択し、`Subject` を入力して `Create` を押します

外部サービスアカウントの JWT における `sub` クレームは、チームレベルの **Service Accounts** タブでチーム管理者が設定した subject と一致させる必要があります。このクレームは [JWT の検証]({{< relref "#jwt-validation" >}}) で確認されます。`aud` クレームも人間のユーザー用 JWT と同様の要件となります。

[外部サービスアカウントの JWT で W&B にアクセスする場合]({{< relref "#using-the-jwt-to-access-wb" >}})、初回 JWT の発行やリフレッシュをワークフローに組み込むことで自動化しやすくなります。外部サービスアカウントで記録した run を人間のユーザーに帰属させたい場合、`WANDB_USERNAME` または `WANDB_USER_EMAIL` などの環境変数を AI ワークフローに設定できます。これは組み込みサービスアカウントと同様の方法です。

{{% alert %}}
W&B では、データの機密性レベルに応じて、組み込みサービスアカウントと外部サービスアカウントの組み合わせ利用を推奨しています。これにより柔軟性とシンプルさのバランスが取れます。
{{% /alert %}}
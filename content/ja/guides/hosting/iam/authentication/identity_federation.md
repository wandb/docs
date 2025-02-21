---
title: Use federated identities with SDK
menu:
  default:
    identifier: ja-guides-hosting-iam-authentication-identity_federation
    parent: authentication
---

Use identity federation を使用して、W&B SDK を通じて組織の認証情報を使用してサインインします。あなたの W&B 組織管理者が組織のために SSO を設定している場合、あなたはすでにあなたの組織の認証情報を使用して W&B アプリ UI にサインインしています。その意味では、identity federation は W&B SDK のための SSO のようなものですが、JSON Web Tokens (JWTs) を直接使用します。identity federation を APIキー の代替として使用することができます。

[RFC 7523](https://datatracker.ietf.org/doc/html/rfc7523) は、SDK における identity federation の基盤を形成します。

{{% alert %}}
Identity federation は、すべてのプラットフォームタイプの `Enterprise` プランで利用可能な `Preview` として提供されています - SaaS Cloud, 専用クラウド, および自己管理インスタンスです。質問があれば、あなたの W&B チームに問い合わせてください。
{{% /alert %}}

{{% alert %}}
このドキュメントの目的では、用語 `identity provider` と `JWT issuer` は同義として使用されています。この機能の文脈では、両方とも同じことを指します。
{{% /alert %}}

## JWT issuer 設定

最初のステップとして、組織管理者はあなたの W&B 組織と公開アクセス可能な JWT issuer 間の federation を設定する必要があります。

* 組織ダッシュボードの **設定** タブに移動します
* **認証** オプションで `JWT issuer を設定` を押します
* テキストボックスに JWT issuer の URL を追加し、`作成` を押します

W&B は自動的にパス `${ISSUER_URL}/.well-known/oidc-configuration` の OIDC ディスカバリードキュメントを探し、ディスカバリードキュメント内の関連する URL で JSON Web Key セット (JWKS) を探します。JWKS は、JWT が関連する identity provider によって発行されたことを確認するためのリアルタイム検証に使用されます。

## JWT を使用して W&B にアクセスする

一旦 JWT issuer があなたの W&B 組織に対して設定されると、ユーザーはその identity provider から発行された JWT を使用して関連する W&B プロジェクトにアクセスを開始できます。JWT 使用のメカニズムは以下の通りです:

* 組織内で利用可能なメカニズムの1つを使用して identity provider にサインインする必要があります。API や SDK を使用して自動化された方法でアクセスできるプロバイダーもあれば、関連する UI を使用してのみアクセスできるプロバイダーもあります。詳細は、あなたの W&B 組織管理者または JWT issuer オーナーにお問い合わせください。
* identity provider にサインインした後に JWT を取得したら、それを安全な場所にファイルとして保存し、環境変数 `WANDB_IDENTITY_TOKEN_FILE` に絶対ファイルパスを設定します。
* W&B SDK や CLI を使用してあなたの W&B プロジェクトにアクセスします。SDK や CLI は自動的に JWT を検出し、それが正常に検証された後に W&B アクセストークンと交換するはずです。W&B アクセストークンは、AI ワークフローを有効にするために、すなわち run、メトリクス、アーティファクトなどをログするために使用される関連する API にアクセスするために使用されます。アクセストークンはデフォルトでパス `~/.config/wandb/credentials.json` に保存されます。環境変数 `WANDB_CREDENTIALS_FILE` を指定することでパスを変更できます。

{{% alert %}}
JWT は、APIキー、パスワードなどの長期的な認証情報の欠点に対処するための短期的な認証情報として設計されています。identity provider に設定された JWT の有効期限に応じて、JWT を継続的に更新し、それが環境変数 `WANDB_IDENTITY_TOKEN_FILE` にリファレンスされたファイルに保存されるようにする必要があります。

W&B アクセストークンにもデフォルトの有効期限があり、それを過ぎると、SDK または CLI は JWT を使用して自動的にそれを更新しようとします。その時点でユーザーの JWT も期限切れで更新されていない場合、認証エラーが発生する可能性があります。可能であれば、JWT の取得と有効期限後の更新メカニズムを W&B SDK または CLI を使用する AI ワークロードの一部として実装する必要があります。
{{% /alert %}}

### JWT validation

W&B アクセストークンとプロジェクトにアクセスするために JWT を交換するワークフローの一環として、JWT は以下の検証を受けます:

* JWT の署名は W&B 組織レベルで JWKS を使用して検証されます。これは防御の第一線であり、これが失敗する場合、あなたの JWKS または JWT の署名方法に問題があることを意味します。
* JWT の `iss` クレームは、組織レベルで設定された issuer URL と一致している必要があります。
* JWT の `sub` クレームは、W&B 組織で設定されたユーザーのメールアドレスと一致している必要があります。
* JWT の `aud` クレームは、AI ワークフローの一部としてアクセスしているプロジェクトを管理する W&B 組織の名前と一致している必要があります。専用クラウドまたは自己管理インスタンスの場合、インスタンスレベルの環境変数 `SKIP_AUDIENCE_VALIDATION` を `true` に設定してオーディエンスクレームの検証をスキップするか、`wandb` をオーディエンスとして使用することができます。
* JWT の `exp` クレームがトークンの有効期限を過ぎているかどうかを確認し、更新が必要か確認されます。

## 外部サービスアカウント

W&B は長期間有効な APIキー を使用した組み込みのサービスアカウントを長期間サポートしてきました。SDK および CLI の identity federation 機能により、組織レベルで設定された同じ issuer によって発行された JWT を使用する外部サービスアカウントを導入することができます。チーム管理者は組み込みのサービスアカウントのように、チームの範囲内で外部サービスアカウントを設定することができます。

外部サービスアカウントを設定するには:

* **サービスアカウント** タブに移動します
* `新しいサービスアカウント` を押します
* サービスアカウントの名前を指定し、`Federated Identity` を `Authentication Method` として選択し、`Subject` を入力して `作成` を押します

外部サービスアカウントの JWT の `sub` クレームは、チーム管理者がチームレベルの **サービスアカウント** タブでその subject として設定したものと同一である必要があります。このクレームは [JWT validation]({{< relref path="#jwt-validation" lang="ja" >}}) の一部として検証されます。`aud` クレームの要件は、人間のユーザー JWT の場合と類似しています。

[外部サービスアカウントの JWT を使用して W&B にアクセスする]({{< relref path="#using-the-jwt-to-access-wb" lang="ja" >}}) ように、自動化されたワークフローを生成して最初の JWT を生成し、継続的に更新するのが通常は容易です。外部サービスアカウントを使用してログされた run を人間のユーザーに帰属させたい場合、組み込みのサービスアカウントで行うのと同様に、環境変数 `WANDB_USERNAME` または `WANDB_USER_EMAIL` を AI ワークフローに設定することができます。

{{% alert %}}
W&B は、異なるデータ感度レベルを持つあなたの AI ワークロード全体で、組み込みと外部サービスアカウントの組み合わせを使用して、柔軟性とシンプルさのバランスを取ることを推奨しています。
{{% /alert %}}
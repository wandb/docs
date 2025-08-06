---
title: SDK でフェデレーテッドアイデンティティを使う
menu:
  default:
    identifier: ja-guides-hosting-iam-authentication-identity_federation
    parent: authentication
---

W&B SDK を使って組織の資格情報でサインインするには、アイデンティティフェデレーションを利用します。すでに W&B の組織管理者が組織向けの SSO を設定している場合は、W&B アプリ UI へのサインインに組織の資格情報を使っているはずです。つまり、アイデンティティフェデレーションは W&B SDK 版の SSO のようなもので、JSON Web Token (JWT) を直接利用します。APIキーの代替としてアイデンティティフェデレーションを利用できます。

[RFC 7523](https://datatracker.ietf.org/doc/html/rfc7523) が SDK 向けアイデンティティフェデレーションの基本仕様になっています。

{{% alert %}}
アイデンティティフェデレーションは、すべてのプラットフォームタイプ（SaaS クラウド、専用クラウド、セルフマネージドインスタンス）の `Enterprise` プラン向けに `Preview` 機能としてご利用いただけます。ご質問があれば W&B チームにご連絡ください。
{{% /alert %}}

{{% alert %}}
このドキュメントでは `identity provider` と `JWT issuer` を同じ意味で使用しています。どちらもこの機能の文脈では同一のものを指します。
{{% /alert %}}

## JWT issuer のセットアップ

最初のステップとして、組織管理者が W&B の組織とパブリックにアクセス可能な JWT issuer のフェデレーション設定を行う必要があります。

* 組織ダッシュボードの **Settings** タブへ移動
* **Authentication** オプションで `Set up JWT Issuer` をクリック
* テキストボックスに JWT issuer の URL を入力し `Create` をクリック

W&B は自動的に `${ISSUER_URL}/.well-known/oidc-configuration` にある OIDC Discovery ドキュメントを参照し、ドキュメント内に記載された JSON Web Key Set (JWKS) の URL を探します。JWKS は、JWT のリアルタイム検証のために利用され、発行元が適切な Identity Provider であることを保証します。

## JWT を利用した W&B へのアクセス

JWT issuer が組織の W&B に対して設定されれば、ユーザーはそのアイデンティティプロバイダーから発行された JWT を使って該当する W&B Projects へアクセスできるようになります。JWT の利用フローは以下の通りです。

* 組織で利用可能な方法を使って Identity Provider へサインインします。プロバイダーによっては API や SDK を使って自動化できますが、専用の UI からしかアクセスできない場合もあります。詳細は W&B 組織管理者や JWT issuer の管理者にご確認ください。
* Identity Provider へのサインイン後に取得した JWT を安全な場所にファイルで保存し、その絶対パスを環境変数 `WANDB_IDENTITY_TOKEN_FILE` に設定します。
* W&B SDK または CLI を使って W&B Project にアクセスします。SDK や CLI は自動的に JWT を検出し、JWT が正しく認証されたあとに W&B アクセストークンと交換します。W&B アクセストークンは、AI ワークフローのための API アクセス（run、メトリクス、Artifacts などのログ用）に使用されます。デフォルトでは `~/.config/wandb/credentials.json` に保存されますが、環境変数 `WANDB_CREDENTIALS_FILE` で保存先を指定できます。

{{% alert %}}
JWT は APIキー や パスワードなど長期利用の資格情報が持つ課題を解決するための短命な資格情報です。Identity Provider で設定されている JWT の有効期限に応じて、JWT を継続的にリフレッシュし、環境変数 `WANDB_IDENTITY_TOKEN_FILE` で指定したファイルに常に最新の JWT を保存してください。

W&B アクセストークンにもデフォルトの有効期限があります。有効期限切れ後は SDK または CLI が自動的に JWT を使って再取得を試みます。その際、ユーザーの JWT もすでに有効期限切れでリフレッシュされていない場合、認証エラーとなる可能性があります。可能であれば、JWT 取得およびリフレッシュの仕組みも W&B SDK または CLI を利用する AI ワークロードに組み込んでください。
{{% /alert %}}

### JWT の検証

JWT と W&B アクセストークンの交換、さらに Project へのアクセスワークフローでは、JWT に対し以下の検証が行われます。

* JWT の署名は、W&B 組織レベルで JWKS を使い検証されます。これが最初の防御線で、失敗した場合は JWKS または JWT の署名方法に問題があります。
* JWT 内の `iss` クレームが組織に設定された issuer URL と一致している必要があります。
* JWT 内の `sub` クレームが W&B 組織に登録されたユーザーのメールアドレスと一致している必要があります。
* JWT 内の `aud` クレームが、その AI ワークフローでアクセスするプロジェクトを有する W&B 組織名と一致している必要があります。[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) や [セルフマネージド]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) インスタンスの場合、インスタンスレベルの環境変数 `SKIP_AUDIENCE_VALIDATION` を `true` に設定して audience 検証を省略したり、`wandb` を audience に指定することも可能です。
* JWT 内の `exp` クレームで有効期限切れかどうかを確認し、切れていればリフレッシュが必要です。

## 外部サービスアカウント

W&B は従来、長期利用の APIキー を使った組み込みサービスアカウントをサポートしてきました。SDK や CLI のアイデンティティフェデレーション機能により、同一の issuer から発行された JWT を使って認証する外部サービスアカウントも利用できるようになります。チーム管理者は組み込みサービスアカウント同様、チーム単位で外部サービスアカウントを設定できます。

外部サービスアカウントを設定するには:

* チームの **Service Accounts** タブに移動
* `New service account` をクリック
* サービスアカウント名を入力し、`Authentication Method` として `Federated Identity` を選択、`Subject` を設定して `Create` をクリック

外部サービスアカウントの JWT 内の `sub` クレームは、チームレベルの **Service Accounts** タブで管理者が設定した Subject と一致する必要があります。そのクレームは [JWT の検証]({{< relref path="#jwt-validation" lang="ja" >}}) の一部として確認されます。`aud` クレームの要件も、通常のユーザーの JWT と同様です。

[外部サービスアカウントの JWT を使って W&B にアクセスする場合]({{< relref path="#using-the-jwt-to-access-wb" lang="ja" >}})、初回の JWT の生成や継続的なリフレッシュを自動化しやすいのが一般的です。外部サービスアカウントで記録した run を特定のユーザーに紐付けたい場合は、AI ワークフローで `WANDB_USERNAME` または `WANDB_USER_EMAIL` の環境変数を設定できます。これは組み込みサービスアカウントの場合と同じやり方です。

{{% alert %}}
W&B では、AI ワークロード全体で機密度に応じて組み込み型と外部サービスアカウントを使い分けることで、柔軟性とシンプルさのバランスをとることを推奨しています。
{{% /alert %}}
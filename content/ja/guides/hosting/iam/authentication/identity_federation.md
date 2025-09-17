---
title: SDK で フェデレーテッド ID を使用する
menu:
  default:
    identifier: ja-guides-hosting-iam-authentication-identity_federation
    parent: authentication
---

W&B SDK を通じて組織の認証情報でサインインするために、アイデンティティフェデレーションを使います。W&B の Organization 管理者が組織向けに SSO を設定している場合、すでに W&B のアプリ UI へのサインインに組織の認証情報を使っています。その意味で、アイデンティティフェデレーションは W&B SDK 向けの SSO のようなもので、JSON Web Tokens (JWTs) を直接用います。API キーの代替としてアイデンティティフェデレーションを利用できます。

[RFC 7523](https://datatracker.ietf.org/doc/html/rfc7523) は、SDK におけるアイデンティティフェデレーションの基盤となる仕様です。

{{% alert %}}
アイデンティティフェデレーションは、すべてのプラットフォームタイプ（SaaS クラウド、専用クラウド、セルフマネージドのインスタンス）の `Enterprise` プランで `Preview` 提供中です。ご不明点は W&B の担当チームにお問い合わせください。
{{% /alert %}}

{{% alert %}}
本ドキュメントでは、`identity provider` と `JWT issuer` という用語を同義として扱います。本機能の文脈ではどちらも同じものを指します。
{{% /alert %}}

## JWT issuer の設定

まず最初に、Organization の管理者が、あなたの W&B Organization と、公開アクセス可能な JWT issuer の間でフェデレーションを設定する必要があります。

* 組織のダッシュボードの **Settings** タブに移動します
* **Authentication** の項目で `Set up JWT Issuer` を押します
* テキストボックスに JWT issuer の URL を入力し、`Create` を押します

W&B は `${ISSUER_URL}/.well-known/openid-configuration` のパスで OIDC のディスカバリードキュメントを自動的に探し、そこに記載された関連 URL から JSON Web Key Set (JWKS) を取得しようとします。JWKS は JWT が適切な identity provider によって発行されたことをリアルタイムで検証するために使用されます。

## JWT を使って W&B にアクセスする

W&B Organization に対して JWT issuer が設定されると、その issuer が発行する JWT を使って、ユーザーは該当する W&B の Projects にアクセスできるようになります。JWT の利用手順は次のとおりです。

* 組織で利用可能な方法のいずれかで identity provider にサインインします。プロバイダによっては API や SDK で自動的にアクセスできる場合もあれば、専用の UI からのみアクセスできる場合もあります。詳細は W&B の Organization 管理者または JWT issuer の管理者に問い合わせてください。
* identity provider へのサインイン後に取得した JWT を、安全な場所のファイルに保存し、その絶対パスを環境変数 `WANDB_IDENTITY_TOKEN_FILE` に設定します。
* W&B SDK または CLI を使って W&B の Project にアクセスします。SDK または CLI は JWT を自動検出し、JWT の検証に成功すると W&B のアクセストークンに交換します。W&B のアクセストークンは、AI ワークフローに必要な各種 API（たとえば Runs、メトリクス、Artifacts をログする等）へのアクセスに使用されます。アクセストークンはデフォルトで `~/.config/wandb/credentials.json` に保存されます。保存先は環境変数 `WANDB_CREDENTIALS_FILE` で変更できます。

{{% alert %}}
JWT は、API キーやパスワードなどの長期クレデンシャルの欠点を補うための短命なクレデンシャルとして設計されています。identity provider で設定された JWT の有効期限に応じて、JWT を継続的にリフレッシュし、その内容が環境変数 `WANDB_IDENTITY_TOKEN_FILE` が参照するファイルに保存されていることを必ず確認してください。

W&B のアクセストークンにもデフォルトの有効期限があり、期限後は SDK または CLI があなたの JWT を使って自動的に更新を試みます。その時点でユーザーの JWT も期限切れで更新されていない場合、認証エラーになる可能性があります。可能であれば、W&B SDK または CLI を使用する AI ワークロードの一部として、JWT の取得と期限切れ後の更新を自動化する実装を行うことを推奨します。
{{% /alert %}}

### JWT の検証

JWT を W&B のアクセストークンに交換して Project にアクセスするワークフローの一環として、JWT には次の検証が行われます。

* JWT の署名は、W&B Organization レベルの JWKS を使って検証されます。これは第一の防御線であり、ここで失敗する場合は、JWKS か JWT の署名方法に問題があることを意味します。
* JWT の `iss` クレームは、Organization レベルで設定した issuer URL と一致している必要があります。
* JWT の `sub` クレームは、W&B Organization で設定されているユーザーのメールアドレスと一致している必要があります。
* JWT の `aud` クレームは、AI ワークフローの一環としてアクセスする Project を保持する W&B Organization の名前と一致している必要があります。[専用クラウド]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ja" >}}) または [セルフマネージド]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) のインスタンスでは、インスタンスレベルの環境変数 `SKIP_AUDIENCE_VALIDATION` を `true` に設定して audience クレームの検証をスキップするか、audience として `wandb` を使用できます。
* JWT の `exp` クレームを確認し、トークンが有効か、期限切れで更新が必要かをチェックします。

## 外部サービスアカウント

W&B は長年にわたり、長期有効な API キーを持つ組み込みのサービスアカウントをサポートしてきました。SDK と CLI のアイデンティティフェデレーション機能により、Organization レベルで設定されたのと同じ issuer が発行した JWT を使う限り、認証に JWT を用いる外部サービスアカウントも利用できます。Team の管理者は、組み込みのサービスアカウントと同様に、Team のスコープ内で外部サービスアカウントを設定できます。

外部サービスアカウントを設定するには:

* Team の **Service Accounts** タブに移動します
* `New service account` を押します
* サービスアカウントの名前を入力し、`Authentication Method` として `Federated Identity` を選択し、`Subject` を入力して `Create` を押します

外部サービスアカウントの JWT に含まれる `sub` クレームは、Team レベルの **Service Accounts** タブで Team 管理者が Subject として設定した値と同一である必要があります。このクレームは[JWT の検証]({{< relref path="#jwt-validation" lang="ja" >}})の一部として検証されます。`aud` クレームの要件は、人間のユーザーの JWT と同様です。

[外部サービスアカウントの JWT を使って W&B にアクセスする]({{< relref path="#using-the-jwt-to-access-wb" lang="ja" >}})場合、初回の JWT 生成と継続的なリフレッシュを自動化する方が一般的に容易です。外部サービスアカウントでログした Runs の帰属先を人間のユーザーにしたい場合は、組み込みのサービスアカウントと同様に、AI ワークフローで環境変数 `WANDB_USERNAME` または `WANDB_USER_EMAIL` を設定できます。

{{% alert %}}
W&B は、データ機密性のレベルが異なる AI ワークロード全体で、組み込みと外部のサービスアカウントを併用し、柔軟性とシンプルさのバランスを取ることを推奨します。
{{% /alert %}}
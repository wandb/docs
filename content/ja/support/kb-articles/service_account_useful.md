---
title: サービスアカウントとは何か、そしてなぜ便利なのか？
menu:
  support:
    identifier: ja-support-kb-articles-service_account_useful
support:
- 管理者
toc_hide: true
type: docs
url: /support/:filename
---

サービスアカウント（エンタープライズ限定機能）は、人間ではない、もしくは機械によるユーザーを表します。これにより、チームや Projects 間で共通のタスクの自動化や、特定のユーザーに紐付かない作業の自動化が可能です。サービスアカウントは各チーム内で作成でき、その APIキー を使って、そのチーム内の Projects へ読み書きできます。

サービスアカウントは、例えば定期的な再トレーニングやナイトリービルドなど、自動で wandb へログを送信するジョブのトラッキングに役立ちます。必要に応じて、こうしたマシン起動の run に [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) `WANDB_USERNAME` または `WANDB_USER_EMAIL` を指定してユーザー名を紐付けることもできます。

詳細は [Team Service Account の振る舞い]({{< relref path="/guides/models/app/settings-page/teams.md#team-service-account-behavior" lang="ja" >}}) をご覧ください。

チームのサービスアカウント用 APIキー は `<WANDB_HOST_URL>/<your-team-name>/service-accounts` で取得できます。または、チームの **Team settings** から **Service Accounts** タブに進んでください。

新しくサービスアカウントを作成するには:
* チームの **Service Accounts** タブ内で **+ New service account** ボタンを押します
* **Name** 欄に名前を入力します
* 認証方式で **Generate API key (Built-in)** を選択します
* **Create** ボタンを押します
* 作成されたサービスアカウントの **Copy API key** ボタンをクリックし、そのキーをシークレットマネージャや安全かつ アクセス 可能な場所に保管してください

{{% alert %}}
**Built-in** サービスアカウントに加えて、W&B は [SDK・CLI向けのアイデンティティフェデレーション]({{< relref path="/guides/hosting/iam/authentication/identity_federation.md#external-service-accounts" lang="ja" >}})を使った **External service accounts** にも対応しています。アイデンティティプロバイダで管理されていて、JSON Web Token (JWT) を発行できるサービスアイデンティティを使って W&B の作業を自動化したい場合は、External service accounts をご利用ください。
{{% /alert %}}